# probe suite — slurm_selftest and io_probe (adopted from fasrc/reframe-fasrc)

**Date:** 2026-07-03
**Location:** `slurm/`, new top-level `io/`
**Status:** approved design, pre-implementation
**Branch base:** `main` (all prior suites merged).
**Provenance:** concepts adapted from
https://github.com/fasrc/reframe-fasrc (`checks/system/slurm/slurm.py` and
`checks/system/io/ior_check.py`), BSD-licensed, CSCS-derived. Re-cast as
standalone verdict CLIs in this repo's conventions; no ReFrame dependency.

## Purpose

Two quick probes (seconds, not benchmarks):

| Tool | One-liner |
|---|---|
| `slurm/slurm_selftest` | Canary job verifying the scheduler itself behaves: env propagation, GPU grant, GRES accounting, cgroup OOM enforcement, node placement. |
| `io/io_probe` | Quick filesystem responsiveness probe on a target directory: bounded sequential write/read MB/s + small-file metadata latency. |

### Non-goals (deferred to a future benchmark plan)

Per user direction, NO long-running or load-generating benchmarks now. The
reframe-fasrc survey identified these as worth adopting later, with designs
sketched during this round:

- `gpu_burn_check` — gpu-burn / gpu-fryer stress with Gflop/s + temperature
  verdicts and a `gpu_health` before/after bracket. The FASRC gpu-fryer
  image exists at `/n/sw/singularity_images/FAS/gpu-fryer/gpu-fryer_1.1.0.sif`
  (verified 2026-07-03), so the fryer engine needs zero setup when built.
- `gpu_bw_check` — H2D/D2H/D2D and GPU-P2P bandwidth from the BSD-licensed
  CUDA sources in reframe-fasrc's `memory_bandwidth`/`p2p_bandwidth` checks
  (nvcc in-job compile; cuda modules up to 12.9.1 available).
- `io_check` — IOR-based storage benchmark (`ior` module exists), with the
  target directory as a required flag (never hardcoded paths).

Also surveyed and rejected: OSU/halo/HPCG MPI benchmarks (overlap
`ib_bw_pair`/`nccl_check` at a layer this cluster doesn't prioritize), CPU
microbenchmarks (GPU-centric cluster), `software/user_codes` checks
(wget/singularity-pull at runtime, FASRC-workflow-specific).

## Shared conventions

- Python 3.6-compatible stdlib executables, no `.py` extension,
  `#!/usr/bin/env python3`, executable bit. `slurm_selftest` reuses
  `slurm/slurmlib.py`; `io_probe` is self-contained stdlib (new `io/` dir
  has no shared module yet — YAGNI).
- Exit codes: `0` all pass / report-only, `1` a probe FAILed (or a `--min-*`
  gate missed), `3` setup/probe error as `<tool>: error: <msg>` on stderr.
  argparse usage errors stay exit 2.
- Every subprocess call has a timeout; unparseable output is exit 3, never a
  silent pass; skipped sub-checks are reported as SKIP, never dropped.
- Pure parser/verdict cores with replay seams (`--from-file` on
  slurm_selftest replays a captured job output; io_probe's math is pure over
  measured numbers and its CLI is testable against a tmpdir).
- File headers note the reframe-fasrc provenance.

## Tool designs

### slurm/slurm_selftest

```
slurm_selftest -A ACCOUNT -p PARTITION [--gpu] [--node-pattern REGEX]
               [--checks LIST] [--timeout MIN] [--json] [--from-file FILE]
```

- Submits ONE small sbatch job (`-N 1`, 2 tasks where needed, `-t 5`,
  `--mem=1G`; plus `--gres=gpu:1` when `--gpu`). The job script runs the
  sub-checks sequentially, each emitting exactly one marker line to stdout:
  `SELFTEST <name> PASS|FAIL|SKIP <detail>`.
- Sub-checks (default: all applicable; `--checks env,oom` selects):
  - `env` — exports `SELFTEST_VAR=TEST123456` at submit time via sbatch
    `--export=ALL,SELFTEST_VAR=...`; the script runs
    `srun -n 2 bash -c 'echo $SELFTEST_VAR'` and PASSes when both task
    outputs match.
  - `placement` — `hostname -s` matched against `--node-pattern` regex;
    SKIP when the flag is absent.
  - `gpu_grant` (requires `--gpu`) — `nvidia-smi -L` inside the job lists
    ≥1 GPU; FAIL when nvidia-smi errors or lists none; SKIP without
    `--gpu`.
  - `gres_accounting` (requires `--gpu`) — `scontrol show job
    $SLURM_JOB_ID` contains `gpu:1` in `TresPerNode=` (or `Gres=` on older
    Slurm); SKIP without `--gpu`.
  - `oom_enforcement` — a step run as `srun -n 1 --mem=512M python3 -c
    '<allocate ~2GB in 64MB chunks, touching each>'`; PASS when the step
    FAILS (nonzero rc) AND the job/step stderr shows an OOM indicator
    ("Out Of Memory", "oom-kill", "exceeded memory limit", case-insensitive);
    FAIL when the allocator survives (enforcement broken). The allocator
    self-limits at 2GB and exits 0 if it ever completes, so a broken cgroup
    cannot hang the job.
- The tool waits for the job (poll + `--timeout`, default 10 min, cancel on
  expiry → exit 3), reads the job output file, and parses the marker lines
  (pure function `parse_markers(text) -> [{name, status, detail}]`).
  Missing markers for requested checks are reported as
  `NO-RESULT` and count as failures (a crashed script must not pass).
- Output: one line per sub-check + summary; `--json` emits the list.
  Exit 1 when any requested check is FAIL or NO-RESULT; 0 otherwise
  (SKIPs alone don't fail); 3 for submission/timeout/read errors.
- `--from-file FILE` parses a captured job output instead of submitting
  (test seam; window/account flags unused in that mode).

### io/io_probe

```
io_probe --dir TARGET [--size MB] [--meta-files N]
         [--min-write MBS] [--min-read MBS] [--max-meta-ms MS]
         [--keep] [--json]
```

- Runs WHERE INVOKED (no Slurm round-trip, like the fabric/ probes); wrap
  in `srun` to probe from a compute node. Documented explicitly.
- Creates a private scratch subdir `TARGET/.io_probe.<pid>/`; refuses (exit
  3) if TARGET is missing or the subdir can't be created. Always removes
  the subdir afterwards (`finally`), unless `--keep`.
- Measurements (defaults: `--size 256` MB, `--meta-files 100`):
  1. **write**: stream `--size` MB of zeros to one file in 4 MB chunks,
     `flush` + `os.fsync` before stopping the clock → write MB/s.
  2. **read**: re-reads the just-written file in 4 MB chunks → read MB/s.
     This is page-cache-assisted (dropping caches needs root), so the probe
     labels the number honestly — "read is page-cache-assisted on this
     host" in the text output, `read_cached: true` in JSON — rather than
     pretending it measured the backing store.
  3. **metadata**: create `--meta-files` empty files (create+close), stat
     them all, delete them all; report mean ms per create/stat/delete.
- Pure core `verdict(metrics, gates) -> (status, reasons)`: FAIL (exit 1)
  only for gates the user set (`--min-write`, `--min-read`,
  `--max-meta-ms` vs the worst of the three metadata means); report-only
  exit 0 otherwise.
- Output: block of metrics with units + the cache caveat + verdict line;
  `--json` emits `{dir, size_mb, write_mbs, read_mbs, read_cached: true,
  meta_ms: {create, stat, delete}, gates, status}`.
- Total runtime target: < 30 s on a healthy filesystem at defaults (256 MB
  + 300 metadata ops).

## Error handling

- slurm_selftest: sbatch rejection, poll timeout (job cancelled), missing
  or unreadable output file → exit 3. A job that ran but produced no
  markers → every requested check NO-RESULT → exit 1 (the scheduler DID
  misbehave in some way worth flagging, and exit 3 would hide it behind
  "probe error").
- io_probe: unwritable target, scratch-dir creation failure, IO errors
  mid-measurement → exit 3 with the OS error; cleanup always attempted and
  a cleanup failure is a printed warning, never masks the result.

## Testing

Offline stdlib-`unittest`:

- `slurm/tests/test_slurm_selftest.py` — marker parsing (PASS/FAIL/SKIP,
  malformed lines, NO-RESULT synthesis for requested-but-missing checks),
  verdict/exit mapping, `--from-file` CLI paths (0 all-pass, 1 with a FAIL,
  1 with missing markers, 3 missing file), job-script generation asserts
  (markers present for selected checks; no `%`-format residue).
- `io/tests/test_io_probe.py` (new `io/tests/` scaffolding, loader
  pattern) — runs the REAL probe against a tempdir with tiny sizes
  (`--size 1 --meta-files 5`): metrics present and positive, JSON schema,
  gate pass/fail exit codes (set an impossible `--min-write 1e12` → exit
  1), unwritable dir → exit 3, cleanup verified (scratch dir gone),
  `--keep` leaves it.
- Live smoke: `slurm_selftest -A kempner_dev -p kempner_eng --gpu` (full
  battery incl. OOM canary) and `io_probe --dir` on netscratch, both end
  to end.

## Documentation

Top-level `README.md`: `slurm_selftest` row in the slurm/ table; new `io/`
section + directory row in the layout diagram and requirements table
(`io/`: any node, Python 3 stdlib only). Note the reframe-fasrc provenance
in both tool rows.
