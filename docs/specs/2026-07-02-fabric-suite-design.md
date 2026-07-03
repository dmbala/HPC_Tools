# fabric suite expansion — fleet_snapshot, nccl_check, ib_bw_pair, topo_verify

**Date:** 2026-07-02
**Location:** `fabric/`
**Status:** approved design, pre-implementation

## Purpose

Four additions that grow `fabric/` from single-node passive probes into a
fleet-and-benchmark diagnostic suite:

| Tool | One-liner |
|---|---|
| `fleet_snapshot` | Run `ib_snapshot.sh` across a nodelist via Slurm and aggregate: version drift, degraded links, down ports, error counters, unreached nodes. |
| `topo_verify` | Diff one node's current snapshot against a blessed "golden" snapshot; detect hardware/topology drift after maintenance. |
| `nccl_check` | NCCL all-reduce bandwidth smoke test (1 or 2 nodes) with optional pass/fail threshold. |
| `ib_bw_pair` | Point-to-point `ib_write_bw`/`ib_read_bw` between two named nodes with a bandwidth verdict. |

Non-goals: ssh/pdsh fan-out (Slurm only), continuous monitoring daemons,
multi-node NCCL rings beyond 2 nodes, switch-side (ibdiagnet) diagnostics,
inventing a second snapshot format (everything builds on `ib_snapshot.sh`
schema_version 1 JSON).

## Shared conventions

- Python 3.6-compatible stdlib executables in `fabric/`, no `.py` extension,
  `#!/usr/bin/env python3`, executable bit set.
- Exit codes: `0` pass/clean, `1` problem found (drift, anomaly, bandwidth
  below threshold), `3` probe/setup error (missing binary, job never started,
  timeout, unparseable output). Operational failures must never surface as a
  health verdict (same contract as `gpu/gpu_health`).
- Plain-text reports to stdout; `--json` where the output is data.
- Node access is Slurm-only: tools submit/attach with `sbatch`/`srun` using
  `-A <account> -p <partition>` flags. No ssh assumed, no root, read-only
  probes (the benchmarks generate traffic but change no state).
- Architecture per tool: pure parse/compare/aggregate functions + thin Slurm
  runner functions + `main()`. All verdict logic must be reachable offline.

## Tool designs

### fleet_snapshot

```
fleet_snapshot (-p PARTITION | -w NODELIST) [-A ACCOUNT] [--gres GRES]
               [--timeout MIN] [--out DIR] [--json]
```

1. Expand targets: `-w` nodelist via `scontrol show hostnames`, or all nodes
   of `-p` via `sinfo -p P -N -o %N` (deduplicated).
2. Submit one small job per node (`-c 2 --mem=4G -t 5 -w <node>`, plus
   `--gres` only if given) running `fabric/ib_snapshot.sh
   <outdir>/<host>.json`. Output dir defaults to
   `results/snapshots/fleet_<runid>/` where runid is a UTC timestamp.
3. Poll job states via `squeue`/`sacct` until all complete or `--timeout`
   (default 30 min) expires; cancel stragglers, mark those nodes unreached.
4. Aggregate all collected JSONs (pure function over a list of snapshot
   dicts):
   - **version drift**: NVIDIA driver, kernel (`uname`), HCA firmware
     (parsed from `ibv_devinfo` text) — modal value vs outliers;
   - **link anomalies**: per-port `state` != ACTIVE, `phys_state` != LinkUp,
     `rate` below the fleet-modal rate for that port position;
   - **error counters**: nonzero error-class counters (same list as
     `counter_delta.sh`: symbol_error, link_downed, port_xmit_discards,
     port_rcv_errors, port_rcv_remote_physical_errors, local_link_integrity_errors,
     excessive_buffer_overrun_errors);
   - **unreached nodes**: submitted but no snapshot within timeout.
5. Report: text summary grouped by anomaly type, plus `summary.json` in the
   output dir. `--json` prints summary.json to stdout instead.
6. Exit 1 if any anomaly (including ≥1 unreached node), 0 if fleet clean,
   3 if the run itself failed (no nodes expanded, sbatch rejected, zero
   snapshots collected).

Notes: without `--gres`, GPU/topology sections of each snapshot are empty on
cgroup-isolated nodes — expected; IB data (the point of the fleet sweep) is
always collected. The aggregator must tolerate missing/error-marked sections
(ib_snapshot embeds `{"error": ...}` stanzas rather than failing).

### topo_verify

```
topo_verify [--golden FILE] [--current FILE] [--save-golden] [--json]
```

- Golden default: `results/golden/<hostname>.json`. `--save-golden` runs
  `ib_snapshot.sh` (or takes `--current FILE`) and blesses it as the golden,
  then exits 0.
- Verify mode: obtain current snapshot (fresh `ib_snapshot.sh` run into a
  temp file, or `--current FILE` for offline use), load golden (missing
  golden → exit 3 with instructions), then compare **identity fields only**:
  - GPU inventory: count, `name`, `pci_bus_id` per index;
  - HCA set: names present;
  - per-port: `state`, `phys_state`, `rate`, `link_layer`;
  - `ibdev2netdev` mapping tuples (hca, port, netdev);
  - topology matrix: the GPU/NIC rows of `nvidia-smi topo -m` raw text,
    whitespace-normalized.
  Volatile fields (counters, temps, clocks, memory_used, timestamps) are
  ignored. Driver/kernel/CUDA version changes are reported as **informational
  drift** (listed, but do not by themselves set exit 1 unless `--strict`).
- Report: one line per drift finding — `field: golden=X current=Y` — grouped
  hardware vs informational; `--json` emits the findings list.
- Exit 0 no hardware drift, 1 hardware drift (or any drift with `--strict`),
  3 probe error.
- Comparison is a pure function `compare(golden_dict, current_dict) ->
  {"hardware": [...], "informational": [...]}`.

### nccl_check

```
nccl_check -A ACCOUNT -p PARTITION [-N NODES] [-G GPUS_PER_NODE] [-w NODELIST]
           [--min-busbw GBPS] [--binary PATH] [--counters] [--timeout MIN]
```

- Defaults: `-N 1`, `-G 4` (Kempner nodes have 4 GPUs), sweep `-b 8 -e 1G -f 2 -g 1`.
- Binary discovery order: `--binary`, `$NCCL_TESTS_BIN`, `all_reduce_perf` on
  `$PATH`, then common locations (`/usr/local/cuda/nccl-tests`,
  `/opt/nccl-tests/build`). Not found → exit 3 with an actionable message
  (how to point the tool at a build).
- Submits one Slurm job: `sbatch -N N --ntasks-per-node=G --gpus-per-node=G
  -c 8 --mem=32G -t <timeout>` running `srun all_reduce_perf ...`; waits for
  completion, reads the job's output file.
- Parses the nccl-tests table (columns: size … busbw for out-of-place and
  in-place) into rows; reports peak and large-message (max size) busbw.
- Verdict: with `--min-busbw`, exit 1 if large-message out-of-place busbw is
  below it; without, report-only exit 0 (no baked-in thresholds — expected
  busbw is GPU-generation and rail-count specific; the README documents
  typical values). Unparseable output or failed job → exit 3.
- `--counters`: bracket the run with `ib_snapshot.sh` before/after on the
  job's nodes (inside the same job script) and run `counter_delta.sh` on the
  pair; counter deltas are reported and force exit 1.
- Parsing is a pure function over the captured output text.

### ib_bw_pair

```
ib_bw_pair -A ACCOUNT -p PARTITION -w NODEA,NODEB [--test write|read]
           [--device DEV] [--min-gbps N] [--timeout MIN]
```

- Exactly two nodes required in `-w` (exit 3 otherwise).
- Submits one 2-node Slurm job whose script: reads the port `rate` from
  sysfs on the server node; starts `ib_write_bw` (or `ib_read_bw`) server on
  node A (`srun -w A --overlap ... &`), then client on node B pointing at A;
  captures the client's report table.
- Parses the perftest output (BW average column, MsgRate) for the largest
  message size.
- Verdict: `--min-gbps` if given; else pass when BW average ≥ 80% of the
  port's sysfs rate (the 0.8 factor is a named constant). Exit 0 pass,
  1 below threshold, 3 setup/parse failure.
- `--device` pins the HCA (`-d mlx5_X`) for rail-specific tests; default
  lets perftest choose.
- Parsing is a pure function over the captured output text.

## Error handling

- Every subprocess call has a timeout; Slurm submission failures, expansion
  failures (`scontrol`/`sinfo` errors), and missing binaries are exit-3
  errors with one-line stderr messages prefixed `<tool>: error:`.
- Aggregation/comparison functions never raise on missing snapshot sections;
  a missing section becomes an explicit "not available" note in the report,
  never a silent pass (the `gpu_health` n/a rule).
- Jobs that outlive `--timeout` are `scancel`ed by job id; partial results
  are still reported.

## Testing

Offline stdlib-`unittest` suite under `fabric/tests/` with fixtures in
`fabric/tests/fixtures/`:

- two+ synthetic `ib_snapshot` JSONs (healthy pair, plus doctored variants:
  degraded rate, down port, nonzero error counter, differing driver/FW) for
  the fleet aggregator and the topo_verify comparator, including golden ==
  current, hardware drift, informational-only drift, and missing-section
  cases;
- a captured/representative `all_reduce_perf` output table (plus a truncated
  variant) for the nccl_check parser and threshold verdicts;
- a captured/representative `ib_write_bw` client report (plus a truncated
  variant) for the ib_bw_pair parser and both verdict modes (explicit
  `--min-gbps` and %-of-rate);
- CLI-level tests for exit codes that don't need Slurm: topo_verify with
  `--golden`/`--current` files (0/1/3 paths), parsers via module import.
  Slurm-dependent paths are covered by documented manual smoke commands in
  the README, not unit tests.

## Documentation

Top-level `README.md`: add the four tools to the `fabric/` section table;
extend the fabric requirements row (adds: Slurm client with a valid account,
`perftest` package for `ib_bw_pair`, an `nccl-tests` build for `nccl_check`);
add usage examples including tested Kempner invocations (`-A kempner_dev
-p kempner_eng`). Document typical expected bandwidth values (H100/H200
NVLink intra-node, NDR inter-node) as prose guidance, not code defaults.
