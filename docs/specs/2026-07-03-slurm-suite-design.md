# slurm suite expansion — job_failures, queue_wait, gpu_idle_now, fairshare_report, frag_report

**Date:** 2026-07-03
**Location:** `slurm/`
**Status:** approved design, pre-implementation
**Branch base:** `gpu-suite` (gpu_idle_now imports `gpu/gpulib.py`; also keeps the suites merge-ordered: gpu-health → gpu-suite → slurm-suite).

## Purpose

Five additions that extend `slurm/` from utilization summaries to failure
triage, scheduling health, live waste detection, share accounting, and
fragmentation analysis:

| Tool | One-liner |
|---|---|
| `job_failures` | Window failure post-mortem: exit codes, OOM, timeouts, cancellations, node failures — per user/account/partition. |
| `queue_wait` | Submit→Start wait distributions by partition, QOS, and GPU-count bucket. |
| `gpu_idle_now` | Live detector of running jobs whose GPUs are idle right now (the real-time gate behind `gpu/fleet_util`'s retrospective numbers). |
| `fairshare_report` | sshare snapshot: share vs effective usage per account, over/under-served rankings. |
| `frag_report` | Free CPU/GPU/memory shards per node and "how many N-GPU jobs could start right now" per partition. |

### Honest scope cuts (verified against what the data can support)

- `queue_wait` does NOT claim to separate priority wait from resource wait —
  `sacct` carries no pending-reason history.
- `fairshare_report` is a point-in-time snapshot — `sshare` keeps no history;
  `--json` output enables external archiving/trending later.
- Near-timeout risk detection ("job finished at 99% of its limit") is out of
  scope for v1 of `job_failures`.

## Shared conventions

- Python 3.6-compatible stdlib executables in `slurm/`, no `.py` extension,
  `#!/usr/bin/env python3`, executable bit. Exception: `gpu_idle_now`
  additionally imports `gpu/gpulib.py` (adds `../gpu` to `sys.path` relative
  to its own resolved location) and therefore needs the jobstats config +
  `requests` at runtime — its siblings need only Slurm client commands.
- Exit codes: `0` report/clean, `1` findings (`gpu_idle_now` only: at least
  one fully-idle job), `3` probe/setup error as `<tool>: error: <msg>` on
  stderr. argparse usage errors stay exit 2.
- Windows via `--days N` (default 7) or `--since ISO --until ISO`
  (job_failures, queue_wait) — same semantics as the gpu suite (reuse the
  parsing convention; these two tools implement it locally to stay
  stdlib-only, matching the gpulib behavior exactly).
- `--json` on every tool emits the structured result instead of the table.
- Every subprocess call has a timeout (sacct/sshare 120 s, squeue/scontrol/
  sinfo 60 s).
- Unparseable rows, excluded rows, and truncations are counted and reported,
  never silent.
- Architecture per tool: pure parse/classify/aggregate core + thin command
  runners + `main()`. Replay seams: `--from-file FILE` where the tool has a
  single text input (job_failures, queue_wait: sacct text; fairshare_report:
  sshare text; frag_report: scontrol text), `--from-json FILE` bundle for
  gpu_idle_now (`{"alloc": <prom result>, "smact": <prom result>,
  "squeue": <text>}`). Prometheus error messages never contain the endpoint
  URL (inherited gpulib behavior).

## Tool designs

### job_failures

```
job_failures [-u USER | -A ACCOUNT | -p PARTITION] [--days N | --since ISO
             --until ISO] [--top N] [--json] [--from-file SACCT_TXT]
```

- Exactly one of `-u`/`-A`/`-p` (default: `-u $USER`). `-A`/`-p` add `-a`
  to sacct (all users).
- One query: `sacct -X -n -P -o JobIDRaw,User,Account,Partition,State,
  ExitCode,DerivedExitCode,Elapsed,Timelimit,ReqMem,AllocTRES,NodeList,
  JobName -S <since> -E <until>` plus the selector flag.
- Pure core `classify(rows)` — per job, first match wins:
  - `OUT_OF_MEMORY*` state → `oom`
  - `TIMEOUT*` → `timeout`
  - `NODE_FAIL*` → `node_fail`
  - `PREEMPTED*` → `preempted`
  - `CANCELLED` and `CANCELLED by <uid>` → `cancelled` (the uid is retained
    in the detail; distinguishing admin-vs-self cancellation needs uid→user
    mapping and is out of scope for v1)
  - `FAILED*` → `failed` (keyed by ExitCode `rc:signal`)
  - `COMPLETED` → `completed`
  - `RUNNING`/`PENDING` → `active` (excluded from failure rate)
  - anything else → `other` (retained and shown)
- Pure core `summarize(classified, top_n)`: totals per class, failure rate
  (= non-completed terminal / all terminal), top exit codes with counts, top
  failing JobNames, top failing users (only meaningful under `-A`/`-p`),
  nodes ranked by node_fail+oom incidence.
- Output: summary block + ranked tables. Exit 0 report, 3 on sacct failure.

### queue_wait

```
queue_wait (-p PARTITION | -A ACCOUNT | -u USER) [--days N | --since ISO
           --until ISO] [--json] [--from-file SACCT_TXT]
```

- Exactly one of `-p`/`-A`/`-u` (default: `-u $USER`, same rule as
  job_failures). One query: `sacct -X -n -P -o JobIDRaw,Partition,QOS,
  Submit,Start,State,AllocTRES -S <since> -E <until>` plus selector
  (`-p`/`-A` add `-a`).
- Pure core `waits(rows)`: per started job, wait_s = Start − Submit (ISO
  parse). Rows with no Start (`Unknown`/`None`, still pending) and rows with
  Start < Submit (clock skew / requeue artifacts) are counted into
  `excluded` buckets and reported. GPU bucket from AllocTRES `gres/gpu=N`:
  `0`, `1`, `2-4`, `>4`.
- Pure core `distribution(waits_list)`: count, p50, p90, max (nearest-rank
  percentiles).
- Output: one table grouped by partition, one by QOS, one by GPU bucket —
  columns `GROUP N P50 P90 MAX` (durations humanized, e.g. `2h 13m`);
  excluded-row note. Exit 0 report, 3 on sacct failure. If `sacct -a` is
  refused by the site, the stderr from sacct is surfaced in the error and
  exit is 3.

### gpu_idle_now

```
gpu_idle_now [--threshold PCT] [--window MIN] [-p PARTITION] [--json]
             [--from-json FILE]
```

- Defaults: threshold 10 (%), window 30 (minutes).
- Query 1 (`alloc`): instant `nvidia_gpu_jobId > 0` — value is the jobid,
  labels carry lowercase `uuid` + `host`.
- Query 2 (`smact`): instant
  `avg_over_time(DCGM_FI_PROF_SM_ACTIVE[<window*60>s:])` — labels carry
  uppercase `UUID`; joined python-side via `gpulib.series_uuid` both-casing
  rule.
- Pure core `group_jobs(alloc_result, smact_result, threshold)` → per jobid:
  total GPUs, idle GPUs (avg sm×100 < threshold), unmatched GPUs (no smact
  sample — counted, never guessed), mean SM_ACT%, nodes. A job is
  `fully_idle` when every matched GPU is idle and ≥1 GPU matched.
- Enrichment: one `squeue -j <ids> -h -o "%A|%u|%a|%P|%L"` for user,
  account, partition, time-left of flagged (fully or partially idle) jobs;
  squeue failure degrades to `-` fields with a stderr warning.
- `-p PARTITION` filters the report to jobs whose squeue partition matches
  (applied post-join; unenriched jobs can't be filtered and are shown with a
  note).
- Output: table `JOBID USER ACCOUNT PARTITION GPUS IDLE UNMATCHED SM_ACT%
  TIME_LEFT`, fully-idle jobs first, then partially idle; summary line.
  Exit 1 if ≥1 fully-idle job, 0 otherwise, 3 on query failure.
- Replay bundle roles: `"alloc"`, `"smact"` (instant results), `"squeue"`
  (text).

### fairshare_report

```
fairshare_report [-A ACCOUNT] [--top N] [--json] [--from-file SSHARE_TXT]
```

- One query: `sshare -a -P -o Account,User,RawShares,NormShares,RawUsage,
  EffectvUsage,FairShare` (`-A` narrows to one account subtree).
- Pure core `parse_sshare(text)`: keep top-level account rows (leading
  single space, no User — sshare indents hierarchy; account rows have empty
  User field); skip the `root` row and per-user rows for the rollup, but
  count them.
- Pure core `rank(accounts, top_n)`: for each account with NormShares > 0:
  `ratio = EffectvUsage / NormShares` (over-served > 1, under-served < 1;
  accounts with zero usage listed separately); top-N by ratio descending
  (over-served) and ascending among accounts with usage (under-served);
  FairShare factor column.
- Output: summary (accounts counted, skipped rows) + two ranked tables.
  Point-in-time note in the footer. Exit 0 report, 3 on sshare failure.

### frag_report

```
frag_report [-p PARTITION] [--cpus-per-gpu N] [--mem-per-gpu GIB] [--json]
            [--from-file SCONTROL_TXT]
```

- Defaults: 8 CPUs and 64 GiB per GPU (the "job shape" used for the
  schedulability counts).
- One pass: `scontrol show node -o` (one line per node; `-p` filters via
  the node's `Partitions=` field).
- Pure core `parse_nodes(text)`: per node — name, partitions, state, free
  CPUs (`CPUTot − CPUAlloc`), free memory MiB (`RealMemory − AllocMem`),
  total/alloc/free GPUs (CfgTRES/AllocTRES `gres/gpu=`). Nodes whose State
  contains DOWN/DRAIN/MAINT/NOT_RESPONDING are set aside into an
  `unavailable` bucket (counted, listed, excluded from free-capacity math).
- Pure core `fragmentation(nodes, cpus_per_gpu, mem_per_gpu_mib)`: per
  partition — free-GPU distribution (nodes with 0/1/2/3/4+ free GPUs), and
  for shapes 1/2/4 GPUs: how many jobs of that shape could start right now
  (per node: `min(free_gpus // n, free_cpus // (n*cpg), free_mem //
  (n*mpg))`, summed).
- Output: per-partition table (`PARTITION NODES FREE_GPU_DIST FIT_1G FIT_2G
  FIT_4G`) + unavailable-node note + shape footnote. Exit 0 report, 3 on
  scontrol failure.

## Error handling

- sacct/sshare/scontrol are required inputs → their failure (missing binary,
  timeout, nonzero exit) is exit 3 with the command's stderr in the message.
- squeue in gpu_idle_now is enrichment → degrade with a stderr warning.
- gpu_idle_now inherits gpulib's URL-free RuntimeError contract; gpulib
  import failure (running where gpu/gpulib.py is absent) is exit 3 with a
  message naming the expected relative location.

## Testing

Offline stdlib-`unittest` under `slurm/tests/` (new: `__init__.py`,
`loader.py` in the fabric/gpu pattern, fixtures directory):

- fixtures: captured-shape sacct failure rows (every class incl.
  `CANCELLED by <uid>`, OOM, TIMEOUT, NODE_FAIL, exit-code variants),
  sacct queue rows (incl. Unknown Start and Start<Submit skew), sshare -P
  output (root row, account rows, user rows), scontrol show node -o lines
  (healthy, allocated, drained), plus in-test alloc/smact instant results
  for gpu_idle_now;
- unit tests per pure core: classification precedence, failure-rate math,
  percentile math (nearest-rank), GPU bucketing, idle/unmatched grouping
  and the fully-idle rule, sshare ratio ranking and row filtering, TRES and
  node-line parsing, shape-fit arithmetic;
- CLI exit-path tests via the replay flags (0/1/3);
- live smoke afterward: all five against the real cluster (sacct/sshare/
  scontrol data is abundant; gpu_idle_now runs against live Prometheus).

## Documentation

Top-level `README.md` `slurm/` table: five new rows; requirements row gains
one sentence: `gpu_idle_now` additionally needs the jobstats config +
`requests` (it queries Prometheus like the `gpu/` window tools).
