# jobstats_history

A compact **per-job utilization table** — `CPU% / MEM% / GPU% / GMEM%` for each
job — computed in **bulk** from each job's `sacct` `AdminComment` blob. One
`sacct` query, **no per-job `jobstats` calls and no job-count cap**. For
completed jobs the numbers match `jobstats` exactly. Companion to
`stotal_kempner` / `seff_history`.

## Synopsis

```bash
./jobstats_history                       # your jobs over the last 1 day
./jobstats_history -D 7                  # your jobs over the last 7 days
./jobstats_history -N 20                 # your most recent 20 jobs
./jobstats_history -u bdesinghu -D 7     # a specific user
./jobstats_history -A kempner_dev -D 7   # narrow to an account
./jobstats_history -p kempner_h100 -D 7  # narrow to a partition
./jobstats_history --gpu -D 7            # GPU columns only (drops non-GPU jobs)
./jobstats_history -d 17487020           # per-node / per-GPU breakdown for one job
./jobstats_history 17487020 17410278     # specific job IDs (bypass time selection)
./jobstats_history --gpu --dcgm -D 7     # + time-averaged DCGM profiling columns
./jobstats_history -d --gpu --dcgm 17487020   # per-GPU breakdown incl. DCGM cols
./jobstats_history --gpu --dcgm --csv -D 7 > out.csv   # CSV incl. DCGM cols
./jobstats_history --gpu --diagnose -D 7       # advisory DIAG verdict per job
./jobstats_history --describe                  # explain every column (no job needed)
```

Selects your jobs (default `$USER`) over a time scope; with no scope at all it
defaults to the **last 1 day**. Give explicit `JOBID`s to bypass selection
entirely.

## Options

### Positional

| Argument | Description |
|---|---|
| `jobids` | Specific job IDs to report on. Bypasses all time selection (the `-u`/`-A`/`-p` filters are **not** applied). The header `User:` line then shows the jobs' **actual owner(s)** from `sacct`, which may differ from you. |

### Selection

| Flag | Description |
|---|---|
| `-u, --user USER` | User whose jobs to select. **Default:** the current user (`$USER`). |
| `-A, --account ACCOUNT` | Narrow to this account. |
| `-p, --partition PARTITION` | Narrow to this partition. |
| `-t, --state {all,completed,failed}` | Job-state filter. **Default:** `all`. (`completed`/`failed` default to `-D 1`.) |

### Time scope

| Flag | Description |
|---|---|
| `-N, --lastn N` | The most recent **N jobs**. |
| `-D, --days N` | Jobs in the **last N days**. |
| `-S, --starttime TIME` | Window start (sacct format). |
| `-E, --endtime TIME` | Window end (sacct format). |

With no scope at all, defaults to the **last 1 day**.

### Columns / view

| Flag | Description |
|---|---|
| `--cpu` | Show **CPU** columns only. |
| `--gpu` | Show **GPU** columns only — **non-GPU jobs are dropped**. |
| `--full` | Show **all** columns (**default**). |
| `--dcgm` | Add time-averaged **DCGM profiling** columns (`SM_ACT% / OCC% / TENSOR% / DRAM% / POWER_W`). GPU jobs only; pulled live from the jobstats Prometheus (see Notes). |
| `--diagnose` | Add an advisory **`DIAG`** column (`idle` / `underfed` / `low-occ` / `mem-bound` / `no-tensor` / `ok`) per job (or per GPU with `-d`). GPU jobs only; fetches the DCGM data like `--dcgm` (combine the two to show both the numbers and the verdict). Heuristic — see `README_jobstats_dcgm.md`. |
| `--min-runtime N` | Jobs shorter than N seconds get `DIAG=short` (**default 180**). |
| `--workers N` | Max concurrent `--dcgm`/`--diagnose` query-sets (**default 8**). The queries are I/O-bound, so `N>1` speeds up wide selections **even on a single CPU**; `--workers 1` serializes them. |
| `-d, --details` | Per-node / per-GPU breakdown for each job. |

If none of `--cpu/--gpu/--full` is given, behaves like the default (full).

### Output

| Flag | Description |
|---|---|
| `-n, --noheader` | Suppress the header / context block. |
| `--describe`, `--description`, `--explain` | Print a plain-English description of each column and **exit** (no job ID needed; CPU/MEM/GPU/GMEM + the DCGM columns). Add `--diagnose` for the DIAG-tag legend. For the full DCGM metric set, see `jobstats_dcgm --describe --all`. |
| `--csv` | Machine-readable output. Works for the summary **and** the `-d` detail view (detail CSV adds a leading `JOBID` column so per-GPU rows are identifiable). |
| `--timeout SECONDS` | Seconds per `sacct` / Prometheus call (**default 60**; `0` disables). |
| `-h, --help` | Show help and exit. |

## Output columns

Summary view (`--full`):

| Column | Meaning |
|---|---|
| `JOBID` | Slurm job ID |
| `STATE` | Slurm job state |
| `NODES` | number of nodes |
| `GPUS` | number of GPUs (from `AllocTRES`) |
| `CPU%` | `100 × Σ node.total_time / (elapsed × Σ cpus)` |
| `MEM%` | `100 × Σ used_memory / Σ total_memory` |
| `GPU%` | mean GPU duty cycle over all GPUs |
| `GMEM%` | `100 × Σ gpu_used_memory / Σ gpu_total_memory` (peak) |
| `RUNTIME` | elapsed time |
| `NAME` | job name |

`-d` adds a per-job block with a `NODE / GPU / CPU% / CPU-MEM / GPU% / GPU-MEM /
GMEM%` table.

With `--dcgm`, these extra **time-averaged** columns are appended (GPU jobs
only). They are *not* in the blob — they come from Prometheus:

| Column | Metric | Meaning (time-averaged over the run) |
|---|---|---|
| `SM_ACT%` | `DCGM_FI_PROF_SM_ACTIVE` | fraction of time ≥1 warp was resident, averaged across SMs |
| `OCC%` | `DCGM_FI_PROF_SM_OCCUPANCY` | fraction of warp slots filled (true occupancy) |
| `TENSOR%` | `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` | fraction of time the tensor pipe was active |
| `DRAM%` | `DCGM_FI_PROF_DRAM_ACTIVE` | fraction of time the HBM interface was busy |
| `POWER_W` | `DCGM_FI_DEV_POWER_USAGE` | mean board power (W) |

`GPU%` (the blob's duty cycle) is the coarse "a kernel ran" signal; expect
`GPU% ≥ SM_ACT% ≥ OCC%`, with `TENSOR%` high only for tensor-core workloads.
For the full DCGM field set (ENGINE/FP16/FP32/FP64/MEMCP/PWRmax/ENERGY/TEMP/
SMCLK), use the standalone `jobstats_dcgm` tool.

Jobs with no jobstats blob (`JS1:None` / `JS1:Short` / missing) print a row
showing the real Slurm state and `(no jobstats data)`.

## Notes

- **Data source:** one bulk
  `sacct -X -P -n --units=G -o JobID,State,JobName,Elapsed,NNodes,AllocTRES,AdminComment`
  query; each `AdminComment` is a `JS1:`-prefixed base64-gzip JSON blob, decoded
  in-process. No per-job round-trips → fast even for hundreds of jobs.
- **Parity:** for **completed** jobs the four metrics match `jobstats` exactly.
  **Running** jobs can differ by ~1% because `jobstats` re-queries live data
  while the stored blob is the last snapshot.
- **GPU% is the time-averaged duty cycle** (fraction of time a kernel ran), not
  SM/tensor-core occupancy. **GMEM% is peak** GPU memory, not time-averaged.
- **`--dcgm` is opt-in and is the only path that touches the network.** Without
  it the tool stays blob-only (one `sacct` query, no per-job calls). With it,
  each GPU job gets a small set of Prometheus queries (endpoint + sampling
  period imported from the jobstats `config`, so credentials are not
  duplicated; must run where `jobstats` lives). GPUs are joined to the job **by
  UUID** via the `nvidia_gpu_jobId` companion — the DCGM `gpu` index and the
  slurm `minor_number` disagree, so only the UUID is stable across the two
  exporters. Each value is averaged over the job's `[start,end]` window, exactly
  the way `jobstats` averages the duty cycle.
- **`--dcgm` cost & concurrency.** Each GPU job costs one query-set (1 discovery
  + 5 metric queries). These run **concurrently across jobs** via a thread pool
  (`--workers`, default 8); the queries are network I/O-bound, so this overlaps
  the Prometheus round-trips and speeds up wide selections even on a single CPU
  (measured ~7× on 12 jobs at `--workers 12`, with no extra CPU). The pool is
  bounded to stay polite to the Prometheus endpoint. Still, very wide windows
  also make the plain `sacct` *selection* slow — narrow with `-N`/`-D` there.
