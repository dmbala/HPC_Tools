# kempner-jobstats

*Quick-look and deep-dive GPU efficiency reporting for Slurm jobs — companions to
Princeton/FASRC `jobstats`.*

Two command-line tools for looking at the **GPU (and CPU) efficiency** of your
Slurm jobs:

- **`jobstats_history`** — the **quick look**. A compact per-job utilization
  table across *many* jobs / days, computed in bulk. Use it first, to see at a
  glance which jobs used their GPUs well and which didn't.
- **`jobstats_dcgm`** — the **closer look**. The full set of DCGM GPU profiling
  metrics for *one or a few specific* jobs. Use it second, once `jobstats_history`
  has pointed you at a job worth investigating.

Both read the same data `jobstats` uses, so the numbers line up with it.

---

## 1. Quick look: did your GPUs actually work? → `jobstats_history`

**Start here.** One command lines up every GPU job you ran over the last few days
and shows — at a glance — whether the GPU was *doing work* or just sitting there
powered on:

```bash
./jobstats_history --gpu --dcgm -D 5
```

`--gpu` keeps GPU jobs only, `--dcgm` adds the *real* utilization columns, `-D 5`
covers the last 5 days. Example:

```
  User:      bdesinghu
  Select:    last 5 days

JOBID        STATE      GPUS GPU% GMEM% SM_ACT% OCC%  TENSOR% DRAM% POWER_W RUNTIME      NAME
--------------------------------------------------------------------------------------------------------
17487020     COMPLETED  4    91   78    62.4    38.1  71.0    45.2  642     1-04:11:30   train_llama
17487044     COMPLETED  4    88   61    11.1    6.0   0.0     7.8   120     08:22:10     sweep_lr
17490318     COMPLETED  1    70   44    48.6    29.4  0.0     33.0  523     03:10:00     eval_run
17491002     FAILED     8    3    9     0.4     0.2   0.0     0.1   74      00:04:55     debug_ddp
--------------------------------------------------------------------------------------------------------
Mean:                        63   48    30.6    18.4  17.8    21.5  340
```

**Here's the catch this view exposes.** Look at `sweep_lr`: its `GPU%` is **88** —
by the usual duty-cycle number it looks busy. But `SM_ACT%` is **11**: the GPU was
*switched on* the whole run while its cores did almost nothing. `GPU%` alone would
have fooled you; the DCGM columns tell the truth.

Reading left to right:

| Column | The question it answers |
|---|---|
| `GPU%` | Was *a* kernel running? (duty cycle — coarse; high is necessary, not sufficient) |
| `SM_ACT%` | Were the cores actually busy? **← the honest efficiency number** |
| `OCC%` | How full were the cores? (low → small kernels / weak parallelism) |
| `TENSOR%` | Were tensor cores used? (~0 on an ML job → no mixed precision) |
| `DRAM%` | Was memory bandwidth the bottleneck? |
| `POWER_W` | Sanity check — near-idle watts mean the GPU was coasting |

The **`Mean:`** row sums it up: across these five days your jobs averaged 63% duty
cycle but only ~31% real SM activity — a lot of allocated-but-idle GPU time.
`train_llama` is genuinely healthy (cores hot, tensor cores at 71%); `sweep_lr`
and `eval_run` were **underfed** (busy in time, idle in compute); `debug_ddp` never
really started.

> **Lighter glance, no network.** Drop `--dcgm` and you get just `GPU%`/`GMEM%`
> from a single, instant `sacct` query — no Prometheus, works anywhere, no
> job-count cap. Add `--dcgm` when you want the honest utilization columns; they
> cost a quick Prometheus fetch per GPU job (run concurrently) and must run where
> `jobstats` lives.

**Want the tool to grade each job for you?** Add `--diagnose` for a one-word
verdict per job (`idle` / `underfed` / `low-occ` / `mem-bound` / `no-tensor` /
`ok`):

```bash
./jobstats_history --gpu --diagnose -D 5
```

```
JOBID        STATE      GPUS GPU% GMEM% DIAG       RUNTIME      NAME
17487020     COMPLETED  4    91   78    ok         1-04:11:30   train_llama
17487044     COMPLETED  4    88   61    underfed   08:22:10     sweep_lr
17490318     COMPLETED  1    70   44    no-tensor  03:10:00     eval_run
17491002     FAILED     8    3    9     idle       00:04:55     debug_ddp
```

More quick-look variants:

```bash
./jobstats_history                       # your jobs over the last 1 day (default)
./jobstats_history -N 20                 # your most recent 20 jobs
./jobstats_history -A kempner_dev -D 7   # everyone on an account, last 7 days
./jobstats_history -t failed -D 7        # only the jobs that failed
```

Two of those jobs clearly deserve a closer look — that's the next tool.

---

## 2. Closer look: *why* was this job inefficient? → `jobstats_dcgm`

Once `jobstats_history` flags a job, point `jobstats_dcgm` at that **specific job
ID**. It pulls the rich DCGM profiling metrics that `jobstats` does **not** store
— how hard the SMs and tensor cores actually worked, memory bandwidth, power,
clocks, energy — and shows them **per GPU**.

```bash
./jobstats_dcgm 17487044
```

Example output:

```
Job 17487044  [COMPLETED]  sweep_lr   (2026-05-30 21:05 .. 05:27, 30060s)
  NODE             GPU  DUTY%   SM_ACT%   OCC%    TENSOR%   DRAM%    POWER_W
  ----------------------------------------------------------------------------
  holygpu8a01      0    89      11.3      6.2     0.0       8.1      121
  holygpu8a01      1    87      10.9      5.8     0.0       7.4      118
  ...
  Overall               88      11.1      6.0     0.0       7.8      120
```

This explains the catch from the quick look: `DUTY%` was ~88% (the GPU was
*occupied* almost the whole run) but `SM_ACT%` was only ~11% and `TENSOR%` was 0 —
the GPU was **resident but barely loaded** (host/input bound, tiny batches), and
no tensor cores were used. That is the "underfed" diagnosis, with the numbers to
back it up.

The six default columns and how to read them:

| Column | Measures | How to read it |
|---|---|---|
| `DUTY%` | GPU ran *any* kernel (coarse) | high is necessary, not sufficient |
| `SM_ACT%` | SMs engaged (≥1 warp resident) | low while `DUTY%` high → GPU parked / underfed |
| `OCC%` | how full the SMs were | low → small kernels / weak parallelism |
| `TENSOR%` | tensor cores doing matmul | ~0 on an ML job → not using mixed precision |
| `DRAM%` | HBM bandwidth busy | high while `SM_ACT%` low → memory-bound |
| `POWER_W` | mean board power draw | near-idle watts → GPU not really working |

Go deeper from here:

```bash
./jobstats_dcgm --all 17487044          # all 28 metrics (clocks, energy, FP pipes, PCIe, NVLink, temps, ...)
./jobstats_dcgm --diagnose 17487044     # add the advisory DIAG tag, per GPU
./jobstats_dcgm 17487044 17490318       # several jobs at once
./jobstats_dcgm --ts 17487044 > job.csv # raw per-scrape time series, for plotting the profile over time
./jobstats_dcgm --describe --all        # explain every metric, no job needed
```

### The two-tool workflow in one line

> **Sweep with `jobstats_history --gpu --dcgm` to find the jobs with low
> `SM_ACT%` / a bad `DIAG`, then profile those specific job IDs with
> `jobstats_dcgm` to see why.**

The same five DCGM columns are available *inside* `jobstats_history --dcgm` for the
quick sweep; `jobstats_dcgm` is the standalone, per-job tool with the **full**
28-metric catalog, per-GPU rows, and time-series output.

---

# In-depth reference

Everything below is the complete option set for both tools.

## `jobstats_history`

A compact **per-job utilization table** — `CPU% / MEM% / GPU% / GMEM%` for each
job — computed in **bulk** from each job's `sacct` `AdminComment` blob (the same
data `jobstats` itself stores). One `sacct` query, **no per-job `jobstats` calls
and no job-count cap**. For completed jobs the four metrics match `jobstats`
exactly; running jobs may differ by ~1% (the blob is the last snapshot, while
`jobstats` re-reads live data). Companion to `stotal_kempner` / `seff_history`.

### Synopsis

```bash
./jobstats_history                       # your jobs over the last 1 day
./jobstats_history -D 7                  # your jobs over the last 7 days
./jobstats_history -N 20                 # your most recent 20 jobs
./jobstats_history -u bdesinghu -D 7     # a specific user
./jobstats_history -A kempner_dev -D 7   # narrow to an account
./jobstats_history -p kempner_h100 -D 7  # narrow to a partition
./jobstats_history -t failed -D 7        # only failed jobs
./jobstats_history --gpu --dcgm -D 5     # GPU jobs + the real DCGM utilization columns
./jobstats_history --gpu --diagnose -D 5 # advisory DIAG verdict per job
./jobstats_history -d --gpu --dcgm 17487020      # per-GPU breakdown incl. DCGM cols
./jobstats_history 17487020 17410278     # specific job IDs (bypass time selection)
./jobstats_history --gpu --dcgm --csv -D 7 > out.csv   # CSV incl. DCGM cols
./jobstats_history --describe            # explain every column (no job needed)
```

Selects your jobs (default `$USER`) over a time scope; with no scope at all it
defaults to the **last 1 day**. Give explicit `JOBID`s to bypass selection
entirely.

### Options

> **Tip — get help any time:** run **`./jobstats_history -h`** for the full
> usage. The Synopsis above covers the common cases; expand the list below for
> every flag.

<details>
<summary><b>Full option list</b> (click to expand)</summary>

| Flag | What it does |
|---|---|
| **Who / which jobs** | |
| `jobids` *(positional)* | Report exactly these job IDs; bypasses time selection and the `-u/-A/-p` filters. The header then shows the jobs' real owner(s) from `sacct`. |
| `-u, --user USER` | Whose jobs to select (default: `$USER`). |
| `-A, --account ACCT` | Narrow to an account. |
| `-p, --partition PART` | Narrow to a partition. |
| `-t, --state {all,completed,failed}` | State filter (default `all`; `completed`/`failed` imply `-D 1`). |
| **Time scope** *(default: last 1 day)* | |
| `-N, --lastn N` | The most recent **N jobs**. |
| `-D, --days N` | Jobs in the **last N days** (not combinable with `-N` or `-S/-E`). |
| `-S, --starttime` / `-E, --endtime TIME` | Explicit window, sacct format (e.g. `2026-05-26`). |
| **Which columns** *(default: `--full`)* | |
| `--cpu` / `--gpu` / `--full` | Pick the metric set. `--gpu` shows GPU columns only and **drops non-GPU jobs**. |
| `--dcgm` | Add DCGM columns `SM_ACT% / OCC% / TENSOR% / DRAM% / POWER_W` (GPU jobs only; queries Prometheus). |
| `--diagnose` | Add the advisory **`DIAG`** verdict (`idle`/`underfed`/`low-occ`/`mem-bound`/`no-tensor`/`ok`). Fetches DCGM data like `--dcgm`; combine the two to show numbers *and* verdict. |
| `-d, --details` | Per-node / per-GPU breakdown for each job. |
| **Tuning & output** | |
| `--min-runtime N` | Runs shorter than N seconds get `DIAG=short` (default 180; sampling noise). |
| `--workers N` | Concurrent `--dcgm` query-sets (default 8; the queries are I/O-bound, so `>1` helps even on one CPU; `1` serializes). |
| `--csv` | Machine-readable output (works for the summary **and** the `-d` view). |
| `-n, --noheader` | Suppress the header / context block. |
| `--describe` | Print a plain-English description of every column and exit (no job needed; add `--diagnose` for the DIAG legend). |
| `--timeout S` | Seconds per `sacct` / Prometheus call (default 60; `0` disables). |
| `-h, --help` | Show help and exit. |

A few details that don't fit a one-liner:

- If you pass none of `--cpu/--gpu/--full`, the tool prompts when run
  interactively, otherwise defaults to `--full`.
- `-t failed` covers `FAILED,TIMEOUT,OUT_OF_MEMORY,NODE_FAIL,CANCELLED,DEADLINE,BOOT_FAIL,PREEMPTED`.
- `--describe` documents the CPU/MEM/GPU/GMEM columns plus the five DCGM columns;
  for the full 28-metric DCGM catalog use `jobstats_dcgm --describe --all`.
- The `-d --csv` detail view prepends a `JOBID` column so per-GPU rows are
  identifiable.
- If a selection spans too many jobs, narrow it with `-N`/`-D`/`-S`/`-E` rather
  than raising `--timeout`.

</details>

### Output columns

<details>
<summary><b>Column reference</b> — summary view, <code>-d</code> detail, and the <code>--dcgm</code> columns (click to expand)</summary>

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
SMCLK/…), use the standalone `jobstats_dcgm` tool.

Jobs with no jobstats blob (`JS1:None` / `JS1:Short` / missing) print a row
showing the real Slurm state and `(no jobstats data)`.

</details>

### Notes

- **Data source:** one bulk `sacct -X -P -n --units=G -o JobID,State,JobName,Elapsed,NNodes,AllocTRES,...,AdminComment`
  query; each `AdminComment` is a `JS1:`-prefixed base64-gzip JSON blob, decoded
  in-process. No per-job round-trips → fast even for hundreds of jobs.
- **Parity:** for **completed** jobs the four blob metrics match `jobstats`
  exactly. **Running** jobs can differ by ~1% because `jobstats` re-queries live
  data while the stored blob is the last snapshot.
- **`GPU%` is the time-averaged duty cycle** (fraction of time a kernel ran), not
  SM/tensor-core occupancy. **`GMEM%` is peak** GPU memory, not time-averaged.
- **`--dcgm`/`--diagnose` are opt-in and are the only path that touches the
  network.** Without them the tool stays blob-only (one `sacct` query). With
  them, each GPU job gets a small set of Prometheus queries (endpoint + sampling
  period imported from the jobstats `config`, so credentials are not duplicated —
  **run this where the `jobstats` command lives**). GPUs are joined to the job
  **by UUID** via the `nvidia_gpu_jobId` companion (the DCGM `gpu` index and the
  slurm `minor_number` disagree; only the UUID is stable across the two
  exporters). Each value is averaged over the job's `[start,end]` window, exactly
  the way `jobstats` averages the duty cycle.
- **`--dcgm` cost & concurrency.** Each GPU job costs one query-set (1 discovery
  + 5 metric queries), run **concurrently across jobs** via a thread pool
  (`--workers`, default 8). Because the queries are network I/O-bound, this
  overlaps the Prometheus round-trips and speeds up wide selections even on a
  single CPU. Very wide windows also make the plain `sacct` *selection* slow —
  narrow with `-N`/`-D` there.

---

## `jobstats_dcgm`

Per-job, **time-averaged DCGM profiling metrics** that `jobstats` does **not**
store in its `AdminComment` blob: SM-active, occupancy, tensor/FP pipe activity,
DRAM-active, power, energy, clocks, temperatures, PCIe/NVLink throughput, and
more. Pulled straight from the jobstats Prometheus endpoint and averaged over
each job's `[start,end]` window (the same way `jobstats` averages the duty
cycle). Companion to `jobstats` / `jobstats_history` / `seff_history`.

`jobstats` itself keeps only one time-averaged GPU number — `nvidia_gpu_duty_cycle`
(its `GPU%`), a coarse "was a kernel running, yes/no" signal. The richer
`DCGM_FI_PROF_*` fields (how hard the SMs / tensor cores actually worked) live in
the same Prometheus but are never written to the blob; this tool exposes them.

> **Requires being run where the `jobstats` command lives** — it imports
> `PROM_SERVER` (and `SAMPLING_PERIOD`) from the jobstats `config` module so the
> endpoint/credentials are never duplicated, and needs the `requests` module
> (which ships with jobstats). Unlike `jobstats_history`, it needs **at least one
> explicit job ID** (or `--describe`).

### Synopsis

```bash
./jobstats_dcgm 17487020                 # default 6 metrics, per GPU, for one job
./jobstats_dcgm 17487020 17410278        # several jobs
./jobstats_dcgm --all 17487020           # all 28 metrics
./jobstats_dcgm --diagnose 17487020      # add the advisory DIAG column, per GPU
./jobstats_dcgm --csv 17487020 > out.csv         # CSV: one row per (job, node, GPU)
./jobstats_dcgm --ts  17487020 > prof.csv        # raw per-scrape time series (for plotting)
./jobstats_dcgm --describe               # explain the default metrics (no job needed)
./jobstats_dcgm --describe --all         # explain all 28 metrics
```

### Options

> **Tip — get help any time:** run **`./jobstats_dcgm -h`** for the full usage,
> or **`./jobstats_dcgm --describe`** to explain every metric without needing a
> job.

<details>
<summary><b>Full option list</b> (click to expand)</summary>

| Flag | What it does |
|---|---|
| `jobids` *(positional)* | One or more Slurm job IDs (required, unless `--describe`). |
| `--all` | Show **all 28** metrics instead of the default 6 (adds ENGINE/HMMA/IMMA/DFMA/FP16/FP32/FP64/MEMCP/PWRmax/ENERGY/FB_USED/FB_FREE/FB_RSVD/PCIE_TX/PCIE_RX/NVLINK/SMCLK/MEMCLK/TEMP/MEMTEMP/ENC/DEC). |
| `--diagnose` | Add the advisory **`DIAG`** column (`idle`/`underfed`/`low-occ`/`mem-bound`/`no-tensor`/`ok`), derived from the default metrics; no extra queries. |
| `--min-runtime N` | Jobs shorter than N seconds get `DIAG=short` (default 180). |
| `--csv` | Machine-readable to stdout — **one row per (job, node, GPU)**; redirect to a file to store. |
| `--timeseries`, `--ts` | Emit the **raw per-scrape time series** over each job's window as CSV (`JOBID,EPOCH,TIME,NODE,GPU,<metrics>`) instead of the window average — for plotting a job's profile over time. |
| `--describe`, `--explain` | Print a plain-English description of each metric and exit (no job needed; respects `--all`; add `--diagnose` for the DIAG legend). |
| `-n, --noheader` | Suppress the table / CSV header row. |
| `--timeout S` | Seconds per HTTP / `sacct` call (default 60). |
| `-h, --help` | Show help and exit. |

</details>

### Output

A human-readable table **per job**, one row per GPU, plus an `Overall` row for
multi-GPU jobs (mean for utilization metrics, `sum` for energy, `max` for peak
power). CPU-only jobs, jobs too short to have samples, or jobs aged out of
Prometheus retention print a short note instead of a table.

### Metric catalog

All `%`-columns from `DCGM_FI_PROF_*` are **time-fractions ×100** ("fraction of
the run during which …"), **not** instantaneous. Expect
`DUTY% ≥ ENGINE% ≥ SM_ACT% ≥ OCC%`, and `TENSOR%`/`HMMA%` high only for
tensor-core workloads.

**Default columns (6)** — shown without `--all`:

| Column | Prometheus metric | Meaning (time-averaged over the run) |
|---|---|---|
| `DUTY%` | `nvidia_gpu_duty_cycle` | jobstats' `GPU%`: frac. time ≥1 kernel ran (coarse) |
| `SM_ACT%` | `DCGM_FI_PROF_SM_ACTIVE` | frac. time ≥1 warp resident, avg over SMs |
| `OCC%` | `DCGM_FI_PROF_SM_OCCUPANCY` | frac. warp slots filled (true occupancy) |
| `TENSOR%` | `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` | frac. time tensor pipe active |
| `DRAM%` | `DCGM_FI_PROF_DRAM_ACTIVE` | frac. time HBM interface busy (bandwidth duty cycle) |
| `POWER_W` | `DCGM_FI_DEV_POWER_USAGE` | mean board power (W) |

<details>
<summary><b>Additional columns with <code>--all</code> (22)</b> (click to expand)</summary>

| Column | Prometheus metric | Meaning |
|---|---|---|
| `ENGINE%` | `DCGM_FI_PROF_GR_ENGINE_ACTIVE` | frac. time the engine had work (finer than `DUTY%`) |
| `HMMA%` | `DCGM_FI_PROF_PIPE_TENSOR_HMMA_ACTIVE` | tensor pipe, fp16/bf16 ops |
| `IMMA%` | `DCGM_FI_PROF_PIPE_TENSOR_IMMA_ACTIVE` | tensor pipe, integer (int8) ops |
| `DFMA%` | `DCGM_FI_PROF_PIPE_TENSOR_DFMA_ACTIVE` | tensor pipe, fp64 ops |
| `FP16%` | `DCGM_FI_PROF_PIPE_FP16_ACTIVE` | non-tensor fp16 pipe activity |
| `FP32%` | `DCGM_FI_PROF_PIPE_FP32_ACTIVE` | non-tensor fp32 pipe activity |
| `FP64%` | `DCGM_FI_PROF_PIPE_FP64_ACTIVE` | non-tensor fp64 pipe activity |
| `MEMCP%` | `DCGM_FI_DEV_MEM_COPY_UTIL` | % time copy engine moved data |
| `PWRmax_W` | `DCGM_FI_DEV_POWER_USAGE` (max) | peak board power (W) |
| `ENERGY_kWh` | `DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION` | energy over the run (kWh) |
| `FB_USED_GB` | `DCGM_FI_DEV_FB_USED` | mean GPU mem used (GiB); jobstats `GMEM%` is peak |
| `FB_FREE_GB` | `DCGM_FI_DEV_FB_FREE` | mean GPU mem free (GiB) |
| `FB_RSVD_GB` | `DCGM_FI_DEV_FB_RESERVED` | mean GPU mem reserved by driver/system (GiB) |
| `PCIE_TX_MBs` | `DCGM_FI_PROF_PCIE_TX_BYTES` | mean PCIe tx throughput (MB/s; rate — see note) |
| `PCIE_RX_MBs` | `DCGM_FI_PROF_PCIE_RX_BYTES` | mean PCIe rx throughput (MB/s; rate — see note) |
| `NVLINK_MBs` | `DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL` | mean NVLink throughput (MiB/s); ~0 single-GPU |
| `SMCLK_MHz` | `DCGM_FI_DEV_SM_CLOCK` | mean SM (core) clock (MHz) |
| `MEMCLK_MHz` | `DCGM_FI_DEV_MEM_CLOCK` | mean memory clock (MHz) |
| `TEMP_C` | `DCGM_FI_DEV_GPU_TEMP` | mean core temperature (°C) |
| `MEMTEMP_C` | `DCGM_FI_DEV_MEMORY_TEMP` | mean memory (HBM) temperature (°C) |
| `ENC%` | `DCGM_FI_DEV_ENC_UTIL` | mean video-encoder (NVENC) utilization |
| `DEC%` | `DCGM_FI_DEV_DEC_UTIL` | mean video-decoder (NVDEC) utilization |

> **Note on rates:** `PCIE_*`/`NVLINK_*` are DCGM rates (not monotonic counters),
> so the window-average is a mean throughput; the MB/s / MiB/s scaling assumes
> per-second values, so treat the *absolute* bandwidth as approximate
> (relative / zero-vs-nonzero is reliable).

</details>

### Reading the metrics — diagnostic patterns

The default 6 metrics combine into a few recognizable signatures (advisory
heuristics; window-averaged, so a multi-phase job blurs — the raw numbers stay
the source of truth):

| Pattern | Signature | Likely cause / action |
|---|---|---|
| **idle / unused** | `DUTY~0`, `SM_ACT~0`, `POWER~idle` | never ran a kernel: crashed early, wrong device, idle interactive session |
| **parked / underfed** | `DUTY` high, `SM_ACT` & `OCC` low | resident but cores idle: host/input bottleneck, tiny batch, launch overhead |
| **low occupancy** | `SM_ACT` high, `OCC` low | warp slots underfilled: small kernels — fuse / tune launch config |
| **memory-bound** | `DRAM` high, `SM_ACT`/`TENSOR` lower | HBM bandwidth is the limiter, not compute |
| **no mixed precision** | `SM_ACT` high, `TENSOR~0` (ML job) | running fp32: enable AMP / bf16 / tf32 |
| **healthy** | `SM_ACT` high, `OCC` ok, `TENSOR`/FP high, `POWER` near TDP | good use; little to change |

The `--diagnose` flag collapses these into a single `DIAG` tag per GPU:

```
DIAG -- advisory tag(s) from the utilization metrics (heuristic, not a verdict;
window-averaged, so multi-phase jobs blur). Several legitimately-low-util
workloads (inference, sampling, sparse HPC, data prep) read as "underfed".
  idle       SM_ACT < 5%                     GPU essentially never ran a kernel
  underfed   SM_ACT < 15%                    resident but cores mostly idle (host/input bound)
  low-occ    SM_ACT >= 40%, OCC < 20%        SMs busy but warp slots underfilled
  mem-bound  DRAM >= 40% and >= SM_ACT       HBM bandwidth is the limiter
  no-tensor  SM_ACT >= 40%, TENSOR < 1%      tensor cores idle (e.g. fp32 ML)
  ok         SM_ACT >= 40%, none of above    healthy compute use
  short      ran under --min-runtime         averages are sampling noise
```

Tags can co-occur (e.g. `mem-bound,no-tensor`). Compare `POWER_W` to the GPU's
TDP (~700 W for H100/H200, ~400 W for A100) — near-idle watts confirm the GPU was
not really working.

---

## References

- **FASRC jobstats** — Harvard FAS Research Computing documentation:
  <https://docs.rc.fas.harvard.edu/kb/jobstats/>
- **Princeton jobstats** — the upstream `jobstats` platform these tools build on:
  <https://princetonuniversity.github.io/jobstats/>

> **For live monitoring**, use **KempnerPulse**:
> <https://github.com/KempnerInstitute/kempnerpulse>. The two tools here look at
> jobs *after the fact* (from stored snapshots / Prometheus history); KempnerPulse
> watches a job's GPU/CPU utilization **in real time** while it runs.

---

*Author: Bala Desinghu, Senior AI/HPC Research Computing Engineer, Kempner
Institute, Harvard.*
