# seff_history

An extension of `seff` that summarizes **job efficiency** (CPU / GPU / Memory /
Time) for a user or account over a period of time, and displays the
distributions as histograms. Based on
[fasrc/seff-account](https://github.com/fasrc/seff-account).

## Synopsis

```bash
./seff_history                         # your efficiency over the last 1 day (full report)
./seff_history -D 7                    # your efficiency over the last 7 days
./seff_history -u bdesinghu -D 30      # a specific user, last 30 days
./seff_history -A kempner_dev -D 7     # an account, last 7 days
./seff_history --gpu -D 7              # GPU metrics only
./seff_history --cpu -D 7              # CPU metrics only
./seff_history -D 7 --no-interactive   # exclude interactive / Jupyter / OnDemand jobs
./seff_history -u bdesinghu -S 2026-05-01 -E 2026-05-31
```

Pass a user (`-u`) **or** an account (`-A`) — they are mutually exclusive.
Default user is `$USER`; default window is the **last 1 day**.

> For accounts you are not a member of you must be a Slurm **Operator** or
> **Administrator**.

## Options

### Selection

| Flag | Description |
|---|---|
| `-u, --user USER` | User to report on. **Default:** the current user (`$USER`). |
| `-A, --account ACCOUNT` | Account to report on (mutually exclusive with `-u`). |
| `-c, --cluster CLUSTER` | Cluster to query. |

### Time scope

| Flag | Description |
|---|---|
| `-S, --starttime TIME` | Start time (sacct format). |
| `-E, --endtime TIME` | End time (sacct format). |
| `-D, --days N` | Report the **last N days** (shorthand for `-S now-Ndays -E now`). **Default:** last 1 day. |

### Which metrics to show

| Flag | Description |
|---|---|
| `--cpu` | Show **CPU** metrics only. |
| `--gpu` | Show **GPU** metrics only (non-GPU jobs are dropped). |
| `--full` | Show **all** metrics (**default**). |

If none of these is given, the tool prompts interactively (defaulting to full),
or just uses `--full` when output is not a terminal.

### Filtering

| Flag | Description |
|---|---|
| `--no-interactive` | Exclude interactive jobs — JobName like `interactive`/`jupyter`/`OnDemand`/`shell`, **OR** a job with a `.interactive` step (i.e. `salloc` / `srun --pty`). |

### Misc

| Flag | Description |
|---|---|
| `--version` | Show the program version and exit. |
| `-h, --help` | Show help and exit. |

## What the report contains

The **full** report includes:

- **Job-state counts** (completed / failed / etc.).
- **Average requested resources** (cores, memory, GPUs, time).
- **Efficiency**, reported two ways:
  - *simple-mean* — the average of per-job efficiencies, and
  - *resource-weighted* — total used ÷ total allocated resource-hours.
- **Most-wasteful jobs** table.
- **GPU compute : memory ratio** and **memory-parked GPU jobs** (jobs holding
  GPU memory but doing little compute).
- **Efficiency-distribution histograms** (GPU first, then CPU/Memory/Time),
  with bins labelled like `0.0 - 10.0%`.

`--cpu` and `--gpu` show focused subsets of the above.

### Definitions

| Metric | Definition |
|---|---|
| CPU efficiency | `cpu-time / (elapsed × requested cores)` |
| GPU efficiency | from **jobstats** (time-averaged GPU duty cycle) |
| Memory efficiency | `MaxRSS / requested memory` |
| Time efficiency | `elapsed / requested time` |
| Resource-weighted | total used ÷ total allocated resource-hours |

## Notes

- **Data source:** `sacct` plus each job's `AdminComment` jobstats blob —
  the same blob `jobstats` decodes. One bulk query, no per-job calls.
- **GPU memory caveat:** the GPU-memory figures are based on **peak** GPU
  memory (high-water mark), not time-averaged usage — a *very approximate*
  estimate. The report prints this caveat next to the relevant numbers.
- **GPU utilization** is the time-averaged **duty cycle** (fraction of time a
  kernel was running), not SM/tensor-core occupancy.
