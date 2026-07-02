# HPC_Tools

A collection of standalone operational tools for Slurm/GPU HPC clusters
(developed against the Kempner Institute cluster at Harvard). Tools are
organized by the subsystem they measure.

```
HPC_Tools/
├── slurm/      # Slurm job accounting & utilization reporting (sacct-based)
├── gpu/        # GPU profiling metrics (DCGM via Prometheus)
├── fabric/     # InfiniBand topology & error-counter diagnostics (read-only)
└── analysis/   # post-processing: history.jsonl → reports / verdicts / trends
```

## Setup

These are operational tools meant to run **on the cluster** against live Slurm /
Prometheus / InfiniBand — not on a laptop. There's nothing to build: clone the
repo onto a shared filesystem and run each tool in place (the executable bit is
preserved by git).

```bash
git clone https://github.com/dmbala/HPC_Tools.git
cd HPC_Tools
./slurm/jobstats_history --help
```

Optionally put the tool dirs on your `PATH`:

```bash
export PATH="$PWD/slurm:$PWD/gpu:$PWD/fabric:$PATH"
```

### Requirements by directory

| Dir | Run where | Needs |
|---|---|---|
| `slurm/` | any node with the Slurm client | `sacct` (and `scontrol` for `stotal_kempner`); Python 3 standard library only. **`seff_history`** additionally needs `numpy`, `pandas`, and `termplotlib` — its shebang points at the FASRC `seff-array` env (`/n/sw/envs/seff-array/bin/python3`); repoint it if that env isn't on your cluster. |
| `gpu/` | wherever `jobstats` is installed | the jobstats `config` / `jobstats` Python modules (auto-discovered on `/usr/local/bin` or `/usr/bin`), the `requests` module (ships with jobstats), and a reachable Prometheus (`PROM_SERVER`, read from the jobstats `config`). **`gpu_health`** is the exception: it runs on the target GPU node and needs only `nvidia-smi` and the Python 3 standard library (no Prometheus, no jobstats modules). |
| `fabric/` | on the target GPU node (no allocation needed) | host tools `nvidia-smi`, `ibstat`, `ibv_devinfo`, `ibdev2netdev`, plus `python3`. All probes are read-only. |
| `analysis/` | anywhere with Python 3 | Python 3 standard library only — runs offline on a `history.jsonl` file. |

## slurm/ — job accounting & utilization

Read from Slurm `sacct` (and the jobstats `AdminComment` blob). No GPU
allocation or special privileges needed.

| Tool | What it does |
|---|---|
| `jobstats_history` | Bulk per-job CPU%/MEM%/GPU%/GMEM% summary computed from each job's sacct AdminComment blob — one sacct query, no per-job `jobstats` calls, no job-count cap. See `README_jobstats_history.md`. |
| `seff_history` | Job-efficiency report for a user/account over a window: state & resource summaries, simple-mean vs resource-weighted efficiency, a "most wasteful jobs" table, and efficiency-distribution histograms. See `README_seff_history.md`. |
| `stotal_kempner` | Total CPU-hours / Mem-hours / GPU-hours / TRES-hours for a user, account, or partition over a window or last-N jobs — restricted to Kempner-owned nodes via `--nodelist`. |

## gpu/ — GPU profiling metrics

Pull per-job DCGM profiling metrics from Prometheus (endpoint taken from the
jobstats `config` module, so credentials are never duplicated). Must run where
the `jobstats` command lives.

| Tool | What it does |
|---|---|
| `jobstats_dcgm` | Per-job, time-averaged DCGM **profiling** metrics that jobstats does *not* store in its blob: SM-active, SM-occupancy, tensor/fp pipe activity, DRAM-active, power, energy. Joins `DCGM_FI_*` to a job's GPUs on UUID. |
| `jobstats_extended.py` | Subclass of `jobstats.Jobstats` that folds those DCGM metrics back **into** the jobstats blob, keyed per-GPU by `minor_number` like the built-in `gpu_utilization`. Lets you test the extended blob before swapping it in cluster-wide. |

## fabric/ — InfiniBand / topology diagnostics

Read-only host probes (`nvidia-smi`, `ibstat`, `ibv_devinfo`, `ibdev2netdev`).
No GPU allocation needed; run in seconds. Default outputs land under
`./results/snapshots/`.

| Tool | What it does |
|---|---|
| `ib_snapshot.sh` | Full static topology snapshot for one node (GPUs, NVLink, IB HCAs, ports, `/sys` counters, GPU↔NIC topology matrix) → one JSON file for diffing/forensics. |
| `affinity_check.sh` | Audits GPU↔IB-NIC PCIe/NUMA affinity from `nvidia-smi topo -m`. Exits non-zero if any GPU lacks a ≥`NODE`-quality link to an IB NIC (`SYS` = 30–50% cross-node BW loss). |
| `counter_delta.sh` | Diffs two snapshots' per-port IB error counters. Exits non-zero if any error-class counter advanced during a workload window. |

Typical workflow:

```bash
fabric/ib_snapshot.sh results/snapshots/before.json
... run workload ...
fabric/ib_snapshot.sh results/snapshots/after.json
fabric/counter_delta.sh results/snapshots/before.json results/snapshots/after.json
```

## analysis/ — history post-processing

Pure-Python post-processors that turn an append-only `history.jsonl` time series
into reports. Run with any `python3`.

| Tool | What it does |
|---|---|
| `scrape_metrics.py` | Walks a results tree and appends normalized rows to `history.jsonl`. |
| `correlate.py` | Per-node verdict cross-referencing training MFU × NCCL busbw × storage × IB counters (OK / DEGRADED_COMPUTE / DEGRADED_NCCL / SLOW_STORAGE / DEGRADED_FABRIC / MULTIPLE). |
| `report.py` | Cluster-wide markdown leaderboard: per-metric p50/p90/min/max + per-host ranking. |
| `trend.py` | Time-series trend with sparklines + regression flagging against a rolling 7-day per-host baseline. |

## Related: the `cluster_bench` benchmark suite

`fabric/` and `analysis/` are the portable subset of the `cluster_bench`
diagnostic suite. The **active benchmark drivers** that feed `analysis/` — NCCL
microbenchmarks and the NeMo-Automodel compute/scaling/storage/acceptance
sbatch drivers — live in `cluster_bench/` and are intentionally *not* copied
here: they depend on the NeMo-Automodel container and the `shared/` launch
infrastructure, so they do not run standalone. Run them from `cluster_bench`;
collate and report their output with the tools in `analysis/`.
