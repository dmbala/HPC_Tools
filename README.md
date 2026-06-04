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
