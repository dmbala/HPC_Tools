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
| `slurm/` | any node with the Slurm client | `sacct` (plus `scontrol` for `stotal_kempner` and `frag_report`, `sshare` for `fairshare_report`, and `squeue` for `gpu_idle_now`); Python 3 standard library only. **`seff_history`** additionally needs `numpy`, `pandas`, and `termplotlib` — its shebang points at the FASRC `seff-array` env (`/n/sw/envs/seff-array/bin/python3`); repoint it if that env isn't on your cluster. **`gpu_idle_now`** additionally needs the jobstats `config` module + `requests` (it queries Prometheus like the `gpu/` window tools) and `gpu/gpulib.py` alongside in the repo. |
| `gpu/` | wherever `jobstats` is installed | the jobstats `config` / `jobstats` Python modules (auto-discovered on `/usr/local/bin` or `/usr/bin`), the `requests` module (ships with jobstats), and a reachable Prometheus (`PROM_SERVER`, read from the jobstats `config`). **`gpu_health`** is the exception: it runs on the target GPU node and needs only `nvidia-smi` and the Python 3 standard library (no Prometheus, no jobstats modules). |
| `fabric/` | on the target GPU node (no allocation needed) | host tools `nvidia-smi`, `ibstat`, `ibv_devinfo`, `ibdev2netdev`, plus `python3`. All probes are read-only. The Slurm-based tools (`fleet_snapshot`, `nccl_check`, `ib_bw_pair`) additionally need the Slurm client and a valid account/partition; `ib_bw_pair` needs the `perftest` package on the nodes; `nccl_check` needs an `nccl-tests` build. `topo_verify`/`fleet_snapshot --from-dir` also run offline on saved snapshots. |
| `analysis/` | anywhere with Python 3 | Python 3 standard library only — runs offline on a `history.jsonl` file. |

## slurm/ — job accounting & utilization

Read from Slurm `sacct` (and the jobstats `AdminComment` blob). No GPU
allocation or special privileges needed.

| Tool | What it does |
|---|---|
| `jobstats_history` | Bulk per-job CPU%/MEM%/GPU%/GMEM% summary computed from each job's sacct AdminComment blob — one sacct query, no per-job `jobstats` calls, no job-count cap. See `README_jobstats_history.md`. |
| `seff_history` | Job-efficiency report for a user/account over a window: state & resource summaries, simple-mean vs resource-weighted efficiency, a "most wasteful jobs" table, and efficiency-distribution histograms. See `README_seff_history.md`. |
| `stotal_kempner` | Total CPU-hours / Mem-hours / GPU-hours / TRES-hours for a user, account, or partition over a window or last-N jobs — restricted to Kempner-owned nodes via `--nodelist`. |
| `job_failures` | Window failure post-mortem: classifies terminal jobs (failed/OOM/timeout/cancelled/node-fail/preempted), ranks exit codes, failing job names/users, and incident nodes. |
| `queue_wait` | Submit→Start wait distributions (p50/p90/max) by partition, QOS, and GPU-count bucket. Does not claim to split priority vs resource wait (sacct can't). |
| `gpu_idle_now` | Live detector of running jobs whose GPUs are idle right now (SM activity below a threshold over a recent window). Exit 1 when a fully-idle job exists. |
| `fairshare_report` | Point-in-time sshare snapshot: share vs effective usage per account, most over/under-served rankings. |
| `frag_report` | Free CPU/GPU/memory shards per node and how many 1/2/4-GPU jobs could start right now, per partition. |

## gpu/ — GPU profiling metrics

Pull per-job DCGM profiling metrics from Prometheus (endpoint taken from the
jobstats `config` module, so credentials are never duplicated). Must run where
the `jobstats` command lives.

| Tool | What it does |
|---|---|
| `jobstats_dcgm` | Per-job, time-averaged DCGM **profiling** metrics that jobstats does *not* store in its blob: SM-active, SM-occupancy, tensor/fp pipe activity, DRAM-active, power, energy. Joins `DCGM_FI_*` to a job's GPUs on UUID. |
| `jobstats_extended.py` | Subclass of `jobstats.Jobstats` that folds those DCGM metrics back **into** the jobstats blob, keyed per-GPU by `minor_number` like the built-in `gpu_utilization`. Lets you test the extended blob before swapping it in cluster-wide. |
| `xid_history` | Window report of **XID errors** fleet-wide, each episode mapped to node, GPU, and the job/user that held the GPU (exit 1 when any found). |
| `energy_report` | Per-user/account **GPU energy (kWh)** over a window, rolled up from DCGM energy counters via the jobstats job→GPU join. |
| `fleet_util` | % of **allocated** GPU-hours with SM activity below a threshold, by partition — "taken" vs "working" for capacity planning. |

## fabric/ — InfiniBand / topology diagnostics

Read-only host probes (`nvidia-smi`, `ibstat`, `ibv_devinfo`, `ibdev2netdev`).
No GPU allocation needed; run in seconds. Default outputs land under
`./results/snapshots/`.

| Tool | What it does |
|---|---|
| `ib_snapshot.sh` | Full static topology snapshot for one node (GPUs, NVLink, IB HCAs, ports, `/sys` counters, GPU↔NIC topology matrix) → one JSON file for diffing/forensics. |
| `affinity_check.sh` | Audits GPU↔IB-NIC PCIe/NUMA affinity from `nvidia-smi topo -m`. Exits non-zero if any GPU lacks a ≥`NODE`-quality link to an IB NIC (`SYS` = 30–50% cross-node BW loss). |
| `counter_delta.sh` | Diffs two snapshots' per-port IB error counters. Exits non-zero if any error-class counter advanced during a workload window. |
| `fleet_snapshot` | Runs `ib_snapshot.sh` across a nodelist/partition via per-node Slurm jobs and aggregates: driver/kernel/FW drift, down or degraded links, nonzero error counters, unreached nodes. `--from-dir` re-aggregates a past run offline. Exit 0 clean / 1 anomalies / 3 error. |
| `topo_verify` | Diffs a node's current snapshot against a blessed golden (`results/golden/<host>.json`, blessed with `--save-golden`). Hardware drift exits 1; driver/kernel changes are informational unless `--strict`. |
| `nccl_check` | NCCL all-reduce bandwidth smoke test in a fresh Slurm allocation (1–2 nodes). Report-only by default; `--min-busbw` makes it a pass/fail gate; `--counters` brackets the run with `ib_snapshot` + `counter_delta`. Needs an `nccl-tests` build (`--binary` or `$NCCL_TESTS_BIN`). |
| `ib_bw_pair` | `ib_write_bw`/`ib_read_bw` between two named nodes in one 2-node Slurm job. Passes when BW average ≥ `--min-gbps`, or ≥ 80% of the port's sysfs rate by default. |

Typical expected values (guidance, not code defaults): H100/H200 NVLink
intra-node all-reduce large-message busbw ≳ 300 GB/s; 2-node NDR (400 Gb/s
per rail) `ib_write_bw` average ≳ 320 Gb/s (80% of rate). Slurm examples:

```bash
fabric/fleet_snapshot -A kempner_dev -p kempner_eng
fabric/topo_verify --save-golden          # on the node, once, post-maintenance
fabric/nccl_check -A kempner_dev -p kempner_eng -N 2 -G 4 --binary /path/to/all_reduce_perf
fabric/ib_bw_pair -A kempner_dev -p kempner_eng -w holygpu8a10401,holygpu8a10402
```

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
