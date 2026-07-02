# gpu/ — GPU profiling metrics

Per-job DCGM profiling metrics pulled from Prometheus. The endpoint is read from
the jobstats `config` module (`PROM_SERVER`), so these must run where the
`jobstats` command lives and credentials are never duplicated.

| Tool | What it does |
|---|---|
| `jobstats_dcgm` | Per-job, time-averaged DCGM **profiling** metrics that jobstats does *not* store in its blob: SM-active, SM-occupancy, tensor/fp pipe activity, DRAM-active, power, energy. Joins `DCGM_FI_*` to a job's GPUs on UUID. |
| `jobstats_extended.py` | Subclass of `jobstats.Jobstats` that folds those DCGM metrics back **into** the jobstats blob, keyed per-GPU by `minor_number` like the built-in `gpu_utilization`. |
| `gpu_health` | Node-local GPU **health** probe: ECC & row-remap, throttling, PCIe/NVLink error counters, with a tiered OK/WARN/FAIL verdict (exit 0/1/2; 3 = probe error). `--json` emits a snapshot; `--from-xml` replays a captured `nvidia-smi -q -x` dump. |

> **Note:** unlike the other tools here, `gpu_health` runs **on the GPU node
> itself** and talks to `nvidia-smi` directly — no Prometheus, no jobstats
> modules, Python 3 stdlib only.

## See also

For a **per-job GPU% / GPU-memory% summary** (coarse duty-cycle from the sacct
AdminComment blob, alongside CPU% and MEM%), use **`../slurm/jobstats_history`**.
It lives under `slurm/` because it's a Slurm `sacct`-based utilization summary
across all resource classes — not a DCGM/Prometheus GPU profiler like the tools
here. Reach for `jobstats_history` for a quick "was the GPU busy" pass, and the
tools here when you need to see how hard the SMs / tensor cores actually worked.
