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

### Running `gpu_health` on a GPU node via Slurm

If you are not already on the node, run it through Slurm (the repo lives on a
shared filesystem, so the node runs it in place):

```bash
# quick interactive check on one GPU
srun -A <account> -p <gpu-partition> --gres=gpu:1 -c 2 --mem=8G -t 10 ./gpu/gpu_health

# check ALL GPUs on a node (Slurm cgroups hide unallocated GPUs, so
# request the whole node's GPUs; add -w <node> to target a specific node)
srun -A <account> -p <gpu-partition> --gres=gpu:4 -c 2 --mem=8G -t 10 -w <node> ./gpu/gpu_health

# JSON snapshot for archiving / diffing
srun -A <account> -p <gpu-partition> --gres=gpu:1 -c 2 --mem=8G -t 10 \
    ./gpu/gpu_health --json results/gpu_health_$(date +%Y%m%d).json

# Kempner example (as tested: H200 node on kempner_eng)
srun -A kempner_dev -p kempner_eng --gres=gpu:1 -c 2 --mem=8G -t 10 ./gpu/gpu_health
```

The exit code passes through Slurm, so `srun ... ./gpu/gpu_health && echo healthy`
works as a scriptable gate (0 OK, 1 WARN, 2 FAIL, 3 probe error). You can also
`ssh` to a node where you have a running job and invoke it directly.

## See also

For a **per-job GPU% / GPU-memory% summary** (coarse duty-cycle from the sacct
AdminComment blob, alongside CPU% and MEM%), use **`../slurm/jobstats_history`**.
It lives under `slurm/` because it's a Slurm `sacct`-based utilization summary
across all resource classes — not a DCGM/Prometheus GPU profiler like the tools
here. Reach for `jobstats_history` for a quick "was the GPU busy" pass, and the
tools here when you need to see how hard the SMs / tensor cores actually worked.
