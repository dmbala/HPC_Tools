# jobstats_extended

`jobstats_extended.py` is a drop-in wrapper around the standard Slurm
`jobstats` report. It keeps the familiar CPU, memory, GPU utilization, and GPU
memory sections, then adds DCGM GPU profiling metrics for completed jobs.

The added DCGM metrics are averaged over the job run time and reported per GPU:

- `SM_ACT%`
- `OCC%`
- `TENSOR%`
- `DRAM%`
- `POWER_W`

## Usage

```bash
./jobstats_extended.py <job_id> [<job_id> ...] [options]
```

Examples:

```bash
./jobstats_extended.py 18583067            # formatted report (like jobstats) + DCGM
./jobstats_extended.py --json 18583067     # extended blob as pretty JSON
./jobstats_extended.py -b 18583067         # JS1:<base64> blob (storage form)
./jobstats_extended.py --no-dcgm 18583067  # report/blob without the DCGM metrics
```

Options:

- `-j`, `--json` — emit the extended blob as pretty JSON instead of the report
- `-b`, `--base64` — emit the `JS1:<base64 gzip JSON>` blob (AdminComment storage form)
- `--no-dcgm` — skip the DCGM augmentation
- `-f`, `--force` — force recompute from Prometheus instead of the cached blob
- `-c`, `--cluster` — cluster name (default: from sacct)
- `-n`, `--no-color` — render the report without colorization
- `-d`, `--debug` — debug output


## Sample output

```bash
./jobstats_extended.py 65867515_0
```

```text
# job 65867515_0: added gpu_sm_active, gpu_sm_occupancy, gpu_tensor_active, gpu_dram_active, gpu_power_usage

================================================================================
                              Slurm Job Statistics                              
================================================================================
         Job ID: 65867515_0
          State: COMPLETED
          Nodes: 3
      CPU Cores: 12
     CPU Memory: 900GB (75GB per CPU-core)
           GPUs: 12
  QOS/Partition: kemp_gpu28_id19/kempner_h100_priority
        Cluster: odyssey
     Start Time: Fri Mar 13, 2026 at 10:30 AM
       Run Time: 2-02:54:04
     Time Limit: 2-12:00:00

                         Overall Utilization                          
================================================================================
  CPU utilization  [|||||||||||||||||||||||||||||||||||||||||||||||97%]
  CPU memory usage [|||||||||||||||||                              34%]
  GPU utilization  [||||||||||||||||||||||||                       48%]
  GPU memory usage [||||||||||||||||||||||                         45%]

                         Detailed Utilization                         
================================================================================
  CPU utilization per node (CPU time used/run time)
      holygpu8a11502: 8-05:50:02/8-11:36:16 (efficiency=97.2%)
      holygpu8a11601: 8-05:52:42/8-11:36:16 (efficiency=97.2%)
      holygpu8a15602: 8-05:55:28/8-11:36:16 (efficiency=97.2%)
  Total used/runtime: 24-17:38:13/25-10:48:48, efficiency=97.2%

  CPU memory usage per node - used/allocated
      holygpu8a11502: 102.3GB/300GB (25.6GB/75GB per core of 4)
      holygpu8a11601: 102.2GB/300GB (25.5GB/75GB per core of 4)
      holygpu8a15602: 102.7GB/300GB (25.7GB/75GB per core of 4)
  Total used/allocated: 307.1GB/900.0GB (25.6GB/75GB per core of 12)

  GPU utilization per node
      holygpu8a11502 (GPU 0): 49.1%
      holygpu8a11502 (GPU 1): 43.0%
      holygpu8a11502 (GPU 2): 50.8%
      holygpu8a11502 (GPU 3): 48.9%
      holygpu8a11601 (GPU 0): 46.9%
      holygpu8a11601 (GPU 1): 46.9%
      holygpu8a11601 (GPU 2): 48.8%
      holygpu8a11601 (GPU 3): 47.7%
      holygpu8a15602 (GPU 0): 48.6%
      holygpu8a15602 (GPU 1): 46.3%
      holygpu8a15602 (GPU 2): 48.8%
      holygpu8a15602 (GPU 3): 48.1%

  GPU memory usage per node - maximum used/total
      holygpu8a11502 (GPU 0): 36.2GB/79.6GB (45.5%)
      holygpu8a11502 (GPU 1): 36.2GB/79.6GB (45.4%)
      holygpu8a11502 (GPU 2): 35.6GB/79.6GB (44.7%)
      holygpu8a11502 (GPU 3): 35.7GB/79.6GB (44.8%)
      holygpu8a11601 (GPU 0): 35.6GB/79.6GB (44.6%)
      holygpu8a11601 (GPU 1): 35.9GB/79.6GB (45.1%)
      holygpu8a11601 (GPU 2): 35.6GB/79.6GB (44.7%)
      holygpu8a11601 (GPU 3): 35.5GB/79.6GB (44.5%)
      holygpu8a15602 (GPU 0): 35.8GB/79.6GB (45.0%)
      holygpu8a15602 (GPU 1): 35.4GB/79.6GB (44.4%)
      holygpu8a15602 (GPU 2): 35.4GB/79.6GB (44.5%)
      holygpu8a15602 (GPU 3): 35.7GB/79.6GB (44.8%)



                     GPU Profiling Metrics (DCGM)                     
================================================================================
  Averaged over the run time. SM/tensor/DRAM are % busy; power is watts.

  Overall (averaged across GPUs)
    SM active     [||||||                                         13%]
    SM occupancy  [|                                               2%]
    Tensor active [|                                               2%]
    DRAM active   [||||                                            8%]
    Power usage   172.9W (avg per GPU)

  Per node / GPU
      holygpu8a11502 (GPU 0): SM active=12.6%  SM occupancy=2.4%  Tensor active=2.3%  DRAM active=7.9%  Power usage=169.2W
      holygpu8a11502 (GPU 1): SM active=12.7%  SM occupancy=2.4%  Tensor active=2.3%  DRAM active=8.0%  Power usage=171.8W
      holygpu8a11502 (GPU 2): SM active=12.5%  SM occupancy=2.3%  Tensor active=2.2%  DRAM active=7.8%  Power usage=172.8W
      holygpu8a11502 (GPU 3): SM active=12.5%  SM occupancy=2.3%  Tensor active=2.2%  DRAM active=7.9%  Power usage=171.8W
      holygpu8a11601 (GPU 0): SM active=12.5%  SM occupancy=2.4%  Tensor active=2.2%  DRAM active=7.9%  Power usage=173.2W
      holygpu8a11601 (GPU 1): SM active=12.4%  SM occupancy=2.3%  Tensor active=2.2%  DRAM active=7.8%  Power usage=174.7W
      holygpu8a11601 (GPU 2): SM active=12.5%  SM occupancy=2.4%  Tensor active=2.2%  DRAM active=7.9%  Power usage=173.0W
      holygpu8a11601 (GPU 3): SM active=12.5%  SM occupancy=2.3%  Tensor active=2.2%  DRAM active=7.9%  Power usage=177.0W
      holygpu8a15602 (GPU 0): SM active=12.5%  SM occupancy=2.4%  Tensor active=2.2%  DRAM active=7.8%  Power usage=171.7W
      holygpu8a15602 (GPU 1): SM active=12.5%  SM occupancy=2.4%  Tensor active=2.2%  DRAM active=7.9%  Power usage=175.8W
      holygpu8a15602 (GPU 2): SM active=12.5%  SM occupancy=2.3%  Tensor active=2.2%  DRAM active=7.9%  Power usage=172.9W
      holygpu8a15602 (GPU 3): SM active=12.5%  SM occupancy=2.3%  Tensor active=2.2%  DRAM active=7.8%  Power usage=171.2W
```

## Metric guide

Standard `jobstats` already reports CPU utilization, CPU memory usage, GPU
utilization, and GPU memory usage. `jobstats_extended.py` preserves those values
and adds DCGM metrics that describe what the GPUs were doing internally.

| Metric | Meaning | How to read it |
| --- | --- | --- |
| `CPU utilization` | CPU time used divided by allocated CPU runtime. | Low values usually mean the job requested more CPU cores than it used. |
| `CPU memory usage` | Peak host memory used compared with allocated memory. | High values can indicate memory pressure; low values can indicate over-requested memory. |
| `GPU utilization` | The standard coarse GPU busy signal. | Useful as a quick activity check, but it does not explain which GPU subsystem was busy. |
| `GPU memory usage` | Peak device memory used compared with GPU memory capacity. | Shows memory footprint, not whether computation or memory bandwidth was the bottleneck. |
| `SM_ACT%` | Percent of time the GPU streaming multiprocessors had active work. | Shows whether CUDA cores were actually active. |
| `OCC%` | Average SM occupancy, or how full the SMs were while kernels ran. | Low occupancy can point to small batches, limited parallelism, register pressure, or launch/configuration limits. |
| `TENSOR%` | Percent of time tensor cores were active. | Important for mixed-precision AI workloads; low values can mean the job is not using tensor-core-friendly kernels. |
| `DRAM%` | Percent of time GPU memory bandwidth was active. | Helps identify memory-bound jobs, data movement overhead, and inefficient memory access patterns. |
| `POWER_W` | Average GPU power draw in watts. | Helps distinguish real compute load from idle, stalled, or lightly loaded GPU allocation time. |

The five DCGM metrics provide a better picture than `GPU utilization` alone
because a single utilization percentage hides the reason a GPU was busy. Two jobs
can both show `GPU utilization=50%`, while one is compute-bound, another is
memory-bandwidth-bound, another is barely filling the SMs, and another is mostly
idle between short kernels. `SM_ACT%`, `OCC%`, `TENSOR%`, `DRAM%`, and `POWER_W`
separate those cases:

- `SM_ACT%` shows whether the GPU compute units were active.
- `OCC%` shows how much parallel work was available when kernels ran.
- `TENSOR%` shows whether tensor cores contributed to the workload.
- `DRAM%` shows whether GPU memory bandwidth was the limiting resource.
- `POWER_W` shows whether the GPU drew power consistent with meaningful work.

Together, these metrics make it easier to tell whether a job needs more GPUs,
fewer GPUs, larger batches, better kernel efficiency, faster input pipelines, or
different resource requests.

## Requirements

Requires an existing `jobstats` install plus the `requests` Python package. The
script imports `jobstats`, `output_formatters`, and `config` (`PROM_SERVER`) from
the system path (`/usr/local/bin`, `/usr/bin`).
