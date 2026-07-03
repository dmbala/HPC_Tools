# gpu suite expansion — xid_history, energy_report, fleet_util

**Date:** 2026-07-03
**Location:** `gpu/`
**Status:** approved design, pre-implementation
**Branch base:** `gpu-health` (reuses its `gpu/tests/` scaffolding and `.gitignore`); merge gpu-health first or stack the PRs.

## Purpose

Three additions that extend `gpu/` from per-job profiling to window/fleet
reporting, all reading the same Prometheus the jobstats stack already feeds:

| Tool | One-liner |
|---|---|
| `xid_history` | Window report of XID errors fleet-wide, each mapped to the node, GPU, and the job/user that held the GPU at the time. |
| `energy_report` | Per-user/account GPU energy (kWh) over a window, rolled up from per-job DCGM energy counters. |
| `fleet_util` | % of *allocated* GPU-hours whose SM activity was below a threshold, by partition — capacity-planning evidence. |

### Dropped from scope: throttle_report

Verified 2026-07-02: this cluster's dcgm-exporter does not scrape
`DCGM_FI_DEV_CLOCK_THROTTLE_REASONS` (zero series in Prometheus, both tag
spellings). A retrospective throttle report cannot be built from data that
was never collected, and a clock-depression proxy would mislead. Revisit
when ops enables the field (dcgm-exporter counters config:
`DCGM_FI_DEV_CLOCK_THROTTLE_REASONS`, or the newer
`DCGM_EXP_CLOCK_EVENTS_COUNT`). Live throttle state on a node is already
covered by `gpu/gpu_health`.

### Verified label schema (2026-07-02, live queries)

- `DCGM_FI_DEV_XID_ERRORS`: labels `Hostname`, `UUID`, `gpu`, `err_code`,
  `err_msg`, `modelName`, `slurm_cluster`; nonzero series exist.
- `nvidia_gpu_jobId`: value is the raw Slurm job id; labels `uuid`
  (lowercase), `host`, `minor_number`.
- `DCGM_FI_PROF_SM_ACTIVE`, `DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION`: labels
  `Hostname`, `UUID`; ~1670 series each.
- Join rule: DCGM series use uppercase `UUID`; the jobstats exporter uses
  lowercase `uuid`. All joins read both casings (`m.get("UUID") or
  m.get("uuid")`), per the sibling `jobstats_dcgm` convention.

## Shared conventions

- Python 3.6-compatible + `requests` (ships with jobstats), extensionless
  executables in `gpu/`, `#!/usr/bin/env python3`, executable bit.
- Prometheus endpoint and sampling period come from the jobstats `config`
  module (search path: `/usr/local/bin`, `/usr/bin`), exactly like
  `jobstats_dcgm`. The endpoint embeds credentials — tools must never print
  it (error messages name the failure, not the URL).
- Shared helper module `gpu/gpulib.py`: config import, instant/range query
  helpers (every call with a timeout), UUID-casing join helpers, window
  parsing (`--days` / `--since`+`--until` ISO), and `err_exit(tool, msg)`
  (`<tool>: error: <msg>` to stderr, exit 3).
- Exit codes: `0` clean/report, `1` findings (only `xid_history`), `3`
  probe/setup error. argparse usage errors remain exit 2.
- `--json` on every tool emits the structured result instead of the table.
- Result caps are never silent: any truncation prints what was dropped.
- Missing/unmatched data is counted and reported ("unmatched"), never
  silently treated as busy, idle, or zero.
- Architecture per tool: pure aggregation core + thin query runners +
  `main()`. Every verdict/aggregation path reachable offline.
- Each tool has a `--from-json FILE` replay flag (reads a captured
  `{"<query-role>": <prometheus-response>, ...}` bundle instead of querying;
  role names per tool below; roles that are keyed per job hold
  `{<jobid>: <response>}` maps, and subprocess inputs ride in the same
  bundle as text under `"sacct"` / `"sinfo"`) — the test seam, same
  pattern as `gpu_health --from-xml`.

## Tool designs

### xid_history

```
xid_history [--days N | --since ISO --until ISO] [--node NODE] [--json]
            [--from-json FILE]
```

- Window default: 7 days, ending now. Step: 300 s.
- Query 1 (`role "xid"`): `query_range` of `DCGM_FI_DEV_XID_ERRORS != 0`
  (optionally `{Hostname="NODE"}` with `--node`) over the window.
- Pure core `episodes(matrix)`: compress consecutive samples per
  (Hostname, gpu, UUID, err_code) into episodes with `first_seen`,
  `last_seen`, `err_code`, `err_msg`, `node`, `gpu`, `uuid`. A gap larger
  than 2 steps starts a new episode.
- Query 2 (`role "jobid"`): for each episode, instant query
  `nvidia_gpu_jobId` at `first_seen` (batched: one query per distinct
  timestamp), matched to the episode python-side on uuid (both casings).
  Job id 0 / no match → "-" (GPU was unallocated or exporter gap).
- Join 3: one batched `sacct -j id1,id2,... -X -n -P
  -o JobIDRaw,User,Account,State` for user/account/state. sacct failure
  degrades to "-" fields with a warning line, not an error exit.
- Output: table sorted by `first_seen` — columns `FIRST_SEEN LAST_SEEN NODE
  GPU XID JOB USER STATE MESSAGE`; summary line with episode count and
  distinct nodes. `--json` emits the episode list.
- Exit 1 if any episodes, 0 if none, 3 on query failure.

### energy_report

```
energy_report (-u USER | -A ACCOUNT) [--days N | --since ISO --until ISO]
              [--top N] [--max-jobs M] [--json] [--from-json FILE]
```

- Window default: 7 days. Exactly one of `-u`/`-A` required (exit 3 via
  err_exit otherwise).
- Job listing: `sacct -X -n -P -o JobIDRaw,User,Account,Start,End,Elapsed,
  AllocTRES,State` for the user/account and window; keep jobs whose
  AllocTRES contains `gres/gpu=`; parse GPU count. Jobs still running use
  now as End.
- Per job (capped at `--max-jobs`, default 200, truncation reported with
  the exact skipped count): discovery instant query (`role "discovery"`,
  one per job) `max_over_time((nvidia_gpu_jobId == <jobid>)[<elapsed>s:])`
  at End → the job's GPU uuids; energy instant query (`role "energy"`)
  `max_over_time(DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION{UUID=~"<uuids>"}
  [<elapsed>s:]) - min_over_time(...)` at End, python-side matched on UUID
  both casings, summed across the job's GPUs, mJ → kWh
  (`/ 3.6e9`). Jobs whose discovery or energy query returns nothing are
  listed as "no data" (counted, shown), not dropped.
- Pure core `rollup(job_rows)`: totals (kWh, GPU-hours from
  Elapsed × gpu_count), top-N jobs by kWh, per-user breakdown when `-A`.
- Output: summary block (window, jobs counted / no-data / skipped, total
  kWh, total GPU-hours, mean W per GPU derived as kWh/GPU-h × 1000), top-N
  table `JOBID USER KWH GPU_H MEAN_W STATE`. Report-only: exit 0, or 3 on
  sacct/query failure.

### fleet_util

```
fleet_util [--days N | --since ISO --until ISO] [--partition P]
           [--threshold PCT] [--step HOURS] [--json] [--from-json FILE]
```

- Window default: 7 days; step default 1 h; threshold default 10 (%).
- Query 1 (`role "alloc"`): `query_range` of `nvidia_gpu_jobId > 0` — the
  per-GPU allocation indicator (labels: lowercase uuid, host).
- Query 2 (`role "smact"`): `query_range` of `DCGM_FI_PROF_SM_ACTIVE`
  (labels: uppercase UUID, Hostname).
- Node→partition map: one `sinfo -N -h -o "%n %P"` call (a node may appear
  in several partitions; attribute its GPU-hours to each partition it is
  in, and say so in the report footer). `--partition` filters to one.
- Pure core `utilization(alloc_matrix, smact_matrix, node_to_parts,
  threshold, step_s)`: per timestep per uuid — allocated if the alloc
  sample > 0; wasted if allocated AND matching smact sample × 100 <
  threshold; unmatched if allocated but no smact sample at that timestep
  (counted separately, never guessed). Aggregates to per-partition and
  overall: allocated GPU-h, wasted GPU-h, wasted %, unmatched GPU-h, plus
  the top 5 nodes by wasted GPU-h.
- Output: per-partition table + overall row + worst-nodes list + unmatched
  note. Report-only: exit 0, or 3 on failure.

## Error handling

- Prometheus non-success / HTTP error / timeout → exit 3 with a one-line
  message that never includes the endpoint URL.
- sacct/sinfo failures: exit 3 where the tool cannot proceed
  (energy_report listing, fleet_util partition map); degrade with a
  printed warning where the data is enrichment (xid_history user lookup).
- Grafana-Cloud query cost is bounded by design: coarse steps, batched
  instant queries, `--max-jobs` cap; all caps reported.

## Testing

Offline stdlib-`unittest` under `gpu/tests/` (extends the gpu-health
scaffolding; loader pattern for extensionless tools):

- fixtures: captured-shape Prometheus JSON (`query_range` matrix with XID
  episodes incl. a gap split; alloc/smact matrices with a wasted GPU, a
  busy GPU, and an unmatched-uuid case; energy min/max vectors) plus sacct
  and sinfo text fixtures;
- unit tests for `episodes` (gap splitting, multi-code), the uuid-casing
  join, energy delta math (mJ→kWh) and no-data accounting, `rollup`
  totals/top-N, `utilization` (threshold boundary, unmatched counting,
  multi-partition attribution);
- CLI exit-path tests via `--from-json` replay: 0 (clean), 1 (xid found),
  3 (bad window args, missing file);
- live smoke after build (real XIDs exist; this session's job ids feed
  energy_report; fleet_util runs on the real week).

## Documentation

`gpu/README.md`: three new table rows + a note that these are
window/fleet tools (vs the per-job profilers) and that they run wherever
jobstats is installed — no allocation needed. Top-level `README.md` gpu/
section: same three rows; requirements unchanged (jobstats config +
requests). Record the throttle_report drop and its exporter knob in both
this spec and the gpu/README note.
