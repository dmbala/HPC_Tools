# gpu_health — node-local GPU health probe

**Date:** 2026-07-02
**Location:** `gpu/gpu_health`
**Status:** approved design, pre-implementation

## Purpose

A node-local, read-only, unprivileged GPU health probe. Run it on any GPU node
and get a per-GPU and node-level OK/WARN/FAIL verdict in a few seconds. It
checks hardware health — not utilization — and is the complement to
`jobstats_dcgm`: that tool tells you how hard the GPUs worked, this one tells
you whether they are trustworthy.

Non-goals: utilization/profiling metrics (covered by `gpu/jobstats_dcgm`),
XID/dmesg scraping (often root-restricted), active tests (`dcgmi diag`),
fleet aggregation (a future tool can consume this tool's JSON output).

## Interface

```
gpu_health [--json [FILE]] [--from-xml FILE]
```

- No flags: human-readable report to stdout, verdict via exit code.
- `--json [FILE]`: emit a JSON snapshot instead of the report; to stdout when
  FILE is omitted.
- `--from-xml FILE`: replay a captured `nvidia-smi -q -x` dump instead of
  running commands (testing/forensics; same replay pattern as
  `fabric/affinity_check.sh --snapshot`). NVLink checks report `n/a` in this
  mode unless a matching `FILE.nvlink` text capture sits next to the XML file.

Exit codes: `0` OK, `1` WARN, `2` FAIL, `3` probe error (no `nvidia-smi`, no
GPUs, XML parse failure). 0/1/2 follow the Nagios convention so the tool can
gate scripts and health checks directly.

## Data collection

Two read-only commands:

1. `nvidia-smi -q -x` — parsed with `xml.etree.ElementTree`. Supplies ECC
   counts (volatile + aggregate), row-remap state, retired-pages state (pre-A100),
   clock-throttle reasons, temperature and slowdown threshold, power draw and
   limit, PCIe replay counters, driver version, GPU name/serial/index.
2. `nvidia-smi nvlink --status -e` (per GPU) — NVLink error counters. Skipped
   cleanly (`n/a`) on GPUs without NVLink.

No DCGM/hostengine dependency, no Prometheus, no root.

## Checks and verdict rules

Each category evaluates to OK/WARN/FAIL per GPU. A GPU's verdict is the worst
of its checks; the node verdict is the worst GPU.

| Tier | Condition |
|---|---|
| FAIL | Volatile (since-boot) uncorrectable ECC > 0 |
| FAIL | Row-remap state `pending` or `failure` (GPU needs reset or RMA) |
| FAIL | Retired pages pending (pre-A100 GPUs) |
| FAIL | HW-slowdown throttle reason active |
| WARN | Aggregate (lifetime) uncorrectable ECC > 0 |
| WARN | Volatile correctable ECC > `CORRECTABLE_ECC_WARN` (default 100) |
| WARN | Thermal or power-cap (SW) throttle reason currently active |
| WARN | PCIe replay counter > 0 |
| WARN | Any nonzero NVLink error counter (CRC, replay, recovery) |
| WARN | Temperature within `TEMP_MARGIN_C` (default 5 °C) of slowdown threshold |

Rules:

- A capability that is absent or disabled (ECC off, no NVLink) reports `n/a`
  and never affects the verdict.
- A field that cannot be parsed (driver renamed/moved it) reports `n/a` with a
  note; the tool never crashes on it and never reports OK for a check it did
  not evaluate.
- All thresholds are named constants at the top of the file
  (`CORRECTABLE_ECC_WARN`, `TEMP_MARGIN_C`).

## Output

**Human report (default):** one block per GPU — index, name, serial — listing
each check with its tier and the tripped values spelled out (e.g.
`ecc: FAIL — volatile uncorrectable DRAM errors: 3`), then a final
`node verdict: FAIL (GPU 2)` line. Plain text, no color.

**JSON snapshot (`--json`):**

```json
{
  "host": "holygpu8a15602",
  "timestamp": "2026-07-02T14:03:11",
  "driver_version": "555.42",
  "verdict": "WARN",
  "gpus": [
    {
      "index": 0,
      "name": "NVIDIA H100",
      "serial": "165342...",
      "verdict": "WARN",
      "checks": {
        "ecc":      {"status": "OK",  "volatile_uncorrectable": 0, "aggregate_uncorrectable": 0, "volatile_correctable": 2, "row_remap": "none"},
        "throttle": {"status": "WARN", "active": ["sw_power_cap"], "temp_c": 62, "slowdown_temp_c": 90, "power_w": 698.0, "power_limit_w": 700.0},
        "pcie":     {"status": "OK",  "replay_counter": 0},
        "nvlink":   {"status": "OK",  "errors": {}}
      }
    }
  ]
}
```

The shape is deliberately snapshot-like (host + timestamp + per-GPU dicts) so a
future fleet aggregator or a before/after diff in the `counter_delta` style can
consume it unchanged.

## Structure

Single executable Python 3 stdlib file, `#!/usr/bin/env python3`, executable
bit set. Small pure core:

- `parse_smi_xml(xml_text) -> dict` — XML in, plain per-GPU dicts out.
- `parse_nvlink(text) -> dict` — NVLink counter text in, per-GPU error dicts out.
- `evaluate(gpu) -> {check: (tier, detail)}` — all verdict logic, pure.
- `render_text(result)` / `render_json(result)` — presentation only.
- `main()` — argparse, command execution, exit-code mapping.

## Error handling

- `nvidia-smi` missing, no GPUs visible, or XML that does not parse: one-line
  error to stderr, exit 3 (distinct from health verdicts so wrappers can tell
  "sick GPU" from "probe could not run").
- Per-field parse failures degrade to `n/a` per the rules above.
- `nvidia-smi nvlink` failing entirely (older drivers) degrades the NVLink
  check to `n/a`, not an error.

## Testing

Offline, via `--from-xml` fixtures under `gpu/tests/fixtures/`:

- one healthy H100 capture (all checks OK), with its companion `.nvlink`
  text capture,
- doctored copies exercising each FAIL and WARN rule (the NVLink WARN rule
  via a doctored `.nvlink` capture),
- an ECC-disabled capture and a no-NVLink capture for the `n/a` paths,
- a truncated/garbled XML for the exit-3 path.

A small stdlib-`unittest` file `gpu/tests/test_gpu_health.py` drives
`parse_smi_xml` + `evaluate` against the fixtures and asserts tiers and exit
codes. No cluster required.

## Documentation

- Add a `gpu_health` row to the `gpu/README.md` tool table, with a note that
  unlike the other `gpu/` tools it runs **on the node** (no Prometheus, no
  jobstats modules).
- Update the top-level `README.md` requirements table: `gpu_health` needs only
  `nvidia-smi` and Python 3 stdlib, and runs on the target GPU node.
