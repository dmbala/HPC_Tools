# gpu_health Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A node-local, read-only, unprivileged GPU health probe (`gpu/gpu_health`) that checks ECC & row-remap, throttling, and PCIe/NVLink errors via `nvidia-smi` and reports tiered OK/WARN/FAIL verdicts per GPU and per node.

**Architecture:** Single executable Python stdlib script. A pure core — `parse_smi_xml()` and `parse_nvlink()` turn command output into plain dicts, `evaluate()` maps one GPU dict to per-check `(tier, detail)` pairs, `build_result()` assembles the JSON-schema snapshot — wrapped by renderers and a `main()` that maps verdicts to exit codes. All tests run offline against fixture captures via the `--from-xml` replay flag.

**Tech Stack:** Python 3 stdlib only (`argparse`, `xml.etree.ElementTree`, `subprocess`, `json`, `re`, `unittest`). No pip packages, no DCGM, no Prometheus.

**Spec:** `docs/specs/2026-07-02-gpu-health-design.md` (approved).

## Global Constraints

- Python 3.6-compatible stdlib only (cluster nodes run RHEL8 system Python 3.6.8): no `subprocess.run(capture_output=...)` / `text=...` (use `stdout=subprocess.PIPE, universal_newlines=True`), no walrus, no dataclasses.
- The tool file is `gpu/gpu_health` — no `.py` extension, `#!/usr/bin/env python3` shebang, executable bit set (matches `gpu/jobstats_dcgm`; the repo README promises "the executable bit is preserved by git").
- Read-only: the only commands the tool may run are `nvidia-smi -q -x` and `nvidia-smi nvlink -e`.
- Exit codes: `0` OK, `1` WARN, `2` FAIL, `3` probe error. Never crash on a missing/renamed XML field — degrade that check to `n/a`.
- Thresholds are module constants: `CORRECTABLE_ECC_WARN = 100`, `TEMP_MARGIN_C = 5`.
- Plain-text output, no ANSI color.
- Tests: stdlib `unittest`, run as `python3 -m unittest discover -s gpu/tests -v` from the repo root. No cluster/GPU required.
- Commit messages: short imperative subject, no attribution footers of any kind.

## File Structure

| File | Responsibility |
|---|---|
| `gpu/gpu_health` (create) | The whole tool: constants, XML/NVLink parsers, verdict logic, renderers, CLI. |
| `gpu/tests/fixtures/healthy.xml` (create) | Captured-style healthy 2-GPU `nvidia-smi -q -x` dump; base for in-test doctoring. |
| `gpu/tests/fixtures/healthy.xml.nvlink` (create) | Companion `nvidia-smi nvlink -e` capture (all counters zero). |
| `gpu/tests/fixtures/ecc_disabled.xml` (create) | 1-GPU dump with ECC disabled and no remap/retired sections → `n/a` paths. |
| `gpu/tests/fixtures/garbled.xml` (create) | Truncated XML → probe-error (exit 3) path. |
| `gpu/tests/test_gpu_health.py` (create) | All unit + CLI tests, grown task by task. |
| `gpu/README.md` (modify) | Add `gpu_health` row + node-local note. |
| `README.md` (modify) | Add `gpu_health` requirements row. |

Per the spec's testing section, each FAIL/WARN rule is exercised offline: verdict rules unit-test `evaluate()` directly on dicts, and the end-to-end CLI tests doctor `healthy.xml` by string replacement instead of committing ten near-identical XML fixtures (DRY).

---

### Task 1: Fixtures, test harness, and `parse_smi_xml`

**Files:**
- Create: `gpu/tests/fixtures/healthy.xml`
- Create: `gpu/tests/fixtures/healthy.xml.nvlink`
- Create: `gpu/tests/fixtures/ecc_disabled.xml`
- Create: `gpu/tests/fixtures/garbled.xml`
- Create: `gpu/tests/test_gpu_health.py`
- Create: `gpu/gpu_health`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: module constants `OK`, `WARN`, `FAIL`, `NA` (strings `"OK"/"WARN"/"FAIL"/"n/a"`), `CORRECTABLE_ECC_WARN`, `TEMP_MARGIN_C`, `EXIT_ERROR = 3`; `parse_smi_xml(xml_text) -> dict` returning `{"driver_version": str|None, "gpus": [gpu_dict, ...]}` and raising `ValueError` on unparseable/GPU-less XML. Each `gpu_dict` has exactly these keys: `index` (int), `name` (str|None), `serial` (str|None), `ecc_enabled` (bool|None), `volatile_correctable`/`volatile_uncorrectable`/`aggregate_uncorrectable` (int|None), `row_remap_pending`/`row_remap_failure`/`retired_pages_pending` (bool|None), `throttle` (dict of `sw_power_cap`/`sw_thermal`/`hw_thermal`/`hw_slowdown` → bool|None, or None), `temp_c`/`slowdown_temp_c` (int|None), `power_w`/`power_limit_w` (float|None), `pcie_replay` (int|None). Test-side helpers `fixture(name)` and the module loader `gh`.

- [ ] **Step 1: Create the fixtures**

`gpu/tests/fixtures/healthy.xml` (Hopper-era layout: `sram_*`/`dram_*` ECC counts, `remapped_rows`, `gpu_power_readings`):

```xml
<?xml version="1.0" ?>
<nvidia_smi_log>
  <timestamp>Thu Jul  2 10:00:00 2026</timestamp>
  <driver_version>550.54.15</driver_version>
  <attached_gpus>2</attached_gpus>
  <gpu id="00000000:19:00.0">
    <product_name>NVIDIA H100 80GB HBM3</product_name>
    <serial>1650000000001</serial>
    <minor_number>0</minor_number>
    <ecc_mode>
      <current_ecc>Enabled</current_ecc>
      <pending_ecc>Enabled</pending_ecc>
    </ecc_mode>
    <ecc_errors>
      <volatile>
        <sram_correctable>0</sram_correctable>
        <sram_uncorrectable>0</sram_uncorrectable>
        <dram_correctable>0</dram_correctable>
        <dram_uncorrectable>0</dram_uncorrectable>
      </volatile>
      <aggregate>
        <sram_correctable>0</sram_correctable>
        <sram_uncorrectable>0</sram_uncorrectable>
        <dram_correctable>0</dram_correctable>
        <dram_uncorrectable>0</dram_uncorrectable>
      </aggregate>
    </ecc_errors>
    <remapped_rows>
      <remapped_row_corr>0</remapped_row_corr>
      <remapped_row_unc>0</remapped_row_unc>
      <remapped_row_pending>No</remapped_row_pending>
      <remapped_row_failure>No</remapped_row_failure>
    </remapped_rows>
    <temperature>
      <gpu_temp>41 C</gpu_temp>
      <gpu_temp_slow_threshold>92 C</gpu_temp_slow_threshold>
    </temperature>
    <clocks_throttle_reasons>
      <clocks_throttle_reason_sw_power_cap>Not Active</clocks_throttle_reason_sw_power_cap>
      <clocks_throttle_reason_sw_thermal_slowdown>Not Active</clocks_throttle_reason_sw_thermal_slowdown>
      <clocks_throttle_reason_hw_thermal_slowdown>Not Active</clocks_throttle_reason_hw_thermal_slowdown>
      <clocks_throttle_reason_hw_slowdown>Not Active</clocks_throttle_reason_hw_slowdown>
    </clocks_throttle_reasons>
    <gpu_power_readings>
      <power_draw>71.50 W</power_draw>
      <current_power_limit>700.00 W</current_power_limit>
    </gpu_power_readings>
    <pci>
      <replay_counter>0</replay_counter>
    </pci>
  </gpu>
  <gpu id="00000000:3B:00.0">
    <product_name>NVIDIA H100 80GB HBM3</product_name>
    <serial>1650000000002</serial>
    <minor_number>1</minor_number>
    <ecc_mode>
      <current_ecc>Enabled</current_ecc>
      <pending_ecc>Enabled</pending_ecc>
    </ecc_mode>
    <ecc_errors>
      <volatile>
        <sram_correctable>0</sram_correctable>
        <sram_uncorrectable>0</sram_uncorrectable>
        <dram_correctable>0</dram_correctable>
        <dram_uncorrectable>0</dram_uncorrectable>
      </volatile>
      <aggregate>
        <sram_correctable>0</sram_correctable>
        <sram_uncorrectable>0</sram_uncorrectable>
        <dram_correctable>0</dram_correctable>
        <dram_uncorrectable>0</dram_uncorrectable>
      </aggregate>
    </ecc_errors>
    <remapped_rows>
      <remapped_row_corr>0</remapped_row_corr>
      <remapped_row_unc>0</remapped_row_unc>
      <remapped_row_pending>No</remapped_row_pending>
      <remapped_row_failure>No</remapped_row_failure>
    </remapped_rows>
    <temperature>
      <gpu_temp>39 C</gpu_temp>
      <gpu_temp_slow_threshold>92 C</gpu_temp_slow_threshold>
    </temperature>
    <clocks_throttle_reasons>
      <clocks_throttle_reason_sw_power_cap>Not Active</clocks_throttle_reason_sw_power_cap>
      <clocks_throttle_reason_sw_thermal_slowdown>Not Active</clocks_throttle_reason_sw_thermal_slowdown>
      <clocks_throttle_reason_hw_thermal_slowdown>Not Active</clocks_throttle_reason_hw_thermal_slowdown>
      <clocks_throttle_reason_hw_slowdown>Not Active</clocks_throttle_reason_hw_slowdown>
    </clocks_throttle_reasons>
    <gpu_power_readings>
      <power_draw>68.20 W</power_draw>
      <current_power_limit>700.00 W</current_power_limit>
    </gpu_power_readings>
    <pci>
      <replay_counter>0</replay_counter>
    </pci>
  </gpu>
</nvidia_smi_log>
```

`gpu/tests/fixtures/healthy.xml.nvlink`:

```text
GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-11111111-1111-1111-1111-111111111111)
         Link 0: Replay Errors: 0
         Link 0: Recovery Errors: 0
         Link 0: CRC Errors: 0
         Link 1: Replay Errors: 0
         Link 1: Recovery Errors: 0
         Link 1: CRC Errors: 0
GPU 1: NVIDIA H100 80GB HBM3 (UUID: GPU-22222222-2222-2222-2222-222222222222)
         Link 0: Replay Errors: 0
         Link 0: Recovery Errors: 0
         Link 0: CRC Errors: 0
         Link 1: Replay Errors: 0
         Link 1: Recovery Errors: 0
         Link 1: CRC Errors: 0
```

`gpu/tests/fixtures/ecc_disabled.xml`:

```xml
<?xml version="1.0" ?>
<nvidia_smi_log>
  <driver_version>535.104.05</driver_version>
  <attached_gpus>1</attached_gpus>
  <gpu id="00000000:65:00.0">
    <product_name>NVIDIA GeForce RTX 4090</product_name>
    <serial>N/A</serial>
    <minor_number>0</minor_number>
    <ecc_mode>
      <current_ecc>Disabled</current_ecc>
      <pending_ecc>Disabled</pending_ecc>
    </ecc_mode>
    <temperature>
      <gpu_temp>35 C</gpu_temp>
      <gpu_temp_slow_threshold>N/A</gpu_temp_slow_threshold>
    </temperature>
    <clocks_throttle_reasons>
      <clocks_throttle_reason_sw_power_cap>Not Active</clocks_throttle_reason_sw_power_cap>
      <clocks_throttle_reason_sw_thermal_slowdown>Not Active</clocks_throttle_reason_sw_thermal_slowdown>
      <clocks_throttle_reason_hw_thermal_slowdown>Not Active</clocks_throttle_reason_hw_thermal_slowdown>
      <clocks_throttle_reason_hw_slowdown>Not Active</clocks_throttle_reason_hw_slowdown>
    </clocks_throttle_reasons>
    <power_readings>
      <power_draw>28.00 W</power_draw>
      <power_limit>450.00 W</power_limit>
    </power_readings>
    <pci>
      <replay_counter>0</replay_counter>
    </pci>
  </gpu>
</nvidia_smi_log>
```

(Note: this fixture deliberately uses the older `power_readings`/`power_limit` tag names to cover the fallback paths.)

`gpu/tests/fixtures/garbled.xml`:

```text
<?xml version="1.0" ?>
<nvidia_smi_log>
  <driver_version>550.54.15</driver_version>
  <gpu id="00000000:19:00.0">
    <product_name>NVIDIA H100
```

- [ ] **Step 2: Write the failing tests with the module loader**

`gpu/tests/test_gpu_health.py`:

```python
"""Tests for gpu/gpu_health. Offline: fixtures only, no GPU needed."""
import importlib.machinery
import importlib.util
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.abspath(os.path.join(HERE, "..", "gpu_health"))
FIXTURES = os.path.join(HERE, "fixtures")

# gpu_health has no .py extension, so load it by explicit path.
_loader = importlib.machinery.SourceFileLoader("gpu_health", TOOL)
_spec = importlib.util.spec_from_loader("gpu_health", _loader)
gh = importlib.util.module_from_spec(_spec)
_loader.exec_module(gh)


def fixture(name):
    with open(os.path.join(FIXTURES, name)) as fh:
        return fh.read()


class TestParseSmiXml(unittest.TestCase):
    def test_healthy_two_gpus(self):
        parsed = gh.parse_smi_xml(fixture("healthy.xml"))
        self.assertEqual(parsed["driver_version"], "550.54.15")
        self.assertEqual(len(parsed["gpus"]), 2)
        g0 = parsed["gpus"][0]
        self.assertEqual(g0["index"], 0)
        self.assertEqual(g0["name"], "NVIDIA H100 80GB HBM3")
        self.assertEqual(g0["serial"], "1650000000001")
        self.assertIs(g0["ecc_enabled"], True)
        self.assertEqual(g0["volatile_correctable"], 0)
        self.assertEqual(g0["volatile_uncorrectable"], 0)
        self.assertEqual(g0["aggregate_uncorrectable"], 0)
        self.assertIs(g0["row_remap_pending"], False)
        self.assertIs(g0["row_remap_failure"], False)
        self.assertIsNone(g0["retired_pages_pending"])
        self.assertEqual(g0["throttle"], {"sw_power_cap": False,
                                          "sw_thermal": False,
                                          "hw_thermal": False,
                                          "hw_slowdown": False})
        self.assertEqual(g0["temp_c"], 41)
        self.assertEqual(g0["slowdown_temp_c"], 92)
        self.assertEqual(g0["power_w"], 71.5)
        self.assertEqual(g0["power_limit_w"], 700.0)
        self.assertEqual(g0["pcie_replay"], 0)
        self.assertEqual(parsed["gpus"][1]["index"], 1)

    def test_doctored_counts_parse(self):
        xml = fixture("healthy.xml").replace(
            "<dram_uncorrectable>0</dram_uncorrectable>",
            "<dram_uncorrectable>3</dram_uncorrectable>", 1)
        g0 = gh.parse_smi_xml(xml)["gpus"][0]
        self.assertEqual(g0["volatile_uncorrectable"], 3)

    def test_ecc_disabled_fixture(self):
        g0 = gh.parse_smi_xml(fixture("ecc_disabled.xml"))["gpus"][0]
        self.assertIs(g0["ecc_enabled"], False)
        self.assertIsNone(g0["volatile_uncorrectable"])
        self.assertIsNone(g0["row_remap_pending"])
        self.assertIsNone(g0["serial"])          # serial is N/A
        self.assertIsNone(g0["slowdown_temp_c"])  # threshold is N/A
        self.assertEqual(g0["power_w"], 28.0)     # old power_readings tags
        self.assertEqual(g0["power_limit_w"], 450.0)

    def test_garbled_raises(self):
        with self.assertRaises(ValueError):
            gh.parse_smi_xml(fixture("garbled.xml"))

    def test_no_gpus_raises(self):
        with self.assertRaises(ValueError):
            gh.parse_smi_xml("<nvidia_smi_log></nvidia_smi_log>")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m unittest discover -s gpu/tests -v`
Expected: ERROR at import time — `FileNotFoundError` for `gpu/gpu_health` (the tool does not exist yet).

- [ ] **Step 4: Write the tool skeleton with `parse_smi_xml`**

`gpu/gpu_health`:

```python
#!/usr/bin/env python3
"""gpu_health -- node-local, read-only GPU health probe.

Checks ECC & row-remap state, clock throttling, and PCIe/NVLink error
counters via nvidia-smi, and reports a tiered OK/WARN/FAIL verdict per GPU
and for the node. Hardware health only -- for utilization/profiling see
jobstats_dcgm in this directory.

Usage:
    gpu_health                 # human-readable report, verdict via exit code
    gpu_health --json [FILE]   # JSON snapshot to stdout (or FILE)
    gpu_health --from-xml F    # replay a captured `nvidia-smi -q -x` dump
                               # (NVLink counters read from F.nvlink if present)

Exit codes: 0 OK, 1 WARN, 2 FAIL, 3 probe error (nvidia-smi missing/failed).
"""

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime

# Verdict thresholds
CORRECTABLE_ECC_WARN = 100  # volatile correctable ECC errors before WARN
TEMP_MARGIN_C = 5           # WARN when temp is within this of slowdown threshold

OK, WARN, FAIL, NA = "OK", "WARN", "FAIL", "n/a"
_SEVERITY = {NA: 0, OK: 0, WARN: 1, FAIL: 2}
_EXIT_FOR = {OK: 0, WARN: 1, FAIL: 2}
EXIT_ERROR = 3


def _find_text(node, *paths):
    """First meaningful text at any of the candidate relative paths, else None.

    Multiple paths cover tag renames across driver versions. 'N/A' and empty
    text count as absent.
    """
    if node is None:
        return None
    for path in paths:
        el = node.find(path)
        if el is not None and el.text:
            text = el.text.strip()
            if text and text != "N/A":
                return text
    return None


def _to_int(text):
    if text is None:
        return None
    m = re.match(r"-?\d+", text)
    return int(m.group()) if m else None


def _to_float(text):
    if text is None:
        return None
    m = re.match(r"-?\d+(\.\d+)?", text)
    return float(m.group()) if m else None


def _yesno(text):
    """'Yes'/'No' -> bool; None/unrecognized -> None."""
    if text is None:
        return None
    return text.strip().lower() == "yes"


def _throttle_flag(node, *paths):
    """'Active'/'Not Active' -> bool; absent -> None."""
    text = _find_text(node, *paths)
    if text is None:
        return None
    return text.strip().lower() == "active"


def _ecc_count(section, kind):
    """Total ECC errors of `kind` ('correctable'|'uncorrectable') in a
    <volatile> or <aggregate> section. Handles both the sram_*/dram_* layout
    (A100/H100) and the older single_bit/double_bit layout. None if absent."""
    if section is None:
        return None
    sram = _to_int(_find_text(section, "sram_%s" % kind))
    dram = _to_int(_find_text(section, "dram_%s" % kind))
    if sram is not None or dram is not None:
        return (sram or 0) + (dram or 0)
    old = "single_bit" if kind == "correctable" else "double_bit"
    return _to_int(_find_text(section, old + "/total"))


def parse_smi_xml(xml_text):
    """Parse `nvidia-smi -q -x` output.

    Returns {'driver_version': str|None, 'gpus': [dict, ...]}.
    Raises ValueError when the XML does not parse or contains no <gpu>.
    Missing individual fields become None (n/a downstream), never an error.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ValueError("nvidia-smi XML did not parse: %s" % exc)
    gpu_nodes = root.findall("gpu")
    if not gpu_nodes:
        raise ValueError("no <gpu> elements in nvidia-smi XML")

    gpus = []
    for position, node in enumerate(gpu_nodes):
        ecc = node.find("ecc_errors")
        volatile = ecc.find("volatile") if ecc is not None else None
        aggregate = ecc.find("aggregate") if ecc is not None else None
        remap = node.find("remapped_rows")
        # driver >= 555 renamed clocks_throttle_reasons -> clocks_event_reasons
        throttle_node = node.find("clocks_throttle_reasons")
        if throttle_node is None:
            throttle_node = node.find("clocks_event_reasons")
        throttle = None
        if throttle_node is not None:
            throttle = {
                "sw_power_cap": _throttle_flag(
                    throttle_node,
                    "clocks_throttle_reason_sw_power_cap",
                    "clocks_event_reason_sw_power_cap"),
                "sw_thermal": _throttle_flag(
                    throttle_node,
                    "clocks_throttle_reason_sw_thermal_slowdown",
                    "clocks_event_reason_sw_thermal_slowdown"),
                "hw_thermal": _throttle_flag(
                    throttle_node,
                    "clocks_throttle_reason_hw_thermal_slowdown",
                    "clocks_event_reason_hw_thermal_slowdown"),
                "hw_slowdown": _throttle_flag(
                    throttle_node,
                    "clocks_throttle_reason_hw_slowdown",
                    "clocks_event_reason_hw_slowdown"),
            }
        ecc_mode = _find_text(node, "ecc_mode/current_ecc")
        minor = _to_int(_find_text(node, "minor_number"))
        gpus.append({
            "index": minor if minor is not None else position,
            "name": _find_text(node, "product_name"),
            "serial": _find_text(node, "serial"),
            "ecc_enabled": None if ecc_mode is None
                           else ecc_mode.lower() == "enabled",
            "volatile_correctable": _ecc_count(volatile, "correctable"),
            "volatile_uncorrectable": _ecc_count(volatile, "uncorrectable"),
            "aggregate_uncorrectable": _ecc_count(aggregate, "uncorrectable"),
            "row_remap_pending": _yesno(_find_text(remap, "remapped_row_pending")),
            "row_remap_failure": _yesno(_find_text(remap, "remapped_row_failure")),
            "retired_pages_pending": _yesno(
                _find_text(node, "retired_pages/pending_retirement")),
            "throttle": throttle,
            "temp_c": _to_int(_find_text(node, "temperature/gpu_temp")),
            "slowdown_temp_c": _to_int(
                _find_text(node, "temperature/gpu_temp_slow_threshold")),
            "power_w": _to_float(_find_text(
                node, "gpu_power_readings/power_draw",
                "power_readings/power_draw")),
            "power_limit_w": _to_float(_find_text(
                node, "gpu_power_readings/current_power_limit",
                "power_readings/current_power_limit",
                "power_readings/power_limit")),
            "pcie_replay": _to_int(_find_text(node, "pci/replay_counter")),
        })
    return {"driver_version": _find_text(root, "driver_version"), "gpus": gpus}


if __name__ == "__main__":
    sys.exit(0)  # replaced by main() in a later task
```

- [ ] **Step 5: Make the tool executable**

Run: `chmod +x gpu/gpu_health && ls -l gpu/gpu_health`
Expected: mode `-rwxr-xr-x`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `python3 -m unittest discover -s gpu/tests -v`
Expected: 5 tests, all PASS.

- [ ] **Step 7: Commit**

```bash
git add gpu/gpu_health gpu/tests
git commit -m "Add gpu_health skeleton: nvidia-smi XML parser with fixtures"
```

---

### Task 2: `parse_nvlink`

**Files:**
- Modify: `gpu/gpu_health` (add one function after `parse_smi_xml`)
- Test: `gpu/tests/test_gpu_health.py` (append a class)

**Interfaces:**
- Consumes: fixtures and loader from Task 1.
- Produces: `parse_nvlink(text) -> dict` mapping `gpu_index (int) -> {counter_name (str): total (int)}`, counter names lowercased with underscores (e.g. `crc_errors`, `replay_errors`, `recovery_errors`), totals summed across links. Returns `{}` for empty/unrecognized input (caller treats `{}` as n/a).

- [ ] **Step 1: Write the failing tests**

Append to `gpu/tests/test_gpu_health.py`:

```python
class TestParseNvlink(unittest.TestCase):
    def test_healthy_capture(self):
        per_gpu = gh.parse_nvlink(fixture("healthy.xml.nvlink"))
        self.assertEqual(sorted(per_gpu), [0, 1])
        self.assertEqual(per_gpu[0], {"replay_errors": 0,
                                      "recovery_errors": 0,
                                      "crc_errors": 0})

    def test_sums_across_links(self):
        text = fixture("healthy.xml.nvlink").replace(
            "Link 1: CRC Errors: 0", "Link 1: CRC Errors: 7", 1)
        per_gpu = gh.parse_nvlink(text)
        self.assertEqual(per_gpu[0]["crc_errors"], 7)
        self.assertEqual(per_gpu[1]["crc_errors"], 0)

    def test_unsupported_output_is_empty(self):
        self.assertEqual(gh.parse_nvlink(""), {})
        self.assertEqual(
            gh.parse_nvlink("NVLink is not supported on this device\n"), {})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest discover -s gpu/tests -v`
Expected: the 3 new tests ERROR with `AttributeError: module 'gpu_health' has no attribute 'parse_nvlink'`; the 5 earlier tests still PASS.

- [ ] **Step 3: Implement `parse_nvlink`**

Add to `gpu/gpu_health` after `parse_smi_xml`:

```python
_NVLINK_GPU_RE = re.compile(r"^GPU (\d+):")
_NVLINK_ERR_RE = re.compile(r"Link \d+:\s*([A-Za-z ]+?):\s*(\d+)\s*$")


def parse_nvlink(text):
    """Parse `nvidia-smi nvlink -e` output.

    Returns {gpu_index: {counter: total-across-links}}; {} when the output
    has no per-GPU sections (no NVLink / unsupported), which callers treat
    as n/a.
    """
    per_gpu = {}
    current = None
    for line in (text or "").splitlines():
        header = _NVLINK_GPU_RE.match(line.strip())
        if header:
            current = int(header.group(1))
            per_gpu.setdefault(current, {})
            continue
        err = _NVLINK_ERR_RE.search(line)
        if err and current is not None:
            key = err.group(1).strip().lower().replace(" ", "_")
            per_gpu[current][key] = per_gpu[current].get(key, 0) + int(err.group(2))
    return per_gpu
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest discover -s gpu/tests -v`
Expected: 8 tests, all PASS.

- [ ] **Step 5: Commit**

```bash
git add gpu/gpu_health gpu/tests/test_gpu_health.py
git commit -m "gpu_health: parse NVLink error counters"
```

---

### Task 3: `evaluate` and `worst` — the verdict rules

**Files:**
- Modify: `gpu/gpu_health` (two functions after `parse_nvlink`)
- Test: `gpu/tests/test_gpu_health.py` (append helper + class)

**Interfaces:**
- Consumes: gpu_dict shape from Task 1, nvlink dict shape from Task 2, tier constants.
- Produces: `evaluate(gpu, nvlink) -> {"ecc"|"throttle"|"pcie"|"nvlink": (tier, detail_str)}` where `nvlink` is that GPU's counter dict or `None` for n/a; `worst(tiers_iterable) -> tier` (NA never worsens; `OK` for empty input).

- [ ] **Step 1: Write the failing tests — one per spec rule**

Append to `gpu/tests/test_gpu_health.py`:

```python
def healthy_gpu(**overrides):
    gpu = {
        "index": 0, "name": "NVIDIA H100", "serial": "1650000000001",
        "ecc_enabled": True,
        "volatile_correctable": 0, "volatile_uncorrectable": 0,
        "aggregate_uncorrectable": 0,
        "row_remap_pending": False, "row_remap_failure": False,
        "retired_pages_pending": None,
        "throttle": {"sw_power_cap": False, "sw_thermal": False,
                     "hw_thermal": False, "hw_slowdown": False},
        "temp_c": 40, "slowdown_temp_c": 92,
        "power_w": 100.0, "power_limit_w": 700.0,
        "pcie_replay": 0,
    }
    gpu.update(overrides)
    return gpu


NO_NVLINK_ERRORS = {"replay_errors": 0, "recovery_errors": 0, "crc_errors": 0}


class TestEvaluate(unittest.TestCase):
    def tiers(self, gpu, nvlink=NO_NVLINK_ERRORS):
        return {k: v[0] for k, v in gh.evaluate(gpu, nvlink).items()}

    def test_all_ok(self):
        self.assertEqual(self.tiers(healthy_gpu()),
                         {"ecc": gh.OK, "throttle": gh.OK,
                          "pcie": gh.OK, "nvlink": gh.OK})

    # --- FAIL rules ---
    def test_volatile_uncorrectable_fails(self):
        self.assertEqual(
            self.tiers(healthy_gpu(volatile_uncorrectable=1))["ecc"], gh.FAIL)

    def test_row_remap_pending_fails(self):
        self.assertEqual(
            self.tiers(healthy_gpu(row_remap_pending=True))["ecc"], gh.FAIL)

    def test_row_remap_failure_fails(self):
        self.assertEqual(
            self.tiers(healthy_gpu(row_remap_failure=True))["ecc"], gh.FAIL)

    def test_retired_pages_pending_fails(self):
        self.assertEqual(
            self.tiers(healthy_gpu(retired_pages_pending=True))["ecc"], gh.FAIL)

    def test_hw_slowdown_fails(self):
        gpu = healthy_gpu()
        gpu["throttle"]["hw_slowdown"] = True
        self.assertEqual(self.tiers(gpu)["throttle"], gh.FAIL)

    # --- WARN rules ---
    def test_aggregate_uncorrectable_warns(self):
        self.assertEqual(
            self.tiers(healthy_gpu(aggregate_uncorrectable=2))["ecc"], gh.WARN)

    def test_correctable_over_threshold_warns(self):
        gpu = healthy_gpu(volatile_correctable=gh.CORRECTABLE_ECC_WARN + 1)
        self.assertEqual(self.tiers(gpu)["ecc"], gh.WARN)

    def test_correctable_at_threshold_ok(self):
        gpu = healthy_gpu(volatile_correctable=gh.CORRECTABLE_ECC_WARN)
        self.assertEqual(self.tiers(gpu)["ecc"], gh.OK)

    def test_power_cap_throttle_warns(self):
        gpu = healthy_gpu()
        gpu["throttle"]["sw_power_cap"] = True
        self.assertEqual(self.tiers(gpu)["throttle"], gh.WARN)

    def test_thermal_throttle_warns(self):
        gpu = healthy_gpu()
        gpu["throttle"]["sw_thermal"] = True
        self.assertEqual(self.tiers(gpu)["throttle"], gh.WARN)

    def test_temp_near_slowdown_warns(self):
        gpu = healthy_gpu(temp_c=92 - gh.TEMP_MARGIN_C, slowdown_temp_c=92)
        self.assertEqual(self.tiers(gpu)["throttle"], gh.WARN)

    def test_temp_below_margin_ok(self):
        gpu = healthy_gpu(temp_c=92 - gh.TEMP_MARGIN_C - 1, slowdown_temp_c=92)
        self.assertEqual(self.tiers(gpu)["throttle"], gh.OK)

    def test_pcie_replay_warns(self):
        self.assertEqual(self.tiers(healthy_gpu(pcie_replay=4))["pcie"], gh.WARN)

    def test_nvlink_errors_warn(self):
        tiers = self.tiers(healthy_gpu(), nvlink={"crc_errors": 3})
        self.assertEqual(tiers["nvlink"], gh.WARN)

    # --- n/a rules: absent capability never affects the verdict ---
    def test_ecc_disabled_is_na(self):
        gpu = healthy_gpu(ecc_enabled=False, volatile_correctable=None,
                          volatile_uncorrectable=None,
                          aggregate_uncorrectable=None,
                          row_remap_pending=None, row_remap_failure=None)
        self.assertEqual(self.tiers(gpu)["ecc"], gh.NA)

    def test_ecc_unreported_is_na(self):
        gpu = healthy_gpu(ecc_enabled=None, volatile_correctable=None,
                          volatile_uncorrectable=None,
                          aggregate_uncorrectable=None,
                          row_remap_pending=None, row_remap_failure=None,
                          retired_pages_pending=None)
        self.assertEqual(self.tiers(gpu)["ecc"], gh.NA)

    def test_no_nvlink_is_na(self):
        self.assertEqual(self.tiers(healthy_gpu(), nvlink=None)["nvlink"], gh.NA)

    def test_no_throttle_data_is_na(self):
        gpu = healthy_gpu(throttle=None, temp_c=None, slowdown_temp_c=None)
        self.assertEqual(self.tiers(gpu)["throttle"], gh.NA)

    def test_pcie_unreported_is_na(self):
        self.assertEqual(self.tiers(healthy_gpu(pcie_replay=None))["pcie"], gh.NA)


class TestWorst(unittest.TestCase):
    def test_ordering(self):
        self.assertEqual(gh.worst([gh.OK, gh.WARN, gh.FAIL]), gh.FAIL)
        self.assertEqual(gh.worst([gh.OK, gh.WARN]), gh.WARN)
        self.assertEqual(gh.worst([gh.OK, gh.OK]), gh.OK)

    def test_na_never_worsens(self):
        self.assertEqual(gh.worst([gh.NA, gh.OK]), gh.OK)
        self.assertEqual(gh.worst([gh.NA]), gh.OK)
        self.assertEqual(gh.worst([]), gh.OK)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest discover -s gpu/tests -v`
Expected: the new tests ERROR with `AttributeError: ... no attribute 'evaluate'` (and `'worst'`); the 8 earlier tests still PASS.

- [ ] **Step 3: Implement `evaluate` and `worst`**

Add to `gpu/gpu_health` after `parse_nvlink`:

```python
def worst(tiers):
    """Worst tier in an iterable. NA never worsens a verdict; OK if empty."""
    result = OK
    for tier in tiers:
        if _SEVERITY.get(tier, 0) > _SEVERITY[result]:
            result = tier
    return result


def evaluate(gpu, nvlink):
    """Verdict rules from the spec: {check: (tier, detail)} for one GPU.

    `nvlink` is this GPU's counter dict from parse_nvlink, or None when
    NVLink data is unavailable (n/a).
    """
    checks = {}

    # ECC & row remap
    ecc_fields = ("volatile_uncorrectable", "aggregate_uncorrectable",
                  "volatile_correctable", "row_remap_pending",
                  "row_remap_failure", "retired_pages_pending")
    if gpu.get("ecc_enabled") is False:
        checks["ecc"] = (NA, "ECC disabled")
    elif all(gpu.get(f) is None for f in ecc_fields):
        checks["ecc"] = (NA, "not reported")
    elif gpu.get("volatile_uncorrectable"):
        checks["ecc"] = (FAIL, "volatile uncorrectable ECC errors: %d"
                         % gpu["volatile_uncorrectable"])
    elif gpu.get("row_remap_failure"):
        checks["ecc"] = (FAIL, "row remap failure (RMA candidate)")
    elif gpu.get("row_remap_pending"):
        checks["ecc"] = (FAIL, "row remap pending (GPU reset required)")
    elif gpu.get("retired_pages_pending"):
        checks["ecc"] = (FAIL, "retired pages pending (GPU reset required)")
    elif gpu.get("aggregate_uncorrectable"):
        checks["ecc"] = (WARN, "lifetime uncorrectable ECC errors: %d"
                         % gpu["aggregate_uncorrectable"])
    elif (gpu.get("volatile_correctable") or 0) > CORRECTABLE_ECC_WARN:
        checks["ecc"] = (WARN, "volatile correctable ECC errors: %d (> %d)"
                         % (gpu["volatile_correctable"], CORRECTABLE_ECC_WARN))
    else:
        checks["ecc"] = (OK, "")

    # Throttling
    throttle = gpu.get("throttle")
    temp, slowdown = gpu.get("temp_c"), gpu.get("slowdown_temp_c")
    active = sorted(k for k, v in (throttle or {}).items() if v)
    if throttle is None and temp is None:
        checks["throttle"] = (NA, "not reported")
    elif "hw_slowdown" in active:
        checks["throttle"] = (FAIL, "HW slowdown throttle active")
    elif active:
        checks["throttle"] = (WARN, "throttling active: %s" % ", ".join(active))
    elif (temp is not None and slowdown is not None
          and temp >= slowdown - TEMP_MARGIN_C):
        checks["throttle"] = (WARN,
                              "temperature %dC within %dC of slowdown "
                              "threshold %dC" % (temp, TEMP_MARGIN_C, slowdown))
    else:
        checks["throttle"] = (OK, "")

    # PCIe
    replay = gpu.get("pcie_replay")
    if replay is None:
        checks["pcie"] = (NA, "not reported")
    elif replay > 0:
        checks["pcie"] = (WARN, "PCIe replay counter: %d" % replay)
    else:
        checks["pcie"] = (OK, "")

    # NVLink
    if nvlink is None:
        checks["nvlink"] = (NA, "no NVLink data")
    else:
        nonzero = {k: v for k, v in nvlink.items() if v}
        if nonzero:
            checks["nvlink"] = (WARN, "NVLink errors: %s" % ", ".join(
                "%s=%d" % item for item in sorted(nonzero.items())))
        else:
            checks["nvlink"] = (OK, "")

    return checks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest discover -s gpu/tests -v`
Expected: 30 tests, all PASS.

- [ ] **Step 5: Commit**

```bash
git add gpu/gpu_health gpu/tests/test_gpu_health.py
git commit -m "gpu_health: tiered verdict rules"
```

---

### Task 4: `build_result` — snapshot assembly

**Files:**
- Modify: `gpu/gpu_health` (two functions after `evaluate`)
- Test: `gpu/tests/test_gpu_health.py` (append a class)

**Interfaces:**
- Consumes: `parse_smi_xml` output, `parse_nvlink` output, `evaluate`, `worst`.
- Produces: `build_result(parsed, nvlink_by_gpu, host=None, timestamp=None) -> dict` in the spec's JSON schema: top-level `host`, `timestamp`, `driver_version`, `verdict`, `gpus`; each gpu has `index`, `name`, `serial`, `verdict`, `checks` with sub-dicts `ecc` (`status`, `detail`, `volatile_uncorrectable`, `aggregate_uncorrectable`, `volatile_correctable`, `row_remap`), `throttle` (`status`, `detail`, `active`, `temp_c`, `slowdown_temp_c`, `power_w`, `power_limit_w`), `pcie` (`status`, `detail`, `replay_counter`), `nvlink` (`status`, `detail`, `errors` — nonzero counters only). `nvlink_by_gpu=None` means NVLink n/a for all GPUs; `host`/`timestamp` default to this host/now (parameters exist for tests).

- [ ] **Step 1: Write the failing tests**

Append to `gpu/tests/test_gpu_health.py`:

```python
class TestBuildResult(unittest.TestCase):
    def build_healthy(self):
        parsed = gh.parse_smi_xml(fixture("healthy.xml"))
        nvlink = gh.parse_nvlink(fixture("healthy.xml.nvlink"))
        return gh.build_result(parsed, nvlink, host="testhost",
                               timestamp="2026-07-02T10:00:00")

    def test_healthy_snapshot(self):
        result = self.build_healthy()
        self.assertEqual(result["host"], "testhost")
        self.assertEqual(result["timestamp"], "2026-07-02T10:00:00")
        self.assertEqual(result["driver_version"], "550.54.15")
        self.assertEqual(result["verdict"], gh.OK)
        self.assertEqual(len(result["gpus"]), 2)
        g0 = result["gpus"][0]
        self.assertEqual(g0["verdict"], gh.OK)
        self.assertEqual(sorted(g0["checks"]),
                         ["ecc", "nvlink", "pcie", "throttle"])
        self.assertEqual(g0["checks"]["ecc"]["row_remap"], "none")
        self.assertEqual(g0["checks"]["nvlink"]["errors"], {})
        self.assertEqual(g0["checks"]["throttle"]["active"], [])

    def test_node_verdict_is_worst_gpu(self):
        xml = fixture("healthy.xml")
        # doctor GPU 1 (second occurrence) to a FAIL: volatile uncorrectable
        first = xml.find("<dram_uncorrectable>0</dram_uncorrectable>")
        second = xml.find("<dram_uncorrectable>0</dram_uncorrectable>",
                          first + 1)
        # occurrences per GPU: volatile then aggregate; GPU 1 volatile is 3rd
        third = xml.find("<dram_uncorrectable>0</dram_uncorrectable>",
                         second + 1)
        xml = (xml[:third]
               + "<dram_uncorrectable>5</dram_uncorrectable>"
               + xml[third + len("<dram_uncorrectable>0</dram_uncorrectable>"):])
        parsed = gh.parse_smi_xml(xml)
        result = gh.build_result(parsed, None, host="h", timestamp="t")
        self.assertEqual(result["gpus"][0]["verdict"], gh.OK)
        self.assertEqual(result["gpus"][1]["verdict"], gh.FAIL)
        self.assertEqual(result["verdict"], gh.FAIL)

    def test_nvlink_none_is_na_everywhere(self):
        parsed = gh.parse_smi_xml(fixture("healthy.xml"))
        result = gh.build_result(parsed, None, host="h", timestamp="t")
        for gpu in result["gpus"]:
            self.assertEqual(gpu["checks"]["nvlink"]["status"], gh.NA)
        self.assertEqual(result["verdict"], gh.OK)  # n/a never worsens

    def test_row_remap_states(self):
        parsed = gh.parse_smi_xml(fixture("healthy.xml").replace(
            "<remapped_row_pending>No</remapped_row_pending>",
            "<remapped_row_pending>Yes</remapped_row_pending>", 1))
        result = gh.build_result(parsed, None, host="h", timestamp="t")
        self.assertEqual(result["gpus"][0]["checks"]["ecc"]["row_remap"],
                         "pending")
        self.assertEqual(result["gpus"][0]["verdict"], gh.FAIL)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest discover -s gpu/tests -v`
Expected: new tests ERROR with `AttributeError: ... no attribute 'build_result'`; all earlier tests PASS.

- [ ] **Step 3: Implement `build_result`**

Add to `gpu/gpu_health` after `evaluate`:

```python
def _row_remap_state(gpu):
    if gpu["row_remap_failure"] is None and gpu["row_remap_pending"] is None:
        return NA
    if gpu["row_remap_failure"]:
        return "failure"
    if gpu["row_remap_pending"]:
        return "pending"
    return "none"


def build_result(parsed, nvlink_by_gpu, host=None, timestamp=None):
    """Assemble the snapshot dict (the --json schema).

    nvlink_by_gpu: parse_nvlink() output, or None when NVLink data is
    entirely unavailable. host/timestamp parameters exist for tests.
    """
    gpus_out = []
    for gpu in parsed["gpus"]:
        nvlink = None
        if nvlink_by_gpu is not None:
            nvlink = nvlink_by_gpu.get(gpu["index"], {})
        checks = evaluate(gpu, nvlink)
        gpus_out.append({
            "index": gpu["index"],
            "name": gpu["name"],
            "serial": gpu["serial"],
            "verdict": worst(tier for tier, _ in checks.values()),
            "checks": {
                "ecc": {
                    "status": checks["ecc"][0],
                    "detail": checks["ecc"][1],
                    "volatile_uncorrectable": gpu["volatile_uncorrectable"],
                    "aggregate_uncorrectable": gpu["aggregate_uncorrectable"],
                    "volatile_correctable": gpu["volatile_correctable"],
                    "row_remap": _row_remap_state(gpu),
                },
                "throttle": {
                    "status": checks["throttle"][0],
                    "detail": checks["throttle"][1],
                    "active": sorted(
                        k for k, v in (gpu["throttle"] or {}).items() if v),
                    "temp_c": gpu["temp_c"],
                    "slowdown_temp_c": gpu["slowdown_temp_c"],
                    "power_w": gpu["power_w"],
                    "power_limit_w": gpu["power_limit_w"],
                },
                "pcie": {
                    "status": checks["pcie"][0],
                    "detail": checks["pcie"][1],
                    "replay_counter": gpu["pcie_replay"],
                },
                "nvlink": {
                    "status": checks["nvlink"][0],
                    "detail": checks["nvlink"][1],
                    "errors": {k: v for k, v in (nvlink or {}).items() if v},
                },
            },
        })
    return {
        "host": host or socket.gethostname().split(".")[0],
        "timestamp": timestamp or datetime.now().isoformat(timespec="seconds"),
        "driver_version": parsed["driver_version"],
        "verdict": worst(g["verdict"] for g in gpus_out),
        "gpus": gpus_out,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest discover -s gpu/tests -v`
Expected: 34 tests, all PASS.

- [ ] **Step 5: Commit**

```bash
git add gpu/gpu_health gpu/tests/test_gpu_health.py
git commit -m "gpu_health: assemble snapshot result"
```

---

### Task 5: Renderers

**Files:**
- Modify: `gpu/gpu_health` (two functions after `build_result`)
- Test: `gpu/tests/test_gpu_health.py` (append a class)

**Interfaces:**
- Consumes: `build_result` output shape.
- Produces: `render_text(result) -> str` (plain-text report, ends with `node verdict: <TIER>` line naming the worst GPU(s) when not OK); `render_json(result) -> str` (`json.dumps(result, indent=2)` plus trailing newline).

- [ ] **Step 1: Write the failing tests**

Append to `gpu/tests/test_gpu_health.py`:

```python
class TestRender(unittest.TestCase):
    def result(self, doctor=None):
        xml = fixture("healthy.xml")
        if doctor:
            xml = doctor(xml)
        parsed = gh.parse_smi_xml(xml)
        nvlink = gh.parse_nvlink(fixture("healthy.xml.nvlink"))
        return gh.build_result(parsed, nvlink, host="testhost",
                               timestamp="2026-07-02T10:00:00")

    def test_text_healthy(self):
        text = gh.render_text(self.result())
        self.assertIn("GPU 0: NVIDIA H100 80GB HBM3", text)
        self.assertIn("GPU 1:", text)
        self.assertTrue(text.rstrip().endswith("node verdict: OK"),
                        msg=text)

    def test_text_names_failing_gpu(self):
        def doctor(xml):
            return xml.replace(
                "<remapped_row_failure>No</remapped_row_failure>",
                "<remapped_row_failure>Yes</remapped_row_failure>", 1)
        text = gh.render_text(self.result(doctor))
        self.assertIn("row remap failure", text)
        self.assertTrue(text.rstrip().endswith("node verdict: FAIL (GPU 0)"),
                        msg=text)

    def test_json_round_trips(self):
        result = self.result()
        parsed_back = __import__("json").loads(gh.render_json(result))
        self.assertEqual(parsed_back, result)
        self.assertTrue(gh.render_json(result).endswith("\n"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest discover -s gpu/tests -v`
Expected: new tests ERROR with `AttributeError: ... no attribute 'render_text'`.

- [ ] **Step 3: Implement the renderers**

Add to `gpu/gpu_health` after `build_result`:

```python
def render_text(result):
    """Plain-text report: header, one block per GPU, node verdict line."""
    lines = ["gpu_health @ %s  (driver %s, %s)"
             % (result["host"], result["driver_version"] or NA,
                result["timestamp"]),
             ""]
    for gpu in result["gpus"]:
        lines.append("GPU %s: %s  (serial %s)"
                     % (gpu["index"], gpu["name"] or "?", gpu["serial"] or NA))
        for name in ("ecc", "throttle", "pcie", "nvlink"):
            check = gpu["checks"][name]
            line = "  %-9s %s" % (name + ":", check["status"])
            if check["detail"]:
                line += " - " + check["detail"]
            lines.append(line)
        lines.append("  verdict:  %s" % gpu["verdict"])
        lines.append("")
    verdict = result["verdict"]
    suffix = ""
    if verdict != OK:
        culprits = [str(g["index"]) for g in result["gpus"]
                    if g["verdict"] == verdict]
        suffix = " (GPU %s)" % ", GPU ".join(culprits)
    lines.append("node verdict: %s%s" % (verdict, suffix))
    return "\n".join(lines)


def render_json(result):
    return json.dumps(result, indent=2) + "\n"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest discover -s gpu/tests -v`
Expected: 37 tests, all PASS.

- [ ] **Step 5: Commit**

```bash
git add gpu/gpu_health gpu/tests/test_gpu_health.py
git commit -m "gpu_health: text and JSON renderers"
```

---

### Task 6: `collect`, `main`, CLI wiring and end-to-end tests

**Files:**
- Modify: `gpu/gpu_health` (replace the placeholder `__main__` block; add `_run`, `collect`, `main`)
- Test: `gpu/tests/test_gpu_health.py` (append a class)

**Interfaces:**
- Consumes: everything above.
- Produces: `collect() -> (xml_text, nvlink_text_or_None)` raising `RuntimeError` on probe failure; `main(argv=None) -> int` exit code; CLI flags `--json [FILE]` and `--from-xml FILE` exactly as in the spec. `main` reads `FILE.nvlink` beside a `--from-xml FILE` when present.

- [ ] **Step 1: Write the failing end-to-end tests**

Append to `gpu/tests/test_gpu_health.py`:

```python
import json as _json
import subprocess
import sys
import tempfile


def run_cli(*argv, **kwargs):
    return subprocess.run([sys.executable, TOOL] + list(argv),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True, **kwargs)


class TestCli(unittest.TestCase):
    def doctored_file(self, old, new):
        """healthy.xml with one string replaced, in a temp file (no .nvlink
        companion, so NVLink reports n/a)."""
        xml = fixture("healthy.xml").replace(old, new, 1)
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".xml",
                                          delete=False)
        self.addCleanup(os.unlink, tmp.name)
        tmp.write(xml)
        tmp.close()
        return tmp.name

    def test_healthy_exits_0(self):
        proc = run_cli("--from-xml",
                       os.path.join(FIXTURES, "healthy.xml"))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("node verdict: OK", proc.stdout)

    def test_warn_exits_1(self):
        path = self.doctored_file(
            "<clocks_throttle_reason_sw_power_cap>Not Active",
            "<clocks_throttle_reason_sw_power_cap>Active")
        proc = run_cli("--from-xml", path)
        self.assertEqual(proc.returncode, 1, proc.stdout)
        self.assertIn("node verdict: WARN", proc.stdout)

    def test_fail_exits_2(self):
        path = self.doctored_file(
            "<dram_uncorrectable>0</dram_uncorrectable>",
            "<dram_uncorrectable>3</dram_uncorrectable>")
        proc = run_cli("--from-xml", path)
        self.assertEqual(proc.returncode, 2, proc.stdout)
        self.assertIn("node verdict: FAIL (GPU 0)", proc.stdout)

    def test_garbled_exits_3(self):
        proc = run_cli("--from-xml",
                       os.path.join(FIXTURES, "garbled.xml"))
        self.assertEqual(proc.returncode, 3)
        self.assertIn("error", proc.stderr.lower())

    def test_missing_from_xml_file_exits_3(self):
        proc = run_cli("--from-xml", "/nonexistent/capture.xml")
        self.assertEqual(proc.returncode, 3)

    def test_missing_nvidia_smi_exits_3(self):
        env = dict(os.environ, PATH="/nonexistent")
        proc = run_cli(env=env)
        self.assertEqual(proc.returncode, 3)
        self.assertIn("nvidia-smi", proc.stderr)

    def test_json_stdout(self):
        proc = run_cli("--from-xml",
                       os.path.join(FIXTURES, "healthy.xml"), "--json")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        snapshot = _json.loads(proc.stdout)
        self.assertEqual(snapshot["verdict"], "OK")
        self.assertEqual(len(snapshot["gpus"]), 2)
        # companion healthy.xml.nvlink was picked up automatically
        self.assertEqual(snapshot["gpus"][0]["checks"]["nvlink"]["status"],
                         "OK")

    def test_json_to_file(self):
        out = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        out.close()
        self.addCleanup(os.unlink, out.name)
        proc = run_cli("--from-xml",
                       os.path.join(FIXTURES, "healthy.xml"),
                       "--json", out.name)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(out.name) as fh:
            self.assertEqual(_json.load(fh)["verdict"], "OK")

    def test_ecc_disabled_is_ok_with_na(self):
        proc = run_cli("--from-xml",
                       os.path.join(FIXTURES, "ecc_disabled.xml"))
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("ecc:      n/a", proc.stdout)
        self.assertIn("nvlink:   n/a", proc.stdout)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest discover -s gpu/tests -v`
Expected: `TestCli` tests FAIL (the placeholder `__main__` exits 0 and prints nothing, so verdict/exit-code assertions fail); everything else PASSes.

- [ ] **Step 3: Implement `collect` and `main`; replace the `__main__` block**

In `gpu/gpu_health`, delete the placeholder block

```python
if __name__ == "__main__":
    sys.exit(0)  # replaced by main() in a later task
```

and append:

```python
def _run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          universal_newlines=True, timeout=60)


def collect():
    """Run nvidia-smi. Returns (xml_text, nvlink_text_or_None).

    Raises RuntimeError when the probe itself cannot run (exit 3 upstream).
    A failing `nvidia-smi nvlink` only degrades NVLink to n/a.
    """
    try:
        smi = _run(["nvidia-smi", "-q", "-x"])
    except OSError:
        raise RuntimeError("nvidia-smi not found on PATH")
    except subprocess.TimeoutExpired:
        raise RuntimeError("nvidia-smi -q -x timed out")
    if smi.returncode != 0:
        raise RuntimeError("nvidia-smi -q -x failed: %s"
                           % (smi.stderr.strip() or "exit %d" % smi.returncode))
    try:
        nvl = _run(["nvidia-smi", "nvlink", "-e"])
        nvlink_text = nvl.stdout if nvl.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        nvlink_text = None
    return smi.stdout, nvlink_text


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="gpu_health",
        description="node-local, read-only GPU health probe "
                    "(ECC/row-remap, throttling, PCIe/NVLink errors); "
                    "exit 0 OK, 1 WARN, 2 FAIL, 3 probe error")
    parser.add_argument("--json", nargs="?", const="-", default=None,
                        metavar="FILE",
                        help="emit a JSON snapshot to FILE (stdout when "
                             "FILE omitted) instead of the report")
    parser.add_argument("--from-xml", metavar="FILE",
                        help="replay a captured `nvidia-smi -q -x` dump; "
                             "NVLink counters are read from FILE.nvlink "
                             "when present")
    args = parser.parse_args(argv)

    try:
        if args.from_xml:
            with open(args.from_xml) as fh:
                xml_text = fh.read()
            nvlink_text = None
            companion = args.from_xml + ".nvlink"
            if os.path.exists(companion):
                with open(companion) as fh:
                    nvlink_text = fh.read()
        else:
            xml_text, nvlink_text = collect()
        parsed = parse_smi_xml(xml_text)
    except (RuntimeError, ValueError, OSError) as exc:
        sys.stderr.write("gpu_health: error: %s\n" % exc)
        return EXIT_ERROR

    nvlink_by_gpu = parse_nvlink(nvlink_text) if nvlink_text else {}
    if not nvlink_by_gpu:
        nvlink_by_gpu = None   # unsupported/absent NVLink -> n/a, not OK

    result = build_result(parsed, nvlink_by_gpu)

    if args.json is not None:
        payload = render_json(result)
        if args.json == "-":
            sys.stdout.write(payload)
        else:
            with open(args.json, "w") as fh:
                fh.write(payload)
    else:
        print(render_text(result))
    return _EXIT_FOR[result["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest discover -s gpu/tests -v`
Expected: 46 tests, all PASS.

- [ ] **Step 5: Sanity-run the CLI by hand**

Run: `./gpu/gpu_health --from-xml gpu/tests/fixtures/healthy.xml; echo "exit=$?"`
Expected: two GPU blocks, `node verdict: OK`, `exit=0`.

Run: `./gpu/gpu_health --from-xml gpu/tests/fixtures/healthy.xml --json | head -5`
Expected: JSON opening with `"host"`, `"timestamp"`, `"driver_version"`.

- [ ] **Step 6: Commit**

```bash
git add gpu/gpu_health gpu/tests/test_gpu_health.py
git commit -m "gpu_health: CLI, collection, and exit-code wiring"
```

---

### Task 7: Documentation

**Files:**
- Modify: `gpu/README.md` (tool table + note)
- Modify: `README.md` (requirements table row)

**Interfaces:**
- Consumes: the finished tool's flags and requirements (Tasks 1–6).
- Produces: docs only; no code.

- [ ] **Step 1: Add the tool row and node-local note to `gpu/README.md`**

In the tool table (after the `jobstats_extended.py` row), add:

```markdown
| `gpu_health` | Node-local GPU **health** probe: ECC & row-remap, throttling, PCIe/NVLink error counters, with a tiered OK/WARN/FAIL verdict (exit 0/1/2; 3 = probe error). `--json` emits a snapshot; `--from-xml` replays a captured `nvidia-smi -q -x` dump. |
```

Directly below the table, add:

```markdown
> **Note:** unlike the other tools here, `gpu_health` runs **on the GPU node
> itself** and talks to `nvidia-smi` directly — no Prometheus, no jobstats
> modules, Python 3 stdlib only.
```

- [ ] **Step 2: Update the requirements table in the top-level `README.md`**

In the "Requirements by directory" table, change the `gpu/` row's "Needs" cell so it ends with:

```markdown
**`gpu_health`** is the exception: it runs on the target GPU node and needs only `nvidia-smi` and the Python 3 standard library (no Prometheus, no jobstats modules).
```

(Keep the existing text about the other `gpu/` tools unchanged.)

- [ ] **Step 3: Run the full suite one last time**

Run: `python3 -m unittest discover -s gpu/tests -v`
Expected: 46 tests, all PASS.

- [ ] **Step 4: Commit**

```bash
git add gpu/README.md README.md
git commit -m "Document gpu_health"
```

---

## Self-Review Notes (spec → plan mapping)

- Purpose/non-goals → Tasks 1–6 build exactly the three check categories; no XID, no dcgmi, no fleet features.
- Interface (`--json [FILE]`, `--from-xml FILE`, exit 0/1/2/3) → Task 6.
- Data collection (two nvidia-smi commands, both read-only) → Task 6 `collect()`.
- Checks & verdict table → Task 3 (every FAIL/WARN/n-a rule has a named test).
- n/a-never-worsens and unknown-field degradation → Tasks 1 (`_find_text` fallbacks), 3 (`worst`, n/a tests), 6 (`ecc_disabled` end-to-end test).
- JSON schema → Task 4 (shape assertions), Task 5 (round-trip).
- Human report incl. `node verdict: FAIL (GPU N)` → Task 5.
- Error handling (exit 3 paths, NVLink degradation) → Task 6 tests.
- Testing section (fixtures, doctored FAIL/WARN coverage, garbled XML) → fixtures in Task 1; rules covered per-rule in Task 3 plus end-to-end doctoring in Task 6.
- Documentation section → Task 7.
