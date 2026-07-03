# Fabric Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four Slurm-mediated fabric diagnostic tools — `fleet_snapshot`, `topo_verify`, `nccl_check`, `ib_bw_pair` — to `fabric/`, per the approved spec.

**Architecture:** Each tool is a Python stdlib executable with a pure core (aggregate / compare / parse / verdict functions, offline-testable against fixtures) wrapped by a thin CLI. Slurm interactions (sbatch/squeue/scancel/scontrol/sinfo) live in a small shared helper module `fabric/fablib.py`; tools add their own directory to `sys.path` to import it. Everything builds on the existing `ib_snapshot.sh` schema_version-1 JSON.

**Tech Stack:** Python 3.6-compatible stdlib (`argparse`, `json`, `re`, `subprocess`, `collections`, `unittest`). Slurm CLI. Existing `fabric/ib_snapshot.sh` and `fabric/counter_delta.sh`.

**Spec:** `docs/specs/2026-07-02-fabric-suite-design.md` (approved).

**One deliberate spec addition:** `fleet_snapshot --from-dir DIR` re-aggregates an existing snapshot directory offline (no Slurm). It exists so a past run can be re-analyzed and so the CLI's 0/1/3 exit paths are testable offline; it adds no new logic beyond skipping submission.

## Global Constraints

- Python 3.6-compatible stdlib only: `subprocess.run(..., stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)`; no `capture_output`/`text` kwargs, no walrus, no dataclasses, no f-strings.
- Tools are `fabric/fleet_snapshot`, `fabric/topo_verify`, `fabric/nccl_check`, `fabric/ib_bw_pair` — no `.py` extension, `#!/usr/bin/env python3` shebang, executable bit set (`chmod +x`).
- Exit codes: `0` pass/clean, `1` problem found (drift / anomaly / bandwidth below threshold / counter deltas), `3` probe or setup error. Error messages go to stderr as `<tool>: error: <msg>`.
- Node access is Slurm-only (`sbatch`/`srun`/`squeue`/`scancel`/`scontrol`/`sinfo`); no ssh.
- Every subprocess call has a timeout.
- Missing snapshot sections become explicit "unavailable" notes, never a silent pass.
- Plain-text output, no ANSI color; `--json` where specified.
- Tests: stdlib `unittest`, run as `python3 -m unittest discover -s fabric/tests -v` from the repo root. No Slurm, GPU, or IB hardware needed for the suite.
- Commit messages: short imperative subject, no attribution footers of any kind.
- Threshold constants: `RATE_FRACTION = 0.8` (ib_bw_pair default pass fraction of port rate). nccl_check has NO built-in bandwidth threshold.

## File Structure

| File | Responsibility |
|---|---|
| `fabric/fablib.py` (create, Task 2) | Shared Slurm/subprocess helpers: `run`, `err_exit`, `expand_nodes`, `submit_sbatch`, `wait_for_jobs`. |
| `fabric/fleet_snapshot` (create, Tasks 1–2) | Fleet sweep: pure `aggregate()` core (Task 1) + Slurm submission CLI (Task 2). |
| `fabric/topo_verify` (create, Task 3) | Golden-snapshot drift: pure `compare()` + CLI. |
| `fabric/nccl_check` (create, Task 4) | NCCL smoke test: `find_binary`, `parse_nccl_table`, `summarize`, verdict + Slurm CLI. |
| `fabric/ib_bw_pair` (create, Task 5) | Pairwise perftest: `parse_perftest`, `_rate_gbps`, verdict + Slurm CLI. |
| `fabric/tests/__init__.py`, `loader.py`, fixtures, 4 test files (Tasks 1–5) | Offline test suite. |
| `README.md` (modify, Task 6) | fabric/ section: 4 new rows, requirements, examples. |

Execution note for the controller: implement on a new branch `fabric-suite` off `main`.

---

### Task 1: Test scaffolding, snapshot fixtures, and the fleet aggregator core

**Files:**
- Create: `fabric/tests/__init__.py` (empty), `fabric/tests/loader.py`
- Create: `fabric/tests/fixtures/snapshot_nodea.json`, `fabric/tests/fixtures/snapshot_nodeb.json`
- Create: `fabric/fleet_snapshot`
- Test: `fabric/tests/test_fleet_snapshot.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `loader.load_tool(name) -> module`, `loader.fixture(name) -> str`, `loader.FIXTURES` (dir path). In `fleet_snapshot`: `ERROR_COUNTERS` tuple; `_rate_gbps(text) -> float|None`; `aggregate(snapshots: {host: dict|None}) -> summary dict` with keys `nodes_total` (int), `nodes_reached` (int), `unreached` (list[str]), `unavailable` (list[str]), `version_drift` (list[str]), `link_anomalies` (list[str]), `error_counters` (list[str]), `clean` (bool); `render_fleet_text(summary) -> str`. The tool file must NOT import `fablib` yet (that arrives in Task 2).

- [ ] **Step 1: Create the loader and fixtures**

`fabric/tests/__init__.py`: empty file.

`fabric/tests/loader.py`:

```python
"""Load the extensionless fabric tools as modules; locate fixtures."""
import importlib.machinery
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")


def load_tool(name):
    path = os.path.abspath(os.path.join(HERE, "..", name))
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def fixture(name):
    with open(os.path.join(FIXTURES, name)) as fh:
        return fh.read()
```

`fabric/tests/fixtures/snapshot_nodea.json` (synthetic but faithful to `ib_snapshot.sh` schema_version 1):

```json
{
  "schema_version": 1,
  "timestamp_utc": "2026-07-02T12:00:00+00:00",
  "hostname": "nodea",
  "system": {
    "uname": "Linux nodea 4.18.0-553.44.1.el8_10.x86_64 #1 SMP x86_64 GNU/Linux",
    "nvidia_driver": "575.57.08",
    "cuda_runtime": "release 12.2"
  },
  "gpus": {
    "gpus": [
      {"index": 0, "name": "NVIDIA H200", "pci_bus_id": "00000000:19:00.0",
       "temp_c": 30.0, "power_draw_w": 70.0, "power_limit_w": 700.0,
       "memory_total_mib": 143771, "memory_used_mib": 0,
       "sm_clock_mhz": 345, "memory_clock_mhz": 2619},
      {"index": 1, "name": "NVIDIA H200", "pci_bus_id": "00000000:3B:00.0",
       "temp_c": 29.0, "power_draw_w": 71.0, "power_limit_w": 700.0,
       "memory_total_mib": 143771, "memory_used_mib": 0,
       "sm_clock_mhz": 345, "memory_clock_mhz": 2619}
    ]
  },
  "nvlink": {"rc": 0, "raw": "GPU 0: NVIDIA H200\n\t Link 0: 26.562 GB/s\n"},
  "topology": {"rc": 0, "raw": "\tGPU0\tGPU1\tNIC0\tCPU Affinity\nGPU0\t X \tNV18\tPXB\t0-47\nGPU1\tNV18\t X \tSYS\t48-95\nNIC0\tPXB\tSYS\t X \t\n"},
  "ib": {
    "hcas": [
      {"name": "mlx5_0",
       "ports": [
         {"port": 1, "state": "4: ACTIVE", "phys_state": "5: LinkUp",
          "rate": "400 Gb/sec (4X NDR)", "link_layer": "InfiniBand",
          "counters": {"symbol_error": 0, "link_downed": 0,
                       "port_xmit_discards": 0, "port_rcv_errors": 0,
                       "port_xmit_data": 123456789}}
       ],
       "ibv_devinfo": "hca_id:\tmlx5_0\n\tfw_ver:\t28.39.1002\n\tport:\t1\n\t\tstate:\tPORT_ACTIVE (4)\n"}
    ]
  },
  "ibdev2netdev": {
    "mapping": [{"hca": "mlx5_0", "port": 1, "netdev": "ibp0", "state": "Up"}],
    "raw": "mlx5_0 port 1 ==> ibp0 (Up)\n"
  },
  "nccl": {"libnccl_symlink_target": null}
}
```

`fabric/tests/fixtures/snapshot_nodeb.json`: identical except `"hostname": "nodeb"`, `"uname": "Linux nodeb 4.18.0-553.44.1.el8_10.x86_64 #1 SMP x86_64 GNU/Linux"`, and `"port_xmit_data": 987654321`. (Same driver, fw, rates — a clean pair. Copy the file and change only those three values.)

- [ ] **Step 2: Write the failing tests**

`fabric/tests/test_fleet_snapshot.py`:

```python
"""Tests for fabric/fleet_snapshot (aggregator core; CLI tested in Task 2)."""
import json
import unittest

from fabric.tests import loader

fs = loader.load_tool("fleet_snapshot")


def snap(name):
    return json.loads(loader.fixture(name))


def clean_pair():
    return {"nodea": snap("snapshot_nodea.json"),
            "nodeb": snap("snapshot_nodeb.json")}


class TestRateGbps(unittest.TestCase):
    def test_parses_leading_number(self):
        self.assertEqual(fs._rate_gbps("400 Gb/sec (4X NDR)"), 400.0)
        self.assertEqual(fs._rate_gbps("100 Gb/sec (4X EDR)"), 100.0)

    def test_none_on_garbage(self):
        self.assertIsNone(fs._rate_gbps(""))
        self.assertIsNone(fs._rate_gbps(None))
        self.assertIsNone(fs._rate_gbps("unknown"))


class TestAggregate(unittest.TestCase):
    def test_clean_fleet(self):
        summary = fs.aggregate(clean_pair())
        self.assertTrue(summary["clean"])
        self.assertEqual(summary["nodes_total"], 2)
        self.assertEqual(summary["nodes_reached"], 2)
        self.assertEqual(summary["unreached"], [])
        self.assertEqual(summary["version_drift"], [])
        self.assertEqual(summary["link_anomalies"], [])
        self.assertEqual(summary["error_counters"], [])
        self.assertEqual(summary["unavailable"], [])

    def test_unreached_node_flagged(self):
        snaps = clean_pair()
        snaps["nodec"] = None
        summary = fs.aggregate(snaps)
        self.assertFalse(summary["clean"])
        self.assertEqual(summary["unreached"], ["nodec"])
        self.assertEqual(summary["nodes_reached"], 2)

    def test_driver_drift_flagged(self):
        snaps = clean_pair()
        snaps["nodeb"]["system"]["nvidia_driver"] = "550.54.15"
        summary = fs.aggregate(snaps)
        self.assertFalse(summary["clean"])
        self.assertTrue(any("nodeb" in d and "driver" in d
                            for d in summary["version_drift"]), summary)

    def test_fw_drift_flagged(self):
        snaps = clean_pair()
        snaps["nodeb"]["ib"]["hcas"][0]["ibv_devinfo"] = (
            "hca_id:\tmlx5_0\n\tfw_ver:\t28.36.1010\n")
        summary = fs.aggregate(snaps)
        self.assertTrue(any("fw" in d for d in summary["version_drift"]), summary)

    def test_down_port_flagged(self):
        snaps = clean_pair()
        port = snaps["nodeb"]["ib"]["hcas"][0]["ports"][0]
        port["state"] = "1: DOWN"
        port["phys_state"] = "3: Disabled"
        summary = fs.aggregate(snaps)
        self.assertFalse(summary["clean"])
        self.assertTrue(any("nodeb" in a and "state" in a
                            for a in summary["link_anomalies"]), summary)

    def test_degraded_rate_flagged(self):
        snaps = clean_pair()
        snaps["nodeb"]["ib"]["hcas"][0]["ports"][0]["rate"] = \
            "200 Gb/sec (4X HDR)"
        summary = fs.aggregate(snaps)
        self.assertTrue(any("rate" in a and "nodeb" in a
                            for a in summary["link_anomalies"]), summary)

    def test_error_counter_flagged(self):
        snaps = clean_pair()
        snaps["nodea"]["ib"]["hcas"][0]["ports"][0]["counters"]["symbol_error"] = 12
        summary = fs.aggregate(snaps)
        self.assertFalse(summary["clean"])
        self.assertTrue(any("symbol_error=12" in e and "nodea" in e
                            for e in summary["error_counters"]), summary)

    def test_traffic_counters_ignored(self):
        # port_xmit_data differs between the fixtures and is NOT an error counter
        summary = fs.aggregate(clean_pair())
        self.assertEqual(summary["error_counters"], [])

    def test_missing_ib_section_noted(self):
        snaps = clean_pair()
        snaps["nodeb"]["ib"] = {"error": "no /sys/class/infiniband", "hcas": []}
        summary = fs.aggregate(snaps)
        self.assertFalse(summary["clean"])
        self.assertTrue(any("nodeb" in u for u in summary["unavailable"]), summary)


class TestRenderFleetText(unittest.TestCase):
    def test_clean_report(self):
        text = fs.render_fleet_text(fs.aggregate(clean_pair()))
        self.assertIn("fleet verdict: CLEAN", text)
        self.assertIn("2/2", text)

    def test_anomalous_report_lists_findings(self):
        snaps = clean_pair()
        snaps["nodeb"]["ib"]["hcas"][0]["ports"][0]["counters"]["symbol_error"] = 3
        text = fs.render_fleet_text(fs.aggregate(snaps))
        self.assertIn("fleet verdict: ANOMALIES", text)
        self.assertIn("symbol_error=3", text)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m unittest discover -s fabric/tests -v`
Expected: import-time ERROR — `FileNotFoundError` for `fabric/fleet_snapshot` (tool does not exist).

- [ ] **Step 4: Write the tool with the aggregator core**

`fabric/fleet_snapshot`:

```python
#!/usr/bin/env python3
"""fleet_snapshot -- run ib_snapshot.sh across a nodelist via Slurm and
aggregate the results: version drift, link anomalies, error counters,
unreached nodes.

Usage:
    fleet_snapshot (-p PARTITION | -w NODELIST) [-A ACCOUNT] [--gres GRES]
                   [--timeout MIN] [--out DIR] [--json]
    fleet_snapshot --from-dir DIR [--json]     # re-aggregate offline

Exit codes: 0 fleet clean, 1 anomalies (incl. unreached nodes), 3 setup error.
"""

import argparse
import collections
import json
import os
import re
import sys

TOOL = "fleet_snapshot"

# Error-class counters (same list counter_delta.sh watches).
ERROR_COUNTERS = ("symbol_error", "link_downed", "port_xmit_discards",
                  "port_rcv_errors", "port_rcv_remote_physical_errors",
                  "local_link_integrity_errors",
                  "excessive_buffer_overrun_errors")

_FW_RE = re.compile(r"fw_ver:\s*(\S+)")
_NUM_RE = re.compile(r"\d+(\.\d+)?")


def _rate_gbps(text):
    """'400 Gb/sec (4X NDR)' -> 400.0; None when absent/unparseable."""
    if not text:
        return None
    m = _NUM_RE.search(text)
    return float(m.group()) if m else None


def _kernel(snap):
    parts = ((snap.get("system") or {}).get("uname") or "").split()
    return parts[2] if len(parts) > 2 else None


def _driver(snap):
    return (snap.get("system") or {}).get("nvidia_driver")


def _fw(snap):
    """Sorted unique fw_ver strings across HCAs, joined; None if none found."""
    versions = set()
    for hca in ((snap.get("ib") or {}).get("hcas") or []):
        versions.update(_FW_RE.findall(hca.get("ibv_devinfo") or ""))
    return ",".join(sorted(versions)) if versions else None


def _ports(snap):
    """Yield (hca_name, port_number, port_dict) for every port."""
    for hca in ((snap.get("ib") or {}).get("hcas") or []):
        for port in (hca.get("ports") or []):
            yield hca.get("name"), port.get("port"), port


def _version_drift(reached, label, getter):
    per_host = dict((h, getter(s)) for h, s in reached.items()
                    if getter(s) is not None)
    if len(set(per_host.values())) <= 1:
        return []
    modal = collections.Counter(per_host.values()).most_common(1)[0][0]
    return sorted("%s: %s=%s (fleet modal: %s)" % (host, label, val, modal)
                  for host, val in per_host.items() if val != modal)


def aggregate(snapshots):
    """Aggregate {host: snapshot-dict or None (unreached)} into a summary."""
    reached = dict((h, s) for h, s in snapshots.items() if s)
    summary = {
        "nodes_total": len(snapshots),
        "nodes_reached": len(reached),
        "unreached": sorted(h for h, s in snapshots.items() if not s),
        "unavailable": [],
        "version_drift": [],
        "link_anomalies": [],
        "error_counters": [],
    }

    for label, getter in (("driver", _driver), ("kernel", _kernel),
                          ("fw", _fw)):
        summary["version_drift"].extend(_version_drift(reached, label, getter))

    rates = collections.defaultdict(dict)  # (hca, port) -> {host: gbps}
    for host in sorted(reached):
        snap = reached[host]
        ib = snap.get("ib") or {}
        if not ib.get("hcas"):
            summary["unavailable"].append(
                "%s: ib section unavailable (%s)"
                % (host, ib.get("error", "no hcas")))
            continue
        for hca, portno, port in _ports(snap):
            where = "%s %s/port%s" % (host, hca, portno)
            state = port.get("state")
            if state and "ACTIVE" not in state:
                summary["link_anomalies"].append(
                    "%s: state=%s" % (where, state))
            phys = port.get("phys_state")
            if phys and "LinkUp" not in phys:
                summary["link_anomalies"].append(
                    "%s: phys_state=%s" % (where, phys))
            gbps = _rate_gbps(port.get("rate"))
            if gbps is not None:
                rates[(hca, portno)][host] = gbps
            counters = port.get("counters") or {}
            for name in ERROR_COUNTERS:
                value = counters.get(name)
                if value:
                    summary["error_counters"].append(
                        "%s: %s=%d" % (where, name, value))

    for (hca, portno), per_host in sorted(rates.items()):
        if len(set(per_host.values())) <= 1:
            continue
        modal = collections.Counter(per_host.values()).most_common(1)[0][0]
        for host, gbps in sorted(per_host.items()):
            if gbps < modal:
                summary["link_anomalies"].append(
                    "%s %s/port%s: rate=%g Gb/s below fleet modal %g Gb/s"
                    % (host, hca, portno, gbps, modal))

    summary["clean"] = not (summary["unreached"] or summary["unavailable"]
                            or summary["version_drift"]
                            or summary["link_anomalies"]
                            or summary["error_counters"])
    return summary


def render_fleet_text(summary):
    lines = ["fleet_snapshot: %d/%d nodes reached"
             % (summary["nodes_reached"], summary["nodes_total"]), ""]
    for key, title in (("unreached", "Unreached nodes"),
                       ("unavailable", "Unavailable data"),
                       ("version_drift", "Version drift"),
                       ("link_anomalies", "Link anomalies"),
                       ("error_counters", "Error counters")):
        items = summary[key]
        if items:
            lines.append("%s (%d):" % (title, len(items)))
            lines.extend("  " + item for item in items)
            lines.append("")
    lines.append("fleet verdict: %s"
                 % ("CLEAN" if summary["clean"] else "ANOMALIES"))
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(0)  # main() arrives in Task 2
```

- [ ] **Step 5: Make it executable and run the tests**

Run: `chmod +x fabric/fleet_snapshot && python3 -m unittest discover -s fabric/tests -v`
Expected: 13 tests, all PASS.

- [ ] **Step 6: Commit**

```bash
git add fabric/fleet_snapshot fabric/tests
git commit -m "Add fleet_snapshot aggregator core with snapshot fixtures"
```

---

### Task 2: `fablib` and the fleet_snapshot CLI

**Files:**
- Create: `fabric/fablib.py`
- Modify: `fabric/fleet_snapshot` (replace the placeholder `__main__` block; add imports + `main`)
- Test: `fabric/tests/test_fleet_snapshot.py` (append a class)

**Interfaces:**
- Consumes: Task 1's `aggregate`/`render_fleet_text`.
- Produces: `fablib.run(cmd, timeout=60, **kw) -> CompletedProcess` (raises `RuntimeError` on missing binary/timeout, never on non-zero exit); `fablib.err_exit(tool, msg)` (stderr + `sys.exit(3)`); `fablib.expand_nodes(nodelist=None, partition=None) -> [str]` (raises `RuntimeError`); `fablib.submit_sbatch(script_text, args) -> job_id str`; `fablib.wait_for_jobs(job_ids, timeout_s, poll_s=15) -> set` (stragglers, already scancel'ed). In `fleet_snapshot`: `main(argv=None) -> int`, flags `-p/-w/-A/--gres/--timeout/--out/--json/--from-dir`, `load_snapshot_dir(path) -> {host: dict|None}`.

- [ ] **Step 1: Write the failing CLI tests**

Append to `fabric/tests/test_fleet_snapshot.py`:

```python
import os
import shutil
import subprocess
import sys
import tempfile

TOOL_PATH = os.path.abspath(os.path.join(loader.HERE, "..", "fleet_snapshot"))


def run_cli(*argv):
    return subprocess.run([sys.executable, TOOL_PATH] + list(argv),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True)


class TestCliFromDir(unittest.TestCase):
    def make_dir(self, doctor=None):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        for name in ("snapshot_nodea.json", "snapshot_nodeb.json"):
            data = json.loads(loader.fixture(name))
            if doctor:
                doctor(data)
            host = data["hostname"]
            with open(os.path.join(tmp, host + ".json"), "w") as fh:
                json.dump(data, fh)
        return tmp

    def test_clean_dir_exits_0(self):
        proc = run_cli("--from-dir", self.make_dir())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("fleet verdict: CLEAN", proc.stdout)

    def test_anomalous_dir_exits_1(self):
        def doctor(data):
            if data["hostname"] == "nodeb":
                data["system"]["nvidia_driver"] = "550.54.15"
        proc = run_cli("--from-dir", self.make_dir(doctor))
        self.assertEqual(proc.returncode, 1, proc.stdout)
        self.assertIn("fleet verdict: ANOMALIES", proc.stdout)

    def test_json_output(self):
        proc = run_cli("--from-dir", self.make_dir(), "--json")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = json.loads(proc.stdout)
        self.assertTrue(summary["clean"])

    def test_corrupt_file_counts_unreached(self):
        d = self.make_dir()
        with open(os.path.join(d, "nodec.json"), "w") as fh:
            fh.write("{not json")
        proc = run_cli("--from-dir", d)
        self.assertEqual(proc.returncode, 1)
        self.assertIn("nodec", proc.stdout)

    def test_empty_dir_exits_3(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        proc = run_cli("--from-dir", tmp)
        self.assertEqual(proc.returncode, 3)
        self.assertIn("error", proc.stderr.lower())

    def test_missing_target_args_exits_3(self):
        proc = run_cli()
        self.assertEqual(proc.returncode, 3)
        self.assertIn("error", proc.stderr.lower())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest discover -s fabric/tests -v`
Expected: the 6 new `TestCliFromDir` tests FAIL (placeholder `__main__` exits 0 with no output); the 13 earlier tests PASS.

- [ ] **Step 3: Create `fabric/fablib.py`**

```python
"""Shared subprocess/Slurm helpers for the fabric/ Python tools."""
import subprocess
import sys
import time


def run(cmd, timeout=60, **kw):
    """subprocess.run with text pipes. Non-zero exit is returned, not raised;
    a missing binary or timeout raises RuntimeError."""
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              universal_newlines=True, timeout=timeout, **kw)
    except OSError:
        raise RuntimeError("command not found: %s" % cmd[0])
    except subprocess.TimeoutExpired:
        raise RuntimeError("command timed out: %s" % " ".join(cmd))


def err_exit(tool, msg):
    sys.stderr.write("%s: error: %s\n" % (tool, msg))
    sys.exit(3)


def expand_nodes(nodelist=None, partition=None):
    """Expand a Slurm nodelist or a partition to sorted unique hostnames."""
    if nodelist:
        r = run(["scontrol", "show", "hostnames", nodelist])
        if r.returncode != 0:
            raise RuntimeError("scontrol show hostnames failed: %s"
                               % r.stderr.strip())
        nodes = r.stdout.split()
    elif partition:
        r = run(["sinfo", "-p", partition, "-N", "-h", "-o", "%N"])
        if r.returncode != 0:
            raise RuntimeError("sinfo failed: %s" % r.stderr.strip())
        nodes = r.stdout.split()
    else:
        raise RuntimeError("need a nodelist or a partition")
    nodes = sorted(set(nodes))
    if not nodes:
        raise RuntimeError("no nodes matched")
    return nodes


def submit_sbatch(script_text, args):
    """Submit a script (text, via stdin) with `sbatch --parsable <args>`.
    Returns the job id string. Raises RuntimeError on rejection."""
    r = run(["sbatch", "--parsable"] + list(args), input=script_text)
    if r.returncode != 0:
        raise RuntimeError("sbatch failed: %s" % r.stderr.strip())
    return r.stdout.strip().split(";")[0]


def wait_for_jobs(job_ids, timeout_s, poll_s=15):
    """Poll squeue until the jobs leave the queue or timeout_s elapses.
    Cancels and returns the set of job ids still queued at timeout."""
    deadline = time.time() + timeout_s
    remaining = set(job_ids)
    while remaining and time.time() < deadline:
        r = run(["squeue", "-h", "-o", "%i", "-j",
                 ",".join(sorted(remaining))], timeout=60)
        if r.returncode == 0:
            remaining = remaining & set(r.stdout.split())
        if remaining:
            time.sleep(poll_s)
    for job_id in sorted(remaining):
        run(["scancel", job_id])
    return remaining
```

- [ ] **Step 4: Wire up the fleet_snapshot CLI**

In `fabric/fleet_snapshot`, delete the placeholder block

```python
if __name__ == "__main__":
    sys.exit(0)  # main() arrives in Task 2
```

and append (note the `fablib` import goes at the TOP of the file, after the stdlib imports):

At the top of the file, after `import sys`, add:

```python
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import fablib
```

At the bottom, append:

```python
def load_snapshot_dir(path):
    """Read every *.json in a directory -> {host: dict|None (corrupt)}."""
    snapshots = {}
    for name in sorted(os.listdir(path)):
        if not name.endswith(".json") or name == "summary.json":
            continue
        host = name[:-len(".json")]
        try:
            with open(os.path.join(path, name)) as fh:
                snapshots[host] = json.load(fh)
        except (OSError, ValueError):
            snapshots[host] = None
    return snapshots


def _sbatch_script(repo_root, out_path):
    return ("#!/bin/bash\n"
            "%s/fabric/ib_snapshot.sh %s\n" % (repo_root, out_path))


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog=TOOL,
        description="run ib_snapshot.sh across a nodelist via Slurm and "
                    "aggregate the results (exit 0 clean, 1 anomalies, "
                    "3 setup error)")
    parser.add_argument("-p", "--partition")
    parser.add_argument("-w", "--nodelist")
    parser.add_argument("-A", "--account")
    parser.add_argument("--gres", help="e.g. gpu:4 to include GPU sections")
    parser.add_argument("--timeout", type=int, default=30,
                        help="minutes to wait for all node jobs (default 30)")
    parser.add_argument("--out", help="output dir (default "
                        "results/snapshots/fleet_<runid>/)")
    parser.add_argument("--json", action="store_true",
                        help="print summary JSON instead of text")
    parser.add_argument("--from-dir", metavar="DIR",
                        help="skip Slurm; aggregate an existing snapshot dir")
    args = parser.parse_args(argv)

    repo_root = os.path.dirname(
        os.path.dirname(os.path.realpath(__file__)))

    try:
        if args.from_dir:
            snapshots = load_snapshot_dir(args.from_dir)
            if not snapshots:
                raise RuntimeError("no snapshot *.json files in %s"
                                   % args.from_dir)
            outdir = args.from_dir
        else:
            if not (args.partition or args.nodelist):
                raise RuntimeError("need -p PARTITION or -w NODELIST "
                                   "(or --from-dir)")
            nodes = fablib.expand_nodes(nodelist=args.nodelist,
                                        partition=args.partition)
            import datetime
            runid = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            outdir = args.out or os.path.join(
                repo_root, "results", "snapshots", "fleet_" + runid)
            os.makedirs(outdir, exist_ok=True)
            job_to_host = {}
            for node in nodes:
                sbatch_args = ["-J", "fleet_snap_" + node, "-w", node,
                               "-N", "1", "-c", "2", "--mem=4G", "-t", "5",
                               "-o", os.path.join(outdir, node + ".log")]
                if args.account:
                    sbatch_args += ["-A", args.account]
                if args.partition:
                    sbatch_args += ["-p", args.partition]
                if args.gres:
                    sbatch_args += ["--gres", args.gres]
                script = _sbatch_script(
                    repo_root, os.path.join(outdir, node + ".json"))
                job_to_host[fablib.submit_sbatch(script, sbatch_args)] = node
            fablib.wait_for_jobs(set(job_to_host), args.timeout * 60)
            snapshots = {}
            for node in nodes:
                path = os.path.join(outdir, node + ".json")
                try:
                    with open(path) as fh:
                        snapshots[node] = json.load(fh)
                except (OSError, ValueError):
                    snapshots[node] = None
    except RuntimeError as exc:
        fablib.err_exit(TOOL, str(exc))

    summary = aggregate(snapshots)
    if not args.from_dir or os.access(outdir, os.W_OK):
        try:
            with open(os.path.join(outdir, "summary.json"), "w") as fh:
                json.dump(summary, fh, indent=2)
        except OSError:
            pass  # summary still printed below
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(render_fleet_text(summary))
    return 0 if summary["clean"] else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m unittest discover -s fabric/tests -v`
Expected: 19 tests, all PASS.

- [ ] **Step 6: Commit**

```bash
git add fabric/fablib.py fabric/fleet_snapshot fabric/tests/test_fleet_snapshot.py
git commit -m "fleet_snapshot: Slurm submission CLI and fablib helpers"
```

---

### Task 3: `topo_verify`

**Files:**
- Create: `fabric/topo_verify`
- Test: `fabric/tests/test_topo_verify.py`

**Interfaces:**
- Consumes: `loader.load_tool`/`fixture`, snapshot fixtures, `fablib` (import only, for `err_exit`/`run`).
- Produces: `compare(golden, current) -> {"hardware": [str], "informational": [str]}`; `main(argv=None) -> int`; flags `--golden FILE`, `--current FILE`, `--save-golden`, `--strict`, `--json`.

- [ ] **Step 1: Write the failing tests**

`fabric/tests/test_topo_verify.py`:

```python
"""Tests for fabric/topo_verify."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from fabric.tests import loader

tv = loader.load_tool("topo_verify")
TOOL_PATH = os.path.abspath(os.path.join(loader.HERE, "..", "topo_verify"))


def snap():
    return json.loads(loader.fixture("snapshot_nodea.json"))


def run_cli(*argv):
    return subprocess.run([sys.executable, TOOL_PATH] + list(argv),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True)


class TestCompare(unittest.TestCase):
    def test_identical_is_clean(self):
        result = tv.compare(snap(), snap())
        self.assertEqual(result, {"hardware": [], "informational": []})

    def test_gpu_count_change_is_hardware(self):
        current = snap()
        current["gpus"]["gpus"].pop()
        result = tv.compare(snap(), current)
        self.assertTrue(any("gpu" in f.lower() for f in result["hardware"]),
                        result)

    def test_port_rate_change_is_hardware(self):
        current = snap()
        current["ib"]["hcas"][0]["ports"][0]["rate"] = "200 Gb/sec (4X HDR)"
        result = tv.compare(snap(), current)
        self.assertTrue(any("rate" in f for f in result["hardware"]), result)

    def test_netdev_mapping_change_is_hardware(self):
        current = snap()
        current["ibdev2netdev"]["mapping"][0]["netdev"] = "ibp1"
        result = tv.compare(snap(), current)
        self.assertTrue(any("ibdev2netdev" in f for f in result["hardware"]),
                        result)

    def test_topology_row_change_is_hardware(self):
        current = snap()
        current["topology"]["raw"] = current["topology"]["raw"].replace(
            "PXB", "SYS")
        result = tv.compare(snap(), current)
        self.assertTrue(any("topology" in f for f in result["hardware"]),
                        result)

    def test_driver_change_is_informational(self):
        current = snap()
        current["system"]["nvidia_driver"] = "580.00.01"
        result = tv.compare(snap(), current)
        self.assertEqual(result["hardware"], [])
        self.assertTrue(any("driver" in f for f in result["informational"]),
                        result)

    def test_volatile_fields_ignored(self):
        current = snap()
        current["gpus"]["gpus"][0]["temp_c"] = 85.0
        current["gpus"]["gpus"][0]["memory_used_mib"] = 100000
        current["ib"]["hcas"][0]["ports"][0]["counters"]["port_xmit_data"] = 1
        current["timestamp_utc"] = "2030-01-01T00:00:00+00:00"
        result = tv.compare(snap(), current)
        self.assertEqual(result, {"hardware": [], "informational": []})

    def test_missing_ib_section_is_hardware_note(self):
        current = snap()
        current["ib"] = {"error": "no /sys/class/infiniband", "hcas": []}
        result = tv.compare(snap(), current)
        self.assertTrue(any("unavailable" in f or "missing" in f
                            for f in result["hardware"]), result)


class TestCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)

    def path_for(self, data, name):
        p = os.path.join(self.tmp, name)
        with open(p, "w") as fh:
            json.dump(data, fh)
        return p

    def test_match_exits_0(self):
        g = self.path_for(snap(), "golden.json")
        c = self.path_for(snap(), "current.json")
        proc = run_cli("--golden", g, "--current", c)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("MATCH", proc.stdout)

    def test_hardware_drift_exits_1(self):
        current = snap()
        current["ib"]["hcas"][0]["ports"][0]["rate"] = "200 Gb/sec (4X HDR)"
        g = self.path_for(snap(), "golden.json")
        c = self.path_for(current, "current.json")
        proc = run_cli("--golden", g, "--current", c)
        self.assertEqual(proc.returncode, 1, proc.stdout)
        self.assertIn("DRIFT", proc.stdout)

    def test_info_drift_exits_0_without_strict(self):
        current = snap()
        current["system"]["nvidia_driver"] = "580.00.01"
        g = self.path_for(snap(), "golden.json")
        c = self.path_for(current, "current.json")
        self.assertEqual(run_cli("--golden", g, "--current", c).returncode, 0)
        self.assertEqual(run_cli("--golden", g, "--current", c,
                                 "--strict").returncode, 1)

    def test_missing_golden_exits_3(self):
        c = self.path_for(snap(), "current.json")
        proc = run_cli("--golden", os.path.join(self.tmp, "nope.json"),
                       "--current", c)
        self.assertEqual(proc.returncode, 3)
        self.assertIn("save-golden", proc.stderr)

    def test_save_golden_from_current(self):
        c = self.path_for(snap(), "current.json")
        g = os.path.join(self.tmp, "golden_out.json")
        proc = run_cli("--save-golden", "--current", c, "--golden", g)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(g) as fh:
            self.assertEqual(json.load(fh)["hostname"], "nodea")

    def test_json_output(self):
        current = snap()
        current["system"]["nvidia_driver"] = "580.00.01"
        g = self.path_for(snap(), "golden.json")
        c = self.path_for(current, "current.json")
        proc = run_cli("--golden", g, "--current", c, "--json")
        findings = json.loads(proc.stdout)
        self.assertEqual(findings["hardware"], [])
        self.assertEqual(len(findings["informational"]), 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest discover -s fabric/tests -v`
Expected: import-time ERROR — `FileNotFoundError` for `fabric/topo_verify`; the 19 earlier tests PASS.

- [ ] **Step 3: Implement `topo_verify`**

`fabric/topo_verify`:

```python
#!/usr/bin/env python3
"""topo_verify -- compare a node's current ib_snapshot against a blessed
"golden" snapshot and report hardware/topology drift.

Usage:
    topo_verify [--golden FILE] [--current FILE] [--strict] [--json]
    topo_verify --save-golden [--current FILE] [--golden FILE]

Default golden: results/golden/<hostname>.json (hostname from the current
snapshot). Without --current, a fresh snapshot is taken via ib_snapshot.sh
(requires being on the node). Identity fields only are compared; volatile
fields (counters, temps, clocks) are ignored. Driver/kernel/CUDA changes are
informational and set exit 1 only with --strict.

Exit codes: 0 match, 1 drift, 3 probe/setup error.
"""

import argparse
import json
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import fablib

TOOL = "topo_verify"
_ROW_RE = re.compile(r"^(GPU|NIC|mlx)\S*", re.IGNORECASE)


def _gpu_identity(snap):
    gpus = (snap.get("gpus") or {}).get("gpus") or []
    return [(g.get("index"), g.get("name"), g.get("pci_bus_id"))
            for g in gpus]


def _hca_ports(snap):
    out = {}
    for hca in ((snap.get("ib") or {}).get("hcas") or []):
        for port in (hca.get("ports") or []):
            key = "%s/port%s" % (hca.get("name"), port.get("port"))
            out[key] = dict((f, port.get(f)) for f in
                            ("state", "phys_state", "rate", "link_layer"))
    return out


def _mapping(snap):
    mapping = (snap.get("ibdev2netdev") or {}).get("mapping") or []
    return sorted((m.get("hca"), m.get("port"), m.get("netdev"))
                  for m in mapping)


def _topo_rows(snap):
    raw = (snap.get("topology") or {}).get("raw") or ""
    rows = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped and _ROW_RE.match(stripped):
            rows.append(" ".join(stripped.split()))
    return rows


def _info_fields(snap):
    system = snap.get("system") or {}
    uname = (system.get("uname") or "").split()
    return {
        "driver": system.get("nvidia_driver"),
        "kernel": uname[2] if len(uname) > 2 else None,
        "cuda": system.get("cuda_runtime"),
    }


def compare(golden, current):
    """Identity-field diff -> {'hardware': [...], 'informational': [...]}."""
    hardware, informational = [], []

    g_ib = (golden.get("ib") or {}).get("hcas")
    c_ib = (current.get("ib") or {}).get("hcas")
    if g_ib and not c_ib:
        hardware.append("ib section unavailable in current snapshot (%s)"
                        % (current.get("ib") or {}).get("error", "no hcas"))

    g_gpus, c_gpus = _gpu_identity(golden), _gpu_identity(current)
    if g_gpus != c_gpus:
        hardware.append("gpu inventory: golden=%s current=%s"
                        % (g_gpus, c_gpus))

    g_ports, c_ports = _hca_ports(golden), _hca_ports(current)
    for key in sorted(set(g_ports) | set(c_ports)):
        if key not in c_ports:
            if c_ib:  # already reported wholesale above when section missing
                hardware.append("%s: missing in current snapshot" % key)
            continue
        if key not in g_ports:
            hardware.append("%s: not present in golden snapshot" % key)
            continue
        for field in ("state", "phys_state", "rate", "link_layer"):
            if g_ports[key][field] != c_ports[key][field]:
                hardware.append("%s %s: golden=%s current=%s"
                                % (key, field, g_ports[key][field],
                                   c_ports[key][field]))

    if _mapping(golden) != _mapping(current):
        hardware.append("ibdev2netdev mapping: golden=%s current=%s"
                        % (_mapping(golden), _mapping(current)))

    if _topo_rows(golden) != _topo_rows(current):
        hardware.append("topology matrix rows changed (golden vs current):\n"
                        "    golden : %s\n    current: %s"
                        % (_topo_rows(golden), _topo_rows(current)))

    g_info, c_info = _info_fields(golden), _info_fields(current)
    for field in sorted(g_info):
        if g_info[field] != c_info[field]:
            informational.append("%s: golden=%s current=%s"
                                 % (field, g_info[field], c_info[field]))

    return {"hardware": hardware, "informational": informational}


def _take_snapshot():
    """Run ib_snapshot.sh into a temp file; return the parsed dict."""
    script = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "ib_snapshot.sh")
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    r = fablib.run([script, tmp.name], timeout=120)
    if r.returncode != 0:
        raise RuntimeError("ib_snapshot.sh failed: %s" % r.stderr.strip())
    with open(tmp.name) as fh:
        snap = json.load(fh)
    os.unlink(tmp.name)
    return snap


def render_text(findings, strict):
    lines = []
    if findings["hardware"]:
        lines.append("Hardware drift (%d):" % len(findings["hardware"]))
        lines.extend("  " + f for f in findings["hardware"])
    if findings["informational"]:
        lines.append("Informational drift (%d)%s:"
                     % (len(findings["informational"]),
                        "" if strict else " (not counted without --strict)"))
        lines.extend("  " + f for f in findings["informational"])
    drift = findings["hardware"] or (strict and findings["informational"])
    lines.append("topo_verify: %s" % ("DRIFT" if drift else "MATCH"))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog=TOOL,
        description="diff a node's snapshot against its golden snapshot "
                    "(exit 0 match, 1 drift, 3 error)")
    parser.add_argument("--golden", metavar="FILE",
                        help="golden snapshot (default "
                             "results/golden/<hostname>.json)")
    parser.add_argument("--current", metavar="FILE",
                        help="use an existing snapshot instead of probing")
    parser.add_argument("--save-golden", action="store_true",
                        help="bless the current snapshot as golden and exit")
    parser.add_argument("--strict", action="store_true",
                        help="informational drift also sets exit 1")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    repo_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    try:
        if args.current:
            with open(args.current) as fh:
                current = json.load(fh)
        else:
            current = _take_snapshot()
    except (OSError, ValueError, RuntimeError) as exc:
        fablib.err_exit(TOOL, "cannot obtain current snapshot: %s" % exc)

    golden_path = args.golden or os.path.join(
        repo_root, "results", "golden",
        "%s.json" % current.get("hostname", "unknown"))

    if args.save_golden:
        try:
            os.makedirs(os.path.dirname(golden_path), exist_ok=True)
            with open(golden_path, "w") as fh:
                json.dump(current, fh, indent=2)
        except OSError as exc:
            fablib.err_exit(TOOL, "cannot write golden: %s" % exc)
        print("golden saved: %s" % golden_path)
        return 0

    try:
        with open(golden_path) as fh:
            golden = json.load(fh)
    except (OSError, ValueError):
        fablib.err_exit(TOOL, "no golden snapshot at %s -- bless one with: "
                        "%s --save-golden" % (golden_path, TOOL))

    findings = compare(golden, current)
    if args.json:
        print(json.dumps(findings, indent=2))
    else:
        print(render_text(findings, args.strict))
    drift = findings["hardware"] or (args.strict and findings["informational"])
    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Make it executable and run the tests**

Run: `chmod +x fabric/topo_verify && python3 -m unittest discover -s fabric/tests -v`
Expected: 33 tests, all PASS.

- [ ] **Step 5: Commit**

```bash
git add fabric/topo_verify fabric/tests/test_topo_verify.py
git commit -m "Add topo_verify golden-snapshot drift checker"
```

---

### Task 4: `nccl_check`

**Files:**
- Create: `fabric/nccl_check`, `fabric/tests/fixtures/nccl_allreduce.out`
- Test: `fabric/tests/test_nccl_check.py`

**Interfaces:**
- Consumes: `fablib` (`run`, `err_exit`, `submit_sbatch`, `wait_for_jobs`), `loader`.
- Produces: `find_binary(cli=None, env=None) -> str|None`; `parse_nccl_table(text) -> [{"size": int, "oop_busbw": float, "ip_busbw": float}]`; `summarize(rows) -> {"peak_busbw": float, "large_busbw": float, "large_size": int}`; `parse_counter_lines(text) -> [str]` (hosts whose counter_delta rc != 0); `main(argv=None) -> int` with flags `-A -p -N -G -w --min-busbw --binary --counters --timeout`.

- [ ] **Step 1: Create the fixture**

`fabric/tests/fixtures/nccl_allreduce.out`:

```text
# nThread 1 nGpus 1 minBytes 8 maxBytes 1073741824 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid 100 on nodea device  0 [0x19] NVIDIA H200
#  Rank  1 Group  0 Pid 101 on nodea device  1 [0x3b] NVIDIA H200
#
#                                                              out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                                 (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
           8             2     float     sum      -1    21.63    0.00    0.00      0    20.87    0.00    0.00      0
        1024           256     float     sum      -1    22.30    0.05    0.09      0    21.94    0.05    0.09      0
     1048576        262144     float     sum      -1    45.10   23.25   43.59      0    44.80   23.41   43.89      0
   134217728      33554432     float     sum      -1   823.50  163.00  305.62      0   820.10  163.66  306.87      0
  1073741824     268435456     float     sum      -1  6120.00  175.45  329.00      0  6100.00  176.02  330.07      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 145.2
```

- [ ] **Step 2: Write the failing tests**

`fabric/tests/test_nccl_check.py`:

```python
"""Tests for fabric/nccl_check (parsing/verdict; Slurm paths are manual)."""
import os
import shutil
import stat
import tempfile
import unittest

from fabric.tests import loader

nc = loader.load_tool("nccl_check")


class TestParseNcclTable(unittest.TestCase):
    def test_parses_data_rows(self):
        rows = nc.parse_nccl_table(loader.fixture("nccl_allreduce.out"))
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0]["size"], 8)
        self.assertEqual(rows[-1]["size"], 1073741824)
        self.assertAlmostEqual(rows[-1]["oop_busbw"], 329.00)
        self.assertAlmostEqual(rows[-1]["ip_busbw"], 330.07)

    def test_truncated_output_is_empty(self):
        text = "\n".join(loader.fixture("nccl_allreduce.out").splitlines()[:8])
        self.assertEqual(nc.parse_nccl_table(text), [])

    def test_garbage_ignored(self):
        self.assertEqual(nc.parse_nccl_table("slurmstepd: error: oom\n"), [])


class TestSummarize(unittest.TestCase):
    def test_peak_and_large(self):
        rows = nc.parse_nccl_table(loader.fixture("nccl_allreduce.out"))
        summary = nc.summarize(rows)
        self.assertAlmostEqual(summary["peak_busbw"], 329.00)
        self.assertAlmostEqual(summary["large_busbw"], 329.00)
        self.assertEqual(summary["large_size"], 1073741824)


class TestFindBinary(unittest.TestCase):
    def _make_exe(self, directory, name="all_reduce_perf"):
        path = os.path.join(directory, name)
        with open(path, "w") as fh:
            fh.write("#!/bin/sh\n")
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
        return path

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_cli_arg_wins(self):
        exe = self._make_exe(self.tmp)
        self.assertEqual(nc.find_binary(cli=exe, env={}), exe)

    def test_cli_arg_missing_is_none(self):
        self.assertIsNone(
            nc.find_binary(cli=os.path.join(self.tmp, "nope"), env={}))

    def test_env_var(self):
        exe = self._make_exe(self.tmp)
        self.assertEqual(nc.find_binary(env={"NCCL_TESTS_BIN": exe}), exe)

    def test_not_found_returns_none(self):
        self.assertIsNone(nc.find_binary(env={"PATH": self.tmp}))


class TestCounterLines(unittest.TestCase):
    def test_nonzero_rc_reported(self):
        text = "COUNTER_DELTA nodea rc=0\nCOUNTER_DELTA nodeb rc=1\n"
        self.assertEqual(nc.parse_counter_lines(text), ["nodeb"])

    def test_absent_lines_empty(self):
        self.assertEqual(nc.parse_counter_lines("no markers here"), [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m unittest discover -s fabric/tests -v`
Expected: import-time ERROR — `FileNotFoundError` for `fabric/nccl_check`; the 33 earlier tests PASS.

- [ ] **Step 4: Implement `nccl_check`**

`fabric/nccl_check`:

```python
#!/usr/bin/env python3
"""nccl_check -- NCCL all-reduce bandwidth smoke test via Slurm.

Usage:
    nccl_check -A ACCOUNT -p PARTITION [-N NODES] [-G GPUS_PER_NODE]
               [-w NODELIST] [--min-busbw GBPS] [--binary PATH]
               [--counters] [--timeout MIN]

Runs all_reduce_perf (-b 8 -e 1G -f 2 -g 1, one task per GPU) in a fresh
Slurm allocation and reports peak / large-message bus bandwidth. With
--min-busbw it is a pass/fail gate on the large-message out-of-place busbw;
without it, report-only. No built-in threshold: expected busbw depends on
GPU generation and rail count (see README). --counters brackets the run
with ib_snapshot + counter_delta on every node; any counter delta fails.

Binary discovery: --binary, $NCCL_TESTS_BIN, all_reduce_perf on $PATH,
then common install dirs.

Exit codes: 0 pass/report, 1 below threshold or counter deltas, 3 setup error.
"""

import argparse
import datetime
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import fablib

TOOL = "nccl_check"
COMMON_BINARY_DIRS = ("/usr/local/cuda/nccl-tests/build",
                      "/opt/nccl-tests/build")
_COUNTER_RE = re.compile(r"^COUNTER_DELTA (\S+) rc=(\d+)", re.MULTILINE)


def find_binary(cli=None, env=None):
    """Locate all_reduce_perf; returns a path or None."""
    env = os.environ if env is None else env
    if cli:
        return cli if os.path.isfile(cli) else None
    env_bin = env.get("NCCL_TESTS_BIN")
    if env_bin:
        return env_bin if os.path.isfile(env_bin) else None
    on_path = shutil.which("all_reduce_perf", path=env.get("PATH"))
    if on_path:
        return on_path
    for d in COMMON_BINARY_DIRS:
        candidate = os.path.join(d, "all_reduce_perf")
        if os.path.isfile(candidate):
            return candidate
    return None


def parse_nccl_table(text):
    """nccl-tests output -> [{'size', 'oop_busbw', 'ip_busbw'}, ...]."""
    rows = []
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) < 13 or line.lstrip().startswith("#"):
            continue
        try:
            rows.append({"size": int(parts[0]),
                         "oop_busbw": float(parts[7]),
                         "ip_busbw": float(parts[11])})
        except ValueError:
            continue
    return rows


def summarize(rows):
    largest = max(rows, key=lambda r: r["size"])
    return {"peak_busbw": max(r["oop_busbw"] for r in rows),
            "large_busbw": largest["oop_busbw"],
            "large_size": largest["size"]}


def parse_counter_lines(text):
    """Hosts whose in-job counter_delta reported a nonzero rc."""
    return sorted(host for host, rc in _COUNTER_RE.findall(text or "")
                  if int(rc) != 0)


def _job_script(binary, gpn, counters, repo_root, outdir):
    lines = ["#!/bin/bash", "set -u"]
    if counters:
        lines += [
            "SNAPDIR=%s/counters" % outdir,
            "mkdir -p $SNAPDIR",
            "srun --ntasks-per-node=1 bash -c "
            "'%s/fabric/ib_snapshot.sh $SNAPDIR/before_$(hostname -s).json'"
            % repo_root,
        ]
    lines.append("srun --ntasks-per-node=%d --gpus-per-node=%d "
                 "%s -b 8 -e 1G -f 2 -g 1" % (gpn, gpn, binary))
    lines.append("rc=$?")
    if counters:
        lines += [
            "srun --ntasks-per-node=1 bash -c "
            "'%s/fabric/ib_snapshot.sh $SNAPDIR/after_$(hostname -s).json'"
            % repo_root,
            "for b in $SNAPDIR/before_*.json; do",
            "  h=$(basename $b); h=${h#before_}; h=${h%.json}",
            "  %s/fabric/counter_delta.sh $b $SNAPDIR/after_$h.json "
            ">/dev/null 2>&1" % repo_root,
            "  echo \"COUNTER_DELTA $h rc=$?\"",
            "done",
        ]
    lines.append("exit $rc")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog=TOOL, description="NCCL all-reduce smoke test via Slurm "
        "(exit 0 pass/report, 1 fail, 3 setup error)")
    parser.add_argument("-A", "--account", required=True)
    parser.add_argument("-p", "--partition", required=True)
    parser.add_argument("-N", "--nodes", type=int, default=1)
    parser.add_argument("-G", "--gpus-per-node", type=int, default=4,
                        dest="gpn")
    parser.add_argument("-w", "--nodelist")
    parser.add_argument("--min-busbw", type=float,
                        help="GB/s; pass/fail gate on large-message busbw")
    parser.add_argument("--binary", help="path to all_reduce_perf")
    parser.add_argument("--counters", action="store_true",
                        help="bracket with ib_snapshot + counter_delta")
    parser.add_argument("--timeout", type=int, default=20,
                        help="minutes (job time limit and wait; default 20)")
    args = parser.parse_args(argv)

    binary = find_binary(cli=args.binary)
    if not binary:
        fablib.err_exit(TOOL, "all_reduce_perf not found -- pass --binary "
                        "PATH or set NCCL_TESTS_BIN to your nccl-tests build")

    repo_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    runid = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    outdir = os.path.join(repo_root, "results", "nccl_check_" + runid)
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, "job.out")

    sbatch_args = ["-J", TOOL, "-A", args.account, "-p", args.partition,
                   "-N", str(args.nodes), "--gpus-per-node=%d" % args.gpn,
                   "-c", "8", "--mem=32G", "-t", str(args.timeout),
                   "-o", outfile]
    if args.nodelist:
        sbatch_args += ["-w", args.nodelist]

    try:
        job = fablib.submit_sbatch(
            _job_script(binary, args.gpn, args.counters, repo_root, outdir),
            sbatch_args)
        stragglers = fablib.wait_for_jobs({job}, args.timeout * 60 + 120)
    except RuntimeError as exc:
        fablib.err_exit(TOOL, str(exc))
    if stragglers:
        fablib.err_exit(TOOL, "job %s did not finish within --timeout "
                        "(cancelled); partial output: %s" % (job, outfile))

    try:
        with open(outfile) as fh:
            output = fh.read()
    except OSError as exc:
        fablib.err_exit(TOOL, "no job output: %s" % exc)

    rows = parse_nccl_table(output)
    if not rows:
        fablib.err_exit(TOOL, "no nccl-tests result table in %s "
                        "(job failed?)" % outfile)
    summary = summarize(rows)
    bad_hosts = parse_counter_lines(output) if args.counters else []

    print("nccl_check: %d node(s) x %d GPU(s), binary %s"
          % (args.nodes, args.gpn, binary))
    print("  peak busbw          : %.2f GB/s" % summary["peak_busbw"])
    print("  large-message busbw : %.2f GB/s (size %d)"
          % (summary["large_busbw"], summary["large_size"]))
    if args.counters:
        print("  counter deltas      : %s"
              % (", ".join(bad_hosts) if bad_hosts else "none"))

    failed = False
    if args.min_busbw is not None:
        ok = summary["large_busbw"] >= args.min_busbw
        print("verdict: %s (threshold %.2f GB/s)"
              % ("PASS" if ok else "FAIL", args.min_busbw))
        failed = not ok
    else:
        print("verdict: REPORT-ONLY (no --min-busbw given)")
    if bad_hosts:
        print("verdict: FAIL (fabric error counters advanced on: %s)"
              % ", ".join(bad_hosts))
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Make it executable and run the tests**

Run: `chmod +x fabric/nccl_check && python3 -m unittest discover -s fabric/tests -v`
Expected: 43 tests, all PASS.

- [ ] **Step 6: Commit**

```bash
git add fabric/nccl_check fabric/tests/fixtures/nccl_allreduce.out fabric/tests/test_nccl_check.py
git commit -m "Add nccl_check NCCL bandwidth smoke test"
```

---

### Task 5: `ib_bw_pair`

**Files:**
- Create: `fabric/ib_bw_pair`, `fabric/tests/fixtures/ib_write_bw.out`
- Test: `fabric/tests/test_ib_bw_pair.py`

**Interfaces:**
- Consumes: `fablib`, `loader`.
- Produces: `parse_perftest(text) -> {"bytes": int, "bw_peak_gbps": float, "bw_avg_gbps": float, "msg_rate_mpps": float}|None` (largest-size row; converts MB/sec to Gb/s when the header says MB); `_rate_gbps(text) -> float|None`; `verdict(bw_avg_gbps, min_gbps, rate_gbps) -> (str, str)` returning `("PASS"|"FAIL"|"REPORT", reason)`; `RATE_FRACTION = 0.8`; `main(argv=None) -> int` with flags `-A -p -w --test --device --min-gbps --timeout`.

- [ ] **Step 1: Create the fixture**

`fabric/tests/fixtures/ib_write_bw.out`:

```text
---------------------------------------------------------------------------------------
                    RDMA_Write BW Test
 Dual-port       : OFF          Device         : mlx5_0
 Number of qps   : 1            Transport type : IB
 Connection type : RC           Using SRQ      : OFF
 TX depth        : 128
 CQ Moderation   : 100
 Mtu             : 4096[B]
 Link type       : IB
 Max inline data : 0[B]
 rdma_cm QPs     : OFF
 Data ex. method : Ethernet
---------------------------------------------------------------------------------------
 #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]
 2          5000             0.052              0.051                3.213725
 65536      5000             361.20             360.88               0.688305
 8388608    5000             389.45             389.20               0.005800
---------------------------------------------------------------------------------------
```

- [ ] **Step 2: Write the failing tests**

`fabric/tests/test_ib_bw_pair.py`:

```python
"""Tests for fabric/ib_bw_pair (parsing/verdict; Slurm path is manual)."""
import unittest

from fabric.tests import loader

bw = loader.load_tool("ib_bw_pair")


class TestParsePerftest(unittest.TestCase):
    def test_parses_largest_row_gbits(self):
        result = bw.parse_perftest(loader.fixture("ib_write_bw.out"))
        self.assertEqual(result["bytes"], 8388608)
        self.assertAlmostEqual(result["bw_avg_gbps"], 389.20)
        self.assertAlmostEqual(result["bw_peak_gbps"], 389.45)
        self.assertAlmostEqual(result["msg_rate_mpps"], 0.0058)

    def test_mb_per_sec_converted(self):
        text = loader.fixture("ib_write_bw.out").replace(
            "[Gb/sec]", "[MB/sec]")
        result = bw.parse_perftest(text)
        # 389.20 MB/s * 8 / 1000 = 3.1136 Gb/s
        self.assertAlmostEqual(result["bw_avg_gbps"], 3.1136, places=3)

    def test_truncated_is_none(self):
        text = "\n".join(loader.fixture("ib_write_bw.out").splitlines()[:13])
        self.assertIsNone(bw.parse_perftest(text))
        self.assertIsNone(bw.parse_perftest(""))


class TestRateGbps(unittest.TestCase):
    def test_rate_line(self):
        self.assertEqual(bw._rate_gbps("400 Gb/sec (4X NDR)"), 400.0)
        self.assertIsNone(bw._rate_gbps(""))


class TestVerdict(unittest.TestCase):
    def test_explicit_threshold(self):
        self.assertEqual(bw.verdict(389.2, 350.0, None)[0], "PASS")
        self.assertEqual(bw.verdict(300.0, 350.0, None)[0], "FAIL")

    def test_rate_fraction_default(self):
        # 0.8 * 400 = 320
        self.assertEqual(bw.verdict(389.2, None, 400.0)[0], "PASS")
        self.assertEqual(bw.verdict(300.0, None, 400.0)[0], "FAIL")

    def test_no_threshold_no_rate_is_report(self):
        status, reason = bw.verdict(389.2, None, None)
        self.assertEqual(status, "REPORT")
        self.assertIn("rate unavailable", reason)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m unittest discover -s fabric/tests -v`
Expected: import-time ERROR — `FileNotFoundError` for `fabric/ib_bw_pair`; the 43 earlier tests PASS.

- [ ] **Step 4: Implement `ib_bw_pair`**

`fabric/ib_bw_pair`:

```python
#!/usr/bin/env python3
"""ib_bw_pair -- point-to-point RDMA bandwidth test between two nodes.

Usage:
    ib_bw_pair -A ACCOUNT -p PARTITION -w NODEA,NODEB [--test write|read]
               [--device DEV] [--min-gbps N] [--timeout MIN]

Submits one 2-node Slurm job: perftest server on node A, client on node B
(ib_write_bw / ib_read_bw -a --report_gbits). Verdict: --min-gbps if given,
else BW average >= RATE_FRACTION (0.8) of the server port's sysfs rate;
REPORT-only if the rate is unavailable.

Exit codes: 0 pass/report, 1 below threshold, 3 setup/parse error.
"""

import argparse
import datetime
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import fablib

TOOL = "ib_bw_pair"
RATE_FRACTION = 0.8  # default pass threshold as a fraction of port rate
_NUM_RE = re.compile(r"\d+(\.\d+)?")
_UNIT_RE = re.compile(r"BW average\[(\w+)/sec\]")


def _rate_gbps(text):
    if not text:
        return None
    m = _NUM_RE.search(text)
    return float(m.group()) if m else None


def parse_perftest(text):
    """perftest client output -> largest-size row dict, or None."""
    unit = None
    rows = []
    for line in (text or "").splitlines():
        m = _UNIT_RE.search(line)
        if m:
            unit = m.group(1)
            continue
        parts = line.split()
        if len(parts) == 5 and unit:
            try:
                rows.append({"bytes": int(parts[0]),
                             "bw_peak": float(parts[2]),
                             "bw_avg": float(parts[3]),
                             "msg_rate_mpps": float(parts[4])})
            except ValueError:
                continue
    if not rows:
        return None
    largest = max(rows, key=lambda r: r["bytes"])
    factor = 1.0 if unit == "Gb" else 8.0 / 1000.0  # MB/sec -> Gb/sec
    return {"bytes": largest["bytes"],
            "bw_peak_gbps": largest["bw_peak"] * factor,
            "bw_avg_gbps": largest["bw_avg"] * factor,
            "msg_rate_mpps": largest["msg_rate_mpps"]}


def verdict(bw_avg_gbps, min_gbps, rate_gbps):
    """-> ('PASS'|'FAIL'|'REPORT', reason)."""
    if min_gbps is not None:
        ok = bw_avg_gbps >= min_gbps
        return ("PASS" if ok else "FAIL",
                "threshold %.1f Gb/s (explicit)" % min_gbps)
    if rate_gbps is not None:
        threshold = RATE_FRACTION * rate_gbps
        ok = bw_avg_gbps >= threshold
        return ("PASS" if ok else "FAIL",
                "threshold %.1f Gb/s (%.0f%% of port rate %.0f Gb/s)"
                % (threshold, RATE_FRACTION * 100, rate_gbps))
    return ("REPORT", "no --min-gbps and port rate unavailable")


def _job_script(test_bin, device, outdir):
    dev_flag = "-d %s" % device if device else ""
    dev_dir = device if device else "$(ls /sys/class/infiniband | head -1)"
    return """#!/bin/bash
set -u
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
A=${NODES[0]}; B=${NODES[1]}
srun -w $A -N1 -n1 --overlap bash -c \
  'cat /sys/class/infiniband/%s/ports/1/rate' > %s/rate.txt 2>/dev/null
srun -w $A -N1 -n1 --overlap %s -a --report_gbits %s > %s/server.log 2>&1 &
sleep 5
srun -w $B -N1 -n1 --overlap %s -a --report_gbits %s $A > %s/client.log 2>&1
rc=$?
wait
exit $rc
""" % (dev_dir, outdir, test_bin, dev_flag, outdir, test_bin, dev_flag,
       outdir)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog=TOOL, description="pairwise RDMA bandwidth test via Slurm "
        "(exit 0 pass/report, 1 fail, 3 setup error)")
    parser.add_argument("-A", "--account", required=True)
    parser.add_argument("-p", "--partition", required=True)
    parser.add_argument("-w", "--nodelist", required=True,
                        help="exactly two nodes: nodeA,nodeB")
    parser.add_argument("--test", choices=("write", "read"), default="write")
    parser.add_argument("--device", help="HCA, e.g. mlx5_0")
    parser.add_argument("--min-gbps", type=float)
    parser.add_argument("--timeout", type=int, default=15,
                        help="minutes (job limit and wait; default 15)")
    args = parser.parse_args(argv)

    try:
        nodes = fablib.expand_nodes(nodelist=args.nodelist)
    except RuntimeError as exc:
        fablib.err_exit(TOOL, str(exc))
    if len(nodes) != 2:
        fablib.err_exit(TOOL, "need exactly two nodes, got %s" % nodes)

    test_bin = "ib_%s_bw" % args.test
    repo_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    runid = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    outdir = os.path.join(repo_root, "results", "ib_bw_pair_" + runid)
    os.makedirs(outdir, exist_ok=True)

    sbatch_args = ["-J", TOOL, "-A", args.account, "-p", args.partition,
                   "-N", "2", "-w", ",".join(nodes), "-c", "2", "--mem=8G",
                   "-t", str(args.timeout),
                   "-o", os.path.join(outdir, "job.out")]
    try:
        job = fablib.submit_sbatch(
            _job_script(test_bin, args.device, outdir), sbatch_args)
        stragglers = fablib.wait_for_jobs({job}, args.timeout * 60 + 120)
    except RuntimeError as exc:
        fablib.err_exit(TOOL, str(exc))
    if stragglers:
        fablib.err_exit(TOOL, "job %s did not finish within --timeout "
                        "(cancelled); see %s" % (job, outdir))

    try:
        with open(os.path.join(outdir, "client.log")) as fh:
            client_out = fh.read()
    except OSError as exc:
        fablib.err_exit(TOOL, "no client output: %s" % exc)
    rate_text = ""
    try:
        with open(os.path.join(outdir, "rate.txt")) as fh:
            rate_text = fh.read()
    except OSError:
        pass

    result = parse_perftest(client_out)
    if not result:
        fablib.err_exit(TOOL, "could not parse perftest output in "
                        "%s/client.log" % outdir)
    rate = _rate_gbps(rate_text)
    status, reason = verdict(result["bw_avg_gbps"], args.min_gbps, rate)

    print("ib_bw_pair: %s  %s -> %s%s"
          % (test_bin, nodes[1], nodes[0],
             "  (dev %s)" % args.device if args.device else ""))
    print("  message size : %d bytes" % result["bytes"])
    print("  BW average   : %.2f Gb/s (peak %.2f)"
          % (result["bw_avg_gbps"], result["bw_peak_gbps"]))
    if rate:
        print("  port rate    : %.0f Gb/s" % rate)
    print("verdict: %s (%s)" % (status, reason))
    return 1 if status == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Make it executable and run the tests**

Run: `chmod +x fabric/ib_bw_pair && python3 -m unittest discover -s fabric/tests -v`
Expected: 50 tests, all PASS.

- [ ] **Step 6: Commit**

```bash
git add fabric/ib_bw_pair fabric/tests/fixtures/ib_write_bw.out fabric/tests/test_ib_bw_pair.py
git commit -m "Add ib_bw_pair pairwise RDMA bandwidth test"
```

---

### Task 6: Documentation and final verification

**Files:**
- Modify: `README.md` (fabric section + requirements table)

**Interfaces:**
- Consumes: the finished tools' flags/behavior (Tasks 1–5).
- Produces: docs only.

- [ ] **Step 1: Update the fabric tool table in `README.md`**

Find the fabric/ section's tool table (rows for `affinity_check.sh`, `counter_delta.sh`, `ib_snapshot.sh`) and append these rows:

```markdown
| `fleet_snapshot` | Runs `ib_snapshot.sh` across a nodelist/partition via per-node Slurm jobs and aggregates: driver/kernel/FW drift, down or degraded links, nonzero error counters, unreached nodes. `--from-dir` re-aggregates a past run offline. Exit 0 clean / 1 anomalies / 3 error. |
| `topo_verify` | Diffs a node's current snapshot against a blessed golden (`results/golden/<host>.json`, blessed with `--save-golden`). Hardware drift exits 1; driver/kernel changes are informational unless `--strict`. |
| `nccl_check` | NCCL all-reduce bandwidth smoke test in a fresh Slurm allocation (1–2 nodes). Report-only by default; `--min-busbw` makes it a pass/fail gate; `--counters` brackets the run with `ib_snapshot` + `counter_delta`. Needs an `nccl-tests` build (`--binary` or `$NCCL_TESTS_BIN`). |
| `ib_bw_pair` | `ib_write_bw`/`ib_read_bw` between two named nodes in one 2-node Slurm job. Passes when BW average ≥ `--min-gbps`, or ≥ 80% of the port's sysfs rate by default. |
```

Directly below the table, add:

```markdown
Typical expected values (guidance, not code defaults): H100/H200 NVLink
intra-node all-reduce large-message busbw ≳ 300 GB/s; 2-node NDR (400 Gb/s
per rail) `ib_write_bw` average ≳ 320 Gb/s (80% of rate). Slurm examples:

​```bash
fabric/fleet_snapshot -A kempner_dev -p kempner_eng
fabric/topo_verify --save-golden          # on the node, once, post-maintenance
fabric/nccl_check -A kempner_dev -p kempner_eng -N 2 -G 4 --binary /path/to/all_reduce_perf
fabric/ib_bw_pair -A kempner_dev -p kempner_eng -w holygpu8a10401,holygpu8a10402
​```
```

(Remove the zero-width characters around the inner code fence when writing the file — they exist only so this plan renders.)

- [ ] **Step 2: Update the fabric requirements row**

In the top-level README "Requirements by directory" table, extend the `fabric/` row's "Needs" cell (keep existing text) with:

```markdown
The Slurm-based tools (`fleet_snapshot`, `nccl_check`, `ib_bw_pair`) additionally need the Slurm client and a valid account/partition; `ib_bw_pair` needs the `perftest` package on the nodes; `nccl_check` needs an `nccl-tests` build. `topo_verify`/`fleet_snapshot --from-dir` also run offline on saved snapshots.
```

- [ ] **Step 3: Run the full suite once**

Run: `python3 -m unittest discover -s fabric/tests -v`
Expected: 50 tests, all PASS (docs-only task; confirms nothing else was touched).

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "Document the fabric suite tools"
```

---

## Self-Review Notes (spec → plan mapping)

- Shared conventions (stdlib, no extension, exec bit, 0/1/3, stderr format, Slurm-only, timeouts) → Global Constraints + every tool's `main`.
- fleet_snapshot steps 1–6 of spec → Task 1 (aggregate: drift/link/counters/unreached/unavailable) + Task 2 (expand, per-node sbatch, wait/cancel, summary.json, exits). Missing-section note rule → `unavailable` category, tested.
- topo_verify (golden path, --save-golden, identity fields, volatile ignored, informational vs hardware, --strict, --json, exit 3 with instructions) → Task 3, each with a named test.
- nccl_check (binary discovery order, sweep flags, table parse, peak/large busbw, no built-in threshold, --min-busbw gate, --counters with counter_delta reuse, exit 3 paths) → Task 4. Slurm runner is manual-smoke per spec's testing section.
- ib_bw_pair (two-node validation, server/client script, --report_gbits parse incl. MB/sec fallback, RATE_FRACTION 0.8 named constant, --device, REPORT when rate unavailable) → Task 5.
- Testing section (synthetic snapshots + doctored variants, captured nccl/perftest outputs + truncated variants, offline CLI exit-code tests, manual smoke for Slurm paths) → fixtures in Tasks 1/4/5, in-test doctoring throughout.
- Documentation section → Task 6.
- Type consistency: `aggregate`/`render_fleet_text` (T1) used by `main` (T2); `fablib` signatures used in T2/T4/T5 match its T2 definition; `loader.load_tool`/`fixture` used in all test files.
