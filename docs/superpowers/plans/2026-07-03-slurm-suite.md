# Slurm Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add five Slurm reporting/triage tools — `job_failures`, `queue_wait`, `fairshare_report`, `frag_report`, `gpu_idle_now` — to `slurm/`, per the approved spec.

**Architecture:** Each tool is an extensionless Python executable with a pure parse/classify/aggregate core (offline-testable via `--from-file`/`--from-json` replay seams) and thin command runners. A shared stdlib-only `slurm/slurmlib.py` holds what the tools repeat: `run` (subprocess with timeout), window parsing (semantics identical to `gpu/gpulib`), `err_exit`, duration humanizing, nearest-rank percentiles, TRES GPU parsing. `gpu_idle_now` alone also imports `gpu/gpulib.py` (Prometheus) via a `../gpu` path insert — hence the branch stacks on `gpu-suite`.

**Tech Stack:** Python 3.6-compatible stdlib; `sacct`/`sshare`/`scontrol`/`squeue`; `gpu/gpulib.py` (+`requests`) for `gpu_idle_now` only; stdlib `unittest`.

**Spec:** `docs/superpowers/specs/2026-07-03-slurm-suite-design.md` (approved).

**Branch:** create `slurm-suite` FROM `gpu-suite` (controller does this before Task 1; the spec's merge order is gpu-health → gpu-suite → slurm-suite).

**Spec interpretation note:** the spec says the window flags are implemented "locally to stay stdlib-only, matching the gpulib behavior exactly." This plan realizes that as ONE local shared module `slurm/slurmlib.py` (stdlib-only, behavior-identical `parse_window`) rather than copy-pasting the logic into each tool — same intent, no verbatim duplication.

## Global Constraints

- Python 3.6-compatible; no f-strings, walrus, dataclasses, or `subprocess.run(capture_output=/text=)` kwargs.
- Tools are `slurm/job_failures`, `slurm/queue_wait`, `slurm/fairshare_report`, `slurm/frag_report`, `slurm/gpu_idle_now` — no `.py` extension, `#!/usr/bin/env python3`, executable bit set.
- Exit codes: `0` report/clean, `1` findings (`gpu_idle_now` only: ≥1 fully-idle job), `3` probe/setup error via `slurmlib.err_exit` (`<tool>: error: <msg>` on stderr). argparse usage errors stay exit 2.
- Subprocess timeouts: sacct/sshare 120 s; squeue/scontrol 60 s.
- Unparseable rows, excluded rows, and truncations are counted and reported, never silent.
- `gpu_idle_now`: Prometheus via `gpu/gpulib.py` (path-inserted `../gpu`); error messages never contain the endpoint URL (inherited gpulib contract); gpulib import failure → exit 3 naming the expected relative location.
- Tests: `python3 -m unittest discover -s slurm/tests -v` from the repo root (new suite; the gpu/fabric suites are separate).
- Commit messages: short imperative subject, no attribution footers of any kind.

## File Structure

| File | Responsibility |
|---|---|
| `slurm/slurmlib.py` (create, Task 1) | Shared stdlib helpers: `err_exit`, `run`, `iso_epoch`, `epoch_iso`, `add_window_args`, `parse_window`, `humanize`, `pctile`, `gpus_from_tres`, `read_text`. |
| `slurm/tests/__init__.py`, `loader.py` (create, Task 1) | Suite scaffolding (fabric/gpu loader pattern). |
| `slurm/job_failures` (Task 2) + `slurm/tests/fixtures/sacct_failures.txt` | Failure classification + rollup. |
| `slurm/queue_wait` (Task 3) + `slurm/tests/fixtures/sacct_queue.txt` | Wait distributions. |
| `slurm/fairshare_report` (Task 4) + `slurm/tests/fixtures/sshare.txt` | Share-vs-usage ranking. |
| `slurm/frag_report` (Task 5) + `slurm/tests/fixtures/scontrol_nodes.txt` | Fragmentation + fit counts. |
| `slurm/gpu_idle_now` (Task 6) | Live idle-GPU gate (in-test bundles, no fixture file). |
| `README.md` (modify, Task 7) | slurm/ table rows + requirements sentence. |

---

### Task 1: `slurmlib` and test scaffolding

**Files:**
- Create: `slurm/slurmlib.py`, `slurm/tests/__init__.py` (empty), `slurm/tests/loader.py`
- Test: `slurm/tests/test_slurmlib.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `slurmlib.err_exit(tool, msg)` (stderr `<tool>: error: <msg>`, exit 3); `slurmlib.run(cmd, timeout) -> stdout str` (raises `RuntimeError` on missing binary, timeout, or nonzero exit with stderr included); `slurmlib.iso_epoch(text) -> int` (raises ValueError); `slurmlib.epoch_iso(epoch) -> str`; `slurmlib.add_window_args(parser)`; `slurmlib.parse_window(args, tool) -> (start, end)` (identical semantics to `gpu/gpulib.parse_window`); `slurmlib.humanize(seconds) -> str`; `slurmlib.pctile(sorted_values, pct) -> value|None` (nearest-rank); `slurmlib.gpus_from_tres(tres) -> int`; `slurmlib.read_text(path, tool) -> str` (err_exit on unreadable). Test-side: `loader.load_tool(name)`, `loader.fixture(name)`, `loader.HERE`, `loader.FIXTURES`.

- [ ] **Step 1: Create the scaffolding**

`slurm/tests/__init__.py`: empty file.

`slurm/tests/loader.py`:

```python
"""Load the extensionless slurm tools as modules; locate fixtures."""
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

- [ ] **Step 2: Write the failing tests**

`slurm/tests/test_slurmlib.py`:

```python
"""Tests for slurm/slurmlib.py (pure parts; run() is exercised live)."""
import argparse
import unittest

from slurm.tests import loader

sl = loader.load_tool("slurmlib.py")


def ns(days=7, since=None, until=None):
    return argparse.Namespace(days=days, since=since, until=until)


class TestHumanize(unittest.TestCase):
    def test_seconds(self):
        self.assertEqual(sl.humanize(45), "45s")

    def test_minutes(self):
        self.assertEqual(sl.humanize(150), "2m 30s")

    def test_hours(self):
        self.assertEqual(sl.humanize(9015), "2h 30m")

    def test_days(self):
        self.assertEqual(sl.humanize(90000), "1d 1h")


class TestPctile(unittest.TestCase):
    def test_p50_nearest_rank(self):
        self.assertEqual(sl.pctile([1, 2, 3, 4], 50), 2)

    def test_p90_of_ten(self):
        self.assertEqual(sl.pctile(list(range(1, 11)), 90), 9)

    def test_empty_is_none(self):
        self.assertIsNone(sl.pctile([], 50))


class TestGpusFromTres(unittest.TestCase):
    def test_plain_and_typed(self):
        self.assertEqual(sl.gpus_from_tres(
            "cpu=16,mem=250G,gres/gpu=1,gres/gpu:nvidia_h200=1"), 1)
        self.assertEqual(sl.gpus_from_tres("cpu=4,gres/gpu:h100=4"), 4)

    def test_no_gpu_is_zero(self):
        self.assertEqual(sl.gpus_from_tres("cpu=4,mem=16G"), 0)
        self.assertEqual(sl.gpus_from_tres(""), 0)


class TestParseWindow(unittest.TestCase):
    def test_days_default(self):
        start, end = sl.parse_window(ns(), "t")
        self.assertEqual(end - start, 7 * 86400)

    def test_since_until(self):
        start, end = sl.parse_window(
            ns(since="2026-07-01T00:00:00", until="2026-07-02T00:00:00"), "t")
        self.assertEqual(end - start, 86400)

    def test_bad_pair_exits_3(self):
        with self.assertRaises(SystemExit) as cm:
            sl.parse_window(ns(since="2026-07-01T00:00:00"), "t")
        self.assertEqual(cm.exception.code, 3)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m unittest discover -s slurm/tests -v`
Expected: import-time ERROR — `FileNotFoundError` for `slurm/slurmlib.py`.

- [ ] **Step 4: Implement `slurm/slurmlib.py`**

```python
"""Shared helpers for the slurm/ reporting tools. Stdlib only."""
import datetime
import math
import subprocess
import sys
import time


def err_exit(tool, msg):
    sys.stderr.write("%s: error: %s\n" % (tool, msg))
    sys.exit(3)


def run(cmd, timeout):
    """Run a command; return stdout. RuntimeError on missing binary,
    timeout, or nonzero exit (the command's stderr is included)."""
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, universal_newlines=True,
                           timeout=timeout)
    except OSError:
        raise RuntimeError("command not found: %s" % cmd[0])
    except subprocess.TimeoutExpired:
        raise RuntimeError("command timed out: %s" % " ".join(cmd))
    if r.returncode != 0:
        raise RuntimeError("%s failed: %s" % (cmd[0], r.stderr.strip()))
    return r.stdout


def iso_epoch(text):
    """'YYYY-mm-ddTHH:MM:SS' -> local epoch int. Raises ValueError."""
    return int(time.mktime(datetime.datetime.strptime(
        text, "%Y-%m-%dT%H:%M:%S").timetuple()))


def epoch_iso(epoch):
    return datetime.datetime.fromtimestamp(epoch).strftime(
        "%Y-%m-%dT%H:%M:%S")


def add_window_args(parser):
    parser.add_argument("--days", type=int, default=7,
                        help="window length ending now (default 7)")
    parser.add_argument("--since", metavar="ISO",
                        help="window start, YYYY-mm-ddTHH:MM:SS")
    parser.add_argument("--until", metavar="ISO",
                        help="window end, YYYY-mm-ddTHH:MM:SS")


def parse_window(args, tool):
    """Same semantics as gpu/gpulib.parse_window. Exit 3 on bad input."""
    if args.since or args.until:
        if not (args.since and args.until):
            err_exit(tool, "--since and --until must be given together")
        try:
            start, end = iso_epoch(args.since), iso_epoch(args.until)
        except ValueError:
            err_exit(tool, "timestamps must be YYYY-mm-ddTHH:MM:SS")
        if end <= start:
            err_exit(tool, "--until must be after --since")
        return start, end
    end = int(time.time())
    return end - args.days * 86400, end


def humanize(seconds):
    """45 -> '45s'; 150 -> '2m 30s'; 9015 -> '2h 30m'; 90000 -> '1d 1h'."""
    seconds = int(seconds)
    if seconds < 60:
        return "%ds" % seconds
    if seconds < 3600:
        return "%dm %ds" % (seconds // 60, seconds % 60)
    if seconds < 86400:
        return "%dh %dm" % (seconds // 3600, seconds % 3600 // 60)
    return "%dd %dh" % (seconds // 86400, seconds % 86400 // 3600)


def pctile(sorted_values, pct):
    """Nearest-rank percentile of a pre-sorted list; None when empty."""
    if not sorted_values:
        return None
    rank = max(1, int(math.ceil(pct / 100.0 * len(sorted_values))))
    return sorted_values[rank - 1]


def gpus_from_tres(tres):
    """GPU count from a TRES string ('gres/gpu=4', or typed variant)."""
    fallback = 0
    for item in (tres or "").split(","):
        if item.startswith("gres/gpu="):
            try:
                return int(item.split("=", 1)[1])
            except ValueError:
                return 0
        if item.startswith("gres/gpu:") and "=" in item:
            try:
                fallback = int(item.split("=", 1)[1])
            except ValueError:
                pass
    return fallback


def read_text(path, tool):
    """Read a replay file; err_exit (3) if unreadable."""
    try:
        with open(path) as fh:
            return fh.read()
    except OSError as exc:
        err_exit(tool, "cannot read replay file: %s" % exc)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m unittest discover -s slurm/tests -v`
Expected: 12 tests, all PASS.

- [ ] **Step 6: Commit**

```bash
git add slurm/slurmlib.py slurm/tests
git commit -m "Add slurmlib shared helpers for slurm reporting tools"
```

---

### Task 2: `job_failures`

**Files:**
- Create: `slurm/job_failures`, `slurm/tests/fixtures/sacct_failures.txt`
- Test: `slurm/tests/test_job_failures.py`

**Interfaces:**
- Consumes: `slurmlib` (Task 1), `loader`.
- Produces: `classify_state(state) -> str` in {`oom`,`timeout`,`node_fail`,`preempted`,`cancelled`,`failed`,`completed`,`active`,`other`}; `parse_rows(text) -> (jobs, malformed_count)` (job dict keys: `jobid,user,account,partition,state,exit_code,derived_exit,elapsed,timelimit,req_mem,alloc_tres,nodelist,job_name,cls`); `summarize(jobs, top_n) -> dict` (keys `counts,terminal,failure_rate_pct,top_exit_codes,top_failing_names,top_failing_users,incident_nodes`); `render(summary, malformed, start, end) -> str`; `main(argv=None) -> int`.

- [ ] **Step 1: Create the fixture**

`slurm/tests/fixtures/sacct_failures.txt` (`-P` rows: `JobIDRaw|User|Account|Partition|State|ExitCode|DerivedExitCode|Elapsed|Timelimit|ReqMem|AllocTRES|NodeList|JobName`):

```text
100|alice|kempner_lab|kempner_h100|COMPLETED|0:0|0:0|02:00:00|08:00:00|128G|cpu=16,gres/gpu=2|holygpu8a10101|train_a
101|alice|kempner_lab|kempner_h100|FAILED|1:0|1:0|00:10:00|08:00:00|128G|cpu=16,gres/gpu=2|holygpu8a10101|train_a
102|bob|kempner_lab|kempner_h100|FAILED|1:0|1:0|00:11:00|08:00:00|64G|cpu=8,gres/gpu=1|holygpu8a10102|train_b
103|bob|kempner_lab|kempner_h100|OUT_OF_MEMORY|0:125|0:125|01:00:00|08:00:00|32G|cpu=8,gres/gpu=1|holygpu8a10102|train_b
104|carol|kempner_lab|kempner_eng|TIMEOUT|0:0|0:1|08:00:01|08:00:00|64G|cpu=8,gres/gpu=1|holygpu8a10201|sweep_c
105|carol|kempner_lab|kempner_eng|CANCELLED by 5001|0:0|0:0|00:30:00|08:00:00|64G|cpu=8,gres/gpu=1|holygpu8a10201|sweep_c
106|dave|kempner_lab|kempner_eng|NODE_FAIL|0:0|0:0|03:00:00|08:00:00|64G|cpu=8,gres/gpu=1|holygpu8a10202|bench_d
107|dave|kempner_lab|kempner_eng|RUNNING|0:0|0:0|01:00:00|08:00:00|64G|cpu=8,gres/gpu=1|holygpu8a10202|bench_d
108|erin|kempner_lab|kempner_eng|PREEMPTED|0:0|0:0|00:20:00|08:00:00|64G|cpu=8|holygpu8a10203|cpu_e
garbage line that is not a row
```

- [ ] **Step 2: Write the failing tests**

`slurm/tests/test_job_failures.py`:

```python
"""Tests for slurm/job_failures."""
import json
import os
import subprocess
import sys
import unittest

from slurm.tests import loader

jf = loader.load_tool("job_failures")
TOOL_PATH = os.path.abspath(os.path.join(loader.HERE, "..", "job_failures"))


def run_cli(*argv):
    return subprocess.run([sys.executable, TOOL_PATH] + list(argv),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True)


class TestClassify(unittest.TestCase):
    def test_oom(self):
        self.assertEqual(jf.classify_state("OUT_OF_MEMORY"), "oom")

    def test_timeout(self):
        self.assertEqual(jf.classify_state("TIMEOUT"), "timeout")

    def test_node_fail(self):
        self.assertEqual(jf.classify_state("NODE_FAIL"), "node_fail")

    def test_preempted(self):
        self.assertEqual(jf.classify_state("PREEMPTED"), "preempted")

    def test_cancelled_with_uid(self):
        self.assertEqual(jf.classify_state("CANCELLED by 5001"), "cancelled")
        self.assertEqual(jf.classify_state("CANCELLED"), "cancelled")

    def test_failed(self):
        self.assertEqual(jf.classify_state("FAILED"), "failed")

    def test_completed(self):
        self.assertEqual(jf.classify_state("COMPLETED"), "completed")

    def test_active_states(self):
        for state in ("RUNNING", "PENDING", "REQUEUED", "SUSPENDED"):
            self.assertEqual(jf.classify_state(state), "active")

    def test_unknown_is_other(self):
        self.assertEqual(jf.classify_state("BOOT_FAIL"), "other")


class TestParseRows(unittest.TestCase):
    def test_parses_and_counts_malformed(self):
        jobs, malformed = jf.parse_rows(loader.fixture("sacct_failures.txt"))
        self.assertEqual(len(jobs), 9)
        self.assertEqual(malformed, 1)
        self.assertEqual(jobs[0]["cls"], "completed")
        self.assertEqual(jobs[3]["cls"], "oom")

    def test_empty(self):
        self.assertEqual(jf.parse_rows(""), ([], 0))


class TestSummarize(unittest.TestCase):
    def summary(self):
        jobs, _ = jf.parse_rows(loader.fixture("sacct_failures.txt"))
        return jf.summarize(jobs, top_n=3)

    def test_failure_rate(self):
        s = self.summary()
        # terminal = 8 (RUNNING excluded); non-completed terminal = 7
        self.assertEqual(s["terminal"], 8)
        self.assertAlmostEqual(s["failure_rate_pct"], 87.5)

    def test_tops(self):
        s = self.summary()
        self.assertEqual(s["top_exit_codes"][0], ("1:0", 2))
        names = dict(s["top_failing_names"])
        self.assertEqual(names["train_b"], 2)
        nodes = dict(s["incident_nodes"])
        self.assertEqual(nodes["holygpu8a10102"], 1)  # the OOM
        self.assertEqual(nodes["holygpu8a10202"], 1)  # the NODE_FAIL


class TestCli(unittest.TestCase):
    def test_report_exit_0(self):
        proc = run_cli("--from-file",
                       os.path.join(loader.FIXTURES, "sacct_failures.txt"))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("failure rate", proc.stdout)
        self.assertIn("87.5", proc.stdout)

    def test_json_output(self):
        proc = run_cli("--from-file",
                       os.path.join(loader.FIXTURES, "sacct_failures.txt"),
                       "--json")
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["summary"]["terminal"], 8)

    def test_two_selectors_exit_3(self):
        proc = run_cli("-u", "alice", "-A", "lab")
        self.assertEqual(proc.returncode, 3)
        self.assertIn("job_failures: error:", proc.stderr)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m unittest discover -s slurm/tests -v`
Expected: import-time ERROR (`FileNotFoundError` for `slurm/job_failures`); the 12 earlier tests PASS.

- [ ] **Step 4: Implement `slurm/job_failures`**

```python
#!/usr/bin/env python3
"""job_failures -- window failure post-mortem from sacct.

Classifies every terminal job in a window (per user, account, or partition)
into completed / failed / oom / timeout / cancelled / node_fail / preempted,
and ranks exit codes, failing job names, failing users, and incident nodes.

Usage:
    job_failures [-u USER | -A ACCOUNT | -p PARTITION]
                 [--days N | --since ISO --until ISO] [--top N] [--json]

Exit codes: 0 report, 3 error.
"""

import argparse
import collections
import getpass
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import slurmlib

TOOL = "job_failures"
SACCT_FIELDS = ("JobIDRaw,User,Account,Partition,State,ExitCode,"
                "DerivedExitCode,Elapsed,Timelimit,ReqMem,AllocTRES,"
                "NodeList,JobName")
FAILURE_CLASSES = ("failed", "oom", "timeout", "node_fail", "cancelled",
                   "preempted")
_KEYS = ("jobid", "user", "account", "partition", "state", "exit_code",
         "derived_exit", "elapsed", "timelimit", "req_mem", "alloc_tres",
         "nodelist", "job_name")


def classify_state(state):
    """Spec precedence; first match wins."""
    s = (state or "").upper()
    if s.startswith("OUT_OF_MEMORY"):
        return "oom"
    if s.startswith("TIMEOUT"):
        return "timeout"
    if s.startswith("NODE_FAIL"):
        return "node_fail"
    if s.startswith("PREEMPTED"):
        return "preempted"
    if s.startswith("CANCELLED"):
        return "cancelled"
    if s.startswith("FAILED"):
        return "failed"
    if s.startswith("COMPLETED"):
        return "completed"
    if s.startswith(("RUNNING", "PENDING", "REQUEUED", "SUSPENDED")):
        return "active"
    return "other"


def parse_rows(text):
    """sacct -P rows -> (job dicts, malformed count)."""
    jobs, malformed = [], 0
    for line in (text or "").splitlines():
        parts = line.split("|")
        if len(parts) < len(_KEYS) or not parts[0].strip().isdigit():
            if line.strip():
                malformed += 1
            continue
        job = dict(zip(_KEYS, parts))
        job["jobid"] = int(job["jobid"])
        job["cls"] = classify_state(job["state"])
        jobs.append(job)
    return jobs, malformed


def summarize(jobs, top_n):
    counts = collections.Counter(j["cls"] for j in jobs)
    terminal = len(jobs) - counts.get("active", 0)
    noncompleted = terminal - counts.get("completed", 0)
    rate = (100.0 * noncompleted / terminal) if terminal else 0.0
    exit_codes = collections.Counter(
        j["exit_code"] for j in jobs if j["cls"] == "failed")
    fail_jobs = [j for j in jobs if j["cls"] in FAILURE_CLASSES]
    names = collections.Counter(j["job_name"] for j in fail_jobs)
    users = collections.Counter(j["user"] for j in fail_jobs)
    nodes = collections.Counter(
        j["nodelist"] for j in jobs if j["cls"] in ("node_fail", "oom"))
    return {"counts": dict(counts), "terminal": terminal,
            "failure_rate_pct": rate,
            "top_exit_codes": exit_codes.most_common(top_n),
            "top_failing_names": names.most_common(top_n),
            "top_failing_users": users.most_common(top_n),
            "incident_nodes": nodes.most_common(top_n)}


def render(summary, malformed, start, end):
    lines = ["job_failures: %s -> %s"
             % (slurmlib.epoch_iso(start), slurmlib.epoch_iso(end)), ""]
    counts = summary["counts"]
    lines.append("  jobs: %d terminal, %d active, %d malformed rows skipped"
                 % (summary["terminal"], counts.get("active", 0), malformed))
    lines.append("  by class: " + "  ".join(
        "%s=%d" % (c, counts[c]) for c in sorted(counts) if c != "active"))
    lines.append("  failure rate: %.1f%% of terminal jobs"
                 % summary["failure_rate_pct"])

    def table(title, pairs, label):
        if not pairs:
            return
        lines.append("")
        lines.append("%s:" % title)
        for value, count in pairs:
            lines.append("  %-40s %6d %s" % (value, count, label))

    table("top exit codes (FAILED jobs)", summary["top_exit_codes"], "jobs")
    table("top failing job names", summary["top_failing_names"], "jobs")
    table("top failing users", summary["top_failing_users"], "jobs")
    table("nodes with OOM / NODE_FAIL incidents", summary["incident_nodes"],
          "incidents")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog=TOOL,
        description="window failure post-mortem from sacct "
                    "(exit 0 report, 3 error)")
    slurmlib.add_window_args(parser)
    parser.add_argument("-u", "--user")
    parser.add_argument("-A", "--account")
    parser.add_argument("-p", "--partition")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--json", action="store_true",
                        help="emit the structured result as JSON")
    parser.add_argument("--from-file", metavar="SACCT_TXT",
                        help="replay captured sacct -P output (testing)")
    args = parser.parse_args(argv)
    selectors = [s for s in (args.user, args.account, args.partition) if s]
    if len(selectors) > 1:
        slurmlib.err_exit(TOOL, "use only one of -u / -A / -p")
    start, end = slurmlib.parse_window(args, TOOL)

    if args.from_file:
        text = slurmlib.read_text(args.from_file, TOOL)
    else:
        cmd = ["sacct", "-X", "-n", "-P", "-o", SACCT_FIELDS,
               "-S", slurmlib.epoch_iso(start), "-E", slurmlib.epoch_iso(end)]
        if args.account:
            cmd += ["-A", args.account, "-a"]
        elif args.partition:
            cmd += ["-r", args.partition, "-a"]
        else:
            cmd += ["-u", args.user or getpass.getuser()]
        try:
            text = slurmlib.run(cmd, timeout=120)
        except RuntimeError as exc:
            slurmlib.err_exit(TOOL, str(exc))

    jobs, malformed = parse_rows(text)
    summary = summarize(jobs, args.top)
    if args.json:
        print(json.dumps({"start": start, "end": end, "malformed": malformed,
                          "summary": summary}, indent=2))
    else:
        print(render(summary, malformed, start, end))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Make it executable and run the tests**

Run: `chmod +x slurm/job_failures && python3 -m unittest discover -s slurm/tests -v`
Expected: 28 tests (12 + 16 new), all PASS.

- [ ] **Step 6: Commit**

```bash
git add slurm/job_failures slurm/tests/fixtures/sacct_failures.txt slurm/tests/test_job_failures.py
git commit -m "Add job_failures window post-mortem"
```

---

### Task 3: `queue_wait`

**Files:**
- Create: `slurm/queue_wait`, `slurm/tests/fixtures/sacct_queue.txt`
- Test: `slurm/tests/test_queue_wait.py`

**Interfaces:**
- Consumes: `slurmlib` (`iso_epoch`, `pctile`, `humanize`, `gpus_from_tres`, window/run/read helpers), `loader`.
- Produces: `parse_queue_rows(text) -> (rows, malformed)` (row keys `jobid,partition,qos,submit,start,state,gpus`); `waits(rows) -> (started, excluded)` where started items are `{"wait_s": int, "partition": str, "qos": str, "bucket": str}` and `excluded = {"pending": n, "skew": n}`; `bucket_for(gpus) -> "0"|"1"|"2-4"|">4"`; `distribution(values) -> {"n","p50","p90","max"}|None`; `group(started, key) -> {value: distribution}`; `render(...)`; `main(argv=None) -> int`.

- [ ] **Step 1: Create the fixture**

`slurm/tests/fixtures/sacct_queue.txt` (`JobIDRaw|Partition|QOS|Submit|Start|State|AllocTRES`):

```text
200|kempner_h100|normal|2026-07-01T10:00:00|2026-07-01T10:30:00|COMPLETED|cpu=16,gres/gpu=2
201|kempner_h100|normal|2026-07-01T10:00:00|2026-07-01T12:00:00|COMPLETED|cpu=16,gres/gpu=2
202|kempner_h100|high|2026-07-01T10:00:00|2026-07-01T10:00:30|COMPLETED|cpu=8,gres/gpu=1
203|kempner_eng|normal|2026-07-01T10:00:00|Unknown|PENDING|cpu=8,gres/gpu=1
204|kempner_eng|normal|2026-07-01T10:00:00|2026-07-01T09:59:00|COMPLETED|cpu=8,gres/gpu=1
205|kempner_eng|normal|2026-07-01T10:00:00|2026-07-01T10:05:00|COMPLETED|cpu=8
206|kempner_eng|normal|2026-07-01T10:00:00|2026-07-01T16:00:00|COMPLETED|cpu=32,gres/gpu=8
```

(Row 203 is pending, row 204 has Start < Submit — both must land in `excluded`.)

- [ ] **Step 2: Write the failing tests**

`slurm/tests/test_queue_wait.py`:

```python
"""Tests for slurm/queue_wait."""
import json
import os
import subprocess
import sys
import unittest

from slurm.tests import loader

qw = loader.load_tool("queue_wait")
TOOL_PATH = os.path.abspath(os.path.join(loader.HERE, "..", "queue_wait"))


def run_cli(*argv):
    return subprocess.run([sys.executable, TOOL_PATH] + list(argv),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True)


def started():
    rows, _ = qw.parse_queue_rows(loader.fixture("sacct_queue.txt"))
    return qw.waits(rows)


class TestWaits(unittest.TestCase):
    def test_excludes_pending_and_skew(self):
        _, excluded = started()
        self.assertEqual(excluded, {"pending": 1, "skew": 1})

    def test_wait_seconds(self):
        rows, _ = started()
        by_job = sorted(w["wait_s"] for w in rows)
        self.assertEqual(by_job, [30, 300, 1800, 7200, 21600])

    def test_buckets(self):
        self.assertEqual(qw.bucket_for(0), "0")
        self.assertEqual(qw.bucket_for(1), "1")
        self.assertEqual(qw.bucket_for(3), "2-4")
        self.assertEqual(qw.bucket_for(8), ">4")


class TestDistribution(unittest.TestCase):
    def test_nearest_rank(self):
        d = qw.distribution([30, 300, 1800, 7200, 21600])
        self.assertEqual(d["n"], 5)
        self.assertEqual(d["p50"], 1800)
        self.assertEqual(d["p90"], 21600)
        self.assertEqual(d["max"], 21600)

    def test_empty_is_none(self):
        self.assertIsNone(qw.distribution([]))


class TestGroup(unittest.TestCase):
    def test_by_partition(self):
        rows, _ = started()
        grouped = qw.group(rows, "partition")
        self.assertEqual(grouped["kempner_h100"]["n"], 3)
        self.assertEqual(grouped["kempner_eng"]["n"], 2)


class TestCli(unittest.TestCase):
    def test_report_exit_0(self):
        proc = run_cli("--from-file",
                       os.path.join(loader.FIXTURES, "sacct_queue.txt"))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("kempner_h100", proc.stdout)
        self.assertIn("excluded", proc.stdout)

    def test_json_output(self):
        proc = run_cli("--from-file",
                       os.path.join(loader.FIXTURES, "sacct_queue.txt"),
                       "--json")
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["excluded"]["pending"], 1)

    def test_bad_window_exits_3(self):
        proc = run_cli("--since", "2026-07-01T00:00:00")
        self.assertEqual(proc.returncode, 3)
        self.assertIn("queue_wait: error:", proc.stderr)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m unittest discover -s slurm/tests -v`
Expected: import-time ERROR (`FileNotFoundError` for `slurm/queue_wait`); 28 earlier tests PASS.

- [ ] **Step 4: Implement `slurm/queue_wait`**

```python
#!/usr/bin/env python3
"""queue_wait -- Submit->Start wait distributions from sacct.

Reports wait-time distributions (count, p50, p90, max; nearest-rank) grouped
by partition, QOS, and GPU-count bucket. Honesty note: sacct carries no
pending-reason history, so this does NOT separate priority wait from
resource wait.

Usage:
    queue_wait (-p PARTITION | -A ACCOUNT | -u USER)
               [--days N | --since ISO --until ISO] [--json]

Exit codes: 0 report, 3 error.
"""

import argparse
import getpass
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import slurmlib

TOOL = "queue_wait"
SACCT_FIELDS = "JobIDRaw,Partition,QOS,Submit,Start,State,AllocTRES"
_KEYS = ("jobid", "partition", "qos", "submit", "start", "state",
         "alloc_tres")


def parse_queue_rows(text):
    rows, malformed = [], 0
    for line in (text or "").splitlines():
        parts = line.split("|")
        if len(parts) < len(_KEYS) or not parts[0].strip().isdigit():
            if line.strip():
                malformed += 1
            continue
        row = dict(zip(_KEYS, parts))
        row["jobid"] = int(row["jobid"])
        row["gpus"] = slurmlib.gpus_from_tres(row.pop("alloc_tres"))
        rows.append(row)
    return rows, malformed


def bucket_for(gpus):
    if gpus <= 0:
        return "0"
    if gpus == 1:
        return "1"
    if gpus <= 4:
        return "2-4"
    return ">4"


def waits(rows):
    """-> (started list, excluded counts). Pending rows (no Start) and
    Start < Submit skew rows are excluded and counted."""
    started, excluded = [], {"pending": 0, "skew": 0}
    for row in rows:
        if row["start"] in ("", "Unknown", "None", "NONE"):
            excluded["pending"] += 1
            continue
        try:
            wait_s = (slurmlib.iso_epoch(row["start"])
                      - slurmlib.iso_epoch(row["submit"]))
        except ValueError:
            excluded["pending"] += 1
            continue
        if wait_s < 0:
            excluded["skew"] += 1
            continue
        started.append({"wait_s": wait_s, "partition": row["partition"],
                        "qos": row["qos"],
                        "bucket": bucket_for(row["gpus"])})
    return started, excluded


def distribution(values):
    if not values:
        return None
    ordered = sorted(values)
    return {"n": len(ordered), "p50": slurmlib.pctile(ordered, 50),
            "p90": slurmlib.pctile(ordered, 90), "max": ordered[-1]}


def group(started, key):
    buckets = {}
    for item in started:
        buckets.setdefault(item[key], []).append(item["wait_s"])
    return dict((name, distribution(vals)) for name, vals in buckets.items())


def render(started, excluded, malformed, start, end):
    lines = ["queue_wait: %s -> %s"
             % (slurmlib.epoch_iso(start), slurmlib.epoch_iso(end)), ""]
    lines.append("  %d started jobs; excluded: %d pending, %d clock-skew; "
                 "%d malformed rows"
                 % (len(started), excluded["pending"], excluded["skew"],
                    malformed))
    lines.append("  (sacct cannot separate priority wait from resource wait)")

    def table(title, key):
        grouped = group(started, key)
        if not grouped:
            return
        lines.append("")
        lines.append("by %s:" % title)
        lines.append("  %-24s %6s %10s %10s %10s"
                     % ("GROUP", "N", "P50", "P90", "MAX"))
        for name in sorted(grouped):
            d = grouped[name]
            lines.append("  %-24s %6d %10s %10s %10s"
                         % (name, d["n"], slurmlib.humanize(d["p50"]),
                            slurmlib.humanize(d["p90"]),
                            slurmlib.humanize(d["max"])))

    table("partition", "partition")
    table("QOS", "qos")
    table("GPU count", "bucket")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog=TOOL,
        description="Submit->Start wait distributions from sacct "
                    "(exit 0 report, 3 error)")
    slurmlib.add_window_args(parser)
    parser.add_argument("-u", "--user")
    parser.add_argument("-A", "--account")
    parser.add_argument("-p", "--partition")
    parser.add_argument("--json", action="store_true",
                        help="emit the structured result as JSON")
    parser.add_argument("--from-file", metavar="SACCT_TXT",
                        help="replay captured sacct -P output (testing)")
    args = parser.parse_args(argv)
    selectors = [s for s in (args.user, args.account, args.partition) if s]
    if len(selectors) > 1:
        slurmlib.err_exit(TOOL, "use only one of -u / -A / -p")
    start, end = slurmlib.parse_window(args, TOOL)

    if args.from_file:
        text = slurmlib.read_text(args.from_file, TOOL)
    else:
        cmd = ["sacct", "-X", "-n", "-P", "-o", SACCT_FIELDS,
               "-S", slurmlib.epoch_iso(start), "-E", slurmlib.epoch_iso(end)]
        if args.account:
            cmd += ["-A", args.account, "-a"]
        elif args.partition:
            cmd += ["-r", args.partition, "-a"]
        else:
            cmd += ["-u", args.user or getpass.getuser()]
        try:
            text = slurmlib.run(cmd, timeout=120)
        except RuntimeError as exc:
            slurmlib.err_exit(TOOL, str(exc))

    rows, malformed = parse_queue_rows(text)
    started, excluded = waits(rows)
    if args.json:
        print(json.dumps({"start": start, "end": end,
                          "excluded": excluded, "malformed": malformed,
                          "by_partition": group(started, "partition"),
                          "by_qos": group(started, "qos"),
                          "by_gpu_bucket": group(started, "bucket")},
                         indent=2))
    else:
        print(render(started, excluded, malformed, start, end))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Make it executable and run the tests**

Run: `chmod +x slurm/queue_wait && python3 -m unittest discover -s slurm/tests -v`
Expected: 37 tests (28 + 9 new), all PASS.

- [ ] **Step 6: Commit**

```bash
git add slurm/queue_wait slurm/tests/fixtures/sacct_queue.txt slurm/tests/test_queue_wait.py
git commit -m "Add queue_wait distribution report"
```

---

### Task 4: `fairshare_report`

**Files:**
- Create: `slurm/fairshare_report`, `slurm/tests/fixtures/sshare.txt`
- Test: `slurm/tests/test_fairshare_report.py`

**Interfaces:**
- Consumes: `slurmlib`, `loader`.
- Produces: `parse_sshare(text) -> (accounts, skipped)` where accounts are `{"account,raw_shares,norm_shares,raw_usage,effectv_usage,fairshare"}` (floats or None) and `skipped = {"root": n, "user_rows": n, "malformed": n}`; `rank(accounts, top_n) -> {"usable","zero_usage","over","under"}` with `ratio` added per account; `render(...)`; `main(argv=None) -> int`.

- [ ] **Step 1: Create the fixture**

`slurm/tests/fixtures/sshare.txt` (`sshare -a -P` shape: header, root row, indented account rows, user rows with User set):

```text
Account|User|RawShares|NormShares|RawUsage|EffectvUsage|FairShare
root|||1.000000|153274818272|1.000000|
 alpha_lab||250|0.000409|22|0.000000|
 beta_lab||912|0.001492|8175000000|0.010000|
 gamma_lab||500|0.000818|125000000000|0.815000|
  gamma_lab|gina|1|0.000100|125000000000|0.815000|0.150000
 delta_lab||500|0.000818|0|0.000000|
not|a|row
```

- [ ] **Step 2: Write the failing tests**

`slurm/tests/test_fairshare_report.py`:

```python
"""Tests for slurm/fairshare_report."""
import json
import os
import subprocess
import sys
import unittest

from slurm.tests import loader

fr = loader.load_tool("fairshare_report")
TOOL_PATH = os.path.abspath(os.path.join(loader.HERE, "..",
                                         "fairshare_report"))


def run_cli(*argv):
    return subprocess.run([sys.executable, TOOL_PATH] + list(argv),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True)


class TestParseSshare(unittest.TestCase):
    def test_accounts_and_skips(self):
        accounts, skipped = fr.parse_sshare(loader.fixture("sshare.txt"))
        self.assertEqual([a["account"] for a in accounts],
                         ["alpha_lab", "beta_lab", "gamma_lab", "delta_lab"])
        self.assertEqual(skipped["root"], 1)
        self.assertEqual(skipped["user_rows"], 1)
        self.assertEqual(skipped["malformed"], 1)

    def test_floats(self):
        accounts, _ = fr.parse_sshare(loader.fixture("sshare.txt"))
        beta = accounts[1]
        self.assertAlmostEqual(beta["norm_shares"], 0.001492)
        self.assertAlmostEqual(beta["effectv_usage"], 0.01)

    def test_empty(self):
        accounts, skipped = fr.parse_sshare("")
        self.assertEqual(accounts, [])
        self.assertEqual(skipped,
                         {"root": 0, "user_rows": 0, "malformed": 0})


class TestRank(unittest.TestCase):
    def ranked(self):
        accounts, _ = fr.parse_sshare(loader.fixture("sshare.txt"))
        return fr.rank(accounts, top_n=2)

    def test_over_served_order(self):
        over = self.ranked()["over"]
        # gamma: 0.815/0.000818 ~ 996; beta: 0.01/0.001492 ~ 6.7
        self.assertEqual([a["account"] for a in over],
                         ["gamma_lab", "beta_lab"])
        self.assertGreater(over[0]["ratio"], 900)

    def test_under_and_zero(self):
        r = self.ranked()
        self.assertEqual([a["account"] for a in r["under"]],
                         ["beta_lab", "gamma_lab"])
        self.assertEqual(r["zero_usage"], 2)  # alpha (0.0) and delta (0)

    def test_counts(self):
        r = self.ranked()
        self.assertEqual(r["usable"], 4)


class TestCli(unittest.TestCase):
    def test_report_exit_0(self):
        proc = run_cli("--from-file",
                       os.path.join(loader.FIXTURES, "sshare.txt"))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("gamma_lab", proc.stdout)
        self.assertIn("point-in-time", proc.stdout)

    def test_json_output(self):
        proc = run_cli("--from-file",
                       os.path.join(loader.FIXTURES, "sshare.txt"), "--json")
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["skipped"]["user_rows"], 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m unittest discover -s slurm/tests -v`
Expected: import-time ERROR (`FileNotFoundError` for `slurm/fairshare_report`); 37 earlier tests PASS.

- [ ] **Step 4: Implement `slurm/fairshare_report`**

```python
#!/usr/bin/env python3
"""fairshare_report -- share vs effective usage per account, from sshare.

Point-in-time snapshot (sshare keeps no history): for every top-level
account, compares NormShares against EffectvUsage (ratio > 1 = over-served)
and ranks the most over- and under-served accounts. --json output makes
periodic archiving trivial if trending is wanted later.

Usage:
    fairshare_report [-A ACCOUNT] [--top N] [--json]

Exit codes: 0 report, 3 error.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import slurmlib

TOOL = "fairshare_report"
SSHARE_FIELDS = ("Account,User,RawShares,NormShares,RawUsage,"
                 "EffectvUsage,FairShare")


def _to_float(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def parse_sshare(text):
    """sshare -a -P rows -> (top-level account dicts, skipped counts)."""
    accounts = []
    skipped = {"root": 0, "user_rows": 0, "malformed": 0}
    for line in (text or "").splitlines():
        parts = line.split("|")
        if len(parts) < 7:
            if line.strip():
                skipped["malformed"] += 1
            continue
        account, user = parts[0].strip(), parts[1].strip()
        if account == "Account":  # header
            continue
        if account == "root":
            skipped["root"] += 1
            continue
        if user:
            skipped["user_rows"] += 1
            continue
        accounts.append({"account": account,
                         "raw_shares": _to_float(parts[2]),
                         "norm_shares": _to_float(parts[3]),
                         "raw_usage": _to_float(parts[4]),
                         "effectv_usage": _to_float(parts[5]),
                         "fairshare": _to_float(parts[6])})
    return accounts, skipped


def rank(accounts, top_n):
    usable = [a for a in accounts
              if a["norm_shares"] and a["norm_shares"] > 0]
    with_usage = []
    zero = 0
    for a in usable:
        usage = a["effectv_usage"]
        if usage is None or usage <= 0:
            a["ratio"] = 0.0
            zero += 1
        else:
            a["ratio"] = usage / a["norm_shares"]
            with_usage.append(a)
    return {"usable": len(usable), "zero_usage": zero,
            "over": sorted(with_usage, key=lambda a: a["ratio"],
                           reverse=True)[:top_n],
            "under": sorted(with_usage, key=lambda a: a["ratio"])[:top_n]}


def _table(lines, title, rows):
    if not rows:
        return
    lines.append("")
    lines.append("%s:" % title)
    lines.append("  %-24s %12s %14s %10s %10s"
                 % ("ACCOUNT", "NORM_SHARES", "EFFECTV_USAGE", "RATIO",
                    "FAIRSHARE"))
    for a in rows:
        lines.append("  %-24s %12.6f %14.6f %10.2f %10s"
                     % (a["account"], a["norm_shares"], a["effectv_usage"],
                        a["ratio"],
                        "%.4f" % a["fairshare"]
                        if a["fairshare"] is not None else "-"))


def render(accounts, skipped, ranked):
    lines = ["fairshare_report: %d top-level accounts (%d with zero usage; "
             "skipped %d user rows, %d malformed)"
             % (ranked["usable"], ranked["zero_usage"],
                skipped["user_rows"], skipped["malformed"])]
    _table(lines, "most over-served (usage >> share)", ranked["over"])
    _table(lines, "most under-served (usage << share)", ranked["under"])
    lines.append("")
    lines.append("note: point-in-time snapshot; sshare keeps no history")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog=TOOL,
        description="share vs effective usage per account from sshare "
                    "(exit 0 report, 3 error)")
    parser.add_argument("-A", "--account",
                        help="narrow to one account subtree")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--json", action="store_true",
                        help="emit the structured result as JSON")
    parser.add_argument("--from-file", metavar="SSHARE_TXT",
                        help="replay captured sshare -P output (testing)")
    args = parser.parse_args(argv)

    if args.from_file:
        text = slurmlib.read_text(args.from_file, TOOL)
    else:
        cmd = ["sshare", "-a", "-P", "-o", SSHARE_FIELDS]
        if args.account:
            cmd += ["-A", args.account]
        try:
            text = slurmlib.run(cmd, timeout=120)
        except RuntimeError as exc:
            slurmlib.err_exit(TOOL, str(exc))

    accounts, skipped = parse_sshare(text)
    ranked = rank(accounts, args.top)
    if args.json:
        print(json.dumps({"skipped": skipped, "ranked": ranked}, indent=2))
    else:
        print(render(accounts, skipped, ranked))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Make it executable and run the tests**

Run: `chmod +x slurm/fairshare_report && python3 -m unittest discover -s slurm/tests -v`
Expected: 45 tests (37 + 8 new), all PASS.

- [ ] **Step 6: Commit**

```bash
git add slurm/fairshare_report slurm/tests/fixtures/sshare.txt slurm/tests/test_fairshare_report.py
git commit -m "Add fairshare_report snapshot ranking"
```

---

### Task 5: `frag_report`

**Files:**
- Create: `slurm/frag_report`, `slurm/tests/fixtures/scontrol_nodes.txt`
- Test: `slurm/tests/test_frag_report.py`

**Interfaces:**
- Consumes: `slurmlib` (`gpus_from_tres`, `err_exit`, `run`, `read_text`), `loader`.
- Produces: `parse_kv(line) -> dict` (space-split `key=value` tokens); `parse_nodes(text) -> (nodes, unavailable, malformed)` (node keys `name,partitions,state,cpu_free,mem_free_mib,gpu_tot,gpu_free`); `fragmentation(nodes, cpus_per_gpu, mem_per_gpu_gib) -> {partition: {"nodes", "gpu_dist" {0,1,2,3,"4+"}, "fit" {1,2,4}}}`; `render(...)`; `main(argv=None) -> int`.

- [ ] **Step 1: Create the fixture**

`slurm/tests/fixtures/scontrol_nodes.txt` (`scontrol show node -o` one-line records, trimmed to the fields the parser reads):

```text
NodeName=nodea CPUAlloc=8 CPUTot=64 State=MIXED Partitions=kempner_eng,kempner_requeue RealMemory=1547361 AllocMem=153600 CfgTRES=cpu=64,mem=1547361M,gres/gpu=4 AllocTRES=cpu=8,mem=150G,gres/gpu=1
NodeName=nodeb CPUAlloc=64 CPUTot=64 State=ALLOCATED Partitions=kempner_eng RealMemory=1547361 AllocMem=1400000 CfgTRES=cpu=64,mem=1547361M,gres/gpu=4 AllocTRES=cpu=64,mem=1400000M,gres/gpu=4
NodeName=nodec CPUAlloc=0 CPUTot=64 State=IDLE+DRAIN Partitions=kempner_eng RealMemory=1547361 AllocMem=0 CfgTRES=cpu=64,mem=1547361M,gres/gpu=4 AllocTRES=
NodeName=noded CPUAlloc=0 CPUTot=64 State=IDLE Partitions=kempner_eng RealMemory=1547361 AllocMem=0 CfgTRES=cpu=64,mem=1547361M,gres/gpu=4 AllocTRES=
```

- [ ] **Step 2: Write the failing tests**

`slurm/tests/test_frag_report.py`:

```python
"""Tests for slurm/frag_report."""
import json
import os
import subprocess
import sys
import unittest

from slurm.tests import loader

fg = loader.load_tool("frag_report")
TOOL_PATH = os.path.abspath(os.path.join(loader.HERE, "..", "frag_report"))


def run_cli(*argv):
    return subprocess.run([sys.executable, TOOL_PATH] + list(argv),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True)


def parsed():
    return fg.parse_nodes(loader.fixture("scontrol_nodes.txt"))


class TestParseKv(unittest.TestCase):
    def test_tokens(self):
        kv = fg.parse_kv("NodeName=x CPUTot=64 CfgTRES=cpu=64,gres/gpu=4")
        self.assertEqual(kv["NodeName"], "x")
        self.assertEqual(kv["CfgTRES"], "cpu=64,gres/gpu=4")


class TestParseNodes(unittest.TestCase):
    def test_counts(self):
        nodes, unavailable, malformed = parsed()
        self.assertEqual([n["name"] for n in nodes],
                         ["nodea", "nodeb", "noded"])
        self.assertEqual([u["name"] for u in unavailable], ["nodec"])
        self.assertEqual(malformed, 0)

    def test_free_math(self):
        nodes, _, _ = parsed()
        nodea = nodes[0]
        self.assertEqual(nodea["cpu_free"], 56)
        self.assertEqual(nodea["mem_free_mib"], 1547361 - 153600)
        self.assertEqual(nodea["gpu_free"], 3)
        nodeb = nodes[1]
        self.assertEqual(nodeb["gpu_free"], 0)

    def test_empty_alloctres(self):
        nodes, _, _ = parsed()
        noded = nodes[2]
        self.assertEqual(noded["gpu_free"], 4)


class TestFragmentation(unittest.TestCase):
    def frag(self):
        nodes, _, _ = parsed()
        return fg.fragmentation(nodes, cpus_per_gpu=8, mem_per_gpu_gib=64)

    def test_gpu_distribution(self):
        eng = self.frag()["kempner_eng"]
        self.assertEqual(eng["gpu_dist"][0], 1)   # nodeb
        self.assertEqual(eng["gpu_dist"][3], 1)   # nodea
        self.assertEqual(eng["gpu_dist"]["4+"], 1)  # noded

    def test_fit_counts(self):
        eng = self.frag()["kempner_eng"]
        # nodea: min(3//1, 56//8, free_mem//65536)=3; noded: min(4,8,23)=4; nodeb: 0
        self.assertEqual(eng["fit"][1], 7)
        # 2-GPU shape: nodea min(1,3,10)=1; noded min(2,4,11)=2
        self.assertEqual(eng["fit"][2], 3)
        # 4-GPU shape: nodea 0; noded min(1,2,5)=1
        self.assertEqual(eng["fit"][4], 1)

    def test_multi_partition(self):
        frag = self.frag()
        self.assertEqual(frag["kempner_requeue"]["nodes"], 1)


class TestCli(unittest.TestCase):
    def test_report_exit_0(self):
        proc = run_cli("--from-file",
                       os.path.join(loader.FIXTURES, "scontrol_nodes.txt"))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("kempner_eng", proc.stdout)
        self.assertIn("unavailable", proc.stdout)

    def test_json_output(self):
        proc = run_cli("--from-file",
                       os.path.join(loader.FIXTURES, "scontrol_nodes.txt"),
                       "--json")
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["partitions"]["kempner_eng"]["fit"]["1"], 7)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m unittest discover -s slurm/tests -v`
Expected: import-time ERROR (`FileNotFoundError` for `slurm/frag_report`); 45 earlier tests PASS.

- [ ] **Step 4: Implement `slurm/frag_report`**

```python
#!/usr/bin/env python3
"""frag_report -- free CPU/GPU/memory shards per node, and how many N-GPU
jobs could start right now, per partition.

Reads one `scontrol show node -o` pass. Nodes in DOWN/DRAIN/MAINT/
NOT_RESPONDING states are excluded from capacity math and listed. The
schedulability counts use a configurable job shape (default 8 CPUs and
64 GiB per GPU).

Usage:
    frag_report [-p PARTITION] [--cpus-per-gpu N] [--mem-per-gpu GIB]
                [--json]

Exit codes: 0 report, 3 error.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import slurmlib

TOOL = "frag_report"
BAD_STATES = ("DOWN", "DRAIN", "MAINT", "NOT_RESPONDING")
SHAPES = (1, 2, 4)


def parse_kv(line):
    """Space-split key=value tokens -> dict (values keep their commas)."""
    kv = {}
    for token in line.split():
        if "=" in token:
            key, value = token.split("=", 1)
            kv[key] = value
    return kv


def _to_int(text):
    try:
        return int(text)
    except (TypeError, ValueError):
        return 0


def parse_nodes(text):
    """-> (usable node dicts, unavailable [{name,state}], malformed)."""
    nodes, unavailable, malformed = [], [], 0
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        kv = parse_kv(line)
        name = kv.get("NodeName")
        if not name:
            malformed += 1
            continue
        state = kv.get("State", "")
        if any(bad in state.upper() for bad in BAD_STATES):
            unavailable.append({"name": name, "state": state})
            continue
        gpu_tot = slurmlib.gpus_from_tres(kv.get("CfgTRES", ""))
        gpu_alloc = slurmlib.gpus_from_tres(kv.get("AllocTRES", ""))
        nodes.append({
            "name": name,
            "partitions": [p for p in kv.get("Partitions", "").split(",")
                           if p],
            "state": state,
            "cpu_free": _to_int(kv.get("CPUTot")) - _to_int(kv.get("CPUAlloc")),
            "mem_free_mib": _to_int(kv.get("RealMemory"))
                            - _to_int(kv.get("AllocMem")),
            "gpu_tot": gpu_tot,
            "gpu_free": max(0, gpu_tot - gpu_alloc),
        })
    return nodes, unavailable, malformed


def fragmentation(nodes, cpus_per_gpu, mem_per_gpu_gib):
    """Per-partition free-GPU distribution and N-GPU-job fit counts."""
    mem_per_gpu_mib = mem_per_gpu_gib * 1024
    per_part = {}
    for node in nodes:
        for part in (node["partitions"] or ["(none)"]):
            stats = per_part.setdefault(
                part, {"nodes": 0,
                       "gpu_dist": {0: 0, 1: 0, 2: 0, 3: 0, "4+": 0},
                       "fit": {1: 0, 2: 0, 4: 0}})
            stats["nodes"] += 1
            free = node["gpu_free"]
            stats["gpu_dist"][free if free < 4 else "4+"] += 1
            for shape in SHAPES:
                fits = min(free // shape,
                           node["cpu_free"] // (shape * cpus_per_gpu),
                           node["mem_free_mib"]
                           // (shape * mem_per_gpu_mib))
                stats["fit"][shape] += max(0, fits)
    return per_part


def render(per_part, unavailable, malformed, cpus_per_gpu, mem_per_gpu_gib):
    lines = ["frag_report: job shape = %d CPUs + %d GiB per GPU"
             % (cpus_per_gpu, mem_per_gpu_gib), ""]
    lines.append("%-24s %6s %-22s %7s %7s %7s"
                 % ("PARTITION", "NODES", "FREE_GPUS(0/1/2/3/4+)",
                    "FIT_1G", "FIT_2G", "FIT_4G"))
    for part in sorted(per_part):
        s = per_part[part]
        dist = "/".join(str(s["gpu_dist"][k]) for k in (0, 1, 2, 3, "4+"))
        lines.append("%-24s %6d %-22s %7d %7d %7d"
                     % (part, s["nodes"], dist,
                        s["fit"][1], s["fit"][2], s["fit"][4]))
    lines.append("")
    lines.append("%d node(s) unavailable (down/drain/maint), excluded; "
                 "%d malformed lines" % (len(unavailable), malformed))
    for u in unavailable[:10]:
        lines.append("  %s (%s)" % (u["name"], u["state"]))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog=TOOL,
        description="free-shard fragmentation and N-GPU-job fit counts "
                    "per partition (exit 0 report, 3 error)")
    parser.add_argument("-p", "--partition",
                        help="restrict to nodes in this partition")
    parser.add_argument("--cpus-per-gpu", type=int, default=8,
                        dest="cpus_per_gpu")
    parser.add_argument("--mem-per-gpu", type=int, default=64,
                        dest="mem_per_gpu", metavar="GIB")
    parser.add_argument("--json", action="store_true",
                        help="emit the structured result as JSON")
    parser.add_argument("--from-file", metavar="SCONTROL_TXT",
                        help="replay captured scontrol show node -o output")
    args = parser.parse_args(argv)

    if args.from_file:
        text = slurmlib.read_text(args.from_file, TOOL)
    else:
        try:
            text = slurmlib.run(["scontrol", "show", "node", "-o"],
                                timeout=60)
        except RuntimeError as exc:
            slurmlib.err_exit(TOOL, str(exc))

    nodes, unavailable, malformed = parse_nodes(text)
    if args.partition:
        nodes = [n for n in nodes if args.partition in n["partitions"]]
        unavailable = [u for u in unavailable]  # kept; shown regardless
    per_part = fragmentation(nodes, args.cpus_per_gpu, args.mem_per_gpu)
    if args.partition:
        per_part = dict((k, v) for k, v in per_part.items()
                        if k == args.partition)
    if args.json:
        payload = {"partitions": dict(
            (part, {"nodes": s["nodes"],
                    "gpu_dist": dict((str(k), v)
                                     for k, v in s["gpu_dist"].items()),
                    "fit": dict((str(k), v) for k, v in s["fit"].items())})
            for part, s in per_part.items()),
            "unavailable": unavailable, "malformed": malformed}
        print(json.dumps(payload, indent=2))
    else:
        print(render(per_part, unavailable, malformed, args.cpus_per_gpu,
                     args.mem_per_gpu))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Make it executable and run the tests**

Run: `chmod +x slurm/frag_report && python3 -m unittest discover -s slurm/tests -v`
Expected: 54 tests (45 + 9 new), all PASS.

- [ ] **Step 6: Commit**

```bash
git add slurm/frag_report slurm/tests/fixtures/scontrol_nodes.txt slurm/tests/test_frag_report.py
git commit -m "Add frag_report fragmentation analysis"
```

---

### Task 6: `gpu_idle_now`

**Files:**
- Create: `slurm/gpu_idle_now`
- Test: `slurm/tests/test_gpu_idle_now.py`

**Interfaces:**
- Consumes: `slurmlib` (`err_exit`), `gpu/gpulib.py` (`load_prom_server`, `query`, `series_uuid`, `load_bundle`), `loader`.
- Produces: `group_jobs(alloc_result, smact_result, threshold_pct) -> [job dict]` (keys `jobid,gpus,idle,unmatched,mean_smact,fully_idle,partially_idle,nodes`; sorted fully-idle first); `attach_squeue(jobs, text)` (`%A|%u|%a|%P|%L` rows → user/account/partition/time_left); `render(...)`; `main(argv=None) -> int`. Replay bundle roles: `"alloc"`, `"smact"` (instant results), `"squeue"` (text).

- [ ] **Step 1: Write the failing tests**

`slurm/tests/test_gpu_idle_now.py`:

```python
"""Tests for slurm/gpu_idle_now."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

from slurm.tests import loader

gi = loader.load_tool("gpu_idle_now")
TOOL_PATH = os.path.abspath(os.path.join(loader.HERE, "..", "gpu_idle_now"))

ALLOC = [
    {"metric": {"uuid": "GPU-a1", "host": "nodea:9445"},
     "value": [0, "301"]},
    {"metric": {"uuid": "GPU-a2", "host": "nodea:9445"},
     "value": [0, "301"]},
    {"metric": {"uuid": "GPU-b1", "host": "nodeb:9445"},
     "value": [0, "302"]},
    {"metric": {"uuid": "GPU-b2", "host": "nodeb:9445"},
     "value": [0, "302"]},
    {"metric": {"uuid": "GPU-c1", "host": "nodec:9445"},
     "value": [0, "303"]},
]
SMACT = [
    {"metric": {"UUID": "GPU-a1"}, "value": [0, "0.02"]},
    {"metric": {"UUID": "GPU-a2"}, "value": [0, "0.03"]},
    {"metric": {"UUID": "GPU-b1"}, "value": [0, "0.05"]},
    {"metric": {"UUID": "GPU-b2"}, "value": [0, "0.90"]},
]
SQUEUE = "301|alice|kempner_lab|kempner_eng|4:00:00\n" \
         "302|bob|kempner_lab|kempner_eng|1:00:00\n"


def run_cli(*argv):
    return subprocess.run([sys.executable, TOOL_PATH] + list(argv),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True)


class TestGroupJobs(unittest.TestCase):
    def jobs(self):
        return dict((j["jobid"], j)
                    for j in gi.group_jobs(ALLOC, SMACT, 10.0))

    def test_fully_idle(self):
        job = self.jobs()[301]
        self.assertTrue(job["fully_idle"])
        self.assertEqual(job["gpus"], 2)
        self.assertEqual(job["idle"], 2)
        self.assertAlmostEqual(job["mean_smact"], 2.5)

    def test_partially_idle(self):
        job = self.jobs()[302]
        self.assertFalse(job["fully_idle"])
        self.assertTrue(job["partially_idle"])
        self.assertEqual(job["idle"], 1)

    def test_unmatched_only_not_fully_idle(self):
        job = self.jobs()[303]
        self.assertFalse(job["fully_idle"])
        self.assertEqual(job["unmatched"], 1)
        self.assertIsNone(job["mean_smact"])

    def test_sorted_fully_idle_first(self):
        ordered = gi.group_jobs(ALLOC, SMACT, 10.0)
        self.assertEqual(ordered[0]["jobid"], 301)


class TestAttachSqueue(unittest.TestCase):
    def test_enrichment(self):
        jobs = gi.group_jobs(ALLOC, SMACT, 10.0)
        gi.attach_squeue(jobs, SQUEUE)
        by_id = dict((j["jobid"], j) for j in jobs)
        self.assertEqual(by_id[301]["user"], "alice")
        self.assertEqual(by_id[302]["time_left"], "1:00:00")
        self.assertNotIn("user", by_id[303])


class TestCli(unittest.TestCase):
    def bundle_path(self, alloc, smact):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                          delete=False)
        self.addCleanup(os.unlink, tmp.name)
        json.dump({"alloc": alloc, "smact": smact, "squeue": SQUEUE}, tmp)
        tmp.close()
        return tmp.name

    def test_fully_idle_exits_1(self):
        proc = run_cli("--from-json", self.bundle_path(ALLOC, SMACT))
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertIn("alice", proc.stdout)
        self.assertIn("fully idle", proc.stdout)

    def test_clean_exits_0(self):
        busy = [{"metric": {"UUID": "GPU-a1"}, "value": [0, "0.9"]},
                {"metric": {"UUID": "GPU-a2"}, "value": [0, "0.8"]}]
        proc = run_cli("--from-json",
                       self.bundle_path(ALLOC[:2], busy))
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_missing_bundle_exits_3(self):
        proc = run_cli("--from-json", "/nonexistent.json")
        self.assertEqual(proc.returncode, 3)
        self.assertIn("gpu_idle_now: error:", proc.stderr)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest discover -s slurm/tests -v`
Expected: import-time ERROR (`FileNotFoundError` for `slurm/gpu_idle_now`); 54 earlier tests PASS.

- [ ] **Step 3: Implement `slurm/gpu_idle_now`**

```python
#!/usr/bin/env python3
"""gpu_idle_now -- running jobs whose GPUs are idle right now.

Two Prometheus queries (via gpu/gpulib): which GPU belongs to which running
job (nvidia_gpu_jobId > 0) and each GPU's SM activity averaged over a recent
window. A job whose every matched GPU is below the threshold is FULLY IDLE
-- the live gate behind gpu/fleet_util's retrospective numbers. GPUs with no
SM data are counted as unmatched, never guessed.

Runs wherever jobstats is installed (needs the jobstats config + requests).

Usage:
    gpu_idle_now [--threshold PCT] [--window MIN] [-p PARTITION] [--json]

Exit codes: 0 no fully-idle jobs, 1 fully-idle jobs found, 3 error.
"""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, _HERE)
import slurmlib

sys.path.insert(0, os.path.join(_HERE, "..", "gpu"))
try:
    import gpulib
except ImportError:
    slurmlib.err_exit("gpu_idle_now",
                      "cannot import gpu/gpulib.py (expected in ../gpu "
                      "relative to this tool)")

TOOL = "gpu_idle_now"


def group_jobs(alloc_result, smact_result, threshold_pct):
    """Join allocation and SM-activity instant results per job."""
    smact = {}
    for series in smact_result:
        uuid = gpulib.series_uuid(series.get("metric", {}))
        try:
            smact[uuid] = float(series["value"][1])
        except (TypeError, ValueError, KeyError, IndexError):
            continue
    jobs = {}
    for series in alloc_result:
        labels = series.get("metric", {})
        uuid = gpulib.series_uuid(labels)
        try:
            jobid = int(float(series["value"][1]))
        except (TypeError, ValueError, KeyError, IndexError):
            continue
        if jobid <= 0:
            continue
        node = (labels.get("host") or labels.get("Hostname")
                or "?").split(":")[0].split(".")[0]
        job = jobs.setdefault(jobid, {"jobid": jobid, "gpus": 0, "idle": 0,
                                      "unmatched": 0, "sm_values": [],
                                      "nodes": set()})
        job["gpus"] += 1
        job["nodes"].add(node)
        sm = smact.get(uuid)
        if sm is None:
            job["unmatched"] += 1
        else:
            job["sm_values"].append(sm * 100.0)
            if sm * 100.0 < threshold_pct:
                job["idle"] += 1
    out = []
    for job in jobs.values():
        matched = job["gpus"] - job["unmatched"]
        values = job.pop("sm_values")
        job["mean_smact"] = (sum(values) / len(values)) if values else None
        job["fully_idle"] = matched >= 1 and job["idle"] == matched
        job["partially_idle"] = job["idle"] > 0 and not job["fully_idle"]
        job["nodes"] = sorted(job["nodes"])
        out.append(job)
    out.sort(key=lambda j: (not j["fully_idle"], not j["partially_idle"],
                            -j["idle"], j["jobid"]))
    return out


def attach_squeue(jobs, text):
    """squeue '%A|%u|%a|%P|%L' rows -> user/account/partition/time_left."""
    info = {}
    for line in (text or "").splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 5 and parts[0].isdigit():
            info[int(parts[0])] = {"user": parts[1], "account": parts[2],
                                   "partition": parts[3],
                                   "time_left": parts[4]}
    for job in jobs:
        job.update(info.get(job["jobid"], {}))
    return jobs


def render(jobs, threshold, window_min, partition):
    flagged = [j for j in jobs if j["fully_idle"] or j["partially_idle"]]
    fully = sum(1 for j in jobs if j["fully_idle"])
    partial = sum(1 for j in jobs if j["partially_idle"])
    lines = ["gpu_idle_now: %d running GPU job(s); %d fully idle, "
             "%d partially idle (SM_ACT < %g%% averaged over %g min)"
             % (len(jobs), fully, partial, threshold, window_min)]
    if partition:
        lines[0] += "  [partition filter: %s]" % partition
    if flagged:
        lines.append("")
        row = "%-10s %-10s %-14s %-18s %5s %5s %5s %8s %10s"
        lines.append(row % ("JOBID", "USER", "ACCOUNT", "PARTITION", "GPUS",
                            "IDLE", "UNMAT", "SM_ACT%", "TIME_LEFT"))
        for j in flagged:
            lines.append(row % (
                j["jobid"], j.get("user", "-"), j.get("account", "-"),
                j.get("partition", "-"), j["gpus"], j["idle"],
                j["unmatched"],
                "%.1f" % j["mean_smact"] if j["mean_smact"] is not None
                else "-",
                j.get("time_left", "-")))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog=TOOL,
        description="running jobs whose GPUs are idle right now "
                    "(exit 0 clean, 1 fully-idle jobs found, 3 error)")
    parser.add_argument("--threshold", type=float, default=10.0,
                        help="idle when SM_ACT%% below this (default 10)")
    parser.add_argument("--window", type=float, default=30.0,
                        help="averaging window in minutes (default 30)")
    parser.add_argument("-p", "--partition",
                        help="show only jobs in this partition")
    parser.add_argument("--json", action="store_true",
                        help="emit the structured result as JSON")
    parser.add_argument("--from-json", metavar="FILE",
                        help="replay a captured query bundle (testing)")
    args = parser.parse_args(argv)

    if args.from_json:
        bundle = gpulib.load_bundle(args.from_json, TOOL)
        alloc = bundle.get("alloc", [])
        smact = bundle.get("smact", [])
        squeue_text = bundle.get("squeue", "")
    else:
        prom = gpulib.load_prom_server(TOOL)
        try:
            alloc = gpulib.query(prom, "nvidia_gpu_jobId > 0")
            smact = gpulib.query(
                prom, "avg_over_time(DCGM_FI_PROF_SM_ACTIVE[%ds:])"
                % int(args.window * 60))
        except RuntimeError as exc:
            gpulib.err_exit(TOOL, str(exc))
        squeue_text = ""

    jobs = group_jobs(alloc, smact, args.threshold)
    flagged_ids = [j["jobid"] for j in jobs
                   if j["fully_idle"] or j["partially_idle"]]
    if not args.from_json and flagged_ids:
        try:
            squeue_text = slurmlib.run(
                ["squeue", "-j", ",".join(str(i) for i in flagged_ids),
                 "-h", "-o", "%A|%u|%a|%P|%L"], timeout=60)
        except RuntimeError as exc:
            sys.stderr.write("warning: squeue enrichment unavailable: "
                             "%s\n" % exc)
    attach_squeue(jobs, squeue_text)

    if args.partition:
        kept = [j for j in jobs
                if j.get("partition") in (args.partition, None)]
        dropped = len(jobs) - len(kept)
        if dropped:
            sys.stderr.write("note: %d job(s) in other partitions hidden\n"
                             % dropped)
        jobs = kept

    if args.json:
        print(json.dumps({"threshold_pct": args.threshold,
                          "window_min": args.window, "jobs": jobs},
                         indent=2))
    else:
        print(render(jobs, args.threshold, args.window, args.partition))
    return 1 if any(j["fully_idle"] for j in jobs) else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Make it executable and run the tests**

Run: `chmod +x slurm/gpu_idle_now && python3 -m unittest discover -s slurm/tests -v`
Expected: 62 tests (54 + 8 new), all PASS.

- [ ] **Step 5: Commit**

```bash
git add slurm/gpu_idle_now slurm/tests/test_gpu_idle_now.py
git commit -m "Add gpu_idle_now live idle-GPU gate"
```

---

### Task 7: Documentation and final verification

**Files:**
- Modify: `README.md`

**Interfaces:** docs only.

- [ ] **Step 1: Add the five rows to the top-level README's slurm/ table**

After the existing `stotal_kempner` row, append:

```markdown
| `job_failures` | Window failure post-mortem: classifies terminal jobs (failed/OOM/timeout/cancelled/node-fail/preempted), ranks exit codes, failing job names/users, and incident nodes. |
| `queue_wait` | Submit→Start wait distributions (p50/p90/max) by partition, QOS, and GPU-count bucket. Does not claim to split priority vs resource wait (sacct can't). |
| `gpu_idle_now` | Live detector of running jobs whose GPUs are idle right now (SM activity below a threshold over a recent window). Exit 1 when a fully-idle job exists. |
| `fairshare_report` | Point-in-time sshare snapshot: share vs effective usage per account, most over/under-served rankings. |
| `frag_report` | Free CPU/GPU/memory shards per node and how many 1/2/4-GPU jobs could start right now, per partition. |
```

- [ ] **Step 2: Extend the slurm/ requirements row**

In the "Requirements by directory" table, append to the `slurm/` row's "Needs" cell (keep existing text):

```markdown
**`gpu_idle_now`** additionally needs the jobstats `config` module + `requests` (it queries Prometheus like the `gpu/` window tools) and `gpu/gpulib.py` alongside in the repo.
```

- [ ] **Step 3: Run the full suite once**

Run: `python3 -m unittest discover -s slurm/tests -v`
Expected: 62 tests, all PASS (docs-only task).

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "Document the slurm reporting tools"
```

---

## Self-Review Notes (spec → plan mapping)

- Shared conventions (stdlib, exec bits, exit 0/1/3, timeouts 120/60, counted exclusions, replay seams, window semantics matching gpulib) → Task 1 slurmlib + every tool's main; the "implement locally" spec line realized as the shared slurm-local module (noted in the header).
- job_failures (selector one-of with `-u $USER` default, sacct fields, classification precedence incl. `CANCELLED by <uid>` → cancelled, failure rate = non-completed terminal / terminal, four ranked tables) → Task 2; every class has a named test; fixture covers all classes + malformed row.
- queue_wait (default selector rule shared with job_failures, pending/skew exclusion buckets, nearest-rank percentiles, GPU buckets 0/1/2-4/>4, honesty line in output, `-a` with -A/-p, sacct failure surfaced at exit 3) → Task 3.
- gpu_idle_now (defaults 10%/30min, alloc+smact instant queries, both-casing join via gpulib, unmatched counted never guessed, fully-idle rule "every matched GPU idle and ≥1 matched", squeue enrichment degrades with stderr warning, -p post-join filter with hidden-count note, exit 1 gate, bundle roles alloc/smact/squeue, gpulib import failure exit 3 naming ../gpu) → Task 6.
- fairshare_report (sshare -a -P, root/user-row filtering with counts, ratio = EffectvUsage/NormShares, zero-usage split, over/under top-N, point-in-time footer) → Task 4.
- frag_report (scontrol show node -o, BAD_STATES exclusion with listing, free math from CPUTot/CPUAlloc + RealMemory/AllocMem + CfgTRES/AllocTRES, gpu_dist 0..4+, fit for shapes 1/2/4 with default 8 CPU + 64 GiB per GPU, -p filter) → Task 5; fit arithmetic hand-verified in the tests' comments.
- Testing section (fixtures per input type, classification/percentile/grouping/parsing/fit units, CLI 0/1/3 via replay) → Tasks 1-6; live smoke is post-plan (controller).
- Documentation → Task 7.
- Type consistency: `slurmlib` names used in Tasks 2-6 match Task 1 signatures; `gpulib` usage in Task 6 matches the gpu-suite module (`load_prom_server`, `query`, `series_uuid`, `load_bundle`, `err_exit`); loader API consistent across test files.
