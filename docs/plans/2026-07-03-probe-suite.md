# Probe Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two quick probes adopted from fasrc/reframe-fasrc — `slurm/slurm_selftest` (scheduler-correctness canaries) and `io/io_probe` (filesystem responsiveness) — per the approved spec. No benchmarks.

**Architecture:** Both are extensionless Python executables with pure cores. `slurm_selftest` generates a bash canary job whose sub-checks each print one `SELFTEST <name> PASS|FAIL|SKIP <detail>` marker; the tool submits it, waits, parses markers (pure), and synthesizes NO-RESULT for requested checks that produced no marker. `io_probe` measures write/read MB/s and per-op metadata latency directly where invoked (no Slurm), with a pure `verdict()` over optional gates.

**Tech Stack:** Python 3.6 stdlib; `slurm/slurmlib.py` (existing) for the selftest's run/err helpers; sbatch/squeue/scancel.

**Spec:** `docs/specs/2026-07-03-probe-suite-design.md` (approved).

**Branch:** create `probe-suite` FROM `main` (controller does this before Task 1).

## Global Constraints

- Python 3.6-compatible stdlib only; no f-strings, walrus, dataclasses, or `subprocess.run(capture_output=/text=)` kwargs.
- Tools: `slurm/slurm_selftest`, `io/io_probe` — no `.py` extension, `#!/usr/bin/env python3`, executable bit set.
- Exit codes: `0` all pass / report-only, `1` any FAIL or NO-RESULT (or a `--min-*`/`--max-*` gate missed), `3` setup error via `<tool>: error: <msg>` on stderr. argparse usage errors stay exit 2.
- Every subprocess call has a timeout; skipped sub-checks are reported as SKIP, never dropped; a canary job that produced no markers yields NO-RESULT for every requested check and exits 1 (never a silent pass).
- File headers note the fasrc/reframe-fasrc provenance.
- io_probe always cleans its scratch subdir (`finally`), unless `--keep`; cleanup failure is a warning, never masks the result.
- Tests: `python3 -m unittest discover -s slurm/tests -v` (64 now → 79 after Task 1) and `python3 -m unittest discover -s io/tests -v` (new, 10 after Task 2).
- Commit messages: short imperative subject, no attribution footers of any kind.

## File Structure

| File | Responsibility |
|---|---|
| `slurm/slurm_selftest` (create, Task 1) | Canary job generation, submission/wait, marker parsing, verdict. |
| `slurm/tests/test_slurm_selftest.py` (create, Task 1) | Marker/evaluate/job-script units + `--from-file` CLI paths. |
| `io/io_probe` (create, Task 2) | Write/read/metadata measurements + gate verdict. |
| `io/tests/__init__.py`, `loader.py`, `test_io_probe.py` (create, Task 2) | New suite scaffolding + real-probe tests against a tempdir. |
| `README.md` (modify, Task 3) | Layout diagram + requirements row for `io/`; tool rows. |

---

### Task 1: `slurm_selftest`

**Files:**
- Create: `slurm/slurm_selftest`
- Test: `slurm/tests/test_slurm_selftest.py`

**Interfaces:**
- Consumes: `slurm/slurmlib.py` (`err_exit(tool, msg)`, `run(cmd, timeout)`), `slurm/tests/loader.py` (`load_tool`, `HERE`).
- Produces: `ALL_CHECKS = ("env", "placement", "gpu_grant", "gres_accounting", "oom_enforcement")`; `ENV_TOKEN = "TEST123456"`; `job_script(checks, node_pattern, gpu) -> str` (bash, one marker line per requested check); `parse_markers(text) -> [{"name","status","detail"}]`; `evaluate(markers, requested) -> [{"name","status","detail"}]` in `requested` order with `NO-RESULT` synthesized for missing names; `render(results) -> str`; `main(argv=None) -> int`.

- [ ] **Step 1: Write the failing tests**

`slurm/tests/test_slurm_selftest.py`:

```python
"""Tests for slurm/slurm_selftest."""
import os
import subprocess
import sys
import tempfile
import unittest

from slurm.tests import loader

st = loader.load_tool("slurm_selftest")
TOOL_PATH = os.path.abspath(os.path.join(loader.HERE, "..",
                                         "slurm_selftest"))

SAMPLE = """some slurm noise
SELFTEST env PASS propagated to 2/2 tasks
SELFTEST placement SKIP no --node-pattern given
SELFTEST gpu_grant FAIL no driver-visible gpu
SELFTEST oom_enforcement PASS step killed (rc=137)
trailing noise
"""


def run_cli(*argv):
    return subprocess.run([sys.executable, TOOL_PATH] + list(argv),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True)


def sample_file(text):
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".out", delete=False)
    tmp.write(text)
    tmp.close()
    return tmp.name


class TestParseMarkers(unittest.TestCase):
    def test_parses_statuses_and_detail(self):
        markers = st.parse_markers(SAMPLE)
        self.assertEqual(len(markers), 4)
        self.assertEqual(markers[0],
                         {"name": "env", "status": "PASS",
                          "detail": "propagated to 2/2 tasks"})
        self.assertEqual(markers[2]["status"], "FAIL")

    def test_malformed_lines_ignored(self):
        markers = st.parse_markers("SELFTEST bad\nSELFTEST x MAYBE y\n")
        self.assertEqual(markers, [])

    def test_empty(self):
        self.assertEqual(st.parse_markers(""), [])


class TestEvaluate(unittest.TestCase):
    def test_no_result_synthesized(self):
        results = st.evaluate(st.parse_markers(SAMPLE),
                              ["env", "gres_accounting"])
        self.assertEqual(results[0]["status"], "PASS")
        self.assertEqual(results[1],
                         {"name": "gres_accounting", "status": "NO-RESULT",
                          "detail": "no marker in job output"})

    def test_requested_order_preserved(self):
        results = st.evaluate(st.parse_markers(SAMPLE),
                              ["oom_enforcement", "env"])
        self.assertEqual([r["name"] for r in results],
                         ["oom_enforcement", "env"])

    def test_skip_kept(self):
        results = st.evaluate(st.parse_markers(SAMPLE), ["placement"])
        self.assertEqual(results[0]["status"], "SKIP")


class TestJobScript(unittest.TestCase):
    def test_all_checks_emit_markers(self):
        script = st.job_script(list(st.ALL_CHECKS), r"^holygpu\d+$", True)
        for name in st.ALL_CHECKS:
            self.assertIn("SELFTEST %s " % name, script)
        self.assertNotIn("%s", script)  # no format residue
        self.assertIn(st.ENV_TOKEN, script)

    def test_placement_skip_without_pattern(self):
        script = st.job_script(["placement"], None, False)
        self.assertIn("SELFTEST placement SKIP", script)

    def test_gpu_checks_skip_without_gpu(self):
        script = st.job_script(["gpu_grant", "gres_accounting"], None, False)
        self.assertIn("SELFTEST gpu_grant SKIP", script)
        self.assertIn("SELFTEST gres_accounting SKIP", script)


class TestCli(unittest.TestCase):
    def test_from_file_all_pass_exit_0(self):
        path = sample_file(
            "SELFTEST env PASS ok\nSELFTEST oom_enforcement PASS ok\n")
        self.addCleanup(os.unlink, path)
        proc = run_cli("--from-file", path, "--checks",
                       "env,oom_enforcement")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("PASS", proc.stdout)

    def test_from_file_fail_exit_1(self):
        path = sample_file(SAMPLE)
        self.addCleanup(os.unlink, path)
        proc = run_cli("--from-file", path)
        self.assertEqual(proc.returncode, 1, proc.stdout)
        self.assertIn("gpu_grant", proc.stdout)

    def test_missing_marker_exit_1(self):
        path = sample_file("SELFTEST env PASS ok\n")
        self.addCleanup(os.unlink, path)
        proc = run_cli("--from-file", path)
        self.assertEqual(proc.returncode, 1)
        self.assertIn("NO-RESULT", proc.stdout)

    def test_missing_file_exit_3(self):
        proc = run_cli("--from-file", "/nonexistent.out")
        self.assertEqual(proc.returncode, 3)
        self.assertIn("slurm_selftest: error:", proc.stderr)

    def test_live_mode_needs_account_partition(self):
        proc = run_cli()
        self.assertEqual(proc.returncode, 3)
        self.assertIn("slurm_selftest: error:", proc.stderr)

    def test_unknown_check_exit_3(self):
        path = sample_file(SAMPLE)
        self.addCleanup(os.unlink, path)
        proc = run_cli("--from-file", path, "--checks", "env,bogus")
        self.assertEqual(proc.returncode, 3)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest discover -s slurm/tests -v`
Expected: import-time ERROR (`FileNotFoundError` for `slurm/slurm_selftest`); the 64 existing tests PASS.

- [ ] **Step 3: Implement `slurm/slurm_selftest`**

```python
#!/usr/bin/env python3
"""slurm_selftest -- canary job verifying the scheduler itself behaves.

Adapted from fasrc/reframe-fasrc checks/system/slurm/slurm.py (BSD),
re-cast as a standalone verdict CLI.

One small sbatch job runs seconds-long sub-checks, each printing exactly
one marker line: SELFTEST <name> PASS|FAIL|SKIP <detail>.

  env             sbatch --export propagates a variable into both tasks
  placement       hostname matches --node-pattern (SKIP without the flag)
  gpu_grant       nvidia-smi -L sees >=1 GPU (SKIP without --gpu)
  gres_accounting scontrol show job reports gpu:1 (SKIP without --gpu)
  oom_enforcement a --mem=512M step allocating ~2GB must be OOM-killed

Usage:
    slurm_selftest -A ACCOUNT -p PARTITION [--gpu] [--node-pattern REGEX]
                   [--checks LIST] [--timeout MIN] [--json]

Exit codes: 0 all requested checks PASS/SKIP, 1 any FAIL or NO-RESULT,
3 submission/timeout/setup error.
"""

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import slurmlib

TOOL = "slurm_selftest"
ALL_CHECKS = ("env", "placement", "gpu_grant", "gres_accounting",
              "oom_enforcement")
ENV_TOKEN = "TEST123456"
_MARKER_RE = re.compile(r"^SELFTEST (\S+) (PASS|FAIL|SKIP)[ \t]*(.*)$",
                        re.MULTILINE)
# ~2GB in 64MB zero-filled chunks; no quotes so it nests in bash cleanly.
_ALLOCATOR = "a=[bytearray(64*1024*1024) for _ in range(32)]"


def job_script(checks, node_pattern, gpu):
    """Bash canary script: one SELFTEST marker line per requested check."""
    lines = ["#!/bin/bash", ""]
    if "env" in checks:
        lines += [
            "vals=$(srun -n 2 bash -c 'echo $SELFTEST_VAR')",
            'n=$(echo "$vals" | grep -c "^%s$")' % ENV_TOKEN,
            'if [ "$n" -eq 2 ]; then '
            'echo "SELFTEST env PASS propagated to 2/2 tasks"; '
            'else echo "SELFTEST env FAIL propagated to $n/2 tasks"; fi',
        ]
    if "placement" in checks:
        if node_pattern:
            lines += [
                "host=$(hostname -s)",
                "if echo \"$host\" | grep -Eq '%s'; then "
                'echo "SELFTEST placement PASS $host"; '
                'else echo "SELFTEST placement FAIL $host does not match"; '
                "fi" % node_pattern,
            ]
        else:
            lines.append(
                'echo "SELFTEST placement SKIP no --node-pattern given"')
    if "gpu_grant" in checks:
        if gpu:
            lines += [
                'ngpu=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU ")',
                'if [ "$ngpu" -ge 1 ]; then '
                'echo "SELFTEST gpu_grant PASS $ngpu gpu(s) visible"; '
                'else echo "SELFTEST gpu_grant FAIL no driver-visible gpu"; '
                "fi",
            ]
        else:
            lines.append('echo "SELFTEST gpu_grant SKIP run with --gpu"')
    if "gres_accounting" in checks:
        if gpu:
            lines += [
                'tres=$(scontrol show job $SLURM_JOB_ID | '
                'grep -oE "(TresPerNode|Gres)=[^ ]*")',
                'if echo "$tres" | grep -q "gpu:1"; then '
                'echo "SELFTEST gres_accounting PASS $tres"; '
                'else echo "SELFTEST gres_accounting FAIL $tres"; fi',
            ]
        else:
            lines.append(
                'echo "SELFTEST gres_accounting SKIP run with --gpu"')
    if "oom_enforcement" in checks:
        lines += [
            'srun -n 1 --mem=512M python3 -c "%s" '
            "> oom_out.txt 2> oom_err.txt" % _ALLOCATOR,
            "rc=$?",
            'if [ "$rc" -ne 0 ] && '
            'grep -qiE "out of memory|oom.kill|exceeded memory limit" '
            "oom_err.txt; then "
            'echo "SELFTEST oom_enforcement PASS step killed (rc=$rc)"; '
            'elif [ "$rc" -ne 0 ]; then '
            'echo "SELFTEST oom_enforcement FAIL step died (rc=$rc) '
            'without an OOM message"; '
            'else echo "SELFTEST oom_enforcement FAIL allocator survived '
            'a 512M limit"; fi',
            "rm -f oom_out.txt oom_err.txt",
        ]
    lines.append("exit 0")
    return "\n".join(lines) + "\n"


def parse_markers(text):
    """Job output -> [{'name','status','detail'}] in appearance order."""
    return [{"name": m.group(1), "status": m.group(2),
             "detail": m.group(3).strip()}
            for m in _MARKER_RE.finditer(text or "")]


def evaluate(markers, requested):
    """Results in `requested` order; NO-RESULT for missing markers."""
    by_name = {}
    for marker in markers:
        by_name.setdefault(marker["name"], marker)
    results = []
    for name in requested:
        if name in by_name:
            results.append(by_name[name])
        else:
            results.append({"name": name, "status": "NO-RESULT",
                            "detail": "no marker in job output"})
    return results


def render(results):
    lines = []
    for r in results:
        lines.append("  %-16s %-9s %s" % (r["name"], r["status"],
                                          r["detail"]))
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    lines.append("")
    lines.append("slurm_selftest: " + "  ".join(
        "%s=%d" % (s, counts[s]) for s in sorted(counts)))
    return "\n".join(lines)


def _submit_and_wait(script, args, timeout_s):
    """sbatch the canary, poll until it leaves the queue, return output."""
    sbatch = ["sbatch", "--parsable", "-J", TOOL, "-A", args.account,
              "-p", args.partition, "-N", "1", "-n", "2", "--mem=1G",
              "-t", "5", "-o", "slurm_selftest_%j.out",
              "--export=ALL,SELFTEST_VAR=" + ENV_TOKEN]
    if args.gpu:
        sbatch.append("--gres=gpu:1")
    try:
        out = slurmlib.run(sbatch + ["/dev/stdin"], timeout=60)
    except RuntimeError:
        # some sbatch builds reject /dev/stdin scripts; use a temp file
        import tempfile
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".sh",
                                          delete=False)
        tmp.write(script)
        tmp.close()
        try:
            out = slurmlib.run(sbatch + [tmp.name], timeout=60)
        finally:
            os.unlink(tmp.name)
        jobid = out.strip().split(";")[0]
        return _wait(jobid, timeout_s)
    # /dev/stdin path needs the script on stdin -- not supported by
    # slurmlib.run, so always go through the temp file branch above.
    jobid = out.strip().split(";")[0]
    return _wait(jobid, timeout_s)


def _wait(jobid, timeout_s):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        queued = slurmlib.run(["squeue", "-h", "-j", jobid], timeout=60)
        if not queued.strip():
            break
        time.sleep(10)
    else:
        slurmlib.run(["scancel", jobid], timeout=60)
        raise RuntimeError("canary job %s did not finish within --timeout"
                           % jobid)
    path = "slurm_selftest_%s.out" % jobid
    try:
        with open(path) as fh:
            return fh.read(), path
    except OSError as exc:
        raise RuntimeError("cannot read job output %s: %s" % (path, exc))


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog=TOOL,
        description="scheduler-correctness canary "
                    "(exit 0 pass, 1 fail, 3 error)")
    parser.add_argument("-A", "--account")
    parser.add_argument("-p", "--partition")
    parser.add_argument("--gpu", action="store_true",
                        help="request gpu:1 and run the GPU checks")
    parser.add_argument("--node-pattern", metavar="REGEX",
                        help="expected hostname pattern for placement")
    parser.add_argument("--checks", metavar="LIST",
                        help="comma-separated subset of: %s"
                             % ",".join(ALL_CHECKS))
    parser.add_argument("--timeout", type=int, default=10,
                        help="minutes to wait for the canary (default 10)")
    parser.add_argument("--json", action="store_true",
                        help="emit the structured result as JSON")
    parser.add_argument("--from-file", metavar="FILE",
                        help="parse a captured job output (testing)")
    args = parser.parse_args(argv)

    requested = list(ALL_CHECKS)
    if args.checks:
        requested = [c.strip() for c in args.checks.split(",") if c.strip()]
        unknown = [c for c in requested if c not in ALL_CHECKS]
        if unknown:
            slurmlib.err_exit(TOOL, "unknown check(s): %s"
                              % ", ".join(unknown))

    if args.from_file:
        try:
            with open(args.from_file) as fh:
                text = fh.read()
        except OSError as exc:
            slurmlib.err_exit(TOOL, "cannot read replay file: %s" % exc)
        out_path = args.from_file
    else:
        if not (args.account and args.partition):
            slurmlib.err_exit(TOOL, "-A ACCOUNT and -p PARTITION are "
                              "required (or use --from-file)")
        script = job_script(requested, args.node_pattern, args.gpu)
        try:
            text, out_path = _submit_and_wait(script, args,
                                              args.timeout * 60)
        except RuntimeError as exc:
            slurmlib.err_exit(TOOL, str(exc))

    results = evaluate(parse_markers(text), requested)
    if args.json:
        print(json.dumps({"results": results, "output": out_path},
                         indent=2))
    else:
        print(render(results))
        print("job output: %s" % out_path)
    return 1 if any(r["status"] in ("FAIL", "NO-RESULT")
                    for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
```

NOTE for the implementer: `_submit_and_wait` must ALWAYS use the temp-file
branch (write the script to a temp .sh and pass its path to sbatch) — the
`/dev/stdin` attempt shown above is dead weight; simplify it to:

```python
def _submit_and_wait(script, args, timeout_s):
    """sbatch the canary via a temp script, poll, return (output, path)."""
    import tempfile
    sbatch = ["sbatch", "--parsable", "-J", TOOL, "-A", args.account,
              "-p", args.partition, "-N", "1", "-n", "2", "--mem=1G",
              "-t", "5", "-o", "slurm_selftest_%j.out",
              "--export=ALL,SELFTEST_VAR=" + ENV_TOKEN]
    if args.gpu:
        sbatch.append("--gres=gpu:1")
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False)
    tmp.write(script)
    tmp.close()
    try:
        out = slurmlib.run(sbatch + [tmp.name], timeout=60)
    finally:
        os.unlink(tmp.name)
    jobid = out.strip().split(";")[0]
    return _wait(jobid, timeout_s)
```

Use this simplified version verbatim; do not include the /dev/stdin branch.

- [ ] **Step 4: Make it executable and run the tests**

Run: `chmod +x slurm/slurm_selftest && python3 -m unittest discover -s slurm/tests -v`
Expected: 79 tests (64 + 15 new), all PASS.

- [ ] **Step 5: Commit**

```bash
git add slurm/slurm_selftest slurm/tests/test_slurm_selftest.py
git commit -m "Add slurm_selftest scheduler canary"
```

---

### Task 2: `io_probe`

**Files:**
- Create: `io/io_probe`, `io/tests/__init__.py` (empty), `io/tests/loader.py`
- Test: `io/tests/test_io_probe.py`

**Interfaces:**
- Consumes: nothing from Task 1 (independent).
- Produces: `measure_write(path, size_mb) -> mbs float` (fsync'd); `measure_read(path) -> mbs float`; `measure_meta(dirpath, count) -> {"create","stat","delete"}` mean ms per op; `verdict(metrics, gates) -> (status, reasons)` with status in `PASS|FAIL|REPORT` (`REPORT` when no gate set); `render(metrics, status, reasons) -> str`; `main(argv=None) -> int`.

- [ ] **Step 1: Create `io/tests/loader.py`** (and empty `io/tests/__init__.py`)

```python
"""Load the extensionless io tools as modules."""
import importlib.machinery
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load_tool(name):
    path = os.path.abspath(os.path.join(HERE, "..", name))
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod
```

- [ ] **Step 2: Write the failing tests**

`io/tests/test_io_probe.py`:

```python
"""Tests for io/io_probe. The CLI tests run the REAL probe with tiny sizes."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from io_probe_loader_shim import *  # noqa -- replaced below; see note

# NOTE: the actual import uses the local loader:
from io.tests import loader

ip = loader.load_tool("io_probe")
TOOL_PATH = os.path.abspath(os.path.join(loader.HERE, "..", "io_probe"))


def run_cli(*argv):
    return subprocess.run([sys.executable, TOOL_PATH] + list(argv),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True)


METRICS = {"write_mbs": 500.0, "read_mbs": 2000.0,
           "meta_ms": {"create": 0.5, "stat": 0.1, "delete": 0.4}}


class TestVerdict(unittest.TestCase):
    def test_no_gates_is_report(self):
        status, reasons = ip.verdict(METRICS, {"min_write": None,
                                               "min_read": None,
                                               "max_meta_ms": None})
        self.assertEqual(status, "REPORT")
        self.assertEqual(reasons, [])

    def test_gates_pass(self):
        status, _ = ip.verdict(METRICS, {"min_write": 100.0,
                                         "min_read": 100.0,
                                         "max_meta_ms": 5.0})
        self.assertEqual(status, "PASS")

    def test_write_gate_fail(self):
        status, reasons = ip.verdict(METRICS, {"min_write": 1000.0,
                                               "min_read": None,
                                               "max_meta_ms": None})
        self.assertEqual(status, "FAIL")
        self.assertIn("write", reasons[0])

    def test_meta_gate_uses_worst_mean(self):
        status, reasons = ip.verdict(METRICS, {"min_write": None,
                                               "min_read": None,
                                               "max_meta_ms": 0.45})
        self.assertEqual(status, "FAIL")  # worst is create=0.5


class TestCliRealProbe(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def args(self, *extra):
        return ("--dir", self.tmp, "--size", "1", "--meta-files", "5") \
            + extra

    def test_report_exit_0_and_cleanup(self):
        proc = run_cli(*self.args())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("MB/s", proc.stdout)
        self.assertIn("page-cache-assisted", proc.stdout)
        leftovers = [d for d in os.listdir(self.tmp)
                     if d.startswith(".io_probe.")]
        self.assertEqual(leftovers, [])

    def test_json_schema(self):
        proc = run_cli(*self.args("--json"))
        payload = json.loads(proc.stdout)
        self.assertGreater(payload["write_mbs"], 0)
        self.assertGreater(payload["read_mbs"], 0)
        self.assertTrue(payload["read_cached"])
        self.assertEqual(sorted(payload["meta_ms"]),
                         ["create", "delete", "stat"])
        self.assertEqual(payload["status"], "REPORT")

    def test_impossible_gate_exit_1(self):
        proc = run_cli(*self.args("--min-write", "1e12"))
        self.assertEqual(proc.returncode, 1, proc.stdout)
        self.assertIn("FAIL", proc.stdout)

    def test_keep_leaves_scratch(self):
        proc = run_cli(*self.args("--keep"))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        leftovers = [d for d in os.listdir(self.tmp)
                     if d.startswith(".io_probe.")]
        self.assertEqual(len(leftovers), 1)

    def test_missing_dir_exit_3(self):
        proc = run_cli("--dir", "/nonexistent-io-probe-target")
        self.assertEqual(proc.returncode, 3)
        self.assertIn("io_probe: error:", proc.stderr)

    def test_unwritable_dir_exit_3(self):
        os.chmod(self.tmp, 0o500)
        self.addCleanup(os.chmod, self.tmp, 0o700)
        proc = run_cli(*self.args())
        self.assertEqual(proc.returncode, 3)
        self.assertIn("io_probe: error:", proc.stderr)


if __name__ == "__main__":
    unittest.main()
```

NOTE for the implementer: delete the `io_probe_loader_shim` line — it is a
placeholder artifact; the real import is the `from io.tests import loader`
line directly below it. CAUTION: the package path `io.tests` shadows the
stdlib `io` module inside the test process only when imported as a package
from the repo root — Python resolves `from io.tests import loader`
correctly under `unittest discover -s io/tests` ONLY if `io/` has an
`__init__.py`. Do NOT add `io/__init__.py` (the tools dir must not become a
package). Instead, `io/tests/test_io_probe.py` must load the loader by
path, exactly like this, replacing BOTH import lines above:

```python
import importlib.machinery
import importlib.util
import os as _os

_here = _os.path.dirname(_os.path.abspath(__file__))
_l = importlib.machinery.SourceFileLoader(
    "io_probe_test_loader", _os.path.join(_here, "loader.py"))
_spec = importlib.util.spec_from_loader("io_probe_test_loader", _l)
loader = importlib.util.module_from_spec(_spec)
_l.exec_module(loader)
```

(This avoids any `io` package/stdlib collision entirely; `loader.HERE` and
`loader.load_tool` work as elsewhere.)

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m unittest discover -s io/tests -v`
Expected: import-time ERROR (`FileNotFoundError` for `io/io_probe`).
Also run: `python3 -m unittest discover -s slurm/tests -v`
Expected: 79 tests PASS (untouched by this task).

- [ ] **Step 4: Implement `io/io_probe`**

```python
#!/usr/bin/env python3
"""io_probe -- quick filesystem responsiveness probe (NOT a benchmark).

Adapted in spirit from fasrc/reframe-fasrc checks/system/io (IOR check),
reduced to a seconds-long probe: bounded sequential write/read MB/s plus
small-file metadata latency, against a target directory. Runs where
invoked (wrap in srun to probe from a compute node). The read re-reads the
just-written file, so it is page-cache-assisted -- reported honestly.

Usage:
    io_probe --dir TARGET [--size MB] [--meta-files N]
             [--min-write MBS] [--min-read MBS] [--max-meta-ms MS]
             [--keep] [--json]

Exit codes: 0 report-only or all gates pass, 1 a gate missed, 3 setup/IO
error.
"""

import argparse
import json
import os
import shutil
import sys
import time

TOOL = "io_probe"
CHUNK = 4 * 1024 * 1024  # 4 MB


def err_exit(msg):
    sys.stderr.write("%s: error: %s\n" % (TOOL, msg))
    sys.exit(3)


def measure_write(path, size_mb):
    """Stream size_mb of zeros in CHUNK pieces, fsync, return MB/s."""
    chunk = b"\0" * CHUNK
    chunks = max(1, int(size_mb * 1024 * 1024) // CHUNK)
    start = time.time()
    with open(path, "wb") as fh:
        for _ in range(chunks):
            fh.write(chunk)
        fh.flush()
        os.fsync(fh.fileno())
    elapsed = time.time() - start
    total_mb = chunks * CHUNK / (1024.0 * 1024.0)
    return total_mb / elapsed if elapsed > 0 else float("inf")


def measure_read(path):
    """Re-read the file in CHUNK pieces, return MB/s (page-cache-assisted)."""
    size_mb = os.path.getsize(path) / (1024.0 * 1024.0)
    start = time.time()
    with open(path, "rb") as fh:
        while fh.read(CHUNK):
            pass
    elapsed = time.time() - start
    return size_mb / elapsed if elapsed > 0 else float("inf")


def measure_meta(dirpath, count):
    """create/stat/delete `count` empty files -> mean ms per op."""
    names = [os.path.join(dirpath, "meta%05d" % i) for i in range(count)]
    t0 = time.time()
    for name in names:
        open(name, "w").close()
    t1 = time.time()
    for name in names:
        os.stat(name)
    t2 = time.time()
    for name in names:
        os.unlink(name)
    t3 = time.time()
    per = lambda a, b: (b - a) * 1000.0 / count  # noqa: E731
    return {"create": per(t0, t1), "stat": per(t1, t2),
            "delete": per(t2, t3)}


def verdict(metrics, gates):
    """-> (status, reasons): REPORT when no gate set; else PASS/FAIL."""
    reasons = []
    gated = False
    if gates.get("min_write") is not None:
        gated = True
        if metrics["write_mbs"] < gates["min_write"]:
            reasons.append("write %.1f MB/s below --min-write %.1f"
                           % (metrics["write_mbs"], gates["min_write"]))
    if gates.get("min_read") is not None:
        gated = True
        if metrics["read_mbs"] < gates["min_read"]:
            reasons.append("read %.1f MB/s below --min-read %.1f"
                           % (metrics["read_mbs"], gates["min_read"]))
    if gates.get("max_meta_ms") is not None:
        gated = True
        worst = max(metrics["meta_ms"].values())
        if worst > gates["max_meta_ms"]:
            reasons.append("metadata %.2f ms/op above --max-meta-ms %.2f"
                           % (worst, gates["max_meta_ms"]))
    if not gated:
        return "REPORT", []
    return ("FAIL" if reasons else "PASS"), reasons


def render(metrics, status, reasons):
    lines = ["io_probe: %s  (%.0f MB file, %d metadata files)"
             % (metrics["dir"], metrics["size_mb"],
                metrics["meta_files"])]
    lines.append("  write     : %8.1f MB/s (fsync included)"
                 % metrics["write_mbs"])
    lines.append("  read      : %8.1f MB/s (page-cache-assisted on this "
                 "host)" % metrics["read_mbs"])
    lines.append("  metadata  : create %.2f  stat %.2f  delete %.2f ms/op"
                 % (metrics["meta_ms"]["create"],
                    metrics["meta_ms"]["stat"],
                    metrics["meta_ms"]["delete"]))
    lines.append("verdict: %s%s"
                 % (status, "" if not reasons
                    else " (" + "; ".join(reasons) + ")"))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog=TOOL,
        description="quick filesystem responsiveness probe "
                    "(exit 0 report/pass, 1 gate missed, 3 error)")
    parser.add_argument("--dir", required=True, metavar="TARGET",
                        help="directory to probe (a scratch subdir is "
                             "created inside it)")
    parser.add_argument("--size", type=float, default=256.0, metavar="MB",
                        help="sequential file size (default 256)")
    parser.add_argument("--meta-files", type=int, default=100,
                        dest="meta_files",
                        help="metadata batch size (default 100)")
    parser.add_argument("--min-write", type=float, dest="min_write",
                        metavar="MBS")
    parser.add_argument("--min-read", type=float, dest="min_read",
                        metavar="MBS")
    parser.add_argument("--max-meta-ms", type=float, dest="max_meta_ms",
                        metavar="MS")
    parser.add_argument("--keep", action="store_true",
                        help="keep the scratch subdir")
    parser.add_argument("--json", action="store_true",
                        help="emit the structured result as JSON")
    args = parser.parse_args(argv)
    if args.size <= 0 or args.meta_files <= 0:
        err_exit("--size and --meta-files must be > 0")

    if not os.path.isdir(args.dir):
        err_exit("target directory does not exist: %s" % args.dir)
    scratch = os.path.join(args.dir, ".io_probe.%d" % os.getpid())
    try:
        os.mkdir(scratch)
    except OSError as exc:
        err_exit("cannot create scratch dir: %s" % exc)

    try:
        data_path = os.path.join(scratch, "data.bin")
        try:
            write_mbs = measure_write(data_path, args.size)
            read_mbs = measure_read(data_path)
            meta_ms = measure_meta(scratch, args.meta_files)
        except (OSError, IOError) as exc:
            err_exit("io failure during probe: %s" % exc)
        metrics = {"dir": args.dir, "size_mb": args.size,
                   "meta_files": args.meta_files,
                   "write_mbs": write_mbs, "read_mbs": read_mbs,
                   "read_cached": True, "meta_ms": meta_ms}
        gates = {"min_write": args.min_write, "min_read": args.min_read,
                 "max_meta_ms": args.max_meta_ms}
        status, reasons = verdict(metrics, gates)
        if args.json:
            print(json.dumps(dict(metrics, gates=gates, status=status,
                                  reasons=reasons), indent=2))
        else:
            print(render(metrics, status, reasons))
        return 1 if status == "FAIL" else 0
    finally:
        if not args.keep:
            try:
                shutil.rmtree(scratch)
            except OSError as exc:
                sys.stderr.write("warning: cleanup failed: %s\n" % exc)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Make it executable and run both suites**

Run: `chmod +x io/io_probe && python3 -m unittest discover -s io/tests -v && python3 -m unittest discover -s slurm/tests -v`
Expected: io suite 10 tests PASS; slurm suite 79 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add io/io_probe io/tests
git commit -m "Add io_probe filesystem responsiveness probe"
```

---

### Task 3: Documentation and final verification

**Files:**
- Modify: `README.md`

**Interfaces:** docs only.

- [ ] **Step 1: Update the layout diagram**

In the top-level README's directory diagram, add an `io/` line after `fabric/`:

```text
├── io/         # filesystem responsiveness probes (read/write/metadata)
```

- [ ] **Step 2: Add the tool rows**

In the slurm/ tool table, append:

```markdown
| `slurm_selftest` | Canary job verifying the scheduler itself: env-var propagation, GPU grant, GRES accounting, cgroup OOM enforcement, node placement. Exit 1 on any failed check. Adapted from fasrc/reframe-fasrc. |
```

Add a new `## io/ — filesystem probes` section after the fabric/ section:

```markdown
## io/ — filesystem probes

Quick responsiveness probes (seconds, not benchmarks). Run where invoked —
wrap in `srun` to probe from a compute node.

| Tool | What it does |
|---|---|
| `io_probe` | Bounded sequential write/read MB/s (fsync'd; read is page-cache-assisted and labeled as such) plus small-file create/stat/delete latency against `--dir`. Optional `--min-*` gates; report-only by default. Adapted from fasrc/reframe-fasrc's IOR check, reduced to a probe. |
```

- [ ] **Step 3: Add the requirements row**

In the "Requirements by directory" table, after the fabric/ row:

```markdown
| `io/` | any node | Python 3 standard library only. |
```

- [ ] **Step 4: Run both suites once**

Run: `python3 -m unittest discover -s slurm/tests -v && python3 -m unittest discover -s io/tests -v`
Expected: 79 and 10 tests, all PASS (docs-only task).

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "Document the probe suite tools"
```

---

## Self-Review Notes (spec → plan mapping)

- Marker protocol, sub-check semantics (env 2-task propagation, placement SKIP without pattern, gpu checks SKIP without --gpu, OOM PASS = nonzero rc + message with the survived/no-message FAIL split), NO-RESULT rule and its exit-1 semantics → Task 1, each with a named test; the allocator is quote-free so it nests in bash safely and self-limits at 2 GB.
- Submission details (sbatch flags incl. --export, -t 5, --mem=1G, gres, %j output file; poll/cancel on --timeout; temp-file script submission) → Task 1 `_submit_and_wait` (simplified version is normative per the implementer note).
- io_probe measurements (4 MB chunks, fsync before stopping the clock, cache-honesty labeling in text and JSON, metadata mean ms, scratch-subdir lifecycle with finally-cleanup and --keep, refusal on missing/unwritable target, gate-only FAIL semantics with worst-of-three metadata mean) → Task 2, all exercised by the real-probe CLI tests with tiny sizes.
- Error handling table (exit 3 paths; NO-RESULT → 1 not 3; cleanup warning never masks) → Tasks 1-2 code + tests.
- Docs (layout diagram, tool rows with provenance, io/ requirements) → Task 3.
- Type consistency: `slurmlib.err_exit/run` signatures match the existing module; loader-by-path pattern in io tests avoids the `io` stdlib-name collision (called out explicitly with normative replacement code).
