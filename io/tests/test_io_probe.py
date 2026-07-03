"""Tests for io/io_probe. The CLI tests run the REAL probe with tiny sizes."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

# NOTE: Load the loader by path to avoid stdlib io module collision.
# This avoids any `io` package/stdlib collision entirely.
import importlib.machinery
import importlib.util
import os as _os

_here = _os.path.dirname(_os.path.abspath(__file__))
_l = importlib.machinery.SourceFileLoader(
    "io_probe_test_loader", _os.path.join(_here, "loader.py"))
_spec = importlib.util.spec_from_loader("io_probe_test_loader", _l)
loader = importlib.util.module_from_spec(_spec)
_l.exec_module(loader)

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
