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
