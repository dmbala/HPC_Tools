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
