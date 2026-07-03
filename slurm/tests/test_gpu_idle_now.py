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
