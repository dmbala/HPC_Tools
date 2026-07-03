"""Tests for gpu/energy_report."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

from gpu.tests import loader

er = loader.load_tool("energy_report")
TOOL_PATH = os.path.abspath(os.path.join(loader.HERE, "..", "energy_report"))


def run_cli(*argv):
    return subprocess.run([sys.executable, TOOL_PATH] + list(argv),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True)


class TestParsers(unittest.TestCase):
    def test_elapsed_plain(self):
        self.assertEqual(er._elapsed_seconds("02:30:15"), 9015)

    def test_elapsed_with_days(self):
        self.assertEqual(er._elapsed_seconds("1-01:00:00"), 90000)

    def test_gpus_plain(self):
        self.assertEqual(er._gpus_from_tres(
            "billing=64,cpu=16,gres/gpu=2,mem=128G"), 2)

    def test_gpus_typed_variant_and_none(self):
        self.assertEqual(er._gpus_from_tres("cpu=4,gres/gpu:h100=4,mem=32G"),
                         4)
        self.assertEqual(er._gpus_from_tres("cpu=4,mem=16G"), 0)

    def test_sacct_jobs_filters_non_gpu(self):
        jobs = er.parse_sacct_jobs(loader.fixture("sacct_energy.txt"))
        self.assertEqual([j["jobid"] for j in jobs], [27400001, 27400002])
        self.assertEqual(jobs[0]["gpus"], 2)
        self.assertEqual(jobs[0]["elapsed_s"], 7200)
        self.assertEqual(jobs[0]["user"], "alice")
        self.assertEqual(jobs[0]["state"], "COMPLETED")


class TestEnergyKwh(unittest.TestCase):
    RESULT = [{"metric": {"UUID": "GPU-aaa"}, "value": [0, "3600000000"]},
              {"metric": {"UUID": "GPU-bbb"}, "value": [0, "7200000000"]}]

    def test_sums_and_converts(self):
        kwh = er.energy_kwh(self.RESULT, ["GPU-aaa", "GPU-bbb"])
        self.assertAlmostEqual(kwh, 3.0)

    def test_filters_foreign_uuids(self):
        kwh = er.energy_kwh(self.RESULT, ["GPU-aaa"])
        self.assertAlmostEqual(kwh, 1.0)

    def test_no_match_is_none(self):
        self.assertIsNone(er.energy_kwh(self.RESULT, ["GPU-zzz"]))
        self.assertIsNone(er.energy_kwh([], ["GPU-aaa"]))


class TestUuidRegex(unittest.TestCase):
    def test_hyphens_survive_unescaped(self):
        rx, safe, dropped = er.uuid_regex(
            ["GPU-ff32a749-638d-95a9-0ba8-539ee8e18388", "GPU-aaa"])
        self.assertEqual(
            rx, "GPU-ff32a749-638d-95a9-0ba8-539ee8e18388|GPU-aaa")
        self.assertNotIn("\\", rx)
        self.assertEqual(dropped, 0)

    def test_bad_uuid_dropped(self):
        rx, safe, dropped = er.uuid_regex(['GPU-aaa', 'evil"} or {x=~".*'])
        self.assertEqual(rx, "GPU-aaa")
        self.assertEqual(safe, ["GPU-aaa"])
        self.assertEqual(dropped, 1)

    def test_all_bad_is_none(self):
        rx, safe, dropped = er.uuid_regex(['x"y'])
        self.assertIsNone(rx)
        self.assertEqual(dropped, 1)


class TestRollup(unittest.TestCase):
    def jobs(self):
        return [{"jobid": 1, "user": "alice", "elapsed_s": 7200, "gpus": 2,
                 "state": "COMPLETED", "kwh": 3.0},
                {"jobid": 2, "user": "bob", "elapsed_s": 1800, "gpus": 1,
                 "state": "FAILED", "kwh": None},
                {"jobid": 3, "user": "alice", "elapsed_s": 3600, "gpus": 1,
                 "state": "COMPLETED", "kwh": 1.0}]

    def test_totals_and_top(self):
        summary = er.rollup(self.jobs(), top_n=1)
        self.assertEqual(summary["counted"], 2)
        self.assertEqual(summary["no_data"], 1)
        self.assertAlmostEqual(summary["total_kwh"], 4.0)
        self.assertAlmostEqual(summary["total_gpu_hours"], 5.0)
        self.assertAlmostEqual(summary["mean_w_per_gpu"], 800.0)
        self.assertEqual([j["jobid"] for j in summary["top_jobs"]], [1])

    def test_per_user(self):
        summary = er.rollup(self.jobs(), top_n=5)
        self.assertAlmostEqual(summary["per_user_kwh"]["alice"], 4.0)
        self.assertNotIn("bob", summary["per_user_kwh"])


class TestCli(unittest.TestCase):
    def bundle_path(self, sacct_text):
        bundle = {
            "sacct": sacct_text,
            "discovery": {"27400001": [
                {"metric": {"uuid": "GPU-aaa"}, "value": [0, "27400001"]},
                {"metric": {"uuid": "GPU-bbb"}, "value": [0, "27400001"]}],
                "27400002": []},
            "energy": {"27400001": [
                {"metric": {"UUID": "GPU-aaa"}, "value": [0, "3600000000"]},
                {"metric": {"UUID": "GPU-bbb"}, "value": [0, "7200000000"]}]},
        }
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                          delete=False)
        self.addCleanup(os.unlink, tmp.name)
        json.dump(bundle, tmp)
        tmp.close()
        return tmp.name

    def test_report_exit_0(self):
        proc = run_cli("-u", "alice", "--from-json",
                       self.bundle_path(loader.fixture("sacct_energy.txt")))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("jobs counted: 1", proc.stdout)
        self.assertIn("no data: 1", proc.stdout)
        self.assertIn("3.00 kWh", proc.stdout)

    def test_user_and_account_both_exit_3(self):
        proc = run_cli("-u", "alice", "-A", "lab")
        self.assertEqual(proc.returncode, 3)
        proc = run_cli()
        self.assertEqual(proc.returncode, 3)

    def test_max_jobs_cap_reported(self):
        proc = run_cli("-u", "alice", "--max-jobs", "1", "--from-json",
                       self.bundle_path(loader.fixture("sacct_energy.txt")))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("skipped by --max-jobs: 1", proc.stdout)

    def test_negative_max_jobs_exits_3(self):
        proc = run_cli("-u", "alice", "--max-jobs", "-1", "--from-json",
                       self.bundle_path(loader.fixture("sacct_energy.txt")))
        self.assertEqual(proc.returncode, 3)
        self.assertIn("energy_report: error:", proc.stderr)


if __name__ == "__main__":
    unittest.main()
