"""Tests for gpu/fleet_util."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

from gpu.tests import loader

fu = loader.load_tool("fleet_util")
TOOL_PATH = os.path.abspath(os.path.join(loader.HERE, "..", "fleet_util"))

ALLOC = [
    {"metric": {"uuid": "GPU-aaa", "host": "nodea:9445"},
     "values": [[0, "101"], [3600, "101"]]},
    {"metric": {"uuid": "GPU-bbb", "host": "nodeb:9445"},
     "values": [[0, "102"]]},
]
SMACT = [
    {"metric": {"UUID": "GPU-aaa", "Hostname": "nodea"},
     "values": [[0, "0.05"], [3600, "0.95"]]},
]
SINFO = "nodea kempner_eng*\nnodea kempner_requeue\nnodeb kempner_eng\n"
STEP = 3600


def run_cli(*argv):
    return subprocess.run([sys.executable, TOOL_PATH] + list(argv),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True)


class TestSinfoMap(unittest.TestCase):
    def test_multi_partition_and_star(self):
        mapping = fu.parse_sinfo_map(SINFO)
        self.assertEqual(mapping["nodea"], ["kempner_eng", "kempner_requeue"])
        self.assertEqual(mapping["nodeb"], ["kempner_eng"])

    def test_empty(self):
        self.assertEqual(fu.parse_sinfo_map(""), {})


class TestUtilization(unittest.TestCase):
    def result(self, threshold=10.0, allowed=None):
        return fu.utilization(ALLOC, SMACT, fu.parse_sinfo_map(SINFO),
                              threshold, STEP, allowed_nodes=allowed)

    def test_overall_counts(self):
        overall = self.result()["overall"]
        self.assertAlmostEqual(overall["allocated_h"], 3.0)
        self.assertAlmostEqual(overall["wasted_h"], 1.0)   # aaa@t0 sm=5%
        self.assertAlmostEqual(overall["unmatched_h"], 1.0)  # bbb no smact

    def test_threshold_boundary_not_wasted(self):
        # sm=0.05 -> 5.0%; threshold 5.0 means "below", so NOT wasted
        overall = self.result(threshold=5.0)["overall"]
        self.assertAlmostEqual(overall["wasted_h"], 0.0)

    def test_multi_partition_attribution(self):
        parts = self.result()["partitions"]
        self.assertAlmostEqual(parts["kempner_eng"]["allocated_h"], 3.0)
        self.assertAlmostEqual(parts["kempner_requeue"]["allocated_h"], 2.0)
        self.assertAlmostEqual(parts["kempner_requeue"]["wasted_h"], 1.0)

    def test_allowed_nodes_filter(self):
        overall = self.result(allowed={"nodea"})["overall"]
        self.assertAlmostEqual(overall["allocated_h"], 2.0)
        self.assertAlmostEqual(overall["unmatched_h"], 0.0)

    def test_worst_nodes(self):
        worst = self.result()["worst_nodes"]
        self.assertEqual(worst[0][0], "nodea")
        self.assertAlmostEqual(worst[0][1], 1.0)

    def test_unknown_node_partition(self):
        result = fu.utilization(ALLOC, SMACT, {}, 10.0, STEP)
        self.assertIn("(unknown)", result["partitions"])


class TestCli(unittest.TestCase):
    def bundle_path(self):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                          delete=False)
        self.addCleanup(os.unlink, tmp.name)
        json.dump({"alloc": ALLOC, "smact": SMACT, "sinfo": SINFO}, tmp)
        tmp.close()
        return tmp.name

    def test_report_exit_0(self):
        proc = run_cli("--from-json", self.bundle_path())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("kempner_eng", proc.stdout)
        self.assertIn("(overall)", proc.stdout)

    def test_json_output(self):
        proc = run_cli("--from-json", self.bundle_path(), "--json")
        payload = json.loads(proc.stdout)
        self.assertAlmostEqual(payload["result"]["overall"]["allocated_h"],
                               3.0)

    def test_unknown_partition_exits_3(self):
        proc = run_cli("--from-json", self.bundle_path(),
                       "--partition", "nope")
        self.assertEqual(proc.returncode, 3)


if __name__ == "__main__":
    unittest.main()
