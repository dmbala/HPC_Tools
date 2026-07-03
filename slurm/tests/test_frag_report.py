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

    def test_zero_shape_exits_3(self):
        proc = run_cli("--from-file",
                       os.path.join(loader.FIXTURES, "scontrol_nodes.txt"),
                       "--cpus-per-gpu", "0")
        self.assertEqual(proc.returncode, 3)
        self.assertIn("frag_report: error:", proc.stderr)


if __name__ == "__main__":
    unittest.main()
