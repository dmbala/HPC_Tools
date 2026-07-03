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

    def test_save_golden_bare_filename(self):
        c = self.path_for(snap(), "current.json")
        proc = subprocess.run([sys.executable, TOOL_PATH, "--save-golden",
                               "--current", c, "--golden", "bare_golden.json"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True, cwd=self.tmp)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(os.path.exists(os.path.join(self.tmp,
                                                    "bare_golden.json")))


if __name__ == "__main__":
    unittest.main()
