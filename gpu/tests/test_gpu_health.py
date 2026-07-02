"""Tests for gpu/gpu_health. Offline: fixtures only, no GPU needed."""
import importlib.machinery
import importlib.util
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.abspath(os.path.join(HERE, "..", "gpu_health"))
FIXTURES = os.path.join(HERE, "fixtures")

# gpu_health has no .py extension, so load it by explicit path.
_loader = importlib.machinery.SourceFileLoader("gpu_health", TOOL)
_spec = importlib.util.spec_from_loader("gpu_health", _loader)
gh = importlib.util.module_from_spec(_spec)
_loader.exec_module(gh)


def fixture(name):
    with open(os.path.join(FIXTURES, name)) as fh:
        return fh.read()


class TestParseSmiXml(unittest.TestCase):
    def test_healthy_two_gpus(self):
        parsed = gh.parse_smi_xml(fixture("healthy.xml"))
        self.assertEqual(parsed["driver_version"], "550.54.15")
        self.assertEqual(len(parsed["gpus"]), 2)
        g0 = parsed["gpus"][0]
        self.assertEqual(g0["index"], 0)
        self.assertEqual(g0["name"], "NVIDIA H100 80GB HBM3")
        self.assertEqual(g0["serial"], "1650000000001")
        self.assertIs(g0["ecc_enabled"], True)
        self.assertEqual(g0["volatile_correctable"], 0)
        self.assertEqual(g0["volatile_uncorrectable"], 0)
        self.assertEqual(g0["aggregate_uncorrectable"], 0)
        self.assertIs(g0["row_remap_pending"], False)
        self.assertIs(g0["row_remap_failure"], False)
        self.assertIsNone(g0["retired_pages_pending"])
        self.assertEqual(g0["throttle"], {"sw_power_cap": False,
                                          "sw_thermal": False,
                                          "hw_thermal": False,
                                          "hw_slowdown": False})
        self.assertEqual(g0["temp_c"], 41)
        self.assertEqual(g0["slowdown_temp_c"], 92)
        self.assertEqual(g0["power_w"], 71.5)
        self.assertEqual(g0["power_limit_w"], 700.0)
        self.assertEqual(g0["pcie_replay"], 0)
        self.assertEqual(parsed["gpus"][1]["index"], 1)

    def test_doctored_counts_parse(self):
        xml = fixture("healthy.xml").replace(
            "<dram_uncorrectable>0</dram_uncorrectable>",
            "<dram_uncorrectable>3</dram_uncorrectable>", 1)
        g0 = gh.parse_smi_xml(xml)["gpus"][0]
        self.assertEqual(g0["volatile_uncorrectable"], 3)

    def test_ecc_disabled_fixture(self):
        g0 = gh.parse_smi_xml(fixture("ecc_disabled.xml"))["gpus"][0]
        self.assertIs(g0["ecc_enabled"], False)
        self.assertIsNone(g0["volatile_uncorrectable"])
        self.assertIsNone(g0["row_remap_pending"])
        self.assertIsNone(g0["serial"])          # serial is N/A
        self.assertIsNone(g0["slowdown_temp_c"])  # threshold is N/A
        self.assertEqual(g0["power_w"], 28.0)     # old power_readings tags
        self.assertEqual(g0["power_limit_w"], 450.0)

    def test_garbled_raises(self):
        with self.assertRaises(ValueError):
            gh.parse_smi_xml(fixture("garbled.xml"))

    def test_no_gpus_raises(self):
        with self.assertRaises(ValueError):
            gh.parse_smi_xml("<nvidia_smi_log></nvidia_smi_log>")


if __name__ == "__main__":
    unittest.main()
