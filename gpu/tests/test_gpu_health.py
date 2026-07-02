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


class TestParseNvlink(unittest.TestCase):
    def test_healthy_capture(self):
        per_gpu = gh.parse_nvlink(fixture("healthy.xml.nvlink"))
        self.assertEqual(sorted(per_gpu), [0, 1])
        self.assertEqual(per_gpu[0], {"replay_errors": 0,
                                      "recovery_errors": 0,
                                      "crc_errors": 0})

    def test_sums_across_links(self):
        text = fixture("healthy.xml.nvlink").replace(
            "Link 1: CRC Errors: 0", "Link 1: CRC Errors: 7", 1)
        per_gpu = gh.parse_nvlink(text)
        self.assertEqual(per_gpu[0]["crc_errors"], 7)
        self.assertEqual(per_gpu[1]["crc_errors"], 0)

    def test_unsupported_output_is_empty(self):
        self.assertEqual(gh.parse_nvlink(""), {})
        self.assertEqual(
            gh.parse_nvlink("NVLink is not supported on this device\n"), {})


def healthy_gpu(**overrides):
    gpu = {
        "index": 0, "name": "NVIDIA H100", "serial": "1650000000001",
        "ecc_enabled": True,
        "volatile_correctable": 0, "volatile_uncorrectable": 0,
        "aggregate_uncorrectable": 0,
        "row_remap_pending": False, "row_remap_failure": False,
        "retired_pages_pending": None,
        "throttle": {"sw_power_cap": False, "sw_thermal": False,
                     "hw_thermal": False, "hw_slowdown": False},
        "temp_c": 40, "slowdown_temp_c": 92,
        "power_w": 100.0, "power_limit_w": 700.0,
        "pcie_replay": 0,
    }
    gpu.update(overrides)
    return gpu


NO_NVLINK_ERRORS = {"replay_errors": 0, "recovery_errors": 0, "crc_errors": 0}


class TestEvaluate(unittest.TestCase):
    def tiers(self, gpu, nvlink=NO_NVLINK_ERRORS):
        return {k: v[0] for k, v in gh.evaluate(gpu, nvlink).items()}

    def test_all_ok(self):
        self.assertEqual(self.tiers(healthy_gpu()),
                         {"ecc": gh.OK, "throttle": gh.OK,
                          "pcie": gh.OK, "nvlink": gh.OK})

    # --- FAIL rules ---
    def test_volatile_uncorrectable_fails(self):
        self.assertEqual(
            self.tiers(healthy_gpu(volatile_uncorrectable=1))["ecc"], gh.FAIL)

    def test_row_remap_pending_fails(self):
        self.assertEqual(
            self.tiers(healthy_gpu(row_remap_pending=True))["ecc"], gh.FAIL)

    def test_row_remap_failure_fails(self):
        self.assertEqual(
            self.tiers(healthy_gpu(row_remap_failure=True))["ecc"], gh.FAIL)

    def test_retired_pages_pending_fails(self):
        self.assertEqual(
            self.tiers(healthy_gpu(retired_pages_pending=True))["ecc"], gh.FAIL)

    def test_hw_slowdown_fails(self):
        gpu = healthy_gpu()
        gpu["throttle"]["hw_slowdown"] = True
        self.assertEqual(self.tiers(gpu)["throttle"], gh.FAIL)

    # --- WARN rules ---
    def test_aggregate_uncorrectable_warns(self):
        self.assertEqual(
            self.tiers(healthy_gpu(aggregate_uncorrectable=2))["ecc"], gh.WARN)

    def test_correctable_over_threshold_warns(self):
        gpu = healthy_gpu(volatile_correctable=gh.CORRECTABLE_ECC_WARN + 1)
        self.assertEqual(self.tiers(gpu)["ecc"], gh.WARN)

    def test_correctable_at_threshold_ok(self):
        gpu = healthy_gpu(volatile_correctable=gh.CORRECTABLE_ECC_WARN)
        self.assertEqual(self.tiers(gpu)["ecc"], gh.OK)

    def test_power_cap_throttle_warns(self):
        gpu = healthy_gpu()
        gpu["throttle"]["sw_power_cap"] = True
        self.assertEqual(self.tiers(gpu)["throttle"], gh.WARN)

    def test_thermal_throttle_warns(self):
        gpu = healthy_gpu()
        gpu["throttle"]["sw_thermal"] = True
        self.assertEqual(self.tiers(gpu)["throttle"], gh.WARN)

    def test_temp_near_slowdown_warns(self):
        gpu = healthy_gpu(temp_c=92 - gh.TEMP_MARGIN_C, slowdown_temp_c=92)
        self.assertEqual(self.tiers(gpu)["throttle"], gh.WARN)

    def test_temp_below_margin_ok(self):
        gpu = healthy_gpu(temp_c=92 - gh.TEMP_MARGIN_C - 1, slowdown_temp_c=92)
        self.assertEqual(self.tiers(gpu)["throttle"], gh.OK)

    def test_pcie_replay_warns(self):
        self.assertEqual(self.tiers(healthy_gpu(pcie_replay=4))["pcie"], gh.WARN)

    def test_nvlink_errors_warn(self):
        tiers = self.tiers(healthy_gpu(), nvlink={"crc_errors": 3})
        self.assertEqual(tiers["nvlink"], gh.WARN)

    # --- n/a rules: absent capability never affects the verdict ---
    def test_ecc_disabled_is_na(self):
        gpu = healthy_gpu(ecc_enabled=False, volatile_correctable=None,
                          volatile_uncorrectable=None,
                          aggregate_uncorrectable=None,
                          row_remap_pending=None, row_remap_failure=None)
        self.assertEqual(self.tiers(gpu)["ecc"], gh.NA)

    def test_ecc_unreported_is_na(self):
        gpu = healthy_gpu(ecc_enabled=None, volatile_correctable=None,
                          volatile_uncorrectable=None,
                          aggregate_uncorrectable=None,
                          row_remap_pending=None, row_remap_failure=None,
                          retired_pages_pending=None)
        self.assertEqual(self.tiers(gpu)["ecc"], gh.NA)

    def test_no_nvlink_is_na(self):
        self.assertEqual(self.tiers(healthy_gpu(), nvlink=None)["nvlink"], gh.NA)

    def test_no_throttle_data_is_na(self):
        gpu = healthy_gpu(throttle=None, temp_c=None, slowdown_temp_c=None)
        self.assertEqual(self.tiers(gpu)["throttle"], gh.NA)

    def test_pcie_unreported_is_na(self):
        self.assertEqual(self.tiers(healthy_gpu(pcie_replay=None))["pcie"], gh.NA)


class TestWorst(unittest.TestCase):
    def test_ordering(self):
        self.assertEqual(gh.worst([gh.OK, gh.WARN, gh.FAIL]), gh.FAIL)
        self.assertEqual(gh.worst([gh.OK, gh.WARN]), gh.WARN)
        self.assertEqual(gh.worst([gh.OK, gh.OK]), gh.OK)

    def test_na_never_worsens(self):
        self.assertEqual(gh.worst([gh.NA, gh.OK]), gh.OK)
        self.assertEqual(gh.worst([gh.NA]), gh.OK)
        self.assertEqual(gh.worst([]), gh.OK)


class TestBuildResult(unittest.TestCase):
    def build_healthy(self):
        parsed = gh.parse_smi_xml(fixture("healthy.xml"))
        nvlink = gh.parse_nvlink(fixture("healthy.xml.nvlink"))
        return gh.build_result(parsed, nvlink, host="testhost",
                               timestamp="2026-07-02T10:00:00")

    def test_healthy_snapshot(self):
        result = self.build_healthy()
        self.assertEqual(result["host"], "testhost")
        self.assertEqual(result["timestamp"], "2026-07-02T10:00:00")
        self.assertEqual(result["driver_version"], "550.54.15")
        self.assertEqual(result["verdict"], gh.OK)
        self.assertEqual(len(result["gpus"]), 2)
        g0 = result["gpus"][0]
        self.assertEqual(g0["verdict"], gh.OK)
        self.assertEqual(sorted(g0["checks"]),
                         ["ecc", "nvlink", "pcie", "throttle"])
        self.assertEqual(g0["checks"]["ecc"]["row_remap"], "none")
        self.assertEqual(g0["checks"]["nvlink"]["errors"], {})
        self.assertEqual(g0["checks"]["throttle"]["active"], [])

    def test_node_verdict_is_worst_gpu(self):
        xml = fixture("healthy.xml")
        # doctor GPU 1 (second occurrence) to a FAIL: volatile uncorrectable
        first = xml.find("<dram_uncorrectable>0</dram_uncorrectable>")
        second = xml.find("<dram_uncorrectable>0</dram_uncorrectable>",
                          first + 1)
        # occurrences per GPU: volatile then aggregate; GPU 1 volatile is 3rd
        third = xml.find("<dram_uncorrectable>0</dram_uncorrectable>",
                         second + 1)
        xml = (xml[:third]
               + "<dram_uncorrectable>5</dram_uncorrectable>"
               + xml[third + len("<dram_uncorrectable>0</dram_uncorrectable>"):])
        parsed = gh.parse_smi_xml(xml)
        result = gh.build_result(parsed, None, host="h", timestamp="t")
        self.assertEqual(result["gpus"][0]["verdict"], gh.OK)
        self.assertEqual(result["gpus"][1]["verdict"], gh.FAIL)
        self.assertEqual(result["verdict"], gh.FAIL)

    def test_nvlink_none_is_na_everywhere(self):
        parsed = gh.parse_smi_xml(fixture("healthy.xml"))
        result = gh.build_result(parsed, None, host="h", timestamp="t")
        for gpu in result["gpus"]:
            self.assertEqual(gpu["checks"]["nvlink"]["status"], gh.NA)
        self.assertEqual(result["verdict"], gh.OK)  # n/a never worsens

    def test_row_remap_states(self):
        parsed = gh.parse_smi_xml(fixture("healthy.xml").replace(
            "<remapped_row_pending>No</remapped_row_pending>",
            "<remapped_row_pending>Yes</remapped_row_pending>", 1))
        result = gh.build_result(parsed, None, host="h", timestamp="t")
        self.assertEqual(result["gpus"][0]["checks"]["ecc"]["row_remap"],
                         "pending")
        self.assertEqual(result["gpus"][0]["verdict"], gh.FAIL)


if __name__ == "__main__":
    unittest.main()
