"""Tests for fabric/fleet_snapshot (aggregator core; CLI tested in Task 2)."""
import json
import unittest

from fabric.tests import loader

fs = loader.load_tool("fleet_snapshot")


def snap(name):
    return json.loads(loader.fixture(name))


def clean_pair():
    return {"nodea": snap("snapshot_nodea.json"),
            "nodeb": snap("snapshot_nodeb.json")}


class TestRateGbps(unittest.TestCase):
    def test_parses_leading_number(self):
        self.assertEqual(fs._rate_gbps("400 Gb/sec (4X NDR)"), 400.0)
        self.assertEqual(fs._rate_gbps("100 Gb/sec (4X EDR)"), 100.0)

    def test_none_on_garbage(self):
        self.assertIsNone(fs._rate_gbps(""))
        self.assertIsNone(fs._rate_gbps(None))
        self.assertIsNone(fs._rate_gbps("unknown"))


class TestAggregate(unittest.TestCase):
    def test_clean_fleet(self):
        summary = fs.aggregate(clean_pair())
        self.assertTrue(summary["clean"])
        self.assertEqual(summary["nodes_total"], 2)
        self.assertEqual(summary["nodes_reached"], 2)
        self.assertEqual(summary["unreached"], [])
        self.assertEqual(summary["version_drift"], [])
        self.assertEqual(summary["link_anomalies"], [])
        self.assertEqual(summary["error_counters"], [])
        self.assertEqual(summary["unavailable"], [])

    def test_unreached_node_flagged(self):
        snaps = clean_pair()
        snaps["nodec"] = None
        summary = fs.aggregate(snaps)
        self.assertFalse(summary["clean"])
        self.assertEqual(summary["unreached"], ["nodec"])
        self.assertEqual(summary["nodes_reached"], 2)

    def test_driver_drift_flagged(self):
        snaps = clean_pair()
        snaps["nodeb"]["system"]["nvidia_driver"] = "550.54.15"
        summary = fs.aggregate(snaps)
        self.assertFalse(summary["clean"])
        self.assertTrue(any("nodeb" in d and "driver" in d
                            for d in summary["version_drift"]), summary)

    def test_fw_drift_flagged(self):
        snaps = clean_pair()
        snaps["nodeb"]["ib"]["hcas"][0]["ibv_devinfo"] = (
            "hca_id:\tmlx5_0\n\tfw_ver:\t28.36.1010\n")
        summary = fs.aggregate(snaps)
        self.assertTrue(any("fw" in d for d in summary["version_drift"]), summary)

    def test_down_port_flagged(self):
        snaps = clean_pair()
        port = snaps["nodeb"]["ib"]["hcas"][0]["ports"][0]
        port["state"] = "1: DOWN"
        port["phys_state"] = "3: Disabled"
        summary = fs.aggregate(snaps)
        self.assertFalse(summary["clean"])
        self.assertTrue(any("nodeb" in a and "state" in a
                            for a in summary["link_anomalies"]), summary)

    def test_degraded_rate_flagged(self):
        snaps = clean_pair()
        snaps["nodeb"]["ib"]["hcas"][0]["ports"][0]["rate"] = \
            "200 Gb/sec (4X HDR)"
        summary = fs.aggregate(snaps)
        self.assertTrue(any("rate" in a and "nodeb" in a
                            for a in summary["link_anomalies"]), summary)

    def test_error_counter_flagged(self):
        snaps = clean_pair()
        snaps["nodea"]["ib"]["hcas"][0]["ports"][0]["counters"]["symbol_error"] = 12
        summary = fs.aggregate(snaps)
        self.assertFalse(summary["clean"])
        self.assertTrue(any("symbol_error=12" in e and "nodea" in e
                            for e in summary["error_counters"]), summary)

    def test_traffic_counters_ignored(self):
        # port_xmit_data differs between the fixtures and is NOT an error counter
        summary = fs.aggregate(clean_pair())
        self.assertEqual(summary["error_counters"], [])

    def test_missing_ib_section_noted(self):
        snaps = clean_pair()
        snaps["nodeb"]["ib"] = {"error": "no /sys/class/infiniband", "hcas": []}
        summary = fs.aggregate(snaps)
        self.assertFalse(summary["clean"])
        self.assertTrue(any("nodeb" in u for u in summary["unavailable"]), summary)


class TestRenderFleetText(unittest.TestCase):
    def test_clean_report(self):
        text = fs.render_fleet_text(fs.aggregate(clean_pair()))
        self.assertIn("fleet verdict: CLEAN", text)
        self.assertIn("2/2", text)

    def test_anomalous_report_lists_findings(self):
        snaps = clean_pair()
        snaps["nodeb"]["ib"]["hcas"][0]["ports"][0]["counters"]["symbol_error"] = 3
        text = fs.render_fleet_text(fs.aggregate(snaps))
        self.assertIn("fleet verdict: ANOMALIES", text)
        self.assertIn("symbol_error=3", text)


if __name__ == "__main__":
    unittest.main()
