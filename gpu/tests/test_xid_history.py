"""Tests for gpu/xid_history."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

from gpu.tests import loader

xh = loader.load_tool("xid_history")
TOOL_PATH = os.path.abspath(os.path.join(loader.HERE, "..", "xid_history"))


def matrix():
    return json.loads(loader.fixture("prom_xid.json"))


def run_cli(*argv):
    return subprocess.run([sys.executable, TOOL_PATH] + list(argv),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True)


class TestEpisodes(unittest.TestCase):
    def test_gap_splits_episode(self):
        eps = xh.episodes(matrix())
        self.assertEqual(len(eps), 3)  # 2 from series 1 (gap), 1 from series 2
        first = eps[0]
        self.assertEqual(first["first_seen"], 1751500000.0)
        self.assertEqual(first["last_seen"], 1751500300.0)
        self.assertEqual(first["err_code"], "79")
        self.assertEqual(first["node"], "holygpu8a10302.rc.fas.harvard.edu")

    def test_sorted_by_first_seen(self):
        eps = xh.episodes(matrix())
        seen = [e["first_seen"] for e in eps]
        self.assertEqual(seen, sorted(seen))

    def test_contiguous_stays_one_episode(self):
        m = [{"metric": {"Hostname": "n", "UUID": "GPU-c", "gpu": "1",
                         "err_code": "31", "err_msg": "m"},
              "values": [[0, "31"], [300, "31"], [600, "31"]]}]
        eps = xh.episodes(m)
        self.assertEqual(len(eps), 1)
        self.assertEqual(eps[0]["last_seen"], 600.0)

    def test_empty_matrix(self):
        self.assertEqual(xh.episodes([]), [])


class TestAttach(unittest.TestCase):
    def test_jobid_matched_on_uuid(self):
        eps = xh.episodes(matrix())
        by_ts = {"1751500000": [{"metric": {"uuid": "GPU-aaa"},
                                 "value": [1751500000, "27400001"]}]}
        xh.attach_jobids(eps, by_ts)
        self.assertEqual(eps[0]["jobid"], 27400001)
        self.assertIsNone(eps[1]["jobid"])

    def test_zero_jobid_means_unallocated(self):
        eps = xh.episodes(matrix())
        by_ts = {"1751500000": [{"metric": {"uuid": "GPU-aaa"},
                                 "value": [1751500000, "0"]}]}
        xh.attach_jobids(eps, by_ts)
        self.assertIsNone(eps[0]["jobid"])

    def test_sacct_fills_user_state(self):
        eps = xh.episodes(matrix())
        eps[0]["jobid"] = 27400001
        xh.attach_sacct(eps, "27400001|alice|kempner_lab|FAILED\n")
        self.assertEqual(eps[0]["user"], "alice")
        self.assertEqual(eps[0]["state"], "FAILED")
        self.assertIsNone(eps[1]["user"])


class TestRender(unittest.TestCase):
    def test_table_contains_fields(self):
        eps = xh.episodes(matrix())
        eps[0]["jobid"], eps[0]["user"], eps[0]["state"] = 27400001, "alice", \
            "FAILED"
        text = xh.render(eps, 1751490000, 1751510000)
        self.assertIn("79", text)
        self.assertIn("holygpu8a10302", text)
        self.assertIn("27400001", text)
        self.assertIn("alice", text)
        self.assertIn("3 episode(s) on 2 node(s)", text)

    def test_empty_report(self):
        text = xh.render([], 0, 1)
        self.assertIn("no XID errors", text)


class TestCli(unittest.TestCase):
    def bundle_path(self, xid):
        bundle = {"xid": xid,
                  "jobid": {"1751500000": [{"metric": {"uuid": "GPU-aaa"},
                                            "value": [0, "27400001"]}]},
                  "sacct": "27400001|alice|kempner_lab|FAILED\n"}
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                          delete=False)
        self.addCleanup(os.unlink, tmp.name)
        json.dump(bundle, tmp)
        tmp.close()
        return tmp.name

    def test_findings_exit_1(self):
        proc = run_cli("--from-json", self.bundle_path(matrix()))
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertIn("alice", proc.stdout)

    def test_clean_exit_0(self):
        proc = run_cli("--from-json", self.bundle_path([]))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("no XID errors", proc.stdout)

    def test_json_output(self):
        proc = run_cli("--from-json", self.bundle_path(matrix()), "--json")
        payload = json.loads(proc.stdout)
        self.assertEqual(len(payload["episodes"]), 3)

    def test_missing_bundle_exits_3(self):
        proc = run_cli("--from-json", "/nonexistent.json")
        self.assertEqual(proc.returncode, 3)
        self.assertIn("xid_history: error:", proc.stderr)

    def test_bad_window_exits_3(self):
        proc = run_cli("--since", "2026-07-01T00:00:00")
        self.assertEqual(proc.returncode, 3)


if __name__ == "__main__":
    unittest.main()
