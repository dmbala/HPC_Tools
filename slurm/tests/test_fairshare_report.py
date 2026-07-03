"""Tests for slurm/fairshare_report."""
import json
import os
import subprocess
import sys
import unittest

from slurm.tests import loader

fr = loader.load_tool("fairshare_report")
TOOL_PATH = os.path.abspath(os.path.join(loader.HERE, "..",
                                         "fairshare_report"))


def run_cli(*argv):
    return subprocess.run([sys.executable, TOOL_PATH] + list(argv),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True)


class TestParseSshare(unittest.TestCase):
    def test_accounts_and_skips(self):
        accounts, skipped = fr.parse_sshare(loader.fixture("sshare.txt"))
        self.assertEqual([a["account"] for a in accounts],
                         ["alpha_lab", "beta_lab", "gamma_lab", "delta_lab"])
        self.assertEqual(skipped["root"], 1)
        self.assertEqual(skipped["user_rows"], 1)
        self.assertEqual(skipped["malformed"], 1)

    def test_floats(self):
        accounts, _ = fr.parse_sshare(loader.fixture("sshare.txt"))
        beta = accounts[1]
        self.assertAlmostEqual(beta["norm_shares"], 0.001492)
        self.assertAlmostEqual(beta["effectv_usage"], 0.01)

    def test_empty(self):
        accounts, skipped = fr.parse_sshare("")
        self.assertEqual(accounts, [])
        self.assertEqual(skipped,
                         {"root": 0, "user_rows": 0, "malformed": 0})


class TestRank(unittest.TestCase):
    def ranked(self):
        accounts, _ = fr.parse_sshare(loader.fixture("sshare.txt"))
        return fr.rank(accounts, top_n=2)

    def test_over_served_order(self):
        over = self.ranked()["over"]
        # gamma: 0.815/0.000818 ~ 996; beta: 0.01/0.001492 ~ 6.7
        self.assertEqual([a["account"] for a in over],
                         ["gamma_lab", "beta_lab"])
        self.assertGreater(over[0]["ratio"], 900)

    def test_under_and_zero(self):
        r = self.ranked()
        self.assertEqual([a["account"] for a in r["under"]],
                         ["beta_lab", "gamma_lab"])
        self.assertEqual(r["zero_usage"], 2)  # alpha (0.0) and delta (0)

    def test_counts(self):
        r = self.ranked()
        self.assertEqual(r["usable"], 4)


class TestCli(unittest.TestCase):
    def test_report_exit_0(self):
        proc = run_cli("--from-file",
                       os.path.join(loader.FIXTURES, "sshare.txt"))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("gamma_lab", proc.stdout)
        self.assertIn("point-in-time", proc.stdout)

    def test_json_output(self):
        proc = run_cli("--from-file",
                       os.path.join(loader.FIXTURES, "sshare.txt"), "--json")
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["skipped"]["user_rows"], 1)


if __name__ == "__main__":
    unittest.main()
