"""Tests for fabric/ib_bw_pair (parsing/verdict; Slurm path is manual)."""
import unittest

from fabric.tests import loader

bw = loader.load_tool("ib_bw_pair")


class TestParsePerftest(unittest.TestCase):
    def test_parses_largest_row_gbits(self):
        result = bw.parse_perftest(loader.fixture("ib_write_bw.out"))
        self.assertEqual(result["bytes"], 8388608)
        self.assertAlmostEqual(result["bw_avg_gbps"], 389.20)
        self.assertAlmostEqual(result["bw_peak_gbps"], 389.45)
        self.assertAlmostEqual(result["msg_rate_mpps"], 0.0058)

    def test_mb_per_sec_converted(self):
        text = loader.fixture("ib_write_bw.out").replace(
            "[Gb/sec]", "[MB/sec]")
        result = bw.parse_perftest(text)
        # 389.20 MB/s * 8 / 1000 = 3.1136 Gb/s
        self.assertAlmostEqual(result["bw_avg_gbps"], 3.1136, places=3)

    def test_truncated_is_none(self):
        text = "\n".join(loader.fixture("ib_write_bw.out").splitlines()[:13])
        self.assertIsNone(bw.parse_perftest(text))
        self.assertIsNone(bw.parse_perftest(""))


class TestRateGbps(unittest.TestCase):
    def test_rate_line(self):
        self.assertEqual(bw._rate_gbps("400 Gb/sec (4X NDR)"), 400.0)
        self.assertIsNone(bw._rate_gbps(""))


class TestVerdict(unittest.TestCase):
    def test_explicit_threshold(self):
        self.assertEqual(bw.verdict(389.2, 350.0, None)[0], "PASS")
        self.assertEqual(bw.verdict(300.0, 350.0, None)[0], "FAIL")

    def test_rate_fraction_default(self):
        # 0.8 * 400 = 320
        self.assertEqual(bw.verdict(389.2, None, 400.0)[0], "PASS")
        self.assertEqual(bw.verdict(300.0, None, 400.0)[0], "FAIL")

    def test_no_threshold_no_rate_is_report(self):
        status, reason = bw.verdict(389.2, None, None)
        self.assertEqual(status, "REPORT")
        self.assertIn("rate unavailable", reason)


if __name__ == "__main__":
    unittest.main()
