"""Tests for slurm/slurmlib.py (pure parts; run() is exercised live)."""
import argparse
import unittest

from slurm.tests import loader

sl = loader.load_tool("slurmlib.py")


def ns(days=7, since=None, until=None):
    return argparse.Namespace(days=days, since=since, until=until)


class TestHumanize(unittest.TestCase):
    def test_seconds(self):
        self.assertEqual(sl.humanize(45), "45s")

    def test_minutes(self):
        self.assertEqual(sl.humanize(150), "2m 30s")

    def test_hours(self):
        self.assertEqual(sl.humanize(9015), "2h 30m")

    def test_days(self):
        self.assertEqual(sl.humanize(90000), "1d 1h")


class TestPctile(unittest.TestCase):
    def test_p50_nearest_rank(self):
        self.assertEqual(sl.pctile([1, 2, 3, 4], 50), 2)

    def test_p90_of_ten(self):
        self.assertEqual(sl.pctile(list(range(1, 11)), 90), 9)

    def test_empty_is_none(self):
        self.assertIsNone(sl.pctile([], 50))


class TestGpusFromTres(unittest.TestCase):
    def test_plain_and_typed(self):
        self.assertEqual(sl.gpus_from_tres(
            "cpu=16,mem=250G,gres/gpu=1,gres/gpu:nvidia_h200=1"), 1)
        self.assertEqual(sl.gpus_from_tres("cpu=4,gres/gpu:h100=4"), 4)

    def test_no_gpu_is_zero(self):
        self.assertEqual(sl.gpus_from_tres("cpu=4,mem=16G"), 0)
        self.assertEqual(sl.gpus_from_tres(""), 0)


class TestParseWindow(unittest.TestCase):
    def test_days_default(self):
        start, end = sl.parse_window(ns(), "t")
        self.assertEqual(end - start, 7 * 86400)

    def test_since_until(self):
        start, end = sl.parse_window(
            ns(since="2026-07-01T00:00:00", until="2026-07-02T00:00:00"), "t")
        self.assertEqual(end - start, 86400)

    def test_bad_pair_exits_3(self):
        with self.assertRaises(SystemExit) as cm:
            sl.parse_window(ns(since="2026-07-01T00:00:00"), "t")
        self.assertEqual(cm.exception.code, 3)


if __name__ == "__main__":
    unittest.main()
