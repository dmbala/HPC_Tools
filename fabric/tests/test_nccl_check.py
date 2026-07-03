"""Tests for fabric/nccl_check (parsing/verdict; Slurm paths are manual)."""
import os
import shutil
import stat
import tempfile
import unittest

from fabric.tests import loader

nc = loader.load_tool("nccl_check")


class TestParseNcclTable(unittest.TestCase):
    def test_parses_data_rows(self):
        rows = nc.parse_nccl_table(loader.fixture("nccl_allreduce.out"))
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0]["size"], 8)
        self.assertEqual(rows[-1]["size"], 1073741824)
        self.assertAlmostEqual(rows[-1]["oop_busbw"], 329.00)
        self.assertAlmostEqual(rows[-1]["ip_busbw"], 330.07)

    def test_truncated_output_is_empty(self):
        text = "\n".join(loader.fixture("nccl_allreduce.out").splitlines()[:8])
        self.assertEqual(nc.parse_nccl_table(text), [])

    def test_garbage_ignored(self):
        self.assertEqual(nc.parse_nccl_table("slurmstepd: error: oom\n"), [])


class TestSummarize(unittest.TestCase):
    def test_peak_and_large(self):
        rows = nc.parse_nccl_table(loader.fixture("nccl_allreduce.out"))
        summary = nc.summarize(rows)
        self.assertAlmostEqual(summary["peak_busbw"], 329.00)
        self.assertAlmostEqual(summary["large_busbw"], 329.00)
        self.assertEqual(summary["large_size"], 1073741824)


class TestFindBinary(unittest.TestCase):
    def _make_exe(self, directory, name="all_reduce_perf"):
        path = os.path.join(directory, name)
        with open(path, "w") as fh:
            fh.write("#!/bin/sh\n")
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
        return path

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_cli_arg_wins(self):
        exe = self._make_exe(self.tmp)
        self.assertEqual(nc.find_binary(cli=exe, env={}), exe)

    def test_cli_arg_missing_is_none(self):
        self.assertIsNone(
            nc.find_binary(cli=os.path.join(self.tmp, "nope"), env={}))

    def test_env_var(self):
        exe = self._make_exe(self.tmp)
        self.assertEqual(nc.find_binary(env={"NCCL_TESTS_BIN": exe}), exe)

    def test_not_found_returns_none(self):
        self.assertIsNone(nc.find_binary(env={"PATH": self.tmp}))


class TestCounterLines(unittest.TestCase):
    def test_nonzero_rc_reported(self):
        text = "COUNTER_DELTA nodea rc=0\nCOUNTER_DELTA nodeb rc=1\n"
        self.assertEqual(nc.parse_counter_lines(text), ["nodeb"])

    def test_absent_lines_empty(self):
        self.assertEqual(nc.parse_counter_lines("no markers here"), [])


class TestJobScript(unittest.TestCase):
    def test_counters_script_interpolates_literal_paths(self):
        script = nc._job_script("/x/all_reduce_perf", 4, True,
                                "/repo", "/repo/results/run1")
        self.assertIn("/repo/results/run1/counters/before_$(hostname -s).json",
                      script)
        self.assertIn("/repo/results/run1/counters/after_$(hostname -s).json",
                      script)
        self.assertIn("for b in /repo/results/run1/counters/before_*.json",
                      script)
        self.assertNotIn("$SNAPDIR", script)
        self.assertNotIn("%s", script)
        self.assertIn("h=${h%.json}", script)

    def test_no_counters_script_minimal(self):
        script = nc._job_script("/x/all_reduce_perf", 4, False, "/repo", "/o")
        self.assertNotIn("ib_snapshot", script)
        self.assertIn("--ntasks-per-node=4 --gpus-per-node=4", script)
        self.assertNotIn("%s", script)


if __name__ == "__main__":
    unittest.main()
