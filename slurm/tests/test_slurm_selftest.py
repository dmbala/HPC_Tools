"""Tests for slurm/slurm_selftest."""
import os
import subprocess
import sys
import tempfile
import unittest

from slurm.tests import loader

st = loader.load_tool("slurm_selftest")
TOOL_PATH = os.path.abspath(os.path.join(loader.HERE, "..",
                                         "slurm_selftest"))

SAMPLE = """some slurm noise
SELFTEST env PASS propagated to 2/2 tasks
SELFTEST placement SKIP no --node-pattern given
SELFTEST gpu_grant FAIL no driver-visible gpu
SELFTEST oom_enforcement PASS step killed (rc=137)
trailing noise
"""


def run_cli(*argv):
    return subprocess.run([sys.executable, TOOL_PATH] + list(argv),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True)


def sample_file(text):
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".out", delete=False)
    tmp.write(text)
    tmp.close()
    return tmp.name


class TestParseMarkers(unittest.TestCase):
    def test_parses_statuses_and_detail(self):
        markers = st.parse_markers(SAMPLE)
        self.assertEqual(len(markers), 4)
        self.assertEqual(markers[0],
                         {"name": "env", "status": "PASS",
                          "detail": "propagated to 2/2 tasks"})
        self.assertEqual(markers[2]["status"], "FAIL")

    def test_malformed_lines_ignored(self):
        markers = st.parse_markers("SELFTEST bad\nSELFTEST x MAYBE y\n")
        self.assertEqual(markers, [])

    def test_empty(self):
        self.assertEqual(st.parse_markers(""), [])


class TestEvaluate(unittest.TestCase):
    def test_no_result_synthesized(self):
        results = st.evaluate(st.parse_markers(SAMPLE),
                              ["env", "gres_accounting"])
        self.assertEqual(results[0]["status"], "PASS")
        self.assertEqual(results[1],
                         {"name": "gres_accounting", "status": "NO-RESULT",
                          "detail": "no marker in job output"})

    def test_requested_order_preserved(self):
        results = st.evaluate(st.parse_markers(SAMPLE),
                              ["oom_enforcement", "env"])
        self.assertEqual([r["name"] for r in results],
                         ["oom_enforcement", "env"])

    def test_skip_kept(self):
        results = st.evaluate(st.parse_markers(SAMPLE), ["placement"])
        self.assertEqual(results[0]["status"], "SKIP")


class TestJobScript(unittest.TestCase):
    def test_all_checks_emit_markers(self):
        script = st.job_script(list(st.ALL_CHECKS), r"^holygpu\d+$", True)
        for name in st.ALL_CHECKS:
            self.assertIn("SELFTEST %s " % name, script)
        self.assertNotIn("%s", script)  # no format residue
        self.assertIn(st.ENV_TOKEN, script)

    def test_placement_skip_without_pattern(self):
        script = st.job_script(["placement"], None, False)
        self.assertIn("SELFTEST placement SKIP", script)

    def test_gpu_checks_skip_without_gpu(self):
        script = st.job_script(["gpu_grant", "gres_accounting"], None, False)
        self.assertIn("SELFTEST gpu_grant SKIP", script)
        self.assertIn("SELFTEST gres_accounting SKIP", script)


class TestCli(unittest.TestCase):
    def test_from_file_all_pass_exit_0(self):
        path = sample_file(
            "SELFTEST env PASS ok\nSELFTEST oom_enforcement PASS ok\n")
        self.addCleanup(os.unlink, path)
        proc = run_cli("--from-file", path, "--checks",
                       "env,oom_enforcement")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("PASS", proc.stdout)

    def test_from_file_fail_exit_1(self):
        path = sample_file(SAMPLE)
        self.addCleanup(os.unlink, path)
        proc = run_cli("--from-file", path)
        self.assertEqual(proc.returncode, 1, proc.stdout)
        self.assertIn("gpu_grant", proc.stdout)

    def test_missing_marker_exit_1(self):
        path = sample_file("SELFTEST env PASS ok\n")
        self.addCleanup(os.unlink, path)
        proc = run_cli("--from-file", path)
        self.assertEqual(proc.returncode, 1)
        self.assertIn("NO-RESULT", proc.stdout)

    def test_missing_file_exit_3(self):
        proc = run_cli("--from-file", "/nonexistent.out")
        self.assertEqual(proc.returncode, 3)
        self.assertIn("slurm_selftest: error:", proc.stderr)

    def test_live_mode_needs_account_partition(self):
        proc = run_cli()
        self.assertEqual(proc.returncode, 3)
        self.assertIn("slurm_selftest: error:", proc.stderr)

    def test_unknown_check_exit_3(self):
        path = sample_file(SAMPLE)
        self.addCleanup(os.unlink, path)
        proc = run_cli("--from-file", path, "--checks", "env,bogus")
        self.assertEqual(proc.returncode, 3)


if __name__ == "__main__":
    unittest.main()
