"""Tests for gpu/gpulib.py (pure parts; HTTP paths are exercised live)."""
import argparse
import json
import os
import tempfile
import unittest

from gpu.tests import loader

gl = loader.load_tool("gpulib.py")


def ns(days=7, since=None, until=None):
    return argparse.Namespace(days=days, since=since, until=until)


class TestSeriesUuid(unittest.TestCase):
    def test_uppercase(self):
        self.assertEqual(gl.series_uuid({"UUID": "GPU-a"}), "GPU-a")

    def test_lowercase(self):
        self.assertEqual(gl.series_uuid({"uuid": "GPU-b"}), "GPU-b")

    def test_absent(self):
        self.assertIsNone(gl.series_uuid({"host": "x"}))


class TestParseWindow(unittest.TestCase):
    def test_days_default(self):
        start, end = gl.parse_window(ns(), "t")
        self.assertEqual(end - start, 7 * 86400)

    def test_since_until(self):
        start, end = gl.parse_window(
            ns(since="2026-07-01T00:00:00", until="2026-07-02T00:00:00"), "t")
        self.assertEqual(end - start, 86400)

    def test_since_without_until_exits_3(self):
        with self.assertRaises(SystemExit) as cm:
            gl.parse_window(ns(since="2026-07-01T00:00:00"), "t")
        self.assertEqual(cm.exception.code, 3)

    def test_bad_iso_exits_3(self):
        with self.assertRaises(SystemExit) as cm:
            gl.parse_window(ns(since="yesterday", until="today"), "t")
        self.assertEqual(cm.exception.code, 3)

    def test_reversed_window_exits_3(self):
        with self.assertRaises(SystemExit) as cm:
            gl.parse_window(
                ns(since="2026-07-02T00:00:00", until="2026-07-01T00:00:00"),
                "t")
        self.assertEqual(cm.exception.code, 3)


class TestLoadBundle(unittest.TestCase):
    def test_valid_bundle(self):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                          delete=False)
        self.addCleanup(os.unlink, tmp.name)
        json.dump({"xid": []}, tmp)
        tmp.close()
        self.assertEqual(gl.load_bundle(tmp.name, "t"), {"xid": []})

    def test_missing_bundle_exits_3(self):
        with self.assertRaises(SystemExit) as cm:
            gl.load_bundle("/nonexistent/bundle.json", "t")
        self.assertEqual(cm.exception.code, 3)


class TestGetGuards(unittest.TestCase):
    def _fake_requests(self, body_text, status=200):
        import types
        fake = types.ModuleType("requests")

        class Resp(object):
            status_code = status

            def json(self):
                json_mod = __import__("json")
                return json_mod.loads(body_text)

        class RequestException(Exception):
            pass

        fake.RequestException = RequestException
        fake.get = lambda *a, **kw: Resp()
        return fake

    def test_non_json_body_raises_runtimeerror(self):
        sys_mod = __import__("sys")
        sys_mod.modules["requests"] = self._fake_requests("<html>oops")
        try:
            with self.assertRaises(RuntimeError) as cm:
                gl.query("http://x", "up")
            self.assertIn("non-JSON", str(cm.exception))
            self.assertNotIn("http://x", str(cm.exception))
        finally:
            del sys_mod.modules["requests"]

    def test_malformed_success_payload_raises_runtimeerror(self):
        sys_mod = __import__("sys")
        sys_mod.modules["requests"] = self._fake_requests(
            '{"status": "success"}')
        try:
            with self.assertRaises(RuntimeError) as cm:
                gl.query("http://x", "up")
            self.assertIn("malformed", str(cm.exception))
        finally:
            del sys_mod.modules["requests"]


if __name__ == "__main__":
    unittest.main()
