"""Shared helpers for the gpu/ Prometheus window/fleet tools.

Imports of the jobstats `config` module and `requests` are lazy (inside
functions) so this module -- and the tools' pure cores -- import cleanly on
machines without a jobstats install (e.g. in the offline test suite).
"""
import datetime
import json
import sys
import time

for _p in ("/usr/local/bin", "/usr/bin"):
    if _p not in sys.path:
        sys.path.append(_p)


def err_exit(tool, msg):
    sys.stderr.write("%s: error: %s\n" % (tool, msg))
    sys.exit(3)


def load_prom_server(tool):
    """PROM_SERVER from the jobstats config; err_exit if unavailable."""
    try:
        from config import PROM_SERVER
        return PROM_SERVER
    except Exception as exc:
        err_exit(tool, "cannot import jobstats config (PROM_SERVER): %s -- "
                 "run where the jobstats command lives" % exc)


def _get(prom, path, params, timeout, kind):
    """Shared HTTP plumbing. Never lets the endpoint URL into an error."""
    import requests
    try:
        resp = requests.get(prom + path, params=params, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError("prometheus %s failed: %s"
                           % (kind, exc.__class__.__name__))
    if resp.status_code != 200:
        raise RuntimeError("prometheus %s failed: HTTP %d"
                           % (kind, resp.status_code))
    try:
        payload = resp.json()
    except ValueError:
        raise RuntimeError("prometheus %s failed: non-JSON response" % kind)
    if not isinstance(payload, dict) or payload.get("status") != "success":
        raise RuntimeError("prometheus %s failed: %s"
                           % (kind, payload.get("error", "non-success")
                              if isinstance(payload, dict) else "bad payload"))
    try:
        return payload["data"]["result"]
    except (KeyError, TypeError):
        raise RuntimeError("prometheus %s failed: malformed response" % kind)


def query(prom, promql, when=None, timeout=60):
    """Instant query -> result list. Raises RuntimeError on failure."""
    params = {"query": promql}
    if when is not None:
        params["time"] = when
    return _get(prom, "/api/v1/query", params, timeout, "query")


def query_range(prom, promql, start, end, step, timeout=120):
    """Range query -> matrix list. Raises RuntimeError on failure."""
    return _get(prom, "/api/v1/query_range",
                {"query": promql, "start": start, "end": end, "step": step},
                timeout, "range query")


def series_uuid(labels):
    """UUID from series labels, either casing (DCGM: UUID, exporter: uuid)."""
    return labels.get("UUID") or labels.get("uuid")


def iso_epoch(text):
    """'YYYY-mm-ddTHH:MM:SS' -> local-time epoch int. Raises ValueError."""
    return int(time.mktime(datetime.datetime.strptime(
        text, "%Y-%m-%dT%H:%M:%S").timetuple()))


def add_window_args(parser):
    parser.add_argument("--days", type=int, default=7,
                        help="window length ending now (default 7)")
    parser.add_argument("--since", metavar="ISO",
                        help="window start, YYYY-mm-ddTHH:MM:SS")
    parser.add_argument("--until", metavar="ISO",
                        help="window end, YYYY-mm-ddTHH:MM:SS")


def parse_window(args, tool):
    """(--days | --since+--until) -> (start_epoch, end_epoch); exit 3 on bad."""
    if args.since or args.until:
        if not (args.since and args.until):
            err_exit(tool, "--since and --until must be given together")
        try:
            start, end = iso_epoch(args.since), iso_epoch(args.until)
        except ValueError:
            err_exit(tool, "timestamps must be YYYY-mm-ddTHH:MM:SS")
        if end <= start:
            err_exit(tool, "--until must be after --since")
        return start, end
    end = int(time.time())
    return end - args.days * 86400, end


def load_bundle(path, tool):
    """Read a --from-json replay bundle; exit 3 if unreadable."""
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        err_exit(tool, "cannot read replay bundle: %s" % exc)
