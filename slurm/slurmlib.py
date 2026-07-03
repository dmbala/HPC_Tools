"""Shared helpers for the slurm/ reporting tools. Stdlib only."""
import datetime
import math
import subprocess
import sys
import time


def err_exit(tool, msg):
    sys.stderr.write("%s: error: %s\n" % (tool, msg))
    sys.exit(3)


def run(cmd, timeout):
    """Run a command; return stdout. RuntimeError on missing binary,
    timeout, or nonzero exit (the command's stderr is included)."""
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, universal_newlines=True,
                           timeout=timeout)
    except OSError:
        raise RuntimeError("command not found: %s" % cmd[0])
    except subprocess.TimeoutExpired:
        raise RuntimeError("command timed out: %s" % " ".join(cmd))
    if r.returncode != 0:
        raise RuntimeError("%s failed: %s" % (cmd[0], r.stderr.strip()))
    return r.stdout


def iso_epoch(text):
    """'YYYY-mm-ddTHH:MM:SS' -> local epoch int. Raises ValueError."""
    return int(time.mktime(datetime.datetime.strptime(
        text, "%Y-%m-%dT%H:%M:%S").timetuple()))


def epoch_iso(epoch):
    return datetime.datetime.fromtimestamp(epoch).strftime(
        "%Y-%m-%dT%H:%M:%S")


def add_window_args(parser):
    parser.add_argument("--days", type=int, default=7,
                        help="window length ending now (default 7)")
    parser.add_argument("--since", metavar="ISO",
                        help="window start, YYYY-mm-ddTHH:MM:SS")
    parser.add_argument("--until", metavar="ISO",
                        help="window end, YYYY-mm-ddTHH:MM:SS")


def parse_window(args, tool):
    """Same semantics as gpu/gpulib.parse_window. Exit 3 on bad input."""
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


def humanize(seconds):
    """45 -> '45s'; 150 -> '2m 30s'; 9015 -> '2h 30m'; 90000 -> '1d 1h'."""
    seconds = int(seconds)
    if seconds < 60:
        return "%ds" % seconds
    if seconds < 3600:
        return "%dm %ds" % (seconds // 60, seconds % 60)
    if seconds < 86400:
        return "%dh %dm" % (seconds // 3600, seconds % 3600 // 60)
    return "%dd %dh" % (seconds // 86400, seconds % 86400 // 3600)


def pctile(sorted_values, pct):
    """Nearest-rank percentile of a pre-sorted list; None when empty."""
    if not sorted_values:
        return None
    rank = max(1, int(math.ceil(pct / 100.0 * len(sorted_values))))
    return sorted_values[rank - 1]


def gpus_from_tres(tres):
    """GPU count from a TRES string ('gres/gpu=4', or typed variant)."""
    fallback = 0
    for item in (tres or "").split(","):
        if item.startswith("gres/gpu="):
            try:
                return int(item.split("=", 1)[1])
            except ValueError:
                return 0
        if item.startswith("gres/gpu:") and "=" in item:
            try:
                fallback = int(item.split("=", 1)[1])
            except ValueError:
                pass
    return fallback


def read_text(path, tool):
    """Read a replay file; err_exit (3) if unreadable."""
    try:
        with open(path) as fh:
            return fh.read()
    except OSError as exc:
        err_exit(tool, "cannot read replay file: %s" % exc)
