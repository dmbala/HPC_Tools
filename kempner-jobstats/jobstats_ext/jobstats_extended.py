#!/usr/bin/env python3
"""jobstats_extended -- a non-invasive superset of the stock ``jobstats`` that
folds extra DCGM profiling metrics (SM-active, SM-occupancy, tensor-pipe,
DRAM-active, power) INTO the jobstats blob, keyed per-GPU by ``minor_number``
exactly like the built-in ``gpu_utilization``.

It subclasses ``jobstats.Jobstats`` so it reuses all of jobstats'
sacct/Prometheus/serialization logic -- the only new code is the DCGM
augmentation and an optional report section.

Why a join is needed: the ``DCGM_FI_*`` series live in a different exporter than
``nvidia_gpu_duty_cycle`` and carry ``UUID`` (not ``minor_number``), and the
``gpu`` index != Slurm ``minor_number``. So we (1) map UUID -> (node, minor)
from the ``nvidia_gpu_jobId`` companion series -- the same job->GPU mapping
jobstats already uses -- then (2) query each DCGM metric by UUID and store it
back under ``minor_number`` so the new keys line up with ``gpu_utilization`` in
the blob.

The new keys are strictly additive: stock readers ignore them, so an extended
blob round-trips through existing consumers unchanged.

This assumes a DCGM-exporter setup where the duty-cycle exporter labels series
with lowercase ``uuid``/``minor_number``/``host`` and the DCGM exporter labels
them with ``UUID``; label casing/availability can vary by exporter version.

Usage (``jobstats``, ``output_formatters`` and ``config`` must be importable):
    jobstats_extended.py 18583067            # formatted report + DCGM section
    jobstats_extended.py --json 18583067     # extended blob as pretty JSON
    jobstats_extended.py -b 18583067         # JS1:<base64> blob (storage form)
    jobstats_extended.py --no-dcgm 18583067  # report/blob without DCGM metrics

Those three modules ship with the stock jobstats install (``jobstats.py``,
``output_formatters.py``, ``config.py``). Python only imports them if their
directory is on the module search path, so running this script from your own
local directory would otherwise fail with
``ModuleNotFoundError: No module named 'jobstats'``.

To make that work without any setup, the block below auto-discovers the install
dir and adds it to ``sys.path`` before importing. It checks, in order:
  1. ``$JOBSTATS_HOME`` (set this to override, e.g. a non-standard install),
  2. the directory of any ``jobstats`` launcher found on ``$PATH``,
  3. the common locations ``/usr/local/bin``, ``/usr/local/jobstats``,
     ``/opt/jobstats``.
If none of those contain the modules, it prints exactly where it looked and how
to fix it, then exits non-zero -- so the failure is actionable instead of a
bare ImportError traceback.

You can still bypass discovery the usual ways: prepend the install dir to
PYTHONPATH (``PYTHONPATH=/path/to/jobstats:$PYTHONPATH python3
jobstats_extended.py 18583067``; ``$PYTHONPATH`` preserves any paths you already
had), export that in ``~/.bashrc`` to make every invocation work, or ``cd``
into the install dir before running.

Author: Bala Desinghu, Senior AI/HPC Research Computing Engineer,
        Kempner Institute, Harvard
"""

import os
import sys
import shutil
import argparse

import requests


# --- locate the stock jobstats install -------------------------------------
# jobstats.py / output_formatters.py / config.py are not on the default search
# path when this script runs from a local dir, so find their directory and put
# it on sys.path before importing. See the module docstring for the search order.
_TARGET_MODULES = ("jobstats", "output_formatters", "config")


def _jobstats_search_dirs():
    """Candidate dirs that may hold the stock jobstats modules, in priority order.

    $JOBSTATS_HOME first (explicit override), then the dir of any ``jobstats``
    launcher on $PATH, then the usual install locations. Only existing dirs are
    returned, de-duplicated, preserving order.
    """
    candidates = [
        os.environ.get("JOBSTATS_HOME"),
        shutil.which("jobstats") and os.path.dirname(os.path.realpath(shutil.which("jobstats"))),
        "/usr/local/bin",
        "/usr/local/jobstats",
        "/opt/jobstats",
    ]
    seen, dirs = set(), []
    for d in candidates:
        if d and d not in seen and os.path.isdir(d):
            seen.add(d)
            dirs.append(d)
    return dirs


_SEARCH_DIRS = _jobstats_search_dirs()
# insert in reverse so the highest-priority dir ends up first on sys.path
for _d in reversed(_SEARCH_DIRS):
    if _d not in sys.path:
        sys.path.insert(0, _d)

try:
    from jobstats import Jobstats
    from output_formatters import ClassicOutput
    from config import PROM_SERVER
except ImportError as exc:
    # Only intercept a missing jobstats module; let a missing transitive
    # dependency (e.g. requests inside config) surface its real traceback.
    if getattr(exc, "name", None) not in _TARGET_MODULES:
        raise
    prog = os.path.basename(sys.argv[0])
    sys.stderr.write(
        "ERROR: cannot import the stock jobstats modules "
        "(jobstats, output_formatters, config): {}\n\n".format(exc))
    sys.stderr.write("Looked for them in these directories:\n")
    for _d in (_SEARCH_DIRS or ["  (no candidate dirs exist on this host)"]):
        sys.stderr.write("  - {}\n".format(_d))
    sys.stderr.write(
        "\nNone of them contain jobstats.py / output_formatters.py / config.py.\n"
        "Point the script at your jobstats install via any one of:\n"
        "  JOBSTATS_HOME=/path/to/jobstats {0} <jobid>\n"
        "  PYTHONPATH=/path/to/jobstats:$PYTHONPATH {0} <jobid>\n"
        "  cd /path/to/jobstats && {0} <jobid>\n".format(prog))
    sys.exit(1)
# ---------------------------------------------------------------------------


# Extra metrics folded into the blob:
#   (blob_key, prometheus_metric, scale, label, unit)
# scale 100 turns DCGM 0-1 fractions into percent so jobstats' round(v, 1) keeps
# precision and they read like gpu_utilization (0-100). All are time-averaged
# over the job window, like the duty cycle. label/unit are only used by the
# human-readable report.
DCGM_EXTRA = [
    ("gpu_sm_active",     "DCGM_FI_PROF_SM_ACTIVE",          100, "SM active",     "%"),
    ("gpu_sm_occupancy",  "DCGM_FI_PROF_SM_OCCUPANCY",       100, "SM occupancy",  "%"),
    ("gpu_tensor_active", "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE", 100, "Tensor active", "%"),
    ("gpu_dram_active",   "DCGM_FI_PROF_DRAM_ACTIVE",        100, "DRAM active",   "%"),
    ("gpu_power_usage",   "DCGM_FI_DEV_POWER_USAGE",         1,   "Power usage",   "W"),
]


class JobstatsExtended(Jobstats):
    """jobstats.Jobstats + DCGM profiling metrics folded into the blob."""

    def __init__(self, *args, dcgm=True, **kwargs):
        self.dcgm_added = []
        super().__init__(*args, **kwargs)
        # sp_node is populated by the parent (from the cached blob or a fresh
        # Prometheus pass); augment it afterwards, only for GPU jobs that ran
        # long enough for the parent to have collected anything.
        if dcgm and getattr(self, "gpus", 0) and self.sp_node:
            try:
                self._augment_with_dcgm()
            except Exception as exc:
                self.debug_print("DCGM augmentation skipped: %s" % exc)

    def _prom(self, query):
        """Instant query at the job end time; returns the result list (or [])."""
        resp = requests.get(self.prom_server + "/api/v1/query",
                            params={"query": query, "time": self.end},
                            timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") == "success":
            return payload["data"]["result"]
        return []

    def _augment_with_dcgm(self):
        cluster_sel = "slurm_cluster='%s'" % self.cluster if self.cluster else ""
        # 1) UUID -> (node, minor) for this job's GPUs. The UUID is not stored
        #    in the blob, so this companion query is always needed.
        discovery = self._prom(
            "max_over_time((nvidia_gpu_jobId{%s} == %s)[%ds:])"
            % (cluster_sel, self.jobidraw, self.diff))
        uuid_to = {}
        for series in discovery:
            labels = series["metric"]
            uuid = labels.get("uuid")
            if uuid and labels.get("minor_number") is not None and "host" in labels:
                uuid_to[uuid] = (labels["host"].split(":")[0], labels["minor_number"])
        if not uuid_to:
            return
        # UUIDs are hex + hyphen, so they are safe to drop into an RE2 alternation.
        uuid_rx = "^(" + "|".join(uuid_to) + ")$"
        # 2) each DCGM metric, joined on UUID, stored back under minor_number.
        for key, metric, scale, _label, _unit in DCGM_EXTRA:
            result = self._prom(
                'avg_over_time((%s{UUID=~"%s"})[%ds:])' % (metric, uuid_rx, self.diff))
            # A UUID is unique to one physical GPU, but a metric can come back as
            # more than one series (e.g. replicated scrapers). Average the
            # duplicates so the stored value is deterministic rather than
            # last-writer-wins.
            accumulated = {}  # (node, minor) -> [sum, count]
            for series in result:
                node_minor = uuid_to.get(series["metric"].get("UUID"))
                if not node_minor:
                    continue
                try:
                    value = float(series["value"][1]) * scale
                except (TypeError, ValueError):
                    continue
                bucket = accumulated.setdefault(node_minor, [0.0, 0])
                bucket[0] += value
                bucket[1] += 1
            for (node, minor), (total, count) in accumulated.items():
                self.sp_node.setdefault(node, {}).setdefault(key, {})[minor] = \
                    round(total / count, 1)
            if accumulated:
                self.dcgm_added.append(key)


def _minor_sort_key(minor):
    """Sort numeric GPU minors numerically; fall back to string for the rest."""
    text = str(minor)
    return (0, int(text), "") if text.isdigit() else (1, 0, text)


def dcgm_section(js, fmt):
    """Render the DCGM metrics folded into ``js.sp_node`` as a report section in
    the ClassicOutput style. ``fmt`` is the ClassicOutput instance already used
    for the main report (its txt_* colors and draw_meter are reused). Returns ""
    when no DCGM keys are present."""
    nodes = sorted(js.sp_node)
    if not any(key in js.sp_node[nd] for nd in nodes for key, *_ in DCGM_EXTRA):
        return ""
    width, gutter = fmt.width, "  "
    bold, normal = fmt.txt_bold, fmt.txt_normal
    out = "\n"
    out += f"{bold}GPU Profiling Metrics (DCGM){normal}".center(width) + "\n"
    out += width * "=" + "\n"
    out += f"{gutter}Averaged over the run time. SM/tensor/DRAM are % busy; power is watts.\n\n"
    # overall: average each metric across every GPU on every node
    out += f"{gutter}Overall (averaged across GPUs)\n"
    for key, _metric, _scale, label, unit in DCGM_EXTRA:
        vals = [v for nd in nodes if key in js.sp_node[nd]
                for v in js.sp_node[nd][key].values()]
        if not vals:
            continue
        avg = sum(vals) / len(vals)
        if unit == "%":
            out += f"{gutter}  {label:<14}{fmt.draw_meter(int(round(avg)), 'dcgm')}\n"
        else:
            out += f"{gutter}  {label:<14}{avg:.1f}{unit} (avg per GPU)\n"
    # per node / GPU detail
    out += f"\n{gutter}Per node / GPU\n"
    for nd in nodes:
        d = js.sp_node[nd]
        minors = set()
        for key, *_ in DCGM_EXTRA:
            if key in d:
                minors.update(d[key])
        for minor in sorted(minors, key=_minor_sort_key):
            parts = [f"{label}={d[key][minor]}{unit}"
                     for key, _metric, _scale, label, unit in DCGM_EXTRA
                     if key in d and minor in d[key]]
            out += f"{gutter}    {nd} (GPU {minor}): " + "  ".join(parts) + "\n"
    return out


def main():
    parser = argparse.ArgumentParser(
        description="jobstats with extra DCGM profiling metrics "
                    "(formatted report by default)")
    parser.add_argument("jobids", nargs="+", help="one or more Slurm job IDs")
    parser.add_argument("-j", "--json", action="store_true",
                        help="emit the extended blob as pretty JSON instead of "
                             "the formatted report")
    parser.add_argument("-b", "--base64", action="store_true",
                        help="emit the JS1:<base64 gzip JSON> blob (the "
                             "AdminComment storage form)")
    parser.add_argument("--no-dcgm", dest="dcgm", action="store_false",
                        help="skip the DCGM augmentation (report/blob without "
                             "the extra metrics)")
    parser.add_argument("-f", "--force", action="store_true",
                        help="force recompute from Prometheus instead of the "
                             "cached blob")
    parser.add_argument("-c", "--cluster", default=None,
                        help="cluster (default: from sacct)")
    parser.add_argument("-n", "--no-color", action="store_true",
                        help="render the report without colorization")
    parser.add_argument("-d", "--debug", action="store_true", help="debug output")
    args = parser.parse_args()

    # JSON/base64 output is the raw blob, so let the parent behave like
    # `jobstats -j/-b`; the formatted report needs parse_stats(), so it passes
    # json_or_base64=False.
    raw = args.json or args.base64

    for jobid in args.jobids:
        try:
            js = JobstatsExtended(jobid, cluster=args.cluster,
                                  prom_server=PROM_SERVER, debug=args.debug,
                                  force_recalc=args.force, json_or_base64=raw,
                                  dcgm=args.dcgm)
        except SystemExit:
            # the parent's .error() exits; keep going for the other job ids
            continue
        if args.dcgm:
            print("# job %s: added %s"
                  % (jobid, ", ".join(js.dcgm_added) or "(none)"),
                  file=sys.stderr)
        if args.base64:
            blob = js.report_job_json(True)
            # report_job_json returns the bare base64 payload, or the sentinels
            # 'Short'/'None' when there is nothing to encode. Only a real
            # payload gets the JS1: storage prefix.
            print(blob if blob in ("Short", "None") else "JS1:" + blob)
        elif args.json:
            print(js.report_job_json(False))
        else:
            fmt = ClassicOutput(js)
            report = fmt.output(no_color=args.no_color)
            if args.dcgm:
                report += dcgm_section(js, fmt)
            print(report)


if __name__ == "__main__":
    main()
