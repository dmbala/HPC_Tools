#!/usr/bin/python3
# jobstats_extended - a LOCAL, non-invasive superset of the stock `jobstats`
# that folds extra DCGM profiling metrics (SM-active, SM-occupancy, tensor-pipe,
# DRAM-active, power) INTO the jobstats blob, keyed per-GPU by minor_number
# exactly like the built-in gpu_utilization. Subclasses jobstats.Jobstats, so it
# reuses all of jobstats' sacct/Prometheus/serialization logic -- the only new
# code is the DCGM augmentation. Lets you test the extended blob end-to-end
# before deciding whether to swap it into the root-level install cluster-wide.
#
# Why a join is needed: the DCGM_FI_* series live in a different exporter than
# nvidia_gpu_duty_cycle and carry `UUID`/`gpu` (not `minor_number`), and the
# `gpu` index != slurm `minor_number`. So we (1) map UUID -> (node, minor) from
# the nvidia_gpu_jobId companion -- the same job->GPU mapping jobstats uses --
# then (2) query each DCGM metric by UUID and store it back under minor_number,
# so the new keys align with gpu_utilization in the blob.
#
# Usage:
#   ./jobstats_extended.py 18583067            # extended blob as pretty JSON
#   ./jobstats_extended.py -b 18583067         # JS1:<base64> blob (storage form)
#   ./jobstats_extended.py --no-dcgm 18583067  # behave like stock jobstats -j
#
# Author: Bala Desinghu, Senior AI/HPC Research Computing Engineer, Kempner Institute, Harvard

import sys, json, argparse

for _p in ("/usr/local/bin", "/usr/bin"):
  if _p not in sys.path:
    sys.path.append(_p)
from config import PROM_SERVER
try:
  from config import SAMPLING_PERIOD
except Exception:
  SAMPLING_PERIOD = 60
from jobstats import Jobstats
import requests

# Extra metrics folded into the blob: (blob_key, prometheus_metric, scale).
# scale 100 turns DCGM 0-1 fractions into percent so jobstats' round(v,1) keeps
# precision and they read like gpu_utilization (0-100). All are time-averaged
# over the job window, like the duty cycle.
DCGM_EXTRA = [
    ("gpu_sm_active",     "DCGM_FI_PROF_SM_ACTIVE",          100),
    ("gpu_sm_occupancy",  "DCGM_FI_PROF_SM_OCCUPANCY",       100),
    ("gpu_tensor_active", "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE", 100),
    ("gpu_dram_active",   "DCGM_FI_PROF_DRAM_ACTIVE",        100),
    ("gpu_power_usage",   "DCGM_FI_DEV_POWER_USAGE",         1),
]


class JobstatsExtended(Jobstats):
  """jobstats.Jobstats + DCGM profiling metrics in the blob."""

  def __init__(self, *a, dcgm=True, **kw):
    self.dcgm_added = []
    super().__init__(*a, **kw)
    # sp_node is populated by the parent (from the cached blob or a fresh
    # Prometheus pass); augment it after, only for long-enough GPU jobs.
    if dcgm and getattr(self, 'gpus', 0) and self.sp_node:
      try:
        self._augment_with_dcgm()
      except Exception as e:
        self.debug_print("DCGM augmentation skipped: %s" % e)

  def _prom(self, query):
    """Instant query at the job end time -> result list (empty on error)."""
    r = requests.get(self.prom_server + '/api/v1/query',
                     params={'query': query, 'time': self.end}, timeout=60)
    j = r.json()
    return j['data']['result'] if j.get('status') == 'success' else []

  def _augment_with_dcgm(self):
    cl = "slurm_cluster='%s'" % self.cluster if self.cluster else ''
    # 1) UUID -> (node, minor) for this job's GPUs (uuid is not stored in the
    #    blob, so this companion query is always needed).
    disc = self._prom("max_over_time((nvidia_gpu_jobId{%s} == %s)[%ds:])"
                      % (cl, self.jobidraw, self.diff))
    uuid_to = {}
    for s in disc:
      m = s['metric']
      u = m.get('uuid')
      if u and m.get('minor_number') is not None and 'host' in m:
        uuid_to[u] = (m['host'].split(':')[0], m['minor_number'])
    if not uuid_to:
      return
    rx = "^(" + "|".join(uuid_to) + ")$"   # UUIDs are hex+hyphen, RE2-safe
    # 2) each DCGM metric, joined on UUID, stored back under minor_number.
    for key, metric, scale in DCGM_EXTRA:
      res = self._prom("avg_over_time((%s{UUID=~\"%s\"})[%ds:])" % (metric, rx, self.diff))
      got = False
      for s in res:
        node_minor = uuid_to.get(s['metric'].get('UUID'))
        if not node_minor:
          continue
        try:
          v = round(float(s['value'][1]) * scale, 1)
        except (TypeError, ValueError):
          continue
        node, minor = node_minor
        self.sp_node.setdefault(node, {}).setdefault(key, {})[minor] = v
        got = True
      if got:
        self.dcgm_added.append(key)


if __name__ == "__main__":
  ap = argparse.ArgumentParser(description="jobstats with extra DCGM metrics folded into the blob")
  ap.add_argument("jobids", nargs="+", help="one or more Slurm job IDs")
  ap.add_argument("-b", "--base64", action="store_true",
                  help="emit the JS1:<base64 gzip JSON> blob (the AdminComment storage form)")
  ap.add_argument("--no-dcgm", dest="dcgm", action="store_false",
                  help="skip the DCGM augmentation (behaves like stock `jobstats -j`)")
  ap.add_argument("-f", "--force", action="store_true",
                  help="force recompute from Prometheus instead of the cached blob")
  ap.add_argument("-c", "--cluster", default=None, help="cluster (default: from sacct)")
  ap.add_argument("-d", "--debug", action="store_true", help="debug output")
  args = ap.parse_args()

  for jid in args.jobids:
    try:
      js = JobstatsExtended(jid, cluster=args.cluster, prom_server=PROM_SERVER,
                            debug=args.debug, force_recalc=args.force,
                            json_or_base64=True, dcgm=args.dcgm)
    except SystemExit:          # parent .error() exits; keep going for other ids
      continue
    if args.dcgm:
      print("# job %s: added %s" % (jid, ", ".join(js.dcgm_added) or "(none)"),
            file=sys.stderr)
    if args.base64:
      blob = js.report_job_json(True)
      print("JS1:" + blob if blob not in ("Short", "None") else "JS1:" + blob)
    else:
      print(js.report_job_json(False))
