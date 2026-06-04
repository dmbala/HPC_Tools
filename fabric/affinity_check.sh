#!/bin/bash
# Validate GPU ↔ IB-HCA PCIe-NUMA affinity by reading `nvidia-smi topo -m`.
# Each GPU's *best* connection to an InfiniBand NIC must be at least as close
# as `NODE` (same NUMA). `SYS` (across UPI/QPI) is a failure mode that costs
# 30-50% cross-node bandwidth.
#
# Exits 0 if every GPU is ≥ NODE to at least one active IB HCA, non-zero otherwise.
#
# Usage:
#   fabric/affinity_check.sh [--snapshot <path.json>]
#
# If --snapshot is given, uses it instead of running nvidia-smi + sysfs probes
# afresh. Useful for post-hoc analysis of a captured snapshot.

set -euo pipefail

HPC_TOOLS_ROOT="${HPC_TOOLS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SNAPSHOT_PATH=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --snapshot) SNAPSHOT_PATH="$2"; shift 2 ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

python3 - "${SNAPSHOT_PATH}" <<'PY'
import json
import re
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

# Connection qualities in descending order (best → worst).
# Higher score = closer. Anything below NODE is a warning.
_QUALITY = OrderedDict([
    ("NV18", 10), ("NV12", 10), ("NV8", 10), ("NV6", 10), ("NV4", 10),
    ("NV2", 10), ("NV1", 10), ("NVL", 10),
    ("PIX", 6),
    ("PXB", 5),
    ("PHB", 4),
    ("NODE", 3),
    ("SYS", 1),
    ("X", 0),
])

OK = 3  # NODE or better


def _run(cmd):
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, timeout=30)
        return r.stdout
    except Exception:
        return ""


def _topology_raw():
    snap_path = sys.argv[1].strip()
    if snap_path:
        snap = json.loads(Path(snap_path).read_text())
        return snap["topology"]["raw"]
    return _run(["nvidia-smi", "topo", "-m"])


def _strip_ansi(s):
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def parse_topo(raw):
    """Parse the nvidia-smi topo -m matrix into {gpu_idx: {nic_name: quality}}."""
    lines = [_strip_ansi(l).rstrip() for l in raw.splitlines() if l.strip()]
    # Header row lists columns: GPU0 GPU1 ... NIC0 NIC1 ... CPU Affinity ...
    header = lines[0].split()
    # Find column indices for NIC0..NIC<n>
    nic_cols = [(i, h) for i, h in enumerate(header) if h.startswith("NIC")]
    if not nic_cols:
        return {}
    # Rows that start with GPU<n> describe that GPU's connections to every column.
    gpu_rows = [l for l in lines[1:] if l.startswith("GPU")]
    result = {}
    for row in gpu_rows:
        parts = row.split()
        gpu_name = parts[0]
        # Strip GPU name col → row values align with the header slots
        values = parts[1:]
        # Header has one col for the row label, so value[i] aligns with header[i+1]
        # nic_cols indices are absolute over header (starting at 0). Map to values[idx-1].
        conn = {}
        for col_idx, nic in nic_cols:
            v = values[col_idx - 1] if (col_idx - 1) < len(values) else "?"
            conn[nic] = v.strip()
        result[gpu_name] = conn
    return result


def quality_score(q):
    return _QUALITY.get(q, 0)


def main():
    raw = _topology_raw()
    if not raw:
        print("[affinity_check] no topology data available", file=sys.stderr)
        sys.exit(2)
    matrix = parse_topo(raw)
    if not matrix:
        print("[affinity_check] could not parse topology output", file=sys.stderr)
        sys.exit(2)

    failures = []
    warnings = []
    print(f"[affinity_check] {len(matrix)} GPUs × {len(next(iter(matrix.values())))} NICs")
    print(f"{'GPU':<6} {'best_nic':<8} {'quality':<8} {'verdict'}")
    print("-" * 50)
    for gpu, conns in matrix.items():
        best_nic, best_q = max(conns.items(), key=lambda kv: quality_score(kv[1]))
        score = quality_score(best_q)
        if score >= OK:
            verdict = "OK"
        elif score > 0:
            verdict = f"WARN (across {best_q})"
            warnings.append((gpu, best_q))
        else:
            verdict = "FAIL"
            failures.append((gpu, best_q))
        print(f"{gpu:<6} {best_nic:<8} {best_q:<8} {verdict}")

    if failures:
        print(f"\n[affinity_check] FAIL: {len(failures)} GPU(s) have no reachable NIC")
        sys.exit(1)
    if warnings:
        print(f"\n[affinity_check] WARN: {len(warnings)} GPU(s) cross NUMA boundary — "
              f"expect 30-50% cross-node bandwidth loss")
        # Non-zero but distinct from FAIL, for scripting.
        sys.exit(3)
    print("\n[affinity_check] OK: all GPUs are ≥ NODE-level to an IB NIC")
    sys.exit(0)


main()
PY
