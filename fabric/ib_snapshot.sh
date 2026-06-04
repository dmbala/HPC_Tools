#!/bin/bash
# Static topology snapshot for one node: GPUs, NVLink, IB HCAs, ports, counters,
# GPU↔NIC topology matrix. Saved as a single JSON file for diffing across
# benchmark runs and for forensic comparison when a node goes slow.
#
# Usage:
#   fabric/ib_snapshot.sh [output.json]
#
# Default output: <repo-root>/results/snapshots/<hostname>_<timestamp>.json
#
# Runs outside the container — uses host `ibstat`, `ibv_devinfo`, `ibdev2netdev`,
# `nvidia-smi`. All are read-only probes.

set -euo pipefail

HPC_TOOLS_ROOT="${HPC_TOOLS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
HOST="$(hostname -s)"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${1:-${HPC_TOOLS_ROOT}/results/snapshots/${HOST}_${TS}.json}"
mkdir -p "$(dirname "${OUT}")"

# Helper: emit JSON via a tiny Python stanza so we get correct escaping for
# multiline strings (nvidia-smi topo -m output is a giant formatted table).
python3 - "${OUT}" <<'PY'
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path


def run(cmd, check=False):
    try:
        r = subprocess.run(cmd, shell=isinstance(cmd, str),
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, check=check, timeout=30)
        return {"rc": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
    except subprocess.TimeoutExpired:
        return {"rc": -1, "stdout": "", "stderr": "timeout"}
    except FileNotFoundError:
        return {"rc": 127, "stdout": "", "stderr": "command not found"}


def nvidia_smi_gpus():
    query = "index,name,pci.bus_id,temperature.gpu,power.draw,power.limit,memory.total,memory.used,clocks.current.sm,clocks.current.memory"
    r = run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    if r["rc"] != 0:
        return {"error": r["stderr"], "gpus": []}
    rows = []
    for line in r["stdout"].strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        rows.append({
            "index": int(parts[0]), "name": parts[1], "pci_bus_id": parts[2],
            "temp_c": _to_float(parts[3]), "power_draw_w": _to_float(parts[4]),
            "power_limit_w": _to_float(parts[5]),
            "memory_total_mib": _to_int(parts[6]), "memory_used_mib": _to_int(parts[7]),
            "sm_clock_mhz": _to_int(parts[8]), "memory_clock_mhz": _to_int(parts[9]),
        })
    return {"gpus": rows}


def _to_int(s):
    try:
        return int(float(s))
    except ValueError:
        return None


def _to_float(s):
    try:
        return float(s)
    except ValueError:
        return None


def ib_hcas():
    hca_root = Path("/sys/class/infiniband")
    if not hca_root.exists():
        return {"error": "no /sys/class/infiniband", "hcas": []}
    hcas = []
    for hca_dir in sorted(hca_root.iterdir()):
        name = hca_dir.name
        info = {"name": name, "ports": []}
        for p in sorted((hca_dir / "ports").iterdir()):
            port = {"port": int(p.name)}
            # Read key port files — each is a one-liner.
            for key, path in [
                ("state", p / "state"),
                ("phys_state", p / "phys_state"),
                ("rate", p / "rate"),
                ("link_layer", p / "link_layer"),
            ]:
                port[key] = _read_first(path)
            counters_dir = p / "counters"
            if counters_dir.exists():
                port["counters"] = {c.name: _to_int(_read_first(c)) for c in counters_dir.iterdir() if c.is_file()}
            info["ports"].append(port)
        # Device-level attributes from ibv_devinfo for a richer view (LID, GID, etc.).
        devinfo = run(["ibv_devinfo", "-d", name])
        info["ibv_devinfo"] = devinfo["stdout"]
        hcas.append(info)
    return {"hcas": hcas}


def _read_first(path):
    try:
        return path.read_text().strip()
    except OSError:
        return None


def ibdev2netdev_map():
    r = run(["ibdev2netdev"])
    if r["rc"] != 0:
        return {"error": r["stderr"], "mapping": []}
    mapping = []
    for line in r["stdout"].strip().splitlines():
        # Typical: "mlx5_0 port 1 ==> ibp0 (Up)"
        parts = line.split()
        if len(parts) >= 5 and parts[3] == "==>":
            mapping.append({
                "hca": parts[0], "port": int(parts[2]),
                "netdev": parts[4], "state": parts[5].strip("()") if len(parts) > 5 else None,
            })
    return {"mapping": mapping, "raw": r["stdout"]}


def nvlink_status():
    r = run(["nvidia-smi", "nvlink", "--status"])
    return {"rc": r["rc"], "raw": r["stdout"]}


def topology_matrix():
    r = run(["nvidia-smi", "topo", "-m"])
    return {"rc": r["rc"], "raw": r["stdout"]}


def nccl_version():
    # NCCL version is read from libnccl.so metadata if available; otherwise skip.
    out = Path("/usr/lib/x86_64-linux-gnu/libnccl.so").resolve() if Path("/usr/lib/x86_64-linux-gnu/libnccl.so").exists() else None
    return {"libnccl_symlink_target": str(out) if out else None}


def kernel_and_driver():
    return {
        "uname": run(["uname", "-a"])["stdout"].strip(),
        "nvidia_driver": run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"])["stdout"].splitlines()[0].strip() if run(["nvidia-smi"])["rc"] == 0 else None,
        "cuda_runtime": run("nvcc --version 2>/dev/null | grep release | awk -F', ' '{print $2}' | tr -d '\\n'")["stdout"] or None,
    }


out_path = Path(sys.argv[1])
snapshot = {
    "schema_version": 1,
    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "hostname": os.uname().nodename,
    "system": kernel_and_driver(),
    "gpus": nvidia_smi_gpus(),
    "nvlink": nvlink_status(),
    "topology": topology_matrix(),
    "ib": ib_hcas(),
    "ibdev2netdev": ibdev2netdev_map(),
    "nccl": nccl_version(),
}
out_path.write_text(json.dumps(snapshot, indent=2, default=str))
print(f"[ib_snapshot] {out_path}")
PY
