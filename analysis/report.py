"""Generate a human-readable markdown summary of recent benchmark runs from
history.jsonl. Per metric, shows cluster p50 / p90 / min / max and the per-host
leaderboard sorted by the metric.

Usage:
    python cluster_bench/analysis/report.py \\
        --history $RESULTS_ROOT/cluster_bench/history.jsonl \\
        --out $RESULTS_ROOT/cluster_bench/history.md
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values: list[float], p: float) -> float:
    values = sorted(v for v in values if v is not None)
    if not values:
        return 0.0
    k = min(int(len(values) * p / 100), len(values) - 1)
    return values[k]


def render(history: list[dict]) -> str:
    out = ["# Cluster-bench history", ""]

    # Per-host MFU leaderboard (compute_baseline config).
    mfus = [r for r in history
            if r.get("kind") == "training_benchmark" and r.get("config") == "compute_baseline"]
    latest_by_host = {}
    for r in mfus:
        h = r.get("hostname")
        if not h:
            continue
        if h not in latest_by_host or r["timestamp_utc"] > latest_by_host[h]["timestamp_utc"]:
            latest_by_host[h] = r
    if latest_by_host:
        vals = [r["mfu_percent"] for r in latest_by_host.values() if r["mfu_percent"]]
        out += ["## Per-node MFU (compute_baseline, latest per host)", ""]
        out.append(f"Cluster p50/p90/min/max MFU%: "
                   f"{percentile(vals, 50):.1f} / {percentile(vals, 90):.1f} / "
                   f"{min(vals):.1f} / {max(vals):.1f}  (n={len(vals)})")
        out += ["", "| node | MFU% | TFLOPs/GPU | iter_s | timestamp |",
                "|------|------|------------|--------|-----------|"]
        for host, r in sorted(latest_by_host.items(), key=lambda kv: -kv[1]["mfu_percent"]):
            out.append(f"| {host} | {r['mfu_percent']:.2f} | {r['tflops_per_gpu']:.1f} | "
                       f"{r['avg_iter_time_s']:.4f} | {r['timestamp_utc']} |")
        out.append("")

    # Scaling curve (network_scaling).
    scaling = [r for r in history
               if r.get("kind") == "training_benchmark" and r.get("config") == "network_scaling"]
    if scaling:
        out += ["## Scaling curve (network_scaling, latest run)", ""]
        latest_ts = max(r["timestamp_utc"] for r in scaling)
        latest_points = [r for r in scaling if r["timestamp_utc"] == latest_ts]
        latest_points.sort(key=lambda r: r["world_size"])
        out += ["| world | nnodes | gpus/node | MFU% | TFLOPs/GPU | iter_s |",
                "|-------|--------|-----------|------|------------|--------|"]
        for r in latest_points:
            out.append(f"| {r['world_size']} | {r['nnodes']} | {r['gpus_per_node']} | "
                       f"{r['mfu_percent']:.2f} | {r['tflops_per_gpu']:.1f} | {r['avg_iter_time_s']:.4f} |")
        out.append("")

    # NCCL busbw @ 1 GB, per collective, leaderboard.
    nccl = [r for r in history if r.get("kind") == "nccl_perf" and r.get("size_bytes") == 1073741824]
    if nccl:
        out += ["## NCCL busbw @ 1 GB (latest per host × collective)", ""]
        by_pair = {}
        for r in nccl:
            key = (r["hostname"], r["collective"])
            if key not in by_pair or r["timestamp_utc"] > by_pair[key]["timestamp_utc"]:
                by_pair[key] = r
        collectives = sorted({r["collective"] for r in by_pair.values()})
        hosts = sorted({r["hostname"] for r in by_pair.values()})
        header = "| node | " + " | ".join(f"{c} GB/s" for c in collectives) + " |"
        sep = "|------|" + "|".join("---" for _ in collectives) + "|"
        out += [header, sep]
        for host in hosts:
            cells = [host]
            for c in collectives:
                r = by_pair.get((host, c))
                cells.append(f"{r['busbw_gbps']:.0f}" if r else "—")
            out.append("| " + " | ".join(cells) + " |")
        out.append("")

    # Storage.
    storage = [r for r in history if r.get("kind") == "storage_save"]
    if storage:
        out += ["## Storage throughput (mean per tier)", ""]
        by_tier = defaultdict(list)
        for r in storage:
            if r.get("throughput_gbps_estimate"):
                by_tier[r["tier"]].append(r["throughput_gbps_estimate"])
        out += ["| tier | mean GB/s | p50 | min | max | runs |",
                "|------|-----------|-----|-----|-----|------|"]
        for tier, vals in sorted(by_tier.items()):
            out.append(f"| {tier} | {statistics.mean(vals):.2f} | "
                       f"{percentile(vals, 50):.2f} | {min(vals):.2f} | {max(vals):.2f} | {len(vals)} |")
        out.append("")

    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if not args.history.exists():
        print(f"[report] {args.history} doesn't exist yet — run scrape_metrics.py first")
        return
    rows = [json.loads(l) for l in args.history.read_text().splitlines() if l.strip()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(rows))
    print(f"[report] {len(rows)} rows → {args.out}")


if __name__ == "__main__":
    main()
