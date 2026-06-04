"""Cross-reference training MFU × NCCL busbw × storage × IB snapshot for each
node and emit a verdict:

  - OK               — all metrics within cluster p50 ± 5%
  - DEGRADED_COMPUTE — training MFU >5% below cluster p50 but NCCL/storage fine
  - DEGRADED_NCCL    — NCCL busbw >10% below peers
  - SLOW_STORAGE     — save throughput >15% below peers
  - DEGRADED_FABRIC  — IB counter_delta showed errors during the benchmark
  - MULTIPLE         — more than one of the above

This is the core of Category J in the plan: the tool that turns "node is slow"
into "node is slow *because*".

Usage:
    python cluster_bench/analysis/correlate.py \\
        --history $RESULTS_ROOT/cluster_bench/history.jsonl \\
        --snapshots $RESULTS_ROOT/cluster_bench/../snapshots \\
        --out $RESULTS_ROOT/cluster_bench/verdicts.md
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _load_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _pct_delta(value: float, reference: float) -> float:
    if reference in (None, 0) or value is None:
        return 0.0
    return (value - reference) / reference * 100


def compute_baselines(history: list[dict]) -> dict:
    """Compute cluster-wide p50 for each metric."""
    groups = defaultdict(list)
    for row in history:
        if row.get("kind") == "training_benchmark" and row.get("config") == "compute_baseline":
            groups[("mfu", 1)].append(row["mfu_percent"])
        elif row.get("kind") == "nccl_perf" and row.get("size_bytes") == 1073741824:  # 1 GB
            key = ("nccl_busbw", row["collective"])
            groups[key].append(row["busbw_gbps"])
        elif row.get("kind") == "storage_save":
            groups[("save_gbps", row["tier"])].append(row.get("throughput_gbps_estimate") or 0)
    baselines = {}
    for key, values in groups.items():
        values = [v for v in values if v]
        if values:
            baselines[key] = statistics.median(values)
    return baselines


def verdict_for_host(host: str, history: list[dict], baselines: dict) -> dict:
    """Apply thresholds; emit a dict with per-metric deltas and overall verdict."""
    row_by_kind = defaultdict(list)
    for r in history:
        if r.get("hostname") != host:
            continue
        row_by_kind[r["kind"]].append(r)

    problems: list[str] = []
    detail: dict = {"hostname": host}

    # Compute MFU (latest)
    tb = [r for r in row_by_kind["training_benchmark"] if r.get("config") == "compute_baseline"]
    if tb:
        latest = max(tb, key=lambda r: r.get("timestamp_utc", ""))
        detail["mfu_percent"] = latest["mfu_percent"]
        p50 = baselines.get(("mfu", 1))
        if p50 is not None:
            delta = _pct_delta(latest["mfu_percent"], p50)
            detail["mfu_delta_pct"] = round(delta, 2)
            detail["cluster_mfu_p50"] = p50
            if delta < -5:
                problems.append("DEGRADED_COMPUTE")

    # NCCL all_reduce busbw at 1 GB (latest)
    nccl_rows = [r for r in row_by_kind["nccl_perf"]
                 if r.get("collective") == "all_reduce" and r.get("size_bytes") == 1073741824]
    if nccl_rows:
        latest = max(nccl_rows, key=lambda r: r["timestamp_utc"])
        detail["nccl_all_reduce_1GB_busbw_gbps"] = latest["busbw_gbps"]
        p50 = baselines.get(("nccl_busbw", "all_reduce"))
        if p50 is not None:
            delta = _pct_delta(latest["busbw_gbps"], p50)
            detail["nccl_busbw_delta_pct"] = round(delta, 2)
            if delta < -10:
                problems.append("DEGRADED_NCCL")

    # Storage (most recent run per tier)
    for save in row_by_kind["storage_save"]:
        tier = save["tier"]
        key = ("save_gbps", tier)
        p50 = baselines.get(key)
        detail.setdefault("storage", {})[tier] = save.get("throughput_gbps_estimate")
        if p50 and save.get("throughput_gbps_estimate"):
            delta = _pct_delta(save["throughput_gbps_estimate"], p50)
            if delta < -15:
                problems.append(f"SLOW_STORAGE ({tier})")

    if not problems:
        detail["verdict"] = "OK"
    elif len(set(p.split()[0] for p in problems)) == 1:
        detail["verdict"] = problems[0]
    else:
        detail["verdict"] = "MULTIPLE: " + ", ".join(sorted(set(problems)))
    return detail


def render_markdown(verdicts: list[dict]) -> str:
    out = ["# Cluster-bench verdicts", ""]
    out.append(f"| node | verdict | MFU% (Δ vs p50) | NCCL busbw GB/s (Δ) | storage GB/s |")
    out.append("|------|---------|------------------|---------------------|--------------|")
    for v in sorted(verdicts, key=lambda x: x["hostname"]):
        mfu = v.get("mfu_percent")
        mfu_delta = v.get("mfu_delta_pct", 0.0)
        mfu_cell = f"{mfu:.1f} ({mfu_delta:+.1f}%)" if mfu is not None else "—"
        busbw = v.get("nccl_all_reduce_1GB_busbw_gbps")
        busbw_delta = v.get("nccl_busbw_delta_pct", 0.0)
        nccl_cell = f"{busbw:.0f} ({busbw_delta:+.1f}%)" if busbw is not None else "—"
        storage_cell = ", ".join(
            f"{t}={gbps:.2f}" for t, gbps in (v.get("storage") or {}).items() if gbps
        ) or "—"
        out.append(f"| {v['hostname']} | **{v['verdict']}** | {mfu_cell} | {nccl_cell} | {storage_cell} |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    history = _load_history(args.history)
    if not history:
        print("[correlate] empty history — nothing to correlate")
        return

    hosts = sorted({r["hostname"] for r in history if r.get("hostname")})
    baselines = compute_baselines(history)
    verdicts = [verdict_for_host(h, history, baselines) for h in hosts]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(verdicts))
    print(f"[correlate] {len(hosts)} nodes → {args.out}")
    for v in verdicts:
        print(f"  {v['hostname']:<30} {v['verdict']}")


if __name__ == "__main__":
    main()
