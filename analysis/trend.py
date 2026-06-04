"""Time-series trend + regression report from history.jsonl.

Groups rows by (hostname, kind, metric) into day buckets, computes daily mean,
and flags days where a metric is >threshold below the previous week's median
for the same host+metric. Output: markdown with sparkline-ish bars per series
and a `regressions.md` listing any flagged series.

Usage:
    shared/launch.sh python analysis/trend.py \\
        --history results/history.jsonl \\
        --out results/history.md \\
        --regressions results/regressions.md \\
        --threshold 5.0       # percent
"""
from __future__ import annotations

import argparse
import datetime
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _day(ts: str) -> str:
    if "T" in ts:
        return ts.split("T")[0]
    return ts[:10]


def _bucket(history: list, metric_extractor) -> dict:
    """Group rows by (host, day); value is a list of metric floats."""
    b = defaultdict(lambda: defaultdict(list))
    for row in history:
        v = metric_extractor(row)
        if v is None:
            continue
        host = row.get("hostname") or "(cluster)"
        day = _day(row["timestamp_utc"])
        b[host][day].append(v)
    return b


def _sparkline(values: list) -> str:
    if not values:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    if hi == lo:
        return blocks[4] * len(values)
    out = []
    for v in values:
        idx = int((v - lo) / (hi - lo) * (len(blocks) - 1))
        out.append(blocks[idx])
    return "".join(out)


def analyze_metric(history: list, kind: str, label: str, extractor, threshold: float):
    """Returns (markdown_section, regressions_list)."""
    per_host_day = _bucket([r for r in history if r.get("kind") == kind], extractor)
    if not per_host_day:
        return "", []

    md = [f"### {label}", ""]
    md += ["| host | spark (last 14 days) | latest | 7-day p50 | Δ% vs p50 |",
           "|------|----------------------|--------|-----------|-----------|"]
    regressions = []
    for host, days in sorted(per_host_day.items()):
        sorted_days = sorted(days)
        daily_means = [statistics.mean(days[d]) for d in sorted_days]
        latest_day = sorted_days[-1]
        latest_val = daily_means[-1]

        # Baseline = median of daily means in the 7 days preceding the latest.
        latest_dt = datetime.date.fromisoformat(latest_day)
        week_ago = (latest_dt - datetime.timedelta(days=7)).isoformat()
        baseline_window = [
            statistics.mean(days[d])
            for d in sorted_days
            if week_ago <= d < latest_day
        ]
        if len(baseline_window) >= 2:
            baseline = statistics.median(baseline_window)
            delta_pct = (latest_val - baseline) / baseline * 100
        else:
            baseline = None
            delta_pct = 0.0

        spark_window = daily_means[-14:]
        spark = _sparkline(spark_window)
        baseline_cell = f"{baseline:.2f}" if baseline is not None else "—"
        md.append(f"| {host} | `{spark}` | {latest_val:.2f} | {baseline_cell} | {delta_pct:+.1f}% |")

        if baseline is not None and delta_pct < -threshold:
            regressions.append({
                "host": host,
                "metric": label,
                "latest_day": latest_day,
                "latest_value": round(latest_val, 3),
                "baseline_p50": round(baseline, 3),
                "delta_pct": round(delta_pct, 2),
            })

    md.append("")
    return "\n".join(md), regressions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="Trend markdown output")
    ap.add_argument("--regressions", type=Path, help="Regressions-only markdown output")
    ap.add_argument("--threshold", type=float, default=5.0,
                    help="Regression threshold in percent (default: 5.0)")
    args = ap.parse_args()

    if not args.history.exists():
        print(f"[trend] {args.history} does not exist")
        return
    history = [json.loads(l) for l in args.history.read_text().splitlines() if l.strip()]

    sections, all_regressions = [], []
    for kind, label, extractor in [
        ("training_benchmark", "Training MFU (%)", lambda r: r.get("mfu_percent") if r.get("config") == "compute_baseline" else None),
        ("nccl_perf", "NCCL all_reduce busbw @1 GB (GB/s)",
            lambda r: r.get("busbw_gbps") if r.get("collective") == "all_reduce" and r.get("size_bytes") == 1073741824 else None),
        ("storage_save", "Storage throughput (GB/s, netscratch)",
            lambda r: r.get("throughput_gbps_estimate") if r.get("tier") == "netscratch" else None),
        ("mttr", "MTTR (s)", lambda r: r.get("total_wall_s")),
    ]:
        section, regs = analyze_metric(history, kind, label, extractor, args.threshold)
        if section:
            sections.append(section)
        all_regressions.extend(regs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        f"# Trend report (generated {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')})\n\n"
        + "\n\n".join(sections)
    )
    print(f"[trend] wrote {args.out}")

    if args.regressions:
        lines = [f"# Regressions (>{args.threshold}%) vs 7-day p50", ""]
        if not all_regressions:
            lines.append("_No regressions detected._")
        else:
            lines += ["| host | metric | day | latest | baseline p50 | delta |",
                      "|------|--------|-----|--------|--------------|-------|"]
            for r in sorted(all_regressions, key=lambda x: x["delta_pct"]):
                lines.append(f"| {r['host']} | {r['metric']} | {r['latest_day']} | "
                             f"{r['latest_value']} | {r['baseline_p50']} | {r['delta_pct']:+.2f}% |")
        args.regressions.parent.mkdir(parents=True, exist_ok=True)
        args.regressions.write_text("\n".join(lines) + "\n")
        print(f"[trend] wrote {args.regressions} ({len(all_regressions)} regressions)")


if __name__ == "__main__":
    main()
