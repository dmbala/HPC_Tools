"""Walk results/ and append normalized rows to history.jsonl.

Input directories produced by the Stage-2/3/4 sbatches:

  results/cluster_bench/per_node/<host>_<ts>/benchmark_results.json
  results/cluster_bench/scaling/<ts>/n<N>_g<G>_w<W>/benchmark_results.json
  results/cluster_bench/nccl/<ts>/<collective>_<host>.csv
  results/cluster_bench/storage/<ts>/<tier>/benchmark_results.json
  results/cluster_bench/storage/<ts>/<tier>/total_bytes.txt

Output: `history.jsonl` with one row per metric-point, each row tagged by
`kind` in {training_benchmark, nccl_perf, storage_save}. Rows are idempotent —
re-running skips rows already present (keyed on kind + host + ts + detail).

Usage:
    python cluster_bench/analysis/scrape_metrics.py \\
        --results $RESULTS_ROOT/cluster_bench \\
        --out $RESULTS_ROOT/cluster_bench/history.jsonl
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable


def _stable_id(row: dict) -> str:
    keys = sorted(row)
    canonical = "|".join(f"{k}={row[k]}" for k in keys if k != "_id")
    return hashlib.sha1(canonical.encode()).hexdigest()[:12]


def _parse_ts(ts: str) -> str:
    """Normalize an ISO-like timestamp to UTC isoformat."""
    if "T" in ts and ts.endswith("Z"):
        return ts
    try:
        return datetime.datetime.strptime(ts, "%Y%m%dT%H%M%SZ").isoformat() + "Z"
    except ValueError:
        return ts


def scrape_per_node(root: Path) -> Iterable[dict]:
    """per_node/<host>_<ts>/benchmark_results.json → training_benchmark rows."""
    for run_dir in sorted((root / "per_node").glob("*_*")):
        m = re.match(r"(?P<host>[\w\-\.]+)_(?P<ts>\d{8}T\d{6}Z)$", run_dir.name)
        if not m:
            continue
        j = run_dir / "benchmark_results.json"
        if not j.exists():
            continue
        with j.open() as f:
            data = json.load(f)
        yield {
            "kind": "training_benchmark",
            "hostname": m["host"],
            "timestamp_utc": _parse_ts(m["ts"]),
            "config": "compute_baseline",
            "world_size": 1,
            "mfu_percent": data.get("avg_mfu_percent"),
            "tflops_per_gpu": data.get("tflops_per_gpu"),
            "avg_iter_time_s": data.get("avg_iter_time_seconds"),
        }


def scrape_scaling(root: Path) -> Iterable[dict]:
    """scaling/<ts>/n<N>_g<G>_w<W>/benchmark_results.json → training_benchmark rows."""
    for ts_dir in sorted((root / "scaling").glob("*")):
        if not ts_dir.is_dir():
            continue
        for point_dir in sorted(ts_dir.glob("n*_g*_w*")):
            m = re.match(r"n(?P<n>\d+)_g(?P<g>\d+)_w(?P<w>\d+)$", point_dir.name)
            if not m:
                continue
            j = point_dir / "benchmark_results.json"
            if not j.exists():
                continue
            with j.open() as f:
                data = json.load(f)
            yield {
                "kind": "training_benchmark",
                "timestamp_utc": _parse_ts(ts_dir.name),
                "config": "network_scaling",
                "nnodes": int(m["n"]),
                "gpus_per_node": int(m["g"]),
                "world_size": int(m["w"]),
                "mfu_percent": data.get("avg_mfu_percent"),
                "tflops_per_gpu": data.get("tflops_per_gpu"),
                "avg_iter_time_s": data.get("avg_iter_time_seconds"),
            }


def scrape_nccl(root: Path) -> Iterable[dict]:
    """nccl/<ts>/<collective>_<host>.csv → nccl_perf rows (one per size)."""
    for ts_dir in sorted((root / "nccl").glob("*")):
        if not ts_dir.is_dir():
            continue
        ts = _parse_ts(ts_dir.name.replace("_inter", ""))
        for csv_path in sorted(ts_dir.glob("*.csv")):
            m = re.match(r"(?P<collective>[a-z_]+)_(?P<host>.+?)\.csv$", csv_path.name)
            if not m:
                continue
            with csv_path.open() as f:
                reader = csv.DictReader(f)
                for row in reader:
                    yield {
                        "kind": "nccl_perf",
                        "hostname": m["host"],
                        "timestamp_utc": ts,
                        "collective": m["collective"],
                        "size_bytes": int(row["size_bytes"]),
                        "time_us": float(row["time_us"]),
                        "algbw_gbps": float(row["algbw_gbps"]),
                        "busbw_gbps": float(row["busbw_gbps"]),
                    }


def scrape_storage(root: Path) -> Iterable[dict]:
    """storage/<ts>/<tier>/total_bytes.txt + stdout.log → storage_save rows."""
    for ts_dir in sorted((root / "storage").glob("*")):
        if not ts_dir.is_dir():
            continue
        ts = _parse_ts(ts_dir.name)
        for tier_dir in sorted(ts_dir.iterdir()):
            bytes_file = tier_dir / "total_bytes.txt"
            log = tier_dir / "stdout.log"
            if not (bytes_file.exists() and log.exists()):
                continue
            total_bytes = int(bytes_file.read_text().strip() or 0)
            # Extract per-save durations from the recipe's log. The recipe prints
            # `Consolidating safetensors files from ...` and `Total time taken: X secs.`
            # lines. We average across the `Total time taken` hits.
            durations = []
            for line in log.read_text().splitlines():
                mm = re.search(r"Total time taken:\s*([\d.]+)\s*secs", line)
                if mm:
                    durations.append(float(mm.group(1)))
            if not durations:
                continue
            # Throughput: rough proxy = total_bytes / (sum of durations × number of saves)
            # Not perfectly apples-to-apples with raw dd but good for diffing tiers.
            yield {
                "kind": "storage_save",
                "hostname": "",  # filled by caller if known
                "timestamp_utc": ts,
                "tier": tier_dir.name,
                "num_saves": len(durations),
                "mean_save_seconds": sum(durations) / len(durations),
                "total_bytes_written": total_bytes,
                "throughput_gbps_estimate": (total_bytes / sum(durations) / 1e9) if sum(durations) > 0 else None,
            }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True,
                    help="Root results dir, e.g. $RESULTS_ROOT/cluster_bench")
    ap.add_argument("--out", type=Path, required=True,
                    help="Append-only JSONL file")
    ap.add_argument("--force-rescan", action="store_true",
                    help="Re-read every JSON even if it was ingested before")
    args = ap.parse_args()

    # _id dedup for correctness (no duplicate rows in history.jsonl).
    existing_ids = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            try:
                existing_ids.add(json.loads(line).get("_id"))
            except json.JSONDecodeError:
                pass

    # Manifest for speed: skip entire result subdirs we've already scraped so
    # we don't re-parse every historical JSON on each run. Keyed on the
    # subdir's relative path + mtime so a later edit still gets re-scraped.
    manifest_path = args.out.with_suffix(".scraped_manifest")
    manifest: dict[str, float] = {}
    if manifest_path.exists() and not args.force_rescan:
        try:
            manifest = {
                k: float(v) for k, v in (json.loads(manifest_path.read_text())).items()
            }
        except (json.JSONDecodeError, ValueError):
            manifest = {}

    def _dir_mtime(root: Path) -> float:
        try:
            return root.stat().st_mtime
        except OSError:
            return 0.0

    def _is_fresh(kind: str, subdir: Path) -> bool:
        key = f"{kind}:{subdir.relative_to(args.results)}"
        m = _dir_mtime(subdir)
        prior = manifest.get(key, -1.0)
        if prior >= m:
            return False
        manifest[key] = m
        return True

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_new = 0
    with args.out.open("a") as fout:
        for kind, scraper, bucket_root in [
            ("per_node", scrape_per_node, args.results / "per_node"),
            ("scaling", scrape_scaling, args.results / "scaling"),
            ("nccl", scrape_nccl, args.results / "nccl"),
            ("storage", scrape_storage, args.results / "storage"),
        ]:
            if not bucket_root.exists():
                continue
            # Fast-path: skip subdirs whose mtime hasn't advanced since the
            # last scrape. Still run the scraper against directories that
            # are new or changed.
            fresh_children = [d for d in bucket_root.iterdir()
                              if d.is_dir() and _is_fresh(kind, d)]
            if not fresh_children and not args.force_rescan:
                continue
            for row in scraper(args.results):
                rid = _stable_id(row)
                if rid in existing_ids:
                    continue
                row["_id"] = rid
                fout.write(json.dumps(row) + "\n")
                existing_ids.add(rid)
                n_new += 1

    manifest_path.write_text(json.dumps(manifest))
    print(f"[scrape] appended {n_new} rows → {args.out}")


if __name__ == "__main__":
    main()
