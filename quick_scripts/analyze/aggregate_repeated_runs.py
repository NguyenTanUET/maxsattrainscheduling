#!/usr/bin/env python3
"""
Aggregate repeated benchmark runs.

Reads all CSV files matching <prefix>_run_*.csv in a directory, computes
per-instance mean/std for total_time and prints a comparison table.

Usage:
    python3 quick_scripts/analyze/aggregate_repeated_runs.py results_repeated/<batch_dir>

    # Or with explicit prefixes:
    python3 quick_scripts/analyze/aggregate_repeated_runs.py \
        results_repeated/<batch_dir> --prefixes sc ladder
"""

import argparse
import csv
import glob
import math
import os
import sys
from collections import defaultdict


def load_runs(directory: str, prefix: str) -> dict[str, list[dict]]:
    """Load all <prefix>_run_*.csv files; return {instance_name: [row, ...]}."""
    pattern = os.path.join(directory, f"{prefix}_run_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return {}
    runs: dict[str, list[dict]] = defaultdict(list)
    for f in files:
        with open(f) as fh:
            for row in csv.DictReader(fh):
                runs[row["name"]].append(row)
    return dict(runs)


def stats(values: list[float]) -> tuple[float, float, float, float]:
    """Return (mean, std, min, max). std = sample std (n-1)."""
    n = len(values)
    if n == 0:
        return (float("nan"),) * 4
    mean = sum(values) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    return mean, std, min(values), max(values)


def summarize(runs: dict[str, list[dict]], field: str) -> dict:
    """Return {instance: stats(...)} for the given numeric field."""
    out = {}
    for name, rows in runs.items():
        vals = []
        for r in rows:
            v = r.get(field, "")
            if v == "" or r.get("status") == "timeout":
                continue
            try:
                vals.append(float(v))
            except ValueError:
                continue
        if vals:
            out[name] = (stats(vals), len(vals))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("--prefixes", nargs="+", default=["sc", "ladder"])
    ap.add_argument("--field", default="total_time")
    args = ap.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Not a directory: {args.directory}", file=sys.stderr)
        return 1

    print(f"Aggregating from: {args.directory}")
    print(f"Field:            {args.field}")
    print()

    summaries: dict[str, dict] = {}
    for prefix in args.prefixes:
        runs = load_runs(args.directory, prefix)
        n_files = len(set(r["status"] for rs in runs.values() for r in rs)) if runs else 0
        n_runs_per_inst = max((len(rs) for rs in runs.values()), default=0)
        if not runs:
            print(f"[{prefix}] No CSV files matching {prefix}_run_*.csv — skipping.")
            continue
        print(f"[{prefix}] Found {n_runs_per_inst} runs covering {len(runs)} instances")
        summaries[prefix] = summarize(runs, args.field)

    if len(summaries) < 1:
        return 0

    # Build comparison table.
    all_instances = sorted(set().union(*(s.keys() for s in summaries.values())))
    header = ["instance"]
    for p in args.prefixes:
        if p in summaries:
            header += [f"{p}_mean", f"{p}_std", f"{p}_min", f"{p}_max", f"{p}_n"]
    print()
    print(",".join(header))
    for inst in all_instances:
        row = [inst]
        for p in args.prefixes:
            if p not in summaries:
                continue
            entry = summaries[p].get(inst)
            if entry is None:
                row += ["", "", "", "", "0"]
            else:
                (mean, std, mn, mx), n = entry
                row += [
                    f"{mean:.6f}",
                    f"{std:.6f}",
                    f"{mn:.6f}",
                    f"{mx:.6f}",
                    str(n),
                ]
        print(",".join(row))

    # Aggregate totals.
    print()
    print("=== AGGREGATE TOTALS (sum of per-instance means, ok-only) ===")
    for p in args.prefixes:
        if p not in summaries:
            continue
        total = sum(entry[0][0] for entry in summaries[p].values())
        n_inst = len(summaries[p])
        print(f"  {p}: sum={total:.4f}  instances={n_inst}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
