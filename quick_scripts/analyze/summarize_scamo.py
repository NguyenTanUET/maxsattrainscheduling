#!/usr/bin/env python3
"""Aggregate `[SCAMO-PH1]` log lines from a benchmark run.

Usage:
    python3 quick_scripts/analyze/summarize_scamo.py <log_file>
"""

import re
import sys
from collections import defaultdict


# Parse one SCAMO-PH1 line, e.g.:
# [SCAMO-PH1] iter=12 candidates=27 pairs_total=22 pairs_multi=3
#             overlap_dist={0: 4, 1: 2} groups=1 cliques_in_groups=2 group_sizes={2: 1}
LINE_RE = re.compile(
    r"\[SCAMO-PH1\] "
    r"iter=(?P<iter>\d+) "
    r"candidates=(?P<candidates>\d+) "
    r"pairs_total=(?P<pairs_total>\d+) "
    r"pairs_multi=(?P<pairs_multi>\d+) "
    r"overlap_dist=(?P<overlap_dist>\{[^}]*\}) "
    r"groups=(?P<groups>\d+) "
    r"cliques_in_groups=(?P<cliques_in_groups>\d+) "
    r"group_sizes=(?P<group_sizes>\{[^}]*\})"
)

INSTANCE_RE = re.compile(r"Reading instances_\w+/Instance(\w+)\.txt")


def parse_dict(s: str) -> dict[int, int]:
    """Parse Rust BTreeMap debug format like '{0: 5, 1: 2}' into {0:5, 1:2}."""
    s = s.strip("{}")
    if not s:
        return {}
    out = {}
    for kv in s.split(","):
        k, v = kv.split(":")
        out[int(k.strip())] = int(v.strip())
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <log_file>", file=sys.stderr)
        return 1

    path = sys.argv[1]

    # Per-instance stats
    per_instance: dict[str, dict] = defaultdict(
        lambda: {
            "n_iters_with_cliques": 0,
            "total_candidates": 0,
            "total_pairs_total": 0,
            "total_pairs_multi": 0,
            "total_groups": 0,
            "total_cliques_in_groups": 0,
            "overlap_dist": defaultdict(int),
            "group_sizes": defaultdict(int),
        }
    )

    current_instance: str | None = None

    with open(path) as f:
        for line in f:
            inst_m = INSTANCE_RE.search(line)
            if inst_m:
                current_instance = inst_m.group(1)
                continue

            m = LINE_RE.search(line)
            if not m:
                continue
            if current_instance is None:
                current_instance = "?"

            stats = per_instance[current_instance]
            stats["n_iters_with_cliques"] += 1
            stats["total_candidates"] += int(m.group("candidates"))
            stats["total_pairs_total"] += int(m.group("pairs_total"))
            stats["total_pairs_multi"] += int(m.group("pairs_multi"))
            stats["total_groups"] += int(m.group("groups"))
            stats["total_cliques_in_groups"] += int(m.group("cliques_in_groups"))

            for k, v in parse_dict(m.group("overlap_dist")).items():
                stats["overlap_dist"][k] += v
            for k, v in parse_dict(m.group("group_sizes")).items():
                stats["group_sizes"][k] += v

    if not per_instance:
        print("No SCAMO-PH1 lines found.", file=sys.stderr)
        return 1

    # Per-instance summary
    print(
        f"{'Instance':<14}"
        f"{'iters':>7}"
        f"{'cands':>7}"
        f"{'pairs':>7}"
        f"{'multi':>7}"
        f"{'groups':>8}"
        f"{'in_grp':>8}"
        f"{'overlap_dist':>30}"
        f"{'group_sizes':>20}"
    )
    print("-" * 108)

    grand: dict = {
        "iters": 0,
        "candidates": 0,
        "pairs": 0,
        "pairs_multi": 0,
        "groups": 0,
        "in_groups": 0,
        "overlap_dist": defaultdict(int),
        "group_sizes": defaultdict(int),
    }

    for inst in sorted(per_instance):
        s = per_instance[inst]
        overlap = ", ".join(f"{k}:{v}" for k, v in sorted(s["overlap_dist"].items()))
        sizes = ", ".join(f"{k}:{v}" for k, v in sorted(s["group_sizes"].items()))
        print(
            f"{inst:<14}"
            f"{s['n_iters_with_cliques']:>7}"
            f"{s['total_candidates']:>7}"
            f"{s['total_pairs_total']:>7}"
            f"{s['total_pairs_multi']:>7}"
            f"{s['total_groups']:>8}"
            f"{s['total_cliques_in_groups']:>8}"
            f"{('{' + overlap + '}'):>30}"
            f"{('{' + sizes + '}'):>20}"
        )
        grand["iters"] += s["n_iters_with_cliques"]
        grand["candidates"] += s["total_candidates"]
        grand["pairs"] += s["total_pairs_total"]
        grand["pairs_multi"] += s["total_pairs_multi"]
        grand["groups"] += s["total_groups"]
        grand["in_groups"] += s["total_cliques_in_groups"]
        for k, v in s["overlap_dist"].items():
            grand["overlap_dist"][k] += v
        for k, v in s["group_sizes"].items():
            grand["group_sizes"][k] += v

    # Grand total
    print("=" * 108)
    print(f"GRAND TOTAL across {len(per_instance)} instances:")
    print(f"  iters with cliques:     {grand['iters']}")
    print(f"  total candidates:       {grand['candidates']}")
    print(f"  total resource pairs:   {grand['pairs']}")
    print(f"  pairs with multi:       {grand['pairs_multi']}")
    print(f"  groups detected:        {grand['groups']}")
    print(f"  cliques in groups:      {grand['in_groups']}")
    print(f"  overlap distribution:   {dict(sorted(grand['overlap_dist'].items()))}")
    print(f"  group size distrib:     {dict(sorted(grand['group_sizes'].items()))}")

    if grand["pairs"]:
        pct_multi = grand["pairs_multi"] / grand["pairs"] * 100
        print(f"\n  → % pairs with multiple cliques: {pct_multi:.2f}%")
    if grand["candidates"]:
        pct_grouped = grand["in_groups"] / grand["candidates"] * 100
        print(f"  → % candidates in SCAMO groups:  {pct_grouped:.2f}%")

    print("\nINTERPRETATION:")
    if grand["pairs_multi"] == 0:
        print(
            "  ✗ Zero resource pairs ever have ≥2 cliques in the same iteration.\n"
            "    DDD's per-iteration model produces sparse, non-staircase cliques.\n"
            "    SCAMO encoding is NOT applicable in the current architecture."
        )
    elif grand["groups"] == 0:
        print(
            "  ✗ Resource pairs do have multiple cliques, but consecutive ones\n"
            "    never share visits. No staircase pattern → SCAMO not applicable."
        )
    elif grand["in_groups"] / max(grand["candidates"], 1) < 0.10:
        print(
            f"  ⚠ Only {grand['in_groups']/grand['candidates']*100:.1f}% of cliques\n"
            "    fall into SCAMO groups. Marginal benefit; SCAMO probably not\n"
            "    worth the engineering cost."
        )
    else:
        print(
            f"  ✓ {grand['in_groups']/grand['candidates']*100:.1f}% of cliques are\n"
            "    SCAMO-eligible. Worth implementing Phase 2 encoding."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
