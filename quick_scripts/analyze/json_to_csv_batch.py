#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


PROBLEM_BASE_FIELDS = (
    "index",
    "name",
    "trains",
    "conflicts",
    "avg_tracks",
    "conflicting_visit_pairs",
    "delay_cost_type",
)


def _problem_base(problem: dict) -> dict:
    return {field: problem.get(field) for field in PROBLEM_BASE_FIELDS}


def _is_scalar(value) -> bool:
    return not isinstance(value, (list, dict))


def json_to_rows_compact(data):
    """One row per solve; list/dict fields inside a solve are dropped.
    Matches the historical CSV layout used for the older BigM/MIP runs.
    """
    rows = []
    cols = set()

    for problem in data:
        base = _problem_base(problem)
        for solve in problem.get("solves", []):
            row = dict(base)
            if isinstance(solve, dict):
                for key, value in solve.items():
                    if _is_scalar(value):
                        row[key] = value
            else:
                row["solve"] = solve
            rows.append(row)
            cols.update(row.keys())

    return rows, sorted(cols)


def json_to_rows_trace(data):
    """One row per value_trace event. Each row carries the problem base + the
    scalar solve fields + the trace event fields (prefixed with `trace_`).
    Solves without a value_trace emit a single row with empty trace fields.
    """
    rows = []
    cols = set()

    for problem in data:
        base = _problem_base(problem)
        for solve in problem.get("solves", []):
            solve_scalars = dict(base)
            value_trace = []
            if isinstance(solve, dict):
                for key, value in solve.items():
                    if _is_scalar(value):
                        solve_scalars[key] = value
                    elif key == "value_trace" and isinstance(value, list):
                        value_trace = value
            else:
                solve_scalars["solve"] = solve

            if not value_trace:
                rows.append(solve_scalars)
                cols.update(solve_scalars.keys())
                continue

            for trace_entry in value_trace:
                row = dict(solve_scalars)
                if isinstance(trace_entry, dict):
                    for tk, tv in trace_entry.items():
                        row[f"trace_{tk}"] = tv
                else:
                    row["trace"] = trace_entry
                rows.append(row)
                cols.update(row.keys())

    return rows, sorted(cols)


FORMAT_DISPATCH = {
    "compact": json_to_rows_compact,
    "trace": json_to_rows_trace,
}


def convert_file(inp: Path, overwrite: bool, fmt: str, suffix: str):
    data = json.loads(inp.read_text())
    rows, cols = FORMAT_DISPATCH[fmt](data)

    out_name = inp.stem + suffix + ".csv"
    out = inp.with_name(out_name)
    if out.exists() and not overwrite:
        print(f"Skipping {out} (already exists)")
        return

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out} [{fmt}, {len(rows)} rows]")


def main():
    parser = argparse.ArgumentParser(
        description="Convert result JSON files into CSV files next to them."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="JSON files to convert",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CSV files",
    )
    parser.add_argument(
        "--format",
        choices=sorted(FORMAT_DISPATCH.keys()),
        default="compact",
        help=(
            "Output layout. 'compact' = one row per solve (list/dict fields "
            "such as value_trace are dropped; matches the old BigM/MIP CSV "
            "format). 'trace' = one row per value_trace event, with trace "
            "fields prefixed by 'trace_'."
        ),
    )
    parser.add_argument(
        "--suffix",
        default=None,
        help=(
            "Extra suffix before '.csv' (defaults to '' for compact, "
            "'_trace' for trace). Use '' to always overwrite the plain name."
        ),
    )
    args = parser.parse_args()

    if args.suffix is None:
        suffix = "" if args.format == "compact" else "_trace"
    else:
        suffix = args.suffix

    for raw in args.inputs:
        convert_file(Path(raw), args.overwrite, args.format, suffix)


if __name__ == "__main__":
    main()
