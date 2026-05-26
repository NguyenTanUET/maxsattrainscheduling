#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


SUMMARY_FIELDS = [
    "status",
    "cost",
    "lb",
    "ub",
    "sol_time",
    "time_to_first_solution_ms",
    "time_to_best_value_ms",
    "time_to_prove_best_value_ms",
    "time_to_optimality_ms",
]


def load_solver_results(path: Path):
    data = json.loads(path.read_text())
    rows = {}
    solver_name = None
    for item in data:
        solves = item.get("solves", [])
        if not solves:
            continue
        solve = solves[0]
        solver_name = solve.get("solver_name", path.stem)
        rows[item["name"]] = {
            "delay_cost_type": item.get("delay_cost_type"),
            **{field: solve.get(field) for field in SUMMARY_FIELDS},
        }
    if solver_name is None:
        raise ValueError(f"No solves found in {path}")
    return solver_name, rows


def fastest_solver(values):
    finite = [(name, value) for name, value in values if value is not None]
    if not finite:
        return None
    finite.sort(key=lambda item: item[1])
    return finite[0][0]


def main():
    parser = argparse.ArgumentParser(
        description="Compare value-trace summary metrics across solver JSON outputs."
    )
    parser.add_argument("inputs", nargs="+", help="JSON result files to compare")
    parser.add_argument("-o", "--output", required=True, help="CSV output path")
    args = parser.parse_args()

    loaded = []
    for raw_path in args.inputs:
        path = Path(raw_path)
        solver_name, rows = load_solver_results(path)
        loaded.append((solver_name, rows))

    all_names = sorted(set().union(*(set(rows.keys()) for _, rows in loaded)))

    out_rows = []
    for name in all_names:
        row = {"name": name}
        delay_cost_type = None
        best_value_candidates = []
        proof_candidates = []
        optimality_candidates = []
        cost_values = []

        for solver_name, rows in loaded:
            solve = rows.get(name, {})
            if delay_cost_type is None:
                delay_cost_type = solve.get("delay_cost_type")
            prefix = solver_name
            for field in SUMMARY_FIELDS:
                row[f"{prefix}_{field}"] = solve.get(field)
            best_value_candidates.append(
                (solver_name, solve.get("time_to_best_value_ms"))
            )
            proof_candidates.append(
                (solver_name, solve.get("time_to_prove_best_value_ms"))
            )
            optimality_candidates.append(
                (solver_name, solve.get("time_to_optimality_ms"))
            )
            if solve.get("cost") is not None:
                cost_values.append(solve.get("cost"))

        row["delay_cost_type"] = delay_cost_type
        row["consistent_cost"] = (
            len(set(cost_values)) == 1 if cost_values else None
        )
        row["fastest_best_value_solver"] = fastest_solver(best_value_candidates)
        row["fastest_proof_solver"] = fastest_solver(proof_candidates)
        row["fastest_optimality_solver"] = fastest_solver(optimality_candidates)
        out_rows.append(row)

    fieldnames = sorted({key for row in out_rows for key in row.keys()})
    out_path = Path(args.output)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
