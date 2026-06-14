"""Verify + emit 1 row CSV theo schema Gurobi (mip_ddd) cho CP Optimizer.

Schema:
algorithm_time, avg_tracks, conflicting_visit_pairs, conflicts, cost,
delay_cost_type, index, internal_cost, intervals, iteration, lb, name,
resource_constraints, sol_time, solver_name, solver_time, status, total_time,
trains, travel_constraints, ub

Usage:
    python verify_gurobi.py <instance> <cost_type> <oplrun_out_or_none> \
        <build_time> <solve_wallclock> [<index>]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional, Tuple

from parser import parse_txt
from cost import parse_cost_type
from utils import compute_cost, verify_solution, visit_conflicts
from verify import parse_oplrun_output, extract_schedule_opl


COST_TYPE_LABEL = {
    "finsteps1_3min": "FiniteSteps1_3Min",
    "finsteps1_5min": "FiniteSteps1_5Min",
    "finsteps123": "FiniteSteps123",
    "finsteps12345": "FiniteSteps12345",
    "finsteps139": "FiniteSteps139",
    "infsteps60": "InfiniteSteps60",
    "infsteps180": "InfiniteSteps180",
    "infsteps360": "InfiniteSteps360",
    "cont": "Continuous",
}


def short_name(instance_path: str) -> str:
    p = Path(instance_path)
    stem = p.stem.replace("Instance", "")
    parent = p.parent.name
    prefix = {
        "original": "orig",
        "addstationtime": "addstation",
        "addtracktime": "addtrack",
    }.get(parent, parent)
    return f"{prefix}{stem}"


def parse_oplrun_meta(out_path: str) -> dict:
    """Extract solver_time / lb / ub / status từ oplrun stdout."""
    out = {"solver_time": None, "lb": None, "ub": None, "status": "unknown"}
    try:
        text = Path(out_path).read_text(errors="ignore")
    except FileNotFoundError:
        return out

    # CP Optimizer prints "Time spent in solve : 12.34s"
    m = re.search(r"Time spent in solve\s*:\s*([\d.]+)\s*s", text)
    if m:
        out["solver_time"] = float(m.group(1))
    # Or "! Time = 12.34s"
    if out["solver_time"] is None:
        m = re.search(r"!\s*Time\s*=\s*([\d.]+)\s*s", text)
        if m:
            out["solver_time"] = float(m.group(1))

    # ! Best objective         : 11
    m = re.search(r"Best objective[^\d-]*([-+\d.e]+)", text)
    if m:
        try:
            out["ub"] = float(m.group(1))
        except ValueError:
            pass
    # ! Best bound  / "Lower bound = X"
    m = re.search(r"Best bound[^\d-]*([-+\d.e]+)", text, re.IGNORECASE)
    if m:
        try:
            out["lb"] = float(m.group(1))
        except ValueError:
            pass

    # Status detection.
    if "OPTIMAL SOLUTION FOUND" in text.upper():
        out["status"] = "ok"
    elif "Solution status: Optimal" in text:
        out["status"] = "ok"
    elif re.search(r"OBJECTIVE:\s*\d", text):
        out["status"] = "ok"
    elif "infeasible" in text.lower():
        out["status"] = "infeasible"
    elif "no solution" in text.lower():
        out["status"] = "no_solution"

    if out["status"] == "ok" and out["ub"] is not None and out["lb"] is None:
        out["lb"] = out["ub"]

    return out


def main():
    if len(sys.argv) < 6:
        print(__doc__)
        sys.exit(1)

    instance_path = sys.argv[1]
    cost_str = sys.argv[2]
    out_path = sys.argv[3]
    build_time = float(sys.argv[4])
    solve_wallclock = float(sys.argv[5])
    index = int(sys.argv[6]) if len(sys.argv) > 6 else 0

    cost_type = parse_cost_type(cost_str)
    problem = parse_txt(instance_path)

    n_trains = len(problem.trains)
    n_visit_conflict_pairs = len(visit_conflicts(problem))
    n_resource_conflicts = len(problem.conflicts)
    n_visits = sum(len(t.visits) for t in problem.trains)
    # avg_tracks = single-track visits / #trains (port từ main.rs:1177).
    conflict_set = set(problem.conflicts)
    n_track_visits = sum(
        sum(1 for v in t.visits if (v.resource_id, v.resource_id) in conflict_set)
        for t in problem.trains
    )
    avg_tracks = n_track_visits / n_trains if n_trains > 0 else 0.0

    # Travel constraints = #endBeforeStart trong CP (per consecutive pair).
    travel_constraints = sum(max(0, len(t.visits) - 1) for t in problem.trains)
    # Resource constraints = #noOverlap pairs (cùng metric với MILP).
    resource_constraints = n_visit_conflict_pairs
    # Intervals = total interval_var = total visits.
    intervals = n_visits

    has_out = out_path != "NONE" and Path(out_path).exists()

    cost: Optional[int] = None
    valid: Optional[bool] = None
    internal_cost: Optional[float] = None
    meta = parse_oplrun_meta(out_path) if has_out else {
        "solver_time": None, "lb": None, "ub": None, "status": "no_solution",
    }

    if has_out:
        try:
            visit_values, oplrun_obj, _ = parse_oplrun_output(out_path)
            if visit_values:
                schedule = extract_schedule_opl(problem, visit_values)
                cost = compute_cost(problem, schedule, cost_type)
                v, _ = verify_solution(problem, schedule, cost_type, tolerance=1.0)
                valid = v
                if oplrun_obj is not None:
                    internal_cost = oplrun_obj
        except Exception as exc:
            print(f"# verify failed: {exc}", file=sys.stderr)

    status = meta["status"]
    if has_out and valid:
        status = "ok"
    elif has_out and valid is False:
        status = "invalid"
    elif not has_out and status == "unknown":
        status = "no_solution"

    solver_time = (
        meta["solver_time"] if meta["solver_time"] is not None else solve_wallclock
    )
    sol_time = solve_wallclock
    total_time = build_time + sol_time

    cost_label = COST_TYPE_LABEL.get(cost_str.lower(), cost_str)
    name = short_name(instance_path)
    lb = meta["lb"] if meta["lb"] is not None else ""
    ub = (
        meta["ub"]
        if meta["ub"] is not None
        else (internal_cost if internal_cost is not None else "")
    )
    internal_cost_out = internal_cost if internal_cost is not None else ""

    cols = [
        f"{build_time}",                   # algorithm_time
        f"{avg_tracks}",                   # avg_tracks
        str(n_visit_conflict_pairs),       # conflicting_visit_pairs
        str(n_resource_conflicts),         # conflicts
        ("" if cost is None else str(cost)),
        cost_label,
        str(index),
        f"{internal_cost_out}",            # internal_cost
        str(intervals),
        "1",                               # iteration
        f"{lb}",
        name,
        str(resource_constraints),
        f"{sol_time}",
        "CP_CPLEX",                        # solver_name
        f"{solver_time}",
        status,
        f"{total_time}",
        str(n_trains),
        str(travel_constraints),
        f"{ub}",
    ]
    print(",".join(cols))


if __name__ == "__main__":
    main()
