"""Verify + emit 1 row CSV theo schema Gurobi (mip_ddd).

Schema:
algorithm_time, avg_tracks, conflicting_visit_pairs, conflicts, cost,
delay_cost_type, index, internal_cost, intervals, iteration, lb, name,
resource_constraints, sol_time, solver_name, solver_time, status, total_time,
trains, travel_constraints, ub

Usage:
    python verify_gurobi.py <instance> <solver> <cost_type> <sol_or_none> \
        <cplex_log> <build_time> <solve_wallclock> <build_stdout_log>
        [<index>]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional, Tuple

from parser import parse_txt
from cost import parse_cost_type
from utils import compute_cost, verify_solution, visit_conflicts
from verify import (
    parse_sol,
    extract_schedule_bigm,
    extract_schedule_ti,
    postprocess_ti,
)


# ─────────────────────────────────────────────────────────────────
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


# Mapping variant_dir → name prefix (matching Gurobi naming).
def short_name(instance_path: str) -> str:
    p = Path(instance_path)
    stem = p.stem  # InstanceA1
    # Strip "Instance" → A1
    short = stem.replace("Instance", "")
    # Determine prefix from parent dir.
    parent = p.parent.name
    if parent == "original":
        prefix = "orig"
    elif parent == "addstationtime":
        prefix = "addstation"
    elif parent == "addtracktime":
        prefix = "addtrack"
    else:
        prefix = parent
    return f"{prefix}{short}"


# ─────────────────────────────────────────────────────────────────
def parse_cplex_log(log_path: str) -> dict:
    """Lấy lb, ub, solver_time, status, lazy_applied từ CPLEX log."""
    out = {
        "lb": None,
        "ub": None,
        "solver_time": None,
        "status": "unknown",
        "lazy_applied": None,
    }
    try:
        text = Path(log_path).read_text(errors="ignore")
    except FileNotFoundError:
        return out

    # CPLEX cut summary lines: "Lazy constraints applied:  X"
    # hoặc "User cuts applied: X". Khi không trigger, dòng thiếu hẳn.
    m = re.search(
        r"Lazy constraints?\s*applied\s*:\s*(\d+)", text, re.IGNORECASE
    )
    if m:
        out["lazy_applied"] = int(m.group(1))

    # Solution time (CPLEX-internal): "Solution time = 60.10 sec."
    m = re.search(r"Solution time =\s*([\d.]+)\s*sec", text)
    if m:
        out["solver_time"] = float(m.group(1))

    # Best bound: "Current MIP best bound =  4.9090e+03"
    m = re.search(
        r"Current MIP best bound\s*=\s*([-+\d.e]+)", text, re.IGNORECASE
    )
    if m:
        out["lb"] = float(m.group(1))

    # Optimal integer: "Integer optimal solution:  Objective =  1.10..."
    m = re.search(r"Integer optimal[^\n]*Objective\s*=\s*([-+\d.e]+)", text)
    if m:
        out["ub"] = float(m.group(1))
        out["status"] = "ok"
    else:
        # Best integer feasible during MIP: "Best integer = X.X"
        m2 = re.search(r"Best integer[^=]*=\s*([-+\d.e]+)", text)
        if m2:
            out["ub"] = float(m2.group(1))

    # Status priority order.
    if "MIP - Time limit exceeded, no integer solution" in text:
        out["status"] = "no_solution"
    elif "Integer optimal solution" in text:
        out["status"] = "ok"
    elif "MIP - Integer optimal" in text:
        out["status"] = "ok"
    elif "Time limit exceeded" in text:
        out["status"] = "timeout"
    elif "Infeasible" in text:
        out["status"] = "infeasible"

    # Khi optimal mà CPLEX không in "best bound" riêng → lb = ub (gap=0).
    if out["status"] == "ok" and out["ub"] is not None and out["lb"] is None:
        out["lb"] = out["ub"]

    return out


def parse_build_stats(build_log_path: str) -> dict:
    """Lấy travel/resource/intervals/vars/constraints từ Rust binary stdout log."""
    out = {
        "travel_constraints": 0,
        "resource_constraints": 0,
        "intervals": 0,
        "n_vars": 0,
        "n_constraints": 0,
    }
    try:
        text = Path(build_log_path).read_text(errors="ignore")
    except FileNotFoundError:
        return out

    patterns = {
        "n_vars": r"Vars:\s*(\d+)",
        "n_constraints": r"Constraints:\s*(\d+)",
        "travel_constraints": r"Travel constraints:\s*(\d+)",
        "resource_constraints": r"Resource constraints:\s*(\d+)",
        "intervals": r"Intervals:\s*(\d+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            out[key] = int(m.group(1))
    return out


# ─────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 9:
        print(__doc__)
        sys.exit(1)

    instance_path = sys.argv[1]
    solver = sys.argv[2].lower()
    cost_str = sys.argv[3]
    sol_path = sys.argv[4]
    cplex_log = sys.argv[5]
    build_time = float(sys.argv[6])
    solve_wallclock = float(sys.argv[7])
    build_stdout_log = sys.argv[8]
    index = int(sys.argv[9]) if len(sys.argv) > 9 else 0

    cost_type = parse_cost_type(cost_str)
    problem = parse_txt(instance_path)

    # Problem-level stats.
    n_trains = len(problem.trains)
    n_visit_conflict_pairs = len(visit_conflicts(problem))
    n_resource_conflicts = len(problem.conflicts)
    # avg_tracks = single-track visits / #trains (port từ main.rs:1177).
    conflict_set = set(problem.conflicts)
    n_track_visits = sum(
        sum(1 for v in t.visits if (v.resource_id, v.resource_id) in conflict_set)
        for t in problem.trains
    )
    avg_tracks = n_track_visits / n_trains if n_trains > 0 else 0.0

    # CPLEX log: lb, ub, solver_time, status.
    cplex_info = parse_cplex_log(cplex_log)
    # Build stats: travel/resource/intervals.
    build_info = parse_build_stats(build_stdout_log)

    cost: Optional[int] = None
    valid: Optional[bool] = None
    internal_cost: Optional[float] = None

    has_sol = sol_path != "NONE" and Path(sol_path).exists()

    if has_sol:
        try:
            variables, header = parse_sol(sol_path)
            if solver == "bigm":
                schedule = extract_schedule_bigm(problem, variables)
            elif solver == "ti":
                ti_sched = extract_schedule_ti(problem, variables)
                workdir = str(Path(sol_path).parent)
                schedule = postprocess_ti(
                    problem, ti_sched, workdir=workdir, verbose=False
                )
            else:
                schedule = None

            if schedule is not None:
                cost = compute_cost(problem, schedule, cost_type)
                v, _msg = verify_solution(
                    problem, schedule, cost_type, tolerance=1.0
                )
                valid = v
                if "objectiveValue" in header:
                    try:
                        internal_cost = float(header["objectiveValue"])
                    except ValueError:
                        pass
        except Exception as exc:
            print(f"# verify failed: {exc}", file=sys.stderr)

    # Map status (override if we have valid cost).
    status = cplex_info["status"]
    if has_sol and valid:
        status = "ok"
    elif has_sol and valid is False:
        status = "invalid"
    elif not has_sol and status not in ("no_solution", "infeasible", "timeout"):
        status = "no_solution"

    # Solver_time: CPLEX-internal time; sol_time: wallclock; total = build+sol.
    solver_time = (
        cplex_info["solver_time"]
        if cplex_info["solver_time"] is not None
        else solve_wallclock
    )
    sol_time = solve_wallclock
    total_time = build_time + sol_time

    # Solver label. BigM của CPLEX dùng lazy (CPLEX LP Lazy Constraints).
    solver_label_map = {
        "bigm": "BigM_CPLEX",
        "ti": "TI_CPLEX",
    }
    solver_label = solver_label_map.get(solver, solver)

    # Output CSV row (header tự handle ngoài).
    cost_label = COST_TYPE_LABEL.get(cost_str.lower(), cost_str)
    name = short_name(instance_path)
    lb = cplex_info["lb"] if cplex_info["lb"] is not None else ""
    ub = (
        cplex_info["ub"]
        if cplex_info["ub"] is not None
        else (internal_cost if internal_cost is not None else "")
    )
    internal_cost_out = internal_cost if internal_cost is not None else ""
    intervals = build_info["intervals"]

    # BigM eager: report số constraints declared (2 * pairs).
    # (Eager dùng cho correctness — CPLEX lazy bị presolve làm sai nghiệm.)
    resource_constraints_out = build_info["resource_constraints"]

    cols = [
        f"{build_time}",                   # algorithm_time
        f"{avg_tracks}",                   # avg_tracks
        str(n_visit_conflict_pairs),       # conflicting_visit_pairs
        str(n_resource_conflicts),         # conflicts
        ("" if cost is None else str(cost)),  # cost
        cost_label,                        # delay_cost_type
        str(index),                        # index
        f"{internal_cost_out}",            # internal_cost
        str(intervals),                    # intervals
        "1",                               # iteration
        f"{lb}",                           # lb
        name,                              # name
        str(resource_constraints_out),     # resource_constraints
        f"{sol_time}",                     # sol_time
        solver_label,                      # solver_name
        f"{solver_time}",                  # solver_time
        status,                            # status
        f"{total_time}",                   # total_time
        str(n_trains),                     # trains
        str(build_info["travel_constraints"]),  # travel_constraints
        f"{ub}",                           # ub
    ]
    print(",".join(cols))


if __name__ == "__main__":
    main()
