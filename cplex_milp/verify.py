"""Verify nghiệm CPLEX (.sol file) cho TRP.

Parse XML solution, extract schedule, verify constraints, compute cost.

Cho TI: thực hiện POST-PROCESSING giống Rust:
  1. Extract priorities từ TI schedule
  2. Build minimize_solution LP → file .lp
  3. Solve LP bằng CPLEX CLI
  4. Lấy continuous schedule từ LP solution
  5. Compute cost từ continuous schedule (khớp Rust TI)

Usage:
    python verify.py <instance.txt> <solver> <cost_type> <sol_file>
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

from parser import Problem, parse_txt
from cost import DelayCostType, parse_cost_type
from utils import compute_cost, verify_solution, visit_conflicts


# ────────────────────────────────────────────────────────────
# Parse CPLEX .sol XML
# ────────────────────────────────────────────────────────────
def parse_sol(sol_path: str) -> Tuple[Dict[str, float], Dict]:
    tree = ET.parse(sol_path)
    root = tree.getroot()
    header = {}
    h = root.find(".//header")
    if h is not None:
        header = dict(h.attrib)
    variables = {}
    for var in root.findall(".//variables/variable"):
        name = var.get("name")
        value = var.get("value")
        if name and value:
            try:
                variables[name] = float(value)
            except ValueError:
                pass
    return variables, header


# ────────────────────────────────────────────────────────────
# Extract schedule
# ────────────────────────────────────────────────────────────
def extract_schedule_bigm(
    problem: Problem, variables: Dict[str, float]
) -> Dict[Tuple[int, int], float]:
    schedule = {}
    for ti, train in enumerate(problem.trains):
        for vi in range(len(train.visits)):
            name = f"t_{ti}_{vi}"
            schedule[(ti, vi)] = variables.get(name, 0.0)
    return schedule


def extract_schedule_ti(
    problem: Problem,
    variables: Dict[str, float],
    threshold: float = 0.5,
) -> Dict[Tuple[int, int], float]:
    schedule = {}
    pattern = re.compile(r"^x_(\d+)_(\d+)_(\d+)$")
    selected = {}
    for name, val in variables.items():
        if val < threshold:
            continue
        m = pattern.match(name)
        if not m:
            continue
        ti, vi, slot_t = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if (ti, vi) not in selected or slot_t < selected[(ti, vi)]:
            selected[(ti, vi)] = slot_t

    for ti, train in enumerate(problem.trains):
        for vi in range(len(train.visits)):
            t_val = selected.get((ti, vi))
            if t_val is None:
                t_val = train.visits[vi].earliest
            schedule[(ti, vi)] = float(t_val)
    return schedule


# ────────────────────────────────────────────────────────────
# Post-processing cho TI: extract priorities + minimize LP
# ────────────────────────────────────────────────────────────
def extract_priorities(
    problem: Problem,
    ti_schedule: Dict[Tuple[int, int], float],
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Extract priorities từ TI schedule.

    Port 1:1 từ milp_ti.rs lines 252-258:
        for (a @ (t1, v1), b @ (t2, v2)) in visit_conflicts(&problem) {
            if solution[t1][v1] <= solution[t2][v2] {
                priorities.push((a, b));
            } else {
                priorities.push((b, a));
            }
        }
    """
    priorities = []
    for ((t1, v1), (t2, v2)) in visit_conflicts(problem):
        a = (t1, v1)
        b = (t2, v2)
        if ti_schedule[a] <= ti_schedule[b]:
            priorities.append((a, b))
        else:
            priorities.append((b, a))
    return priorities


def write_minimize_lp(
    problem: Problem,
    priorities: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    lp_path: str,
) -> None:
    """Build minimize_solution LP (port 1:1 từ util/minimize.rs).

    LP simple:
    - Vars: t[i][v] continuous, lb=earliest, obj=1.0
    - Travel: t[i][v+1] - t[i][v] >= travel
    - Priority: t[t1][v1+1] <= t[t2][v2]
    - Minimize: sum(t)
    """
    with open(lp_path, "w") as f:
        f.write("\\ minimize_solution LP from milp_ti post-processing\n")
        f.write("Minimize\n obj:\n")

        # Objective: sum of all t[i][v]
        terms = []
        for ti, train in enumerate(problem.trains):
            for vi in range(len(train.visits)):
                terms.append(f"t_{ti}_{vi}")
        # Write objective in multiline format
        line = "    "
        first = True
        for term in terms:
            prefix = "" if first else " + "
            if len(line) + len(prefix) + len(term) > 400:
                f.write(line + "\n")
                line = "    + " + term
            else:
                line += prefix + term
            first = False
        f.write(line + "\n")

        f.write("Subject To\n")
        c_idx = 0

        # Travel constraints
        for ti, train in enumerate(problem.trains):
            for vi in range(len(train.visits) - 1):
                travel = train.visits[vi].travel_time
                f.write(
                    f" c{c_idx}: t_{ti}_{vi + 1} - t_{ti}_{vi} >= {travel}\n"
                )
                c_idx += 1

        # Priority constraints
        for ((t1, v1), (t2, v2)) in priorities:
            # Skip if no next visit
            if v1 + 1 >= len(problem.trains[t1].visits):
                continue
            f.write(f" c{c_idx}: t_{t1}_{v1 + 1} - t_{t2}_{v2} <= 0\n")
            c_idx += 1

        # Bounds: t[i][v] >= earliest[i][v]
        f.write("Bounds\n")
        for ti, train in enumerate(problem.trains):
            for vi, visit in enumerate(train.visits):
                # No upper bound, just lower
                f.write(f" t_{ti}_{vi} >= {visit.earliest}\n")

        f.write("End\n")


def solve_lp_with_cplex(lp_path: str, sol_path: str, timeout: int = 60) -> bool:
    """Call CPLEX CLI để solve LP file."""
    # Find cplex binary
    cplex_bin = "cplex"
    if "CPLEX_HOME" in os.environ:
        candidate = os.path.join(
            os.environ["CPLEX_HOME"], "bin", "x86-64_linux", "cplex"
        )
        if os.path.exists(candidate):
            cplex_bin = candidate

    if os.path.exists(sol_path):
        os.remove(sol_path)

    cmds = [
        f"set timelimit {timeout}",
        "set threads 4",
        f"read {lp_path}",
        "optimize",
        f"write {sol_path}",
        "quit",
    ]
    proc = subprocess.run(
        [cplex_bin, "-c"] + cmds,
        capture_output=True,
        text=True,
        timeout=timeout + 30,
    )
    return os.path.exists(sol_path)


def parse_continuous_schedule(
    problem: Problem, sol_path: str
) -> Dict[Tuple[int, int], float]:
    """Parse .sol file của minimize_solution → continuous schedule."""
    variables, _ = parse_sol(sol_path)
    schedule = {}
    for ti, train in enumerate(problem.trains):
        for vi in range(len(train.visits)):
            name = f"t_{ti}_{vi}"
            schedule[(ti, vi)] = variables.get(name, 0.0)
    return schedule


def postprocess_ti(
    problem: Problem,
    ti_schedule: Dict[Tuple[int, int], float],
    workdir: str = None,
    verbose: bool = True,
) -> Dict[Tuple[int, int], float]:
    """Full TI post-processing pipeline (port 1:1 Rust milp_ti.rs).

    1. Extract priorities
    2. Build minimize_solution LP
    3. Solve LP via CPLEX CLI
    4. Return continuous schedule
    """
    if workdir is None:
        workdir = tempfile.gettempdir()

    if verbose:
        print(f"  Post-processing TI (port 1:1 Rust)...")

    priorities = extract_priorities(problem, ti_schedule)
    if verbose:
        print(f"    Extracted {len(priorities)} priorities")

    lp_path = os.path.join(workdir, "_ti_minimize.lp")
    sol_path = os.path.join(workdir, "_ti_minimize.sol")
    write_minimize_lp(problem, priorities, lp_path)
    if verbose:
        size = os.path.getsize(lp_path) / 1024
        print(f"    Wrote LP file ({size:.1f} KB)")

    if not solve_lp_with_cplex(lp_path, sol_path):
        if verbose:
            print(f"    LP solve FAILED, falling back to TI schedule")
        return ti_schedule

    continuous_schedule = parse_continuous_schedule(problem, sol_path)
    if verbose:
        print(f"    Got continuous schedule from minimize_solution LP")
    return continuous_schedule


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(1)

    instance_path = sys.argv[1]
    solver = sys.argv[2].lower()
    cost_str = sys.argv[3]
    sol_path = sys.argv[4]

    cost_type = parse_cost_type(cost_str)
    problem = parse_txt(instance_path)

    print(f"Loading {Path(instance_path).name}...")
    print(f"  Solver: {solver}")
    print(f"  Cost type: {cost_type.value}")
    print(f"  Solution file: {sol_path}")

    if not Path(sol_path).exists():
        print(f"ERROR: solution file not found: {sol_path}")
        sys.exit(2)

    try:
        variables, header = parse_sol(sol_path)
    except ET.ParseError as e:
        print(f"ERROR parsing .sol XML: {e}")
        sys.exit(3)

    print(f"\nParsed {len(variables)} variables")

    if solver == "bigm":
        schedule = extract_schedule_bigm(problem, variables)
    elif solver == "ti":
        # Bước 1: extract TI schedule (discretized, có thể vi phạm earliest)
        ti_schedule = extract_schedule_ti(problem, variables)
        # Bước 2: post-process bằng minimize_solution LP qua CPLEX CLI
        # (port 1:1 từ Rust milp_ti.rs lines 246-260)
        workdir = os.path.dirname(os.path.abspath(sol_path))
        schedule = postprocess_ti(problem, ti_schedule, workdir=workdir)
    else:
        print(f"ERROR: unknown solver '{solver}'")
        sys.exit(4)

    cost = compute_cost(problem, schedule, cost_type)
    valid, msg = verify_solution(problem, schedule, cost_type, tolerance=1.0)

    print(f"\n=== RESULTS ===")
    if "objectiveValue" in header:
        print(f"CPLEX TI objective (relaxation): {header['objectiveValue']}")
    if "solutionStatusString" in header:
        print(f"CPLEX status: {header['solutionStatusString']}")
    if "MIPRelativeGap" in header:
        try:
            gap = float(header["MIPRelativeGap"])
            print(f"MIP gap: {gap * 100:.4f}%")
        except ValueError:
            pass
    print(f"Verified cost: {cost}")
    print(f"Valid: {valid} ({msg})")

    if not valid:
        sys.exit(5)


if __name__ == "__main__":
    main()
