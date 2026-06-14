"""Verify nghiệm OPL + CP Optimizer cho TRP.

Parse output text từ oplrun, extract schedule, verify constraints, compute cost.

oplrun output format (do execute block in .mod file):
    OBJECTIVE: 11
    v[1]: start=0 end=0
    v[2]: start=0 end=470
    ...

Usage:
    python verify.py <instance.txt> <cost_type> <out_file>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from parser import Problem, parse_txt
from cost import DelayCostType, parse_cost_type
from utils import compute_cost, verify_solution


def parse_oplrun_output(
    out_path: str,
) -> Tuple[Dict[int, Tuple[int, int]], Optional[float], Optional[str]]:
    """Parse oplrun text output.

    Returns:
        (visit_values, objective, status)
        visit_values = {1-based-idx: (start, end)}
    """
    with open(out_path) as f:
        content = f.read()

    visit_values: Dict[int, Tuple[int, int]] = {}
    objective: Optional[float] = None
    status: Optional[str] = None

    # Status detection
    if "OBJECTIVE:" in content:
        status = "feasible"  # may upgrade to "optimal" based on solver status
    if "OPTIMAL SOLUTION FOUND" in content.upper():
        status = "optimal"
    elif "Solution status: Optimal" in content:
        status = "optimal"
    elif "Feasible solution" in content:
        status = "feasible"
    elif "infeasible" in content.lower():
        status = "infeasible"

    # Objective parsing
    # Format from execute block: "OBJECTIVE: 11"
    m = re.search(r"OBJECTIVE:\s*([-+]?\d+\.?\d*(?:e[-+]?\d+)?)", content)
    if m:
        try:
            objective = float(m.group(1))
        except ValueError:
            pass

    # Backup: parse from oplrun's standard "OBJECTIVE = ..."
    if objective is None:
        m = re.search(
            r"OBJECTIVE\s*[:=]\s*([-+]?\d+\.?\d*(?:e[-+]?\d+)?)",
            content,
            re.IGNORECASE,
        )
        if m:
            try:
                objective = float(m.group(1))
            except ValueError:
                pass

    # Visit values: "v[1]: start=0 end=0"
    var_pattern = re.compile(
        r"v\[(\d+)\]:\s*start\s*=\s*(\d+)\s+end\s*=\s*(\d+)"
    )
    for m in var_pattern.finditer(content):
        idx = int(m.group(1))
        start = int(m.group(2))
        end = int(m.group(3))
        visit_values[idx] = (start, end)

    return visit_values, objective, status


def build_visit_index(problem: Problem) -> List[Tuple[int, int]]:
    """Build flat visit list (1-indexed): index → (train_idx, visit_idx).

    Phải match thứ tự trong export_opl.rs.
    """
    flat: List[Tuple[int, int]] = []
    for ti, train in enumerate(problem.trains):
        for vi, _visit in enumerate(train.visits):
            flat.append((ti, vi))
    return flat


def extract_schedule_opl(
    problem: Problem,
    visit_values: Dict[int, Tuple[int, int]],
) -> Dict[Tuple[int, int], float]:
    """Extract schedule từ OPL output."""
    flat = build_visit_index(problem)
    schedule = {}
    for idx, (ti, vi) in enumerate(flat, start=1):
        if idx in visit_values:
            start, _end = visit_values[idx]
            schedule[(ti, vi)] = float(start)
        else:
            # Fallback to earliest
            schedule[(ti, vi)] = float(problem.trains[ti].visits[vi].earliest)
    return schedule


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    instance_path = sys.argv[1]
    cost_str = sys.argv[2]
    out_path = sys.argv[3]

    cost_type = parse_cost_type(cost_str)
    problem = parse_txt(instance_path)

    print(f"Loading {Path(instance_path).name}...")
    print(f"  Cost type: {cost_type.value}")
    print(f"  Output file: {out_path}")

    if not Path(out_path).exists():
        print(f"ERROR: output file not found: {out_path}")
        sys.exit(2)

    visit_values, oplrun_objective, oplrun_status = parse_oplrun_output(out_path)
    print(f"\nParsed {len(visit_values)} visit values")

    if not visit_values:
        print("ERROR: no visit values found in output")
        print(f"OPLrun status: {oplrun_status}")
        print(f"Check {out_path} for parsing errors")
        sys.exit(3)

    schedule = extract_schedule_opl(problem, visit_values)
    cost = compute_cost(problem, schedule, cost_type)
    valid, msg = verify_solution(problem, schedule, cost_type, tolerance=1.0)

    print(f"\n=== RESULTS ===")
    if oplrun_objective is not None:
        print(f"OPLrun objective: {oplrun_objective}")
    if oplrun_status:
        print(f"OPLrun status: {oplrun_status}")
    print(f"Verified cost: {cost}")
    print(f"Valid: {valid} ({msg})")

    if not valid:
        sys.exit(5)


if __name__ == "__main__":
    main()
