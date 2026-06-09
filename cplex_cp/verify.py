"""Verify nghiệm CP Optimizer cho TRP.

Parse output text từ cpoptimizer CLI, extract schedule, verify constraints,
compute cost.

CP Optimizer output format khác CPLEX:
- Text-based, có header với "Objective values" và "Variables"
- Mỗi interval var hiển thị như: `v_0_0: [start, end, size]`

Usage:
    python verify.py <instance.txt> <cost_type> <out_file>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

from parser import Problem, parse_txt
from cost import DelayCostType, parse_cost_type
from utils import compute_cost, verify_solution


# ────────────────────────────────────────────────────────────
# Parse cpoptimizer output
# ────────────────────────────────────────────────────────────
def parse_cpo_output(
    out_path: str,
) -> Tuple[Dict[str, Tuple[float, float]], Optional[float], Optional[str]]:
    """Parse cpoptimizer text output.

    Returns:
        (interval_values, objective, status)
        interval_values = {var_name: (start, end)}
        objective: float | None
        status: 'optimal' | 'feasible' | 'no_solution' | None
    """
    with open(out_path) as f:
        content = f.read()

    interval_values: Dict[str, Tuple[float, float]] = {}
    objective: Optional[float] = None
    status: Optional[str] = None

    # Status detection
    if (
        "Search completed, model has no solution" in content
        or "Model is infeasible" in content
    ):
        status = "infeasible"
    elif "Search completed, objective" in content:
        status = "optimal"
    elif "Search terminated by limit, objective" in content:
        status = "feasible"
    elif "Solve found" in content:
        # CPO output có "Solve found N solutions"
        if "optimal" in content.lower():
            status = "optimal"
        else:
            status = "feasible"

    # Objective parsing — multiple format variants
    # Format 1: "Search completed, objective = 11"
    m = re.search(r"objective\s*=\s*([-+]?\d+\.?\d*(?:e[-+]?\d+)?)", content)
    if m:
        try:
            objective = float(m.group(1))
        except ValueError:
            pass

    # Format 2: "Best objective: 11.0"
    if objective is None:
        m = re.search(
            r"Best\s+objective\s*[:=]?\s*([-+]?\d+\.?\d*(?:e[-+]?\d+)?)",
            content,
            re.IGNORECASE,
        )
        if m:
            try:
                objective = float(m.group(1))
            except ValueError:
                pass

    # Interval var values
    # Format: "v_0_0: [start, end, size]" hoặc "v_0_0 = intervalVar(...)"
    # CPO output: "v_0_0: [a -- b -- c]" (start, end, size)
    # Or: "v_0_0 = intervalVar(start=N, end=M, size=K)"
    var_pattern = re.compile(
        r"(v_\d+_\d+)\s*[:=]\s*"
        r"(?:intervalVar\s*\(\s*"
        r"start\s*=\s*(\d+)\s*,\s*"
        r"end\s*=\s*(\d+)"
        r"|\[\s*(\d+)\s*[-,]\s*(\d+)"
        r")",
        re.MULTILINE,
    )
    for m in var_pattern.finditer(content):
        name = m.group(1)
        # Either group 2-3 (intervalVar=) or 4-5 (bracket)
        if m.group(2) is not None and m.group(3) is not None:
            start, end = float(m.group(2)), float(m.group(3))
        elif m.group(4) is not None and m.group(5) is not None:
            start, end = float(m.group(4)), float(m.group(5))
        else:
            continue
        interval_values[name] = (start, end)

    return interval_values, objective, status


def extract_schedule_cp(
    problem: Problem,
    interval_values: Dict[str, Tuple[float, float]],
) -> Dict[Tuple[int, int], float]:
    """Extract schedule từ CP interval values."""
    schedule = {}
    for ti, train in enumerate(problem.trains):
        for vi in range(len(train.visits)):
            name = f"v_{ti}_{vi}"
            if name in interval_values:
                start, _end = interval_values[name]
                schedule[(ti, vi)] = start
            else:
                # Fallback to earliest if missing
                schedule[(ti, vi)] = float(train.visits[vi].earliest)
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

    interval_values, cpo_objective, cpo_status = parse_cpo_output(out_path)
    print(f"\nParsed {len(interval_values)} interval variables")

    if not interval_values:
        print("ERROR: no interval values found in output")
        print("CPO status:", cpo_status)
        sys.exit(3)

    schedule = extract_schedule_cp(problem, interval_values)
    cost = compute_cost(problem, schedule, cost_type)
    valid, msg = verify_solution(problem, schedule, cost_type, tolerance=1.0)

    print(f"\n=== RESULTS ===")
    if cpo_objective is not None:
        print(f"CPO objective: {cpo_objective}")
    if cpo_status:
        print(f"CPO status: {cpo_status}")
    print(f"Verified cost: {cost}")
    print(f"Valid: {valid} ({msg})")

    if not valid:
        sys.exit(5)


if __name__ == "__main__":
    main()
