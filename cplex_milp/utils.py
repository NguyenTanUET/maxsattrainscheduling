"""Helper utilities cho TRP:

- visit_conflicts: enumerate cặp (visit1, visit2) tiềm năng xung đột tài nguyên
- detect_conflicts_in_schedule: kiểm tra lịch trình có vi phạm không
- compute_cost: tính tổng cost trễ cho 1 nghiệm
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from parser import Problem
from cost import DelayCostType, visit_delay_cost


VisitId = Tuple[int, int]  # (train_idx, visit_idx)
VisitPair = Tuple[VisitId, VisitId]


def visit_conflicts(problem: Problem) -> List[VisitPair]:
    """Enumerate tất cả cặp visit (từ 2 tàu khác nhau) có thể xung đột tài nguyên.

    Port từ visit_conflicts() trong bigm.rs.
    """
    conflict_set = set(problem.conflicts)
    pairs: List[VisitPair] = []
    n_trains = len(problem.trains)
    for t1 in range(n_trains):
        for t2 in range(t1 + 1, n_trains):
            for v1, visit1 in enumerate(problem.trains[t1].visits):
                for v2, visit2 in enumerate(problem.trains[t2].visits):
                    r1, r2 = visit1.resource_id, visit2.resource_id
                    if (r1, r2) in conflict_set or (r2, r1) in conflict_set:
                        pairs.append(((t1, v1), (t2, v2)))
    return pairs


def check_pair_separated(
    problem: Problem,
    schedule: Dict[VisitId, float],
    pair: VisitPair,
    tolerance: float = 1.0,
) -> bool:
    """Kiểm tra cặp visit có tách biệt thời gian không (không overlap).

    Tính end thực = start + travel_time (đúng cho cả MILP và CP).
    """
    (t1, v1), (t2, v2) = pair
    visit1 = problem.trains[t1].visits[v1]
    visit2 = problem.trains[t2].visits[v2]

    t1_start = schedule[(t1, v1)]
    t1_end = t1_start + visit1.travel_time
    t2_start = schedule[(t2, v2)]
    t2_end = t2_start + visit2.travel_time

    separation = max(t2_start - t1_end, t1_start - t2_end)
    return separation >= -tolerance


def detect_conflicts_in_schedule(
    problem: Problem,
    schedule: Dict[VisitId, float],
    tolerance: float = 1.0,
) -> List[VisitPair]:
    """Trả về list các cặp visit thật sự vi phạm trong schedule."""
    violations = []
    for pair in visit_conflicts(problem):
        if not check_pair_separated(problem, schedule, pair, tolerance=tolerance):
            violations.append(pair)
    return violations


def compute_cost(
    problem: Problem,
    schedule: Dict[VisitId, float],
    cost_type: DelayCostType,
) -> int:
    """Tính tổng cost trễ của 1 nghiệm."""
    total = 0
    for t, train in enumerate(problem.trains):
        for v, visit in enumerate(train.visits):
            if visit.aimed is None:
                continue
            t_val = schedule.get((t, v))
            if t_val is None:
                continue
            delay = int(round(t_val)) - visit.aimed
            total += visit_delay_cost(cost_type, delay)
    return total


def verify_solution(
    problem: Problem,
    schedule: Dict[VisitId, float],
    cost_type: DelayCostType,
    tolerance: float = 1.0,
) -> Tuple[bool, str]:
    """Verify nghiệm thoả mọi ràng buộc TRP.

    Args:
        tolerance: dung sai thời gian (giây). Mặc định 1.0 vì TRP làm việc
                   với integer time. Cần dung sai để khắc phục sai số floating
                   point từ MILP solver.

    Returns:
        (is_valid, message)
    """
    # 1. Cận dưới đầu vào
    for t, train in enumerate(problem.trains):
        for v, visit in enumerate(train.visits):
            t_val = schedule.get((t, v))
            if t_val is None:
                return False, f"Missing schedule for ({t}, {v})"
            if t_val < visit.earliest - tolerance:
                return (
                    False,
                    f"Earliest violation: t[{t}][{v}]={t_val} < "
                    f"{visit.earliest}",
                )

    # 2. Tiền tố trong tàu
    for t, train in enumerate(problem.trains):
        for v in range(len(train.visits) - 1):
            t_cur = schedule[(t, v)]
            t_next = schedule[(t, v + 1)]
            travel = train.visits[v].travel_time
            if t_next < t_cur + travel - tolerance:
                return (
                    False,
                    f"Travel violation: t[{t}][{v + 1}]={t_next} < "
                    f"t[{t}][{v}]+travel = {t_cur + travel}",
                )

    # 3. Xung đột tài nguyên
    conflicts = detect_conflicts_in_schedule(
        problem, schedule, tolerance=tolerance
    )
    if conflicts:
        sample = conflicts[0]
        (t1, v1), (t2, v2) = sample
        v1_obj = problem.trains[t1].visits[v1]
        v2_obj = problem.trains[t2].visits[v2]
        t1_end = schedule[(t1, v1)] + v1_obj.travel_time
        t2_end = schedule[(t2, v2)] + v2_obj.travel_time
        return False, (
            f"Resource conflicts found: {len(conflicts)} pairs. "
            f"Example: t{t1}v{v1}=[{schedule[(t1, v1)]:.2f}-{t1_end:.2f}] vs "
            f"t{t2}v{v2}=[{schedule[(t2, v2)]:.2f}-{t2_end:.2f}]"
        )

    return True, "OK"
