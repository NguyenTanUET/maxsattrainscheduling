"""Delay cost functions cho TRP.

Port từ src/problem.rs (DelayCostThresholds, DelayCostType).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple


class DelayCostType(Enum):
    FINSTEPS_1_3MIN = "finsteps1_3min"
    FINSTEPS_1_5MIN = "finsteps1_5min"
    FINSTEPS_123 = "finsteps123"
    FINSTEPS_12345 = "finsteps12345"
    FINSTEPS_139 = "finsteps139"
    INFSTEPS_60 = "infsteps60"
    INFSTEPS_180 = "infsteps180"
    INFSTEPS_360 = "infsteps360"
    CONTINUOUS = "cont"


@dataclass
class DelayCostThresholds:
    """Stepped cost function: each (threshold, cost) means delay > threshold → cost."""

    thresholds: List[Tuple[int, int]]

    @classmethod
    def f1_3min(cls) -> "DelayCostThresholds":
        return cls(thresholds=[(5 * 60, 1)])

    @classmethod
    def f1_5min(cls) -> "DelayCostThresholds":
        return cls(thresholds=[(3 * 60, 1)])

    @classmethod
    def f123(cls) -> "DelayCostThresholds":
        return cls(thresholds=[(360, 3), (180, 2), (0, 1)])

    @classmethod
    def f12345(cls) -> "DelayCostThresholds":
        return cls(
            thresholds=[(3 * 360, 5), (2 * 360, 4), (360, 3), (180, 2), (0, 1)]
        )

    @classmethod
    def f139(cls) -> "DelayCostThresholds":
        return cls(thresholds=[(360, 9), (180, 3), (0, 1)])

    def eval(self, delay: int) -> int:
        """Trả về cost theo delay. Iterate từ ngưỡng cao xuống thấp."""
        for threshold, cost in self.thresholds:
            if delay > threshold:
                return cost
        return 0


def get_thresholds(cost_type: DelayCostType) -> DelayCostThresholds:
    """Lookup thresholds object cho FiniteSteps type."""
    mapping = {
        DelayCostType.FINSTEPS_1_3MIN: DelayCostThresholds.f1_3min,
        DelayCostType.FINSTEPS_1_5MIN: DelayCostThresholds.f1_5min,
        DelayCostType.FINSTEPS_123: DelayCostThresholds.f123,
        DelayCostType.FINSTEPS_12345: DelayCostThresholds.f12345,
        DelayCostType.FINSTEPS_139: DelayCostThresholds.f139,
    }
    if cost_type not in mapping:
        raise ValueError(f"Not a FiniteSteps type: {cost_type}")
    return mapping[cost_type]()


def get_interval(cost_type: DelayCostType) -> int:
    """Lookup interval cho InfiniteSteps type."""
    mapping = {
        DelayCostType.INFSTEPS_60: 60,
        DelayCostType.INFSTEPS_180: 180,
        DelayCostType.INFSTEPS_360: 360,
    }
    if cost_type not in mapping:
        raise ValueError(f"Not an InfiniteSteps type: {cost_type}")
    return mapping[cost_type]


def visit_delay_cost(cost_type: DelayCostType, delay: int) -> int:
    """Tính cost cho 1 visit dựa trên delay (= t_actual - aimed).

    Port từ visit_delay_cost trong problem.rs.
    """
    if delay <= 0:
        return 0
    if cost_type == DelayCostType.CONTINUOUS:
        return delay
    if cost_type in (
        DelayCostType.INFSTEPS_60,
        DelayCostType.INFSTEPS_180,
        DelayCostType.INFSTEPS_360,
    ):
        interval = get_interval(cost_type)
        return -(-delay // interval)  # ceil division
    return get_thresholds(cost_type).eval(delay)


def parse_cost_type(s: str) -> DelayCostType:
    """Parse từ CLI string sang DelayCostType."""
    key = s.lower().strip()
    aliases = {
        "infsteps123": DelayCostType.INFSTEPS_180,  # alias trong main.rs
    }
    if key in aliases:
        return aliases[key]
    for t in DelayCostType:
        if t.value == key:
            return t
    raise ValueError(
        f"Unknown cost type '{s}'. Valid: {[t.value for t in DelayCostType]}"
    )
