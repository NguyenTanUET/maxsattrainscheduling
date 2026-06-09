"""Parser cho TRP dataset .txt format.

Port từ src/parser.rs (read_txt_file). Mỗi dòng track trong file tạo
ra 2 visits: "in" (tại station) + "out" (trên track).

Hỗ trợ DelayMeasurementType để khớp với Rust baseline:
- FinalStationArrival (DEFAULT): chỉ visit cuối tàu có aimed
- AllStationArrivals: tất cả visit "in" có aimed
- AllStationDepartures: tất cả visit "out" có aimed
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class DelayMeasurementType(Enum):
    FINAL_STATION_ARRIVAL = "final"        # Default — match Rust baseline
    ALL_STATION_ARRIVALS = "arrivals"
    ALL_STATION_DEPARTURES = "departures"


@dataclass(frozen=True)
class Visit:
    resource_id: int
    earliest: int
    aimed: Optional[int]
    travel_time: int


@dataclass
class Train:
    visits: List[Visit] = field(default_factory=list)


@dataclass
class Problem:
    name: str
    trains: List[Train] = field(default_factory=list)
    conflicts: List[Tuple[int, int]] = field(default_factory=list)
    resource_names: List[str] = field(default_factory=list)


def parse_txt(
    filename: str,
    measurement: DelayMeasurementType = DelayMeasurementType.FINAL_STATION_ARRIVAL,
) -> Problem:
    """Parse TRP dataset .txt file.

    Args:
        filename: đường dẫn file
        measurement: kiểu đo trễ (default = FINAL_STATION_ARRIVAL theo Rust)

    Returns:
        Problem object với trains, conflicts, resource_names.
    """
    with open(filename) as f:
        raw_lines = [line.rstrip("\n") for line in f]

    problem = Problem(name=filename, resource_names=["Any station"])
    resources_map: dict[str, int] = {}
    current_visits: Optional[List[Visit]] = None
    next_earliest: Optional[int] = None

    def flush_train():
        nonlocal current_visits, next_earliest
        if current_visits:
            last = current_visits[-1]
            current_visits.append(
                Visit(
                    resource_id=0,
                    earliest=last.earliest + last.travel_time,
                    aimed=None,
                    travel_time=0,
                )
            )
            problem.trains.append(Train(visits=current_visits))
        current_visits = None
        next_earliest = None

    # Peek-ahead để biết is_last_track
    n = len(raw_lines)

    def is_last_track_at(idx: int) -> bool:
        """True nếu line idx là track line cuối của tàu (next line trống hoặc EOF)."""
        if idx + 1 >= n:
            return True
        return not raw_lines[idx + 1].strip()

    for i, line in enumerate(raw_lines + [""]):
        if not line.strip():
            flush_train()
            continue

        fields = line.split()
        if fields[0].startswith("TrainId="):
            current_visits = []
            next_earliest = None
            continue

        if current_visits is None:
            continue

        track_id = fields[0]
        kvs = {}
        for f in fields[2:]:
            if "=" in f:
                k, v = f.split("=", 1)
                kvs[k] = int(v)

        aimed = kvs["AimedDepartureTime"]
        wait_time = kvs["WaitTime"]
        base_time = kvs["BaseTime"]
        run_time = kvs["RunTime"]

        if track_id not in resources_map:
            resources_map[track_id] = len(problem.resource_names)
            problem.resource_names.append(track_id)
        resource_id = resources_map[track_id]

        earliest_in = (
            next_earliest if next_earliest is not None else base_time - wait_time
        )
        earliest_out = base_time
        next_earliest = earliest_out + run_time

        # Determine aimed_in, aimed_out theo measurement type
        is_last = i < n and is_last_track_at(i)
        if measurement == DelayMeasurementType.ALL_STATION_ARRIVALS:
            aimed_in = aimed - wait_time
            aimed_out = None
        elif measurement == DelayMeasurementType.ALL_STATION_DEPARTURES:
            aimed_in = None
            aimed_out = aimed
        else:  # FINAL_STATION_ARRIVAL — default
            aimed_in = None
            aimed_out = aimed if is_last else None

        current_visits.append(
            Visit(
                resource_id=0,
                earliest=earliest_in,
                aimed=aimed_in,
                travel_time=wait_time,
            )
        )
        current_visits.append(
            Visit(
                resource_id=resource_id,
                earliest=earliest_out,
                aimed=aimed_out,
                travel_time=run_time,
            )
        )

    problem.conflicts = [(rid, rid) for rid in resources_map.values()]
    return problem


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else (
        "../instances/original/InstanceA1.txt"
    )
    p = parse_txt(path)
    n_visits = sum(len(t.visits) for t in p.trains)
    n_with_aimed = sum(
        1 for t in p.trains for v in t.visits if v.aimed is not None
    )
    print(f"Instance: {p.name}")
    print(f"  Trains: {len(p.trains)}")
    print(f"  Resources: {len(p.resource_names)}")
    print(f"  Conflicts: {len(p.conflicts)}")
    print(f"  Total visits: {n_visits}")
    print(f"  Visits with aimed (cost contribution): {n_with_aimed}")
    print(f"  First train: {len(p.trains[0].visits)} visits")
