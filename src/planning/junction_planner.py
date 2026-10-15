"""Legal successor traversal and exit-based maneuver classification (BL-A-03)."""

import math
from dataclasses import dataclass

from planning.lane_graph import LaneTopology


@dataclass(frozen=True)
class JunctionExit:
    maneuver: str
    segment_ids: tuple[str, ...]
    heading_change_rad: float


def _heading(points, end=False):
    a, b = points[-2:] if end else points[:2]
    return math.atan2(b[1] - a[1], b[0] - a[0])


def classify_exit(angle):
    if not math.isfinite(angle):
        raise ValueError("Heading must be finite")
    angle = math.atan2(math.sin(angle), math.cos(angle))
    degrees = math.degrees(angle)
    if abs(degrees) <= 25:
        return "straight"
    if 45 <= degrees <= 135:
        return "left"
    if -135 <= degrees <= -45:
        return "right"
    raise ValueError("Ambiguous or unsupported junction heading")


def junction_exits(topology: LaneTopology, approach_id, max_segments=64):
    """Follow every legal branch through the junction, never choose next[0]."""
    if type(max_segments) is not int or max_segments < 1:
        raise ValueError("max_segments must be positive")
    by_id = {lane.segment_id: lane for lane in topology.lanes}
    if approach_id not in by_id or by_id[approach_id].junction_id is not None:
        raise ValueError("Expected a non-junction approach lane")
    approach = by_id[approach_id]
    heading = _heading(approach.points_xyz_m, True)
    result = []
    pending = [(key, (approach_id,)) for key in approach.successor_ids]
    while pending:
        key, visited = pending.pop()
        if key in visited or len(visited) > max_segments:
            raise ValueError("Cycle or traversal limit in junction")
        lane = by_id[key]
        path = visited + (key,)
        if lane.junction_id is None:
            if len(path) < 3:
                raise ValueError("Successor is not a junction traversal")
            change = _heading(lane.points_xyz_m) - heading
            change = math.atan2(math.sin(change), math.cos(change))
            result.append(JunctionExit(classify_exit(change), path, change))
        else:
            if not lane.successor_ids:
                raise ValueError("Junction branch has no legal exit")
            pending.extend((successor, path) for successor in lane.successor_ids)
    if not result:
        raise ValueError("No legal junction exits")
    return tuple(sorted(result, key=lambda x: x.segment_ids))


def choose_exit(topology, approach_id, maneuver):
    if maneuver not in ("left", "right", "straight"):
        raise ValueError("Unsupported maneuver")
    choices = [x for x in junction_exits(topology, approach_id) if x.maneuver == maneuver]
    if len(choices) != 1:
        raise ValueError(f"Expected exactly one legal {maneuver} exit, found {len(choices)}")
    return choices[0]
