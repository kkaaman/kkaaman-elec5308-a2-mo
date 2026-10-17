"""Versioned, open fixed mission selected from explicit legal lane successors."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from autonomy.interfaces import GraphEdge, GraphNode, LaneGraph
from planning.corridor import RouteGeometry
from planning.junction_planner import choose_exit
from planning.route_manager import stitch_route


def _orientation(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def validate_open_route(points):
    route = RouteGeometry(np.asarray(points)[:, :2])
    for i, (a, b) in enumerate(zip(route.points, route.points[1:])):
        for j in range(i + 2, len(route.points) - 1):
            c, d = route.points[j : j + 2]
            if (
                _orientation(a, b, c) * _orientation(a, b, d) < -1e-9
                and _orientation(c, d, a) * _orientation(c, d, b) < -1e-9
            ):
                raise ValueError("Self-intersecting mission")
    return route


@dataclass(frozen=True)
class Mission:
    map_name: str
    opendrive_sha256: str
    segment_ids: tuple[str, ...]
    points_xyz_m: tuple[tuple[float, float, float], ...]
    manoeuvres: tuple[str, ...]
    turn_exit_s_m: float
    source_topology_sha256: str
    map_validation: dict | None = None
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1 or self.map_name not in {"Town10HD", "Town10HD_Opt"}:
            raise ValueError("Unsupported mission/map")
        for digest in (self.opendrive_sha256, self.source_topology_sha256):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Map/topology identity required")
        points = tuple(tuple(float(v) for v in p) for p in self.points_xyz_m)
        if any(len(p) != 3 for p in points):
            raise ValueError("XYZ route required")
        route = validate_open_route(points)
        if not 0 < self.turn_exit_s_m < route.length_m:
            raise ValueError("Turn must be within mission")
        object.__setattr__(self, "points_xyz_m", points)
        object.__setattr__(self, "segment_ids", tuple(self.segment_ids))
        object.__setattr__(self, "manoeuvres", tuple(self.manoeuvres))

    @property
    def points_xy_m(self):
        return tuple(p[:2] for p in self.points_xyz_m)

    @property
    def sha256(self):
        return hashlib.sha256(
            json.dumps(
                asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()

    def save(self, path):
        Path(path).write_text(
            json.dumps(asdict(self), indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8-sig")))


def select_mission(topology):
    """Deterministic candidate selection, followed by separate live validation.

    Follow named legal exits, reject ambiguous forks and self-intersections.
    Selection never infers legality by proximity or chooses next()[0].
    """
    candidates = []
    by_id = {lane.segment_id: lane for lane in topology.lanes}
    for lane in topology.lanes:
        if lane.junction_id is not None:
            continue
        length = sum(math.dist(a, b) for a, b in zip(lane.points_xyz_m, lane.points_xyz_m[1:]))
        if length < 45:
            continue
        for turns in (
            ("right",),
            ("left",),
            ("straight", "right"),
            ("straight", "left"),
            ("right", "straight"),
        ):
            ids = [lane.segment_id]
            try:
                turn_exit = None
                for turn in turns:
                    exit = choose_exit(topology, ids[-1], turn)
                    ids.extend(exit.segment_ids[1:])
                    if turn != "straight":
                        turn_exit = exit.segment_ids[-1]
                points = stitch_route(topology, ids)
                route = validate_open_route(points)
                exit_point = by_id[turn_exit].points_xyz_m[0]
                turn_s = route.project(*exit_point[:2]).s_m
                if not (
                    120 <= route.length_m <= 350 and turn_s >= 40 and route.length_m - turn_s >= 35
                ):
                    continue
                # Start/goal must be separated; this baseline excludes loops.
                if math.dist(points[0], points[-1]) < 30:
                    continue
                candidates.append((len(turns), route.length_m, tuple(ids), points, turns, turn_s))
            except ValueError:
                continue
    if not candidates:
        raise ValueError(
            "No legal open candidate satisfies the configured geometry; inspect topology"
        )
    _, _, ids, points, turns, turn_s = min(candidates, key=lambda x: (x[0], x[1], x[2]))
    return Mission(
        topology.map_name, topology.opendrive_sha256, ids, points, turns, turn_s, topology.sha256
    )


def overlapping_junction_lane(topology, mission_segment_ids, segment_id):
    """True if ``segment_id`` is a junction lane overlapping the mission's own.

    Inside a junction, lanes that diverge from the same approach or merge into
    the same exit share centreline geometry near that shared end, and CARLA's
    ``get_waypoint`` can return either. Such a lane is accepted only if it lies
    in the same junction as one of the mission's junction lanes and shares that
    lane's predecessor or successor. It never admits a non-junction lane, a lane
    of another junction, or a junction lane with no shared entry/exit.
    """
    by_id = {lane.segment_id: lane for lane in topology.lanes}
    lane = by_id.get(segment_id)
    if lane is None or lane.junction_id is None:
        return False
    predecessors = {}
    for item in topology.lanes:
        for successor in item.successor_ids:
            predecessors.setdefault(successor, set()).add(item.segment_id)
    for key in mission_segment_ids:
        own = by_id[key]
        if own.junction_id != lane.junction_id or key == segment_id:
            continue
        if set(own.successor_ids) & set(lane.successor_ids):
            return True
        if predecessors.get(key, set()) & predecessors.get(segment_id, set()):
            return True
    return False


def route_graph(points):
    return LaneGraph(
        (GraphNode("start", *points[0]), GraphNode("goal", *points[-1])),
        (GraphEdge("start", "goal", tuple(map(tuple, points)), 5.0),),
    )
