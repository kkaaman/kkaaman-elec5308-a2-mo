"""Versioned, open fixed mission selected from explicit legal lane successors."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
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


def _trim(points, start_s, end_s):
    """Points of ``points`` between arc lengths ``start_s`` and ``end_s``, ends interpolated."""
    p = np.asarray(points, dtype=float)
    arc = RouteGeometry(p[:, :2]).arc

    def at(s):
        i = min(int(np.searchsorted(arc, s, side="right")) - 1, len(p) - 2)
        f = (s - arc[i]) / (arc[i + 1] - arc[i])
        return tuple(float(v) for v in p[i] + f * (p[i + 1] - p[i]))

    inner = [
        tuple(float(v) for v in q) for q, a in zip(p, arc) if start_s + 0.05 < a < end_s - 0.05
    ]
    return (at(start_s), *inner, at(end_s))


def mission_along_lanes(topology, segment_ids, manoeuvres, *, start_xy_m=None, end_xy_m=None):
    """Mission over an explicit legal lane chain, optionally trimmed to start/end points.

    Used to drive a route chosen for a reason (for example the roads an earlier
    run took) rather than the shortest automatic candidate. The chain is held to
    the same rules as :func:`select_mission`: every transition must be a listed
    successor, and each junction traversal must be the unique legal exit of its
    declared manoeuvre from the lane before it. ``start_xy_m``/``end_xy_m`` are
    projected onto the chain; the mission starts and ends at those projections.
    The turn exit is the first lane after the last junction traversal.
    """
    ids = tuple(segment_ids)
    manoeuvres = tuple(manoeuvres)
    by_id = {lane.segment_id: lane for lane in topology.lanes}
    if any(key not in by_id for key in ids):
        raise ValueError("Unknown route section")
    if by_id[ids[0]].junction_id is not None or by_id[ids[-1]].junction_id is not None:
        raise ValueError("Mission must start and end outside a junction")
    traversed, exit_lane, i = 0, None, 1
    while i < len(ids):
        if by_id[ids[i]].junction_id is None:
            i += 1
            continue
        if traversed >= len(manoeuvres):
            raise ValueError("More junction traversals than declared manoeuvres")
        exit = choose_exit(topology, ids[i - 1], manoeuvres[traversed])
        if ids[i - 1 : i - 1 + len(exit.segment_ids)] != tuple(exit.segment_ids):
            raise ValueError(
                f"{ids[i - 1]} -> {ids[i]} is not the legal {manoeuvres[traversed]} exit"
            )
        traversed += 1
        exit_lane = exit.segment_ids[-1]
        i += len(exit.segment_ids) - 1
    if traversed != len(manoeuvres) or exit_lane is None:
        raise ValueError("Declared manoeuvres must match the junctions traversed (at least one)")
    points = stitch_route(topology, ids)
    route = RouteGeometry(np.asarray(points)[:, :2])
    start_s = 0.0 if start_xy_m is None else route.project(*start_xy_m).s_m
    end_s = route.length_m if end_xy_m is None else route.project(*end_xy_m).s_m
    if end_s - start_s < 50:
        raise ValueError("Trimmed mission must be at least 50 m long and run forwards")
    points = _trim(points, start_s, end_s)
    route = validate_open_route(points)
    if math.dist(points[0][:2], points[-1][:2]) < 30:
        raise ValueError("Start and goal must be separated; this baseline excludes loops")
    turn_s = route.project(*by_id[exit_lane].points_xyz_m[0][:2]).s_m
    return Mission(
        topology.map_name,
        topology.opendrive_sha256,
        ids,
        points,
        manoeuvres,
        turn_s,
        topology.sha256,
    )


def load_route_definition(path):
    """Named explicit route (``configs/missions/*.json``) for :func:`mission_from_definition`."""
    document = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if document.get("schema_version") != 1 or not document.get("name"):
        raise ValueError("Unsupported route definition")
    for key in ("segment_ids", "manoeuvres"):
        if not isinstance(document.get(key), list) or not document[key]:
            raise ValueError(f"Route definition needs a non-empty {key} list")
    return document


def route_summary(definition):
    """Definition fields recorded with the mission (and so in its hash)."""
    summary = dict(name=definition["name"])
    if definition.get("obstacle_s_m") is not None:
        obstacle_s = float(definition["obstacle_s_m"])
        if not math.isfinite(obstacle_s) or obstacle_s <= 0:
            raise ValueError("obstacle_s_m must be positive")
        summary["obstacle_s_m"] = obstacle_s
    return summary


def mission_from_definition(topology, definition):
    """Mission for a route definition. Not drivable live until prepare_mission validates it."""
    point = lambda key: None if definition.get(key) is None else tuple(definition[key])  # noqa: E731
    mission = mission_along_lanes(
        topology,
        definition["segment_ids"],
        definition["manoeuvres"],
        start_xy_m=point("start_xy_m"),
        end_xy_m=point("end_xy_m"),
    )
    return replace(
        mission,
        map_validation=dict(
            status="NOT_VALIDATED_LIVE", route_definition=route_summary(definition)
        ),
    )


def scenario_obstacle_s(mission, scenario):
    """Route s of a scenario obstacle: after the turn exit, else the route's own
    placement (explicit routes), else the scenario's ``obstacle_s_m``."""
    if scenario.get("after_turn_m") is not None:
        return mission.turn_exit_s_m + scenario["after_turn_m"]
    route = (getattr(mission, "map_validation", None) or {}).get("route_definition") or {}
    if route.get("obstacle_s_m") is not None:
        return float(route["obstacle_s_m"])
    return scenario["obstacle_s_m"]


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
