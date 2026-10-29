"""Continuous routes between named destinations for the route manager (E3-02).

A route is one lane sequence stitched by :func:`planning.route_manager.stitch_route`
and trimmed to the origin and destination nodes, in the same ``segment_ids`` /
``point_slice`` / ``points_xyz_m`` form as the frozen baseline route file. The
polyline is installed once per episode; manoeuvres are announcements along it,
never endpoints that reset the path.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from autonomy.interfaces import LaneGraph
from planning.baseline_mission import canonical_sha256
from planning.junction_planner import _heading, classify_exit
from planning.lane_graph import LaneSegment, LaneTopology
from planning.route_manager import stitch_route
from planning.wayfinder.astar import SearchResult, shortest_path
from planning.wayfinder.pois import PoiRegistry, lane_of_node

__all__ = ["DestinationCheck", "Maneuver", "RoutePlanningError", "WayfinderRoute", "check_destination", "plan_named_route", "plan_route"]

MANEUVER_KINDS = ("straight", "left", "right", "arrive")


class RoutePlanningError(ValueError):
    """A route could not be produced; ``reason`` is a stable machine-readable code."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class DestinationCheck:
    """Legality of an origin/destination pair; ``legal`` is False with a reason otherwise."""

    origin_node_id: str
    destination_node_id: str
    legal: bool
    reason: str
    detail: str = ""
    search: SearchResult | None = None


@dataclass(frozen=True)
class Maneuver:
    """One announcement on the route, at ``route_s_m`` metres from the route start."""

    kind: str
    route_s_m: float
    approach_segment_id: str
    exit_segment_id: str
    junction_id: int | None
    heading_change_rad: float

    def __post_init__(self) -> None:
        if self.kind not in MANEUVER_KINDS:
            raise ValueError(f"unsupported manoeuvre kind {self.kind!r}")
        if not math.isfinite(self.route_s_m) or self.route_s_m < 0 or not math.isfinite(self.heading_change_rad):
            raise ValueError("manoeuvre position and heading change must be finite")


@dataclass(frozen=True)
class WayfinderRoute:
    """A planned, continuous and legal route between two lane-boundary nodes."""

    map_name: str
    opendrive_sha256: str
    topology_sha256: str
    origin_name: str
    destination_name: str
    origin_node_id: str
    destination_node_id: str
    node_ids: tuple[str, ...]
    segment_ids: tuple[str, ...]
    point_slice: tuple[int, int]
    points_xyz_m: tuple[tuple[float, float, float], ...]
    maneuvers: tuple[Maneuver, ...]

    @property
    def length_m(self) -> float:
        return sum(math.dist(a[:2], b[:2]) for a, b in zip(self.points_xyz_m, self.points_xyz_m[1:]))

    @property
    def goal_xyz_m(self) -> tuple[float, float, float]:
        return self.points_xyz_m[-1]

    @property
    def turn_count(self) -> int:
        return sum(m.kind in ("left", "right") for m in self.maneuvers)

    def to_document(self) -> dict:
        """Route-file document in the baseline route schema plus manoeuvres."""
        document = {
            "schema_version": 1,
            "route_kind": "wayfinder_v1",
            "coordinate_frame": "right-handed-x-forward-y-left-z-up-metres",
            "map_name": self.map_name,
            "opendrive_sha256": self.opendrive_sha256,
            "topology_sha256": self.topology_sha256,
            "origin": {"name": self.origin_name, "node_id": self.origin_node_id},
            "destination": {"name": self.destination_name, "node_id": self.destination_node_id},
            "segment_ids": list(self.segment_ids),
            "point_slice": list(self.point_slice),
            "points_xyz_m": [list(p) for p in self.points_xyz_m],
            "goal_xyz_m": list(self.goal_xyz_m),
            "length_m": round(self.length_m, 3),
            "maneuvers": [asdict(m) for m in self.maneuvers],
        }
        document["route_sha256"] = canonical_sha256(document)
        return document


def _lane_sequence(topology: LaneTopology, node_ids) -> list[LaneSegment]:
    """Collapse a node path to its lane sequence; refuse a lane visited twice.

    A node path from ``X:exit`` round a block back to ``X:entry`` has distinct nodes
    but repeats lane ``X``; ``stitch_route`` refuses repeated sections, so the loop is
    reported here under its own reason code rather than as a route-manager refusal.
    """
    by_id = {lane.segment_id: lane for lane in topology.lanes}
    lanes: list[LaneSegment] = []
    seen: set[str] = set()
    for node_id in node_ids:
        lane = by_id[node_id.rsplit(":", 1)[0]]
        if lanes and lanes[-1].segment_id == lane.segment_id:
            continue
        if lane.segment_id in seen:
            raise RoutePlanningError("lane_loop", f"route would pass through lane {lane.segment_id} twice")
        seen.add(lane.segment_id)
        lanes.append(lane)
    return lanes


def _node_point(lane: LaneSegment, anchor: str):
    return lane.points_xyz_m[0] if anchor == "entry" else lane.points_xyz_m[-1]


def check_destination(topology: LaneTopology, origin_node_id: str, destination_node_id: str,
                      graph: LaneGraph | None = None) -> DestinationCheck:
    """Report whether a pair of nodes may be routed, without raising.

    ``graph`` may be ``topology.graph`` computed once by a caller planning many
    routes; it is rebuilt here otherwise.
    """
    def refuse(reason, detail):
        return DestinationCheck(origin_node_id, destination_node_id, False, reason, detail)

    for role, node_id in (("origin", origin_node_id), ("destination", destination_node_id)):
        try:
            lane, _anchor = lane_of_node(topology, node_id)
        except ValueError as exc:
            return refuse("unknown_node", f"{role}: {exc}")
        if lane.junction_id is not None:
            return refuse("junction_node", f"{role} {node_id} lies inside junction {lane.junction_id}")
    if origin_node_id == destination_node_id:
        return refuse("same_node", "origin and destination are the same node")
    result = shortest_path(topology.graph if graph is None else graph, origin_node_id, destination_node_id)
    if not result.reachable:
        return refuse("unreachable", f"no legal lane sequence from {origin_node_id} to {destination_node_id}")
    try:
        _lane_sequence(topology, result.node_ids)
    except RoutePlanningError as exc:
        return refuse(exc.reason, exc.detail)
    return DestinationCheck(origin_node_id, destination_node_id, True, "ok", f"{result.length_m:.1f} m", result)


def _maneuvers(topology: LaneTopology, lanes, points, arc) -> tuple[Maneuver, ...]:
    result = []
    index = 0
    while index < len(lanes):
        if lanes[index].junction_id is None:
            index += 1
            continue
        if index == 0 or index == len(lanes) - 1:
            raise RoutePlanningError("junction_node", "route may not start or end inside a junction")
        approach = lanes[index - 1]
        end = index
        while end < len(lanes) and lanes[end].junction_id is not None:
            end += 1
        if end == len(lanes):
            raise RoutePlanningError("junction_node", "route ends inside a junction")
        exit_lane = lanes[end]
        change = _heading(exit_lane.points_xyz_m) - _heading(approach.points_xyz_m, True)
        change = math.atan2(math.sin(change), math.cos(change))
        try:
            kind = classify_exit(change)
        except ValueError as exc:
            raise RoutePlanningError("ambiguous_maneuver", f"{approach.segment_id} -> {exit_lane.segment_id}: {exc}") from exc
        position = points.index(approach.points_xyz_m[-1])
        result.append(Maneuver(kind, arc[position], approach.segment_id, exit_lane.segment_id, lanes[index].junction_id, change))
        index = end
    result.append(Maneuver("arrive", arc[-1], lanes[-1].segment_id, lanes[-1].segment_id, None, 0.0))
    return tuple(result)


def plan_route(topology: LaneTopology, origin_node_id: str, destination_node_id: str,
               origin_name: str = "", destination_name: str = "", graph: LaneGraph | None = None) -> WayfinderRoute:
    """Plan one continuous legal route; raise :class:`RoutePlanningError` otherwise."""
    check = check_destination(topology, origin_node_id, destination_node_id, graph)
    if not check.legal or check.search is None:
        raise RoutePlanningError(check.reason, check.detail)
    search = check.search
    lanes = _lane_sequence(topology, search.node_ids)
    segment_ids = tuple(lane.segment_id for lane in lanes)
    try:
        stitched = stitch_route(topology, segment_ids)
    except ValueError as exc:
        raise RoutePlanningError("route_manager_refused", str(exc)) from exc
    origin_lane, origin_anchor = lane_of_node(topology, origin_node_id)
    destination_lane, destination_anchor = lane_of_node(topology, destination_node_id)
    start = stitched.index(_node_point(origin_lane, origin_anchor))
    stop = stitched.index(_node_point(destination_lane, destination_anchor), start) + 1
    points = stitched[start:stop]
    if len(points) < 2:
        raise RoutePlanningError("degenerate_route", "route needs at least two distinct points")
    arc = [0.0]
    for a, b in zip(points, points[1:]):
        arc.append(arc[-1] + math.dist(a[:2], b[:2]))
    maneuvers = _maneuvers(topology, lanes, points, arc)
    return WayfinderRoute(topology.map_name, topology.opendrive_sha256, topology.sha256,
                          origin_name or origin_node_id, destination_name or destination_node_id,
                          origin_node_id, destination_node_id, search.node_ids, segment_ids,
                          (start, stop), points, maneuvers)


def plan_named_route(topology: LaneTopology, registry: PoiRegistry, origin: str, destination: str,
                     graph: LaneGraph | None = None) -> WayfinderRoute:
    """Plan between two registry names; the registry must match the topology."""
    registry.validate(topology)
    a, b = registry.get(origin), registry.get(destination)
    return plan_route(topology, a.node_id, b.node_id, a.name, b.name, graph)
