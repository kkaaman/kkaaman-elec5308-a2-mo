"""Serializable lane sections and their explicit legal successors (BL-A-02).

This module has no simulator dependency. ``LaneTopology.graph`` exposes the
existing LaneGraph contract, with separate lane and boundary-connection edges.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from autonomy.interfaces import GraphEdge, GraphNode, LaneGraph

SCHEMA_VERSION = 1
COORDINATE_FRAME = "right-handed-x-forward-y-left-z-up-metres"


def _positive(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def lane_segment_id(road_id: int, section_id: int, lane_id: int) -> str:
    """Lane IDs are unique only together with road and section IDs."""
    return f"{road_id}:{section_id}:{lane_id}"


@dataclass(frozen=True)
class LaneSegment:
    """A driving lane section, sampled in travel order, including both ends.

    ``junction_id`` identifies a lane inside a junction. Its entry/exit nodes
    delimit that junction section; adjacent successors identify approach/exit
    lanes. OpenDRIVE s can decrease in the direction of travel.
    """

    road_id: int
    section_id: int
    lane_id: int
    entry_s_m: float
    exit_s_m: float
    points_xyz_m: tuple[tuple[float, float, float], ...]
    successor_ids: tuple[str, ...] = ()
    junction_id: int | None = None

    def __post_init__(self) -> None:
        for name in ("road_id", "section_id", "lane_id"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.road_id < 0 or self.section_id < 0 or self.lane_id == 0:
            raise ValueError("invalid OpenDRIVE lane identity")
        if self.junction_id is not None and (type(self.junction_id) is not int or self.junction_id < 0):
            raise ValueError("junction_id must be a nonnegative integer or None")
        for value in (self.entry_s_m, self.exit_s_m):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("lane s must be finite and nonnegative")
        if self.entry_s_m == self.exit_s_m:
            raise ValueError("lane section must have nonzero longitudinal extent")
        points = tuple(tuple(p) for p in self.points_xyz_m)
        if len(points) < 2 or any(len(p) != 3 for p in points):
            raise ValueError("lane requires at least two XYZ points")
        for p in points:
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in p):
                raise ValueError("lane points must be finite numbers")
        if any(a[:2] == b[:2] for a, b in zip(points, points[1:])):
            raise ValueError("consecutive lane XY points must differ")
        successors = tuple(self.successor_ids)
        if any(not isinstance(s, str) or not s.strip() for s in successors):
            raise ValueError("successor IDs must be nonempty strings")
        if len(set(successors)) != len(successors):
            raise ValueError("duplicate lane successor")
        object.__setattr__(self, "points_xyz_m", points)
        object.__setattr__(self, "successor_ids", tuple(sorted(successors)))
        object.__setattr__(self, "entry_s_m", float(self.entry_s_m))
        object.__setattr__(self, "exit_s_m", float(self.exit_s_m))
        object.__setattr__(self, "points_xyz_m", tuple(tuple(float(v) if v else 0.0 for v in p) for p in points))

    @property
    def segment_id(self) -> str:
        return lane_segment_id(self.road_id, self.section_id, self.lane_id)

    @property
    def entry_node_id(self) -> str:
        return f"{self.segment_id}:entry"

    @property
    def exit_node_id(self) -> str:
        return f"{self.segment_id}:exit"


@dataclass(frozen=True)
class LaneTopology:
    """Versioned map document, without timestamps or simulator handles.

    ``navigation_speed_cap_mps`` is a configured cap, not a measured/signposted
    road speed limit. ``max_connection_gap_m`` guards boundary geometry only;
    it never authorizes a connection absent from ``successor_ids``.
    """

    map_name: str
    opendrive_sha256: str
    sampling_distance_m: float
    navigation_speed_cap_mps: float
    lanes: tuple[LaneSegment, ...]
    max_connection_gap_m: float = 0.25
    schema_version: int = SCHEMA_VERSION
    coordinate_frame: str = COORDINATE_FRAME

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported lane topology schema version")
        if self.coordinate_frame != COORDINATE_FRAME:
            raise ValueError("unsupported coordinate frame")
        if not isinstance(self.map_name, str) or not self.map_name.strip():
            raise ValueError("map_name must be nonempty")
        if not isinstance(self.opendrive_sha256, str) or len(self.opendrive_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.opendrive_sha256):
            raise ValueError("opendrive_sha256 must be a lowercase SHA256 digest")
        for name in ("sampling_distance_m", "navigation_speed_cap_mps", "max_connection_gap_m"):
            _positive(name, getattr(self, name))
            object.__setattr__(self, name, float(getattr(self, name)))
        lanes = tuple(self.lanes)
        if not lanes or any(not isinstance(lane, LaneSegment) for lane in lanes):
            raise ValueError("topology requires LaneSegment records")
        by_id = {lane.segment_id: lane for lane in lanes}
        if len(by_id) != len(lanes):
            raise ValueError("duplicate lane section")
        for lane in lanes:
            for key in lane.successor_ids:
                if key not in by_id:
                    raise ValueError(f"unknown successor {key} from {lane.segment_id}")
                start, end = lane.points_xyz_m[-1], by_id[key].points_xyz_m[0]
                if math.dist(start, end) > self.max_connection_gap_m:
                    raise ValueError(f"disconnected lane boundary {lane.segment_id} -> {key}")
                if start[:2] == end[:2]:
                    raise ValueError(f"coincident connection endpoints {lane.segment_id} -> {key}")
        object.__setattr__(self, "lanes", tuple(sorted(lanes, key=lambda lane: lane.segment_id)))

    @property
    def graph(self) -> LaneGraph:
        """Convert to the existing core contract, preserving forks and merges.

        Boundary nodes are never merged by proximity. Each explicit successor
        becomes its own short directed edge; this avoids inventing turns at a
        junction where several lanes share a geometric endpoint.
        """
        nodes, edges = [], []
        by_id = {lane.segment_id: lane for lane in self.lanes}
        for lane in self.lanes:
            nodes.extend((GraphNode(lane.entry_node_id, *lane.points_xyz_m[0]),
                          GraphNode(lane.exit_node_id, *lane.points_xyz_m[-1])))
            edges.append(GraphEdge(lane.entry_node_id, lane.exit_node_id,
                                   tuple(p[:2] for p in lane.points_xyz_m), self.navigation_speed_cap_mps))
            for key in lane.successor_ids:
                target = by_id[key]
                edges.append(GraphEdge(lane.exit_node_id, target.entry_node_id,
                                       (lane.points_xyz_m[-1][:2], target.points_xyz_m[0][:2]),
                                       self.navigation_speed_cap_mps))
        return LaneGraph(tuple(sorted(nodes, key=lambda n: n.node_id)),
                         tuple(sorted(edges, key=lambda e: (e.start_id, e.end_id))))

    def to_json(self) -> str:
        """Canonical UTF-8 JSON: sorted records/keys, no NaN, newline at EOF."""
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(self.to_json().encode("utf-8"))

    @classmethod
    def from_json(cls, text: str) -> LaneTopology:
        def unique_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = value
            return result

        try:
            data = json.loads(text, object_pairs_hook=unique_keys)
            if not isinstance(data, dict) or "schema_version" not in data or "coordinate_frame" not in data:
                raise ValueError("lane document requires schema_version and coordinate_frame")
            data["lanes"] = tuple(LaneSegment(**lane) for lane in data["lanes"])
            return cls(**data)
        except (TypeError, KeyError) as exc:
            raise ValueError(f"invalid lane topology document: {exc}") from exc

    @classmethod
    def load(cls, path: str | Path) -> LaneTopology:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
