"""Named points of interest snapped to legal lane-boundary nodes (E3-02).

A POI names one graph node of the exported topology (``road:section:lane:entry``
or ``:exit``) and records the node's XYZ so a stale registry is detected when
the topology changes. POIs on junction lanes are refused: a route may not start
or end inside a junction, where no manoeuvre can be announced.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from planning.lane_graph import LaneSegment, LaneTopology

__all__ = ["PointOfInterest", "PoiRegistry", "SNAP_TOLERANCE_M", "lane_of_node"]

SCHEMA_VERSION = 1
SNAP_TOLERANCE_M = 0.05


def lane_of_node(topology: LaneTopology, node_id: str) -> tuple[LaneSegment, str]:
    """Return ``(lane, anchor)`` for a boundary node ID, or raise ``ValueError``."""
    if not isinstance(node_id, str) or node_id.count(":") != 3:
        raise ValueError(f"{node_id!r} is not a lane boundary node ID")
    segment_id, anchor = node_id.rsplit(":", 1)
    if anchor not in ("entry", "exit"):
        raise ValueError(f"{node_id!r} must end in :entry or :exit")
    for lane in topology.lanes:
        if lane.segment_id == segment_id:
            return lane, anchor
    raise ValueError(f"{node_id!r} names an unknown lane section")


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class PointOfInterest:
    name: str
    node_id: str
    xyz_m: tuple[float, float, float]
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("POI name must be a nonempty string")
        if not isinstance(self.node_id, str) or self.node_id.count(":") != 3:
            raise ValueError("POI node_id must be road:section:lane:anchor")
        xyz = tuple(self.xyz_m)
        if len(xyz) != 3 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in xyz):
            raise ValueError("POI xyz_m must hold three finite numbers")
        if not isinstance(self.description, str):
            raise ValueError("POI description must be a string")
        object.__setattr__(self, "xyz_m", tuple(float(v) for v in xyz))

    def validate(self, topology: LaneTopology) -> LaneSegment:
        """Check the node exists, is outside junctions and matches the recorded position."""
        lane, anchor = lane_of_node(topology, self.node_id)
        if lane.junction_id is not None:
            raise ValueError(f"POI {self.name!r} lies on junction lane {lane.segment_id}")
        point = lane.points_xyz_m[0] if anchor == "entry" else lane.points_xyz_m[-1]
        if math.dist(point, self.xyz_m) > SNAP_TOLERANCE_M:
            raise ValueError(f"POI {self.name!r} is {math.dist(point, self.xyz_m):.3f} m from node {self.node_id}")
        return lane


@dataclass(frozen=True)
class PoiRegistry:
    """Named destinations for one exported topology, listed by name."""

    map_name: str
    topology_sha256: str
    pois: tuple[PointOfInterest, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported POI registry schema version")
        if not isinstance(self.map_name, str) or not self.map_name.strip():
            raise ValueError("map_name must be nonempty")
        if not isinstance(self.topology_sha256, str) or len(self.topology_sha256) != 64:
            raise ValueError("topology_sha256 must be a SHA256 digest")
        pois = tuple(self.pois)
        if not pois or any(not isinstance(p, PointOfInterest) for p in pois):
            raise ValueError("registry requires PointOfInterest records")
        if len({p.name for p in pois}) != len(pois):
            raise ValueError("duplicate POI name")
        object.__setattr__(self, "pois", tuple(sorted(pois, key=lambda p: p.name)))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.pois)

    def get(self, name: str) -> PointOfInterest:
        for poi in self.pois:
            if poi.name == name:
                return poi
        raise KeyError(f"unknown POI {name!r}; known: {', '.join(self.names)}")

    def validate(self, topology: LaneTopology) -> None:
        """Every POI must snap to a legal non-junction node of this exact topology."""
        if topology.map_name != self.map_name:
            raise ValueError("POI registry map does not match the topology")
        if topology.sha256 != self.topology_sha256:
            raise ValueError("POI registry was built for a different topology fingerprint")
        for poi in self.pois:
            poi.validate(topology)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, text: str) -> PoiRegistry:
        try:
            data = json.loads(text, object_pairs_hook=_unique_keys)
            data["pois"] = tuple(PointOfInterest(**poi) for poi in data["pois"])
            return cls(**data)
        except (TypeError, KeyError, AttributeError) as exc:
            raise ValueError(f"invalid POI registry: {exc}") from exc

    @classmethod
    def load(cls, path: str | Path) -> PoiRegistry:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
