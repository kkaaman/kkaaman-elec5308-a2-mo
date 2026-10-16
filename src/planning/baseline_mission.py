"""Load and validate one frozen, simulator-independent baseline mission."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from navigation_control import ControllerConfig, Waypoint
from planning.lane_graph import COORDINATE_FRAME, LaneTopology
from planning.route_manager import stitch_route


def canonical_sha256(document):
    """Hash JSON content rather than whitespace or platform-specific newlines."""
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _unique_keys(pairs):
    document = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"Duplicate JSON key: {key}")
        document[key] = value
    return document


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)


def validate_open_polyline(points):
    """Reject degenerate, closed, crossing, touching or overlapping XY segments."""
    if len(points) < 3:
        raise ValueError("An open mission requires at least three points")
    for point in points:
        if len(point) != 3 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in point
        ):
            raise ValueError("Route points must contain finite XYZ coordinates")
    if math.dist(points[0][:2], points[-1][:2]) < 5:
        raise ValueError("Open route endpoints must be separated by at least five metres")
    segments = list(zip(points, points[1:]))
    if any(math.dist(a[:2], b[:2]) <= 1e-7 for a, b in segments):
        raise ValueError("Route contains a repeated consecutive XY point")

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a, b, p):
        return (
            abs(cross(a, b, p)) <= 1e-8
            and min(a[0], b[0]) - 1e-8 <= p[0] <= max(a[0], b[0]) + 1e-8
            and min(a[1], b[1]) - 1e-8 <= p[1] <= max(a[1], b[1]) + 1e-8
        )

    for index, (a, b) in enumerate(segments):
        if index and (
            on_segment(a, b, segments[index - 1][0])
            or on_segment(segments[index - 1][0], a, b)
        ):
            raise ValueError("Adjacent route segments backtrack or overlap")
        for c, d in segments[index + 2:]:
            ab_c, ab_d, cd_a, cd_b = cross(a, b, c), cross(a, b, d), cross(c, d, a), cross(c, d, b)
            crossing = ab_c * ab_d < 0 and cd_a * cd_b < 0
            touching = any((on_segment(a, b, c), on_segment(a, b, d), on_segment(c, d, a), on_segment(c, d, b)))
            if crossing or touching:
                raise ValueError("Route contains a geometric self-intersection or overlap")


@dataclass(frozen=True)
class FrozenMission:
    """Validated mission data; no live route selection occurs during an episode."""

    _config_json: str
    _route_json: str
    route_sha256: str
    config_sha256: str
    controller_config: ControllerConfig

    @property
    def config(self):
        """Return a detached copy so callers cannot alter the frozen identity."""
        return json.loads(self._config_json)

    @property
    def route(self):
        """Return detached route metadata, never the stored source snapshot."""
        return json.loads(self._route_json)

    @property
    def waypoints(self):
        return tuple(Waypoint(*point[:2]) for point in self.route["points_xyz_m"])

    @property
    def spawn_pose(self):
        return dict(self.route["spawn_pose"])

    @property
    def map_name(self):
        return self.route["map_name"]

    @property
    def opendrive_sha256(self):
        return self.route["opendrive_sha256"]

    @property
    def length_m(self):
        return sum(math.dist(a[:2], b[:2]) for a, b in zip(self.route["points_xyz_m"], self.route["points_xyz_m"][1:]))


def load_frozen_mission(config_path):
    """Verify geometry, legal topology, identity and fixed execution settings."""
    config_path = Path(config_path).resolve()
    config = _read_json(config_path)
    if config.get("route_selection_status") != "FROZEN_PENDING_S0_ACCEPTANCE":
        raise ValueError("Mission must reference a frozen route before execution")
    if config.get("set_path_policy") != "once_per_episode_before_first_update":
        raise ValueError("Mission requires one path installation per episode")
    route = _read_json(config_path.parent / config["route_file"])
    if route.get("schema_version") != 1 or route.get("coordinate_frame") != COORDINATE_FRAME:
        raise ValueError("Unsupported mission route schema or coordinates")
    topology = LaneTopology.load(config_path.parent / config["topology_file"])
    if route.get("topology_sha256") != topology.sha256 or route.get("opendrive_sha256") != topology.opendrive_sha256:
        raise ValueError("Frozen route does not match the declared topology fingerprint")
    if route.get("map_name") != topology.map_name or config.get("map") != topology.map_name:
        raise ValueError("Mission, route and topology maps must agree")
    points = tuple(tuple(p) for p in route["points_xyz_m"])
    validate_open_polyline(points)
    stitched = stitch_route(topology, route["segment_ids"])
    start, end = route["point_slice"]
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(stitched):
        raise ValueError("Invalid frozen route slice")
    if points != stitched[start:end]:
        raise ValueError("Frozen route points differ from legal topology geometry")
    if tuple(route.get("goal_xyz_m", ())) != points[-1]:
        raise ValueError("Frozen goal must be the final route point")
    spawn = route["spawn_pose"]
    if set(spawn) != {"x_m", "y_m", "z_m", "yaw_rad"} or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in spawn.values()
    ):
        raise ValueError("Invalid frozen actor spawn pose")
    heading = math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0])
    offset = 1.38641
    rear = (spawn["x_m"] - offset * math.cos(spawn["yaw_rad"]), spawn["y_m"] - offset * math.sin(spawn["yaw_rad"]))
    if math.dist(rear, points[0][:2]) > 0.05 or abs(math.remainder(spawn["yaw_rad"] - heading, 2 * math.pi)) > 0.02:
        raise ValueError("Spawn rear axle and heading must match the route start")
    controller = ControllerConfig(**config["controller"])
    runtime = config["runtime"]
    if runtime.get("fixed_delta_seconds") != 0.05:
        raise ValueError("Baseline physics timestep must remain 0.05 seconds")
    for name in ("cruise_speed_mps", "curve_speed_mps", "curve_preview_m", "curve_heading_threshold_deg"):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid mission parameter: {name}")
    if config["curve_speed_mps"] > config["cruise_speed_mps"] or config["cruise_speed_mps"] > topology.navigation_speed_cap_mps:
        raise ValueError("Mission speeds exceed the declared navigation cap")
    for name in ("camera_width", "camera_height", "settle_ticks", "seed"):
        if type(runtime[name]) is not int or runtime[name] < (0 if name == "seed" else 1):
            raise ValueError(f"Invalid runtime integer: {name}")
    for name in ("max_simulation_seconds", "capture_timeout_s"):
        if isinstance(runtime[name], bool) or not isinstance(runtime[name], (int, float)) or not math.isfinite(runtime[name]) or runtime[name] <= 0:
            raise ValueError(f"Invalid runtime duration: {name}")
    route_hash = canonical_sha256(route)
    config_hash = canonical_sha256({"config": config, "route_sha256": route_hash})
    return FrozenMission(json.dumps(config), json.dumps(route), route_hash, config_hash, controller)
