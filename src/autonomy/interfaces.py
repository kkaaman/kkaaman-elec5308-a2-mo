"""Version 1 simulator-independent packets (BL-F-01).

World and ego coordinates are right-handed: +X forward, +Y left, +Z up.
Angles are radians; pose reference is the rear axle. See docs/interfaces.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from navigation_control.types import ControlCommand, VehicleState

CONTRACT_VERSION = 1


def _finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _identifier(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _frame(value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("frame_id must be a nonnegative integer")


@dataclass(frozen=True)
class EgoPose:
    """Rear-axle pose and speed in the shared world coordinate frame."""

    state: VehicleState
    z_m: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.state, VehicleState):
            raise TypeError("state must be VehicleState")
        _finite("z_m", self.z_m)


@dataclass(frozen=True)
class CameraCalibration:
    """Pinhole intrinsics; image pixels have +u right and +v down.

    camera_to_ego maps optical coordinates (+X right,+Y down,+Z forward)
    to ego coordinates in metres. It is a row-major rigid 4x4 transform.
    """

    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    camera_to_ego: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        for name in ("fx_px", "fy_px", "cx_px", "cy_px"):
            _finite(name, getattr(self, name))
        if self.fx_px <= 0 or self.fy_px <= 0:
            raise ValueError("focal lengths must be positive")
        matrix = np.asarray(self.camera_to_ego, dtype=float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("camera_to_ego must be finite 4x4")
        if not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-7):
            raise ValueError("invalid homogeneous last row")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError("rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
            raise ValueError("rotation must preserve handedness")
        object.__setattr__(self, "camera_to_ego", tuple(tuple(r) for r in matrix))


@dataclass(frozen=True)
class FrameBundle:
    """Aligned RGB/depth observations plus ego pose, never obstacle truth.

    Arrays are copied into read-only snapshots. depth_m contains axial optical
    depth, not radial range. NaN represents invalid pixels; infinity, zero and
    negative values are rejected. None means no depth sensor was provided.
    """

    episode_id: str
    frame_id: int
    sim_time_s: float
    received_monotonic_s: float
    ego: EgoPose
    rgb: np.ndarray
    calibration: CameraCalibration
    depth_m: np.ndarray | None = None
    depth_frame_id: int | None = None
    capture_started_monotonic_s: float | None = None

    def __post_init__(self) -> None:
        _identifier("episode_id", self.episode_id)
        _frame(self.frame_id)
        for name in ("sim_time_s", "received_monotonic_s"):
            _finite(name, getattr(self, name))
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.capture_started_monotonic_s is not None:
            _finite("capture_started_monotonic_s", self.capture_started_monotonic_s)
            if not 0 <= self.capture_started_monotonic_s <= self.received_monotonic_s:
                raise ValueError("Capture clock must precede receipt on the same host")
        if not isinstance(self.ego, EgoPose) or not isinstance(self.calibration, CameraCalibration):
            raise TypeError("ego and calibration must use contract types")
        if not isinstance(self.rgb, np.ndarray) or self.rgb.dtype != np.uint8:
            raise ValueError("rgb must be uint8 RGB")
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3 or 0 in self.rgb.shape:
            raise ValueError("rgb must be nonempty HxWx3")
        rgb = self.rgb.copy()
        rgb.setflags(write=False)
        object.__setattr__(self, "rgb", rgb)
        if self.depth_m is None:
            if self.depth_frame_id is not None:
                raise ValueError("depth_frame_id requires depth_m")
            return
        if self.depth_frame_id != self.frame_id or type(self.depth_frame_id) is not int:
            raise ValueError("RGB and depth frame IDs must match")
        if not isinstance(self.depth_m, np.ndarray) or self.depth_m.shape != rgb.shape[:2]:
            raise ValueError("depth must align with RGB dimensions")
        if self.depth_m.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("depth must be float32 or float64 metres")
        if np.isinf(self.depth_m).any() or (self.depth_m <= 0).any():
            raise ValueError("invalid depth pixels must be NaN")
        depth = self.depth_m.copy()
        depth.setflags(write=False)
        object.__setattr__(self, "depth_m", depth)


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    x_m: float
    y_m: float
    z_m: float = 0.0

    def __post_init__(self) -> None:
        _identifier("node_id", self.node_id)
        for name in ("x_m", "y_m", "z_m"):
            _finite(name, getattr(self, name))


@dataclass(frozen=True)
class GraphEdge:
    start_id: str
    end_id: str
    points_xy_m: tuple[tuple[float, float], ...]
    speed_limit_mps: float

    def __post_init__(self) -> None:
        _identifier("start_id", self.start_id)
        _identifier("end_id", self.end_id)
        _finite("speed_limit_mps", self.speed_limit_mps)
        if self.speed_limit_mps <= 0:
            raise ValueError("speed_limit_mps must be positive")
        points = tuple(tuple(p) for p in self.points_xy_m)
        if len(points) < 2 or any(len(p) != 2 for p in points):
            raise ValueError("edge requires at least two XY points")
        for p in points:
            for v in p:
                _finite("edge point", v)
        if any(a == b for a, b in zip(points, points[1:])):
            raise ValueError("consecutive edge points must differ")
        object.__setattr__(self, "points_xy_m", points)


@dataclass(frozen=True)
class LaneGraph:
    """Directed legal transitions. Absence of an edge forbids that transition."""

    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "edges", tuple(self.edges))
        ids = {n.node_id for n in self.nodes}
        if len(ids) != len(self.nodes):
            raise ValueError("graph node IDs must be unique")
        pairs = set()
        nodes = {n.node_id: n for n in self.nodes}
        for edge in self.edges:
            if edge.start_id not in ids or edge.end_id not in ids:
                raise ValueError("edge references unknown node")
            pair = (edge.start_id, edge.end_id)
            if pair in pairs:
                raise ValueError("duplicate directed edge")
            pairs.add(pair)
            for key, point in (
                (edge.start_id, edge.points_xy_m[0]),
                (edge.end_id, edge.points_xy_m[-1]),
            ):
                node = nodes[key]
                if math.hypot(node.x_m - point[0], node.y_m - point[1]) > 1e-4:
                    raise ValueError("edge endpoints must match graph nodes")


@dataclass(frozen=True)
class Diagnostics:
    """Per-stage wall durations, validity and reasons; no simulator objects."""

    stage_durations_s: tuple[tuple[str, float], ...] = ()
    observation_valid: bool = True
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        durations = tuple(tuple(x) for x in self.stage_durations_s)
        names = set()
        for name, duration in durations:
            _identifier("stage", name)
            _finite("duration", duration)
            if duration < 0 or name in names:
                raise ValueError("stage names must be unique; durations nonnegative")
            names.add(name)
        if type(self.observation_valid) is not bool:
            raise ValueError("observation_valid must be bool")
        reasons = tuple(self.reasons)
        for reason in reasons:
            _identifier("reason", reason)
        if not self.observation_valid and not reasons:
            raise ValueError("invalid observations require a reason")
        object.__setattr__(self, "stage_durations_s", durations)
        object.__setattr__(self, "reasons", reasons)


@dataclass(frozen=True)
class ControlPacket:
    """Command tied to an episode and observation, with simulation expiry."""

    episode_id: str
    source_frame_id: int
    source_sim_time_s: float
    valid_until_sim_time_s: float
    command: ControlCommand
    diagnostics: Diagnostics
    valid_until_monotonic_s: float | None = None

    def __post_init__(self) -> None:
        _identifier("episode_id", self.episode_id)
        _frame(self.source_frame_id)
        for name in ("source_sim_time_s", "valid_until_sim_time_s"):
            _finite(name, getattr(self, name))
        if self.source_sim_time_s < 0 or self.valid_until_sim_time_s < self.source_sim_time_s:
            raise ValueError("invalid command validity interval")
        if self.valid_until_monotonic_s is not None:
            _finite("valid_until_monotonic_s", self.valid_until_monotonic_s)
            if self.valid_until_monotonic_s < 0:
                raise ValueError("Host expiry must be nonnegative")
        if not isinstance(self.command, ControlCommand) or not isinstance(
            self.diagnostics, Diagnostics
        ):
            raise TypeError("command and diagnostics must use contract types")

    def is_current(self, episode_id: str, sim_time_s: float, *, monotonic_s=None) -> bool:
        _finite("sim_time_s", sim_time_s)
        if self.valid_until_monotonic_s is not None:
            if monotonic_s is None:
                return False
            _finite("monotonic_s", monotonic_s)
            if monotonic_s > self.valid_until_monotonic_s:
                return False
        return (
            self.episode_id == episode_id
            and self.source_sim_time_s <= sim_time_s <= self.valid_until_sim_time_s
        )


@runtime_checkable
class World(Protocol):
    """One owner advances time; close restores settings and destroys owned actors."""

    def reset(self, *, seed: int) -> tuple[FrameBundle, LaneGraph]: ...
    def step(self, packet: ControlPacket) -> FrameBundle: ...
    def close(self) -> None: ...


@runtime_checkable
class AutonomyStack(Protocol):
    def reset(self, *, episode_id: str, graph: LaneGraph) -> None: ...
    def step(self, frame: FrameBundle) -> ControlPacket: ...
    def close(self) -> None: ...
