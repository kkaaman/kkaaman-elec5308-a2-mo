"""Plain-data contracts shared by every A2 package.

Nothing here imports a simulator or a neural-network library. Simulator
adapters translate their own objects into these types; core packages only ever
see these types. ``docs/interfaces.md`` explains the conventions with examples.

Conventions (all types):

* Units are SI: metres, seconds, radians, metres per second.
* **World frame** (``map``): right-handed, ``x`` forward along the map's +X,
  ``y`` left, ``z`` up; yaw is counter-clockwise from +X. It matches
  :class:`navigation_control.VehicleState`. (CARLA's +Y points right; the
  CARLA adapter negates ``y`` and yaw at the boundary.)
* **Vehicle frame** (``vehicle``): origin at the rear-axle centre on the
  ground, ``x`` forward, ``y`` left, ``z`` up.
* **Camera optical frame**: ``x`` right, ``y`` down, ``z`` forward along the
  optical axis (OpenCV). Pixel ``(u, v)`` has ``u`` to the right and ``v``
  down from the top-left corner.
* **Clocks are never mixed.** ``sim_time_s`` is simulation time.
  ``host_*_s`` fields are ``time.perf_counter()`` seconds on the machine that
  runs the stack. A remote server's clock is a third domain and never appears
  here directly.
* ``frame_id`` is the simulator's frame counter. Everything derived from a
  frame carries that ``frame_id``.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

import numpy as np

from navigation_control import ControlCommand, VehicleState

__all__ = [
    "ActorTruth",
    "CameraModel",
    "CollisionEvent",
    "ControlCommand",
    "ControlPacket",
    "Detection2D",
    "DetectionBatch",
    "Detector",
    "Diagnostics",
    "DriveMode",
    "EvaluationPacket",
    "Extrinsics",
    "FrameBundle",
    "Gear",
    "ImageBox",
    "LaneGraph",
    "LaneInvasionEvent",
    "LaneSegment",
    "LidarScan",
    "Maneuver",
    "Mission",
    "MotionState",
    "ScenarioEvent",
    "SensorStatus",
    "SimulatorInterface",
    "SensorTimeout",
    "VehicleGeometry",
]


# --------------------------------------------------------------------------- helpers
def _finite(name: str, value: float) -> None:
    if not isinstance(value, numbers.Real) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number, got {value!r}")


def _positive(name: str, value: float) -> None:
    _finite(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def _nonnegative(name: str, value: float) -> None:
    _finite(name, value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative, got {value!r}")


def _probability(name: str, value: float) -> None:
    _finite(name, value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")


def _frozen_mapping(obj: Any, name: str) -> None:
    """Replace a dict field with a read-only copy (dataclass is frozen)."""
    object.__setattr__(obj, name, MappingProxyType(dict(getattr(obj, name))))


# --------------------------------------------------------------------- geometry data
@dataclass(frozen=True)
class Extrinsics:
    """Pose of a sensor body in the vehicle frame.

    Translation in metres; roll, pitch and yaw in radians, applied as
    yaw (about z), then pitch (about y), then roll (about x), right-handed.
    A camera with all angles zero looks along vehicle +x. The camera's optical
    frame is obtained from its body frame by the fixed axis swap
    ``x_opt = -y_body, y_opt = -z_body, z_opt = x_body``.
    """

    x_m: float
    y_m: float
    z_m: float
    roll_rad: float = 0.0
    pitch_rad: float = 0.0
    yaw_rad: float = 0.0

    def __post_init__(self) -> None:
        for name in ("x_m", "y_m", "z_m", "roll_rad", "pitch_rad", "yaw_rad"):
            _finite(name, getattr(self, name))


@dataclass(frozen=True)
class CameraModel:
    """Pinhole intrinsics plus mounting pose. RGB and depth share one model.

    ``depth_kind`` records what the depth image measures: ``"optical_z"``
    (distance along the optical axis) or ``"ray"`` (Euclidean distance to the
    camera centre). The CARLA value must be confirmed on the installed build
    (BL-B-04); until then it is ``"optical_z"`` with ``depth_kind_verified``
    left ``False``.
    """

    width_px: int
    height_px: int
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    mount: Extrinsics
    depth_kind: str = "optical_z"
    depth_kind_verified: bool = False

    def __post_init__(self) -> None:
        for name in ("width_px", "height_px"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        _positive("fx_px", self.fx_px)
        _positive("fy_px", self.fy_px)
        _finite("cx_px", self.cx_px)
        _finite("cy_px", self.cy_px)
        if not (0 <= self.cx_px <= self.width_px and 0 <= self.cy_px <= self.height_px):
            raise ValueError("principal point must lie inside the image")
        if self.depth_kind not in ("optical_z", "ray"):
            raise ValueError("depth_kind must be 'optical_z' or 'ray'")
        if not isinstance(self.mount, Extrinsics):
            raise TypeError("mount must be an Extrinsics")

    @classmethod
    def from_fov(cls, width_px: int, height_px: int, horizontal_fov_rad: float,
                 mount: Extrinsics, **kwargs: Any) -> CameraModel:
        """Square-pixel camera with a centred principal point (CARLA's model)."""
        _positive("horizontal_fov_rad", horizontal_fov_rad)
        if horizontal_fov_rad >= math.pi:
            raise ValueError("horizontal_fov_rad must be below pi")
        f = width_px / (2.0 * math.tan(horizontal_fov_rad / 2.0))
        return cls(width_px, height_px, f, f, width_px / 2.0, height_px / 2.0, mount, **kwargs)


@dataclass(frozen=True)
class VehicleGeometry:
    """Measured geometry of the one vehicle blueprint used for all runs (BL-E-01).

    ``max_steer_rad`` is the physical front-wheel limit, read from the vehicle,
    and is separate from any controller cap. Distances are from the rear-axle
    origin of the vehicle frame.
    """

    blueprint: str
    wheelbase_m: float
    rear_axle_to_front_bumper_m: float
    rear_axle_to_rear_bumper_m: float
    width_m: float
    max_steer_rad: float

    def __post_init__(self) -> None:
        for name in ("wheelbase_m", "rear_axle_to_front_bumper_m", "width_m", "max_steer_rad"):
            _positive(name, getattr(self, name))
        _nonnegative("rear_axle_to_rear_bumper_m", self.rear_axle_to_rear_bumper_m)
        if self.rear_axle_to_front_bumper_m <= self.wheelbase_m:
            raise ValueError("front bumper must be ahead of the front axle")
        if self.max_steer_rad >= math.pi / 2:
            raise ValueError("max_steer_rad must be below pi/2")


# ------------------------------------------------------------------------- ego state
class Gear(str, Enum):
    """Requested transmission direction. Changes only while stopped."""

    FORWARD = "forward"
    REVERSE = "reverse"


class DriveMode(str, Enum):
    """Which controller owns the vehicle (extension hook for parking, E5)."""

    DRIVE = "drive"
    PARK = "park"


@dataclass(frozen=True)
class MotionState:
    """Ego pose in the world frame at the rear axle, with **signed** speed.

    ``speed_mps`` is the longitudinal velocity along the vehicle's +x axis,
    negative when reversing. Use :meth:`to_vehicle_state` to feed the A1
    forward-only controller.
    """

    x_m: float
    y_m: float
    yaw_rad: float
    speed_mps: float
    yaw_rate_rps: float = 0.0
    z_m: float = 0.0

    def __post_init__(self) -> None:
        for name in ("x_m", "y_m", "yaw_rad", "speed_mps", "yaw_rate_rps", "z_m"):
            _finite(name, getattr(self, name))

    def to_vehicle_state(self, reverse_tolerance_mps: float = 0.05) -> VehicleState:
        """Return the forward-only A1 state.

        Small negative speeds from simulator noise (down to
        ``-reverse_tolerance_mps``) are clamped to zero. Real reverse motion
        raises ``ValueError``; it must go to a reverse-capable controller.
        """
        if self.speed_mps < -reverse_tolerance_mps:
            raise ValueError(f"vehicle is reversing ({self.speed_mps:.3f} m/s); "
                             "VehicleState is forward-only")
        return VehicleState(self.x_m, self.y_m, self.yaw_rad, max(0.0, self.speed_mps))


# --------------------------------------------------------------------------- sensors
@dataclass(frozen=True)
class SensorStatus:
    """Whether one sensor delivered a usable reading for this frame."""

    name: str
    valid: bool
    frame_id: Optional[int] = None
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.valid and not self.reason:
            raise ValueError(f"invalid sensor {self.name!r} must give a reason")


@dataclass(frozen=True)
class LidarScan:
    """LiDAR points in the vehicle frame for one frame (extension E2).

    ``points`` is an ``(N, 4)`` float32 array of x, y, z (m) and intensity.
    """

    points: np.ndarray
    frame_id: int

    def __post_init__(self) -> None:
        pts = self.points
        if not isinstance(pts, np.ndarray) or pts.ndim != 2 or pts.shape[1] != 4:
            raise ValueError("points must be an (N, 4) array")
        if pts.dtype != np.float32:
            raise ValueError("points must be float32")


@dataclass(frozen=True)
class FrameBundle:
    """Everything the stack may observe for one simulator frame.

    Every field describes the **same** frame: ``rgb`` and ``depth_m`` were
    captured at ``frame_id`` and ``ego`` is the pose at that frame. Never
    combine an image with a pose from another frame without explicit motion
    compensation.

    * ``rgb``: ``(H, W, 3)`` uint8, **RGB** channel order (not CARLA's BGRA).
    * ``depth_m``: ``(H, W)`` float32 metres, decoded from the raw depth
      image; ``NaN`` marks pixels with no valid depth. ``None`` if the frame
      has no depth sensor.
    * ``host_tick_s``: host time when the stack requested this simulator tick,
      the documented capture-origin approximation for latency.
    * ``host_receive_s``: host time when the bundle was complete.
    * ``sensors``: status of every configured sensor; required sensors that
      are invalid make :attr:`valid` false.
    """

    run_id: str
    frame_id: int
    sim_time_s: float
    host_tick_s: float
    host_receive_s: float
    ego: MotionState
    camera: CameraModel
    rgb: np.ndarray
    depth_m: Optional[np.ndarray] = None
    lidar: Optional[LidarScan] = None
    sensors: tuple[SensorStatus, ...] = ()

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if not isinstance(self.frame_id, int) or self.frame_id < 0:
            raise ValueError("frame_id must be a nonnegative integer")
        _nonnegative("sim_time_s", self.sim_time_s)
        _finite("host_tick_s", self.host_tick_s)
        _finite("host_receive_s", self.host_receive_s)
        if self.host_receive_s < self.host_tick_s:
            raise ValueError("host_receive_s cannot precede host_tick_s")
        if not isinstance(self.ego, MotionState):
            raise TypeError("ego must be a MotionState")
        h, w = self.camera.height_px, self.camera.width_px
        if not isinstance(self.rgb, np.ndarray) or self.rgb.shape != (h, w, 3) or self.rgb.dtype != np.uint8:
            raise ValueError(f"rgb must be a ({h}, {w}, 3) uint8 array matching the camera model")
        if self.depth_m is not None:
            d = self.depth_m
            if not isinstance(d, np.ndarray) or d.shape != (h, w) or d.dtype != np.float32:
                raise ValueError(f"depth_m must be a ({h}, {w}) float32 array")
        if self.lidar is not None and self.lidar.frame_id != self.frame_id:
            raise ValueError("lidar scan belongs to a different frame")
        for status in self.sensors:
            if status.valid and status.frame_id is not None and status.frame_id != self.frame_id:
                raise ValueError(f"sensor {status.name!r} reading is from frame {status.frame_id}")

    @property
    def valid(self) -> bool:
        """True when every listed sensor delivered a reading for this frame."""
        return all(s.valid for s in self.sensors)

    @property
    def frame_age_s(self) -> float:
        """Host seconds between requesting the tick and completing the bundle."""
        return self.host_receive_s - self.host_tick_s


class SensorTimeout(RuntimeError):
    """A sensor did not deliver within the bounded tick timeout.

    The stack treats this as a fault and commands a stop; it never reuses a
    previous moving command.
    """


# -------------------------------------------------------------------------- control
@dataclass(frozen=True)
class ControlPacket:
    """One command for the simulator, tied to the frame that produced it.

    ``command`` is the A1 :class:`navigation_control.ControlCommand`
    (positive-left steering in radians, throttle and brake in [0, 1], never
    both positive). The adapter owns sign flips, normalisation by the physical
    steering limit and gear handling.
    """

    frame_id: int
    command: ControlCommand
    reason: str
    gear: Gear = Gear.FORWARD
    mode: DriveMode = DriveMode.DRIVE
    emergency: bool = False
    hand_brake: bool = False
    host_command_s: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.command, ControlCommand):
            raise TypeError("command must be a ControlCommand")
        if not self.reason:
            raise ValueError("every command needs a reason (for logs and the HUD)")
        if self.emergency and (self.command.throttle > 0 or self.command.brake <= 0):
            raise ValueError("an emergency packet must brake with zero throttle")

    @classmethod
    def stop(cls, frame_id: int, reason: str, brake: float = 1.0,
             host_command_s: Optional[float] = None) -> ControlPacket:
        """Full stop with straight wheels, flagged as an emergency."""
        return cls(frame_id, ControlCommand(0.0, 0.0, brake), reason,
                   emergency=True, host_command_s=host_command_s)


# ---------------------------------------------------------------------- map / route
class Maneuver(str, Enum):
    """Choice at a junction."""

    STRAIGHT = "straight"
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class LaneSegment:
    """One lane between junction boundaries, as exported from the map (BL-A-02).

    ``centerline`` is a tuple of ``(x_m, y_m)`` world-frame points in driving
    order. ``successors`` are IDs of segments legally reachable from its end.
    """

    segment_id: str
    centerline: tuple[tuple[float, float], ...]
    successors: tuple[str, ...] = ()
    in_junction: bool = False
    road_id: Optional[int] = None
    lane_id: Optional[int] = None
    speed_limit_mps: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.segment_id:
            raise ValueError("segment_id must be non-empty")
        if len(self.centerline) < 2:
            raise ValueError(f"segment {self.segment_id!r} needs at least two points")
        for x, y in self.centerline:
            _finite("centerline x", x)
            _finite("centerline y", y)

    @property
    def length_m(self) -> float:
        pts = self.centerline
        return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))


@dataclass(frozen=True)
class LaneGraph:
    """Serializable lane topology for one map. No simulator objects inside."""

    map_name: str
    segments: Mapping[str, LaneSegment]
    source_hash: str = ""

    def __post_init__(self) -> None:
        _frozen_mapping(self, "segments")
        for key, seg in self.segments.items():
            if key != seg.segment_id:
                raise ValueError(f"segment stored under {key!r} has id {seg.segment_id!r}")
            for nxt in seg.successors:
                if nxt not in self.segments:
                    raise ValueError(f"segment {key!r} points to unknown successor {nxt!r}")

    def to_dict(self) -> dict:
        """Deterministic, JSON-ready form (sorted by segment id)."""
        return {
            "map_name": self.map_name,
            "source_hash": self.source_hash,
            "segments": [
                {
                    "segment_id": s.segment_id,
                    "centerline": [list(p) for p in s.centerline],
                    "successors": sorted(s.successors),
                    "in_junction": s.in_junction,
                    "road_id": s.road_id,
                    "lane_id": s.lane_id,
                    "speed_limit_mps": s.speed_limit_mps,
                }
                for _, s in sorted(self.segments.items())
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LaneGraph:
        segments = {}
        for item in data["segments"]:
            seg = LaneSegment(
                segment_id=item["segment_id"],
                centerline=tuple((float(x), float(y)) for x, y in item["centerline"]),
                successors=tuple(item.get("successors", ())),
                in_junction=bool(item.get("in_junction", False)),
                road_id=item.get("road_id"),
                lane_id=item.get("lane_id"),
                speed_limit_mps=item.get("speed_limit_mps"),
            )
            segments[seg.segment_id] = seg
        return cls(data["map_name"], segments, data.get("source_hash", ""))


@dataclass(frozen=True)
class Mission:
    """A fixed route: start pose, junction choices in order, and destination."""

    map_name: str
    start: MotionState
    maneuvers: tuple[Maneuver, ...]
    destination_xy: tuple[float, float]
    name: str = ""
    graph_hash: str = ""

    def __post_init__(self) -> None:
        if any(not isinstance(m, Maneuver) for m in self.maneuvers):
            raise TypeError("maneuvers must be Maneuver values")
        _finite("destination x", self.destination_xy[0])
        _finite("destination y", self.destination_xy[1])


# ------------------------------------------------------------------- detector (C-01)
@dataclass(frozen=True)
class ImageBox:
    """Canonical validated axis-aligned box in pixel coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        for name in ("x_min", "y_min", "x_max", "y_max"):
            _finite(name, getattr(self, name))
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("box must have x_max > x_min and y_max > y_min")

    @property
    def width_px(self) -> float:
        return self.x_max - self.x_min

    @property
    def height_px(self) -> float:
        return self.y_max - self.y_min

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)


@dataclass(frozen=True)
class Detection2D:
    """One image detection. ``track_id`` is only meaningful within its batch's
    ``source`` and ``session_id``; use :meth:`track_key` across sources."""

    class_name: str
    confidence: float
    box: ImageBox
    track_id: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.class_name:
            raise ValueError("class_name must be non-empty")
        _probability("confidence", self.confidence)
        if not isinstance(self.box, ImageBox):
            raise TypeError("box must be an ImageBox")


@dataclass(frozen=True)
class DetectionBatch:
    """Detector output for one frame, with provenance and timing.

    * ``source``: backend id, e.g. ``"local"`` or ``"remote"``.
    * ``session_id``: the tracker session; reset per episode.
    * ``model_version``: model file name plus checksum prefix.
    * ``host_start_s`` / ``host_end_s``: host clock around the call that
      produced the result (for a remote backend, the client-side request and
      response times, never the server's clock).
    * ``valid``: ``False`` means *no usable result* (failure, timeout, stale).
      That is different from a valid batch with no detections.
    """

    frame_id: int
    source: str
    session_id: str
    model_version: str
    detections: tuple[Detection2D, ...]
    tracked: bool
    host_start_s: float
    host_end_s: float
    valid: bool = True
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.source or not self.session_id:
            raise ValueError("source and session_id must be non-empty")
        _finite("host_start_s", self.host_start_s)
        _finite("host_end_s", self.host_end_s)
        if self.host_end_s < self.host_start_s:
            raise ValueError("host_end_s cannot precede host_start_s")
        if not self.valid and not self.reason:
            raise ValueError("an invalid batch must give a reason")
        if not self.valid and self.detections:
            raise ValueError("an invalid batch cannot carry detections")
        if not self.tracked and any(d.track_id is not None for d in self.detections):
            raise ValueError("untracked batches must not carry track ids")

    @property
    def latency_s(self) -> float:
        return self.host_end_s - self.host_start_s

    def track_key(self, detection: Detection2D) -> Optional[str]:
        """Globally unique identity ``source/session/track``, or ``None``."""
        if detection.track_id is None:
            return None
        return f"{self.source}/{self.session_id}/{detection.track_id}"


@runtime_checkable
class Detector(Protocol):
    """What the stack needs from any object detector, local or remote.

    ``reset`` starts a new tracker session and must drop all track history.
    ``infer`` never raises for backend failures: it returns a batch with
    ``valid=False`` and a reason, so the stack can fall back or stop.
    """

    source: str
    model_version: str

    def reset(self, session_id: str) -> None: ...

    def infer(self, rgb: np.ndarray, frame_id: int, track: bool = True) -> DetectionBatch: ...


# ---------------------------------------------------------------------- diagnostics
@dataclass(frozen=True)
class Diagnostics:
    """What the stack decided this frame and why (logged; drawn by the HUD).

    ``speed_caps_mps`` holds each candidate cap, e.g. ``cruise``, ``curve``,
    ``obstacle``, ``endpoint``; the target is their minimum.
    ``timings_ms`` holds per-stage durations on the host clock.
    """

    frame_id: int
    sim_time_s: float
    behaviour: str
    mode: DriveMode
    target_speed_mps: float
    actual_speed_mps: float
    speed_caps_mps: Mapping[str, float] = field(default_factory=dict)
    detector_source: str = ""
    confirm_count: int = 0
    clear_count: int = 0
    nearest_gap_m: Optional[float] = None
    signed_cte_m: Optional[float] = None
    route_progress_m: Optional[float] = None
    timings_ms: Mapping[str, float] = field(default_factory=dict)
    fault: Optional[str] = None

    def __post_init__(self) -> None:
        _frozen_mapping(self, "speed_caps_mps")
        _frozen_mapping(self, "timings_ms")
        _nonnegative("target_speed_mps", self.target_speed_mps)


# ------------------------------------------------------------- evaluation-only truth
@dataclass(frozen=True)
class ActorTruth:
    """Simulator ground truth for one actor. Evaluation only."""

    actor_id: int
    kind: str
    x_m: float
    y_m: float
    yaw_rad: float
    length_m: float
    width_m: float


@dataclass(frozen=True)
class CollisionEvent:
    frame_id: int
    other_actor_id: Optional[int]
    other_kind: str
    impulse_ns: float


@dataclass(frozen=True)
class LaneInvasionEvent:
    frame_id: int
    marking_types: tuple[str, ...]


@dataclass(frozen=True)
class ScenarioEvent:
    """Scripted scenario change, e.g. ``obstacle_spawned``/``obstacle_removed``."""

    frame_id: int
    name: str
    detail: str = ""


@dataclass(frozen=True)
class EvaluationPacket:
    """Ground truth for one frame. **Only the evaluation package may read it.**

    Obstacle actor poses, boxes and light states live here so metrics can be
    computed; they must never reach perception, planning or control.
    """

    frame_id: int
    ego_truth: MotionState
    actors: tuple[ActorTruth, ...] = ()
    collisions: tuple[CollisionEvent, ...] = ()
    lane_invasions: tuple[LaneInvasionEvent, ...] = ()
    events: tuple[ScenarioEvent, ...] = ()


# ------------------------------------------------------------------- the simulator
@runtime_checkable
class SimulatorInterface(Protocol):
    """Lifecycle every world implements (CARLA, the kinematic harness, replay).

    One caller owns the clock: only the stack calls :meth:`tick`. The sequence
    per control cycle is ``apply(packet)`` for the frame just processed, then
    ``tick()`` to advance one fixed step and observe the next frame.

    * :meth:`reset` builds the scenario and returns the first frame.
    * :meth:`tick` advances exactly one step and returns that frame's bundle,
      or raises :class:`SensorTimeout` after ``timeout_s`` host seconds.
    * :meth:`evaluation` returns ground truth for the latest frame, for the
      evaluation package only.
    * :meth:`close` restores world settings and destroys only the actors this
      run created; it is safe to call more than once.
    """

    @property
    def vehicle_geometry(self) -> VehicleGeometry: ...

    def reset(self, scenario: Mapping[str, Any]) -> FrameBundle: ...

    def apply(self, packet: ControlPacket) -> None: ...

    def tick(self, timeout_s: float) -> FrameBundle: ...

    def evaluation(self) -> EvaluationPacket: ...

    def close(self) -> None: ...
