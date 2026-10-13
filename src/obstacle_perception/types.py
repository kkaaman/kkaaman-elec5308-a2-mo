"""Obstacle types shared by every obstacle source (RGB-D, LiDAR, ...).

An :class:`Obstacle` is one physical object believed to be on or near the
road, located in the vehicle frame at the moment its sensor data was captured.
An :class:`ObstacleSourceResult` is everything one source reports for one
frame, including whether the source could see at all, because "no obstacles"
and "could not tell" are different outcomes for the stop/resume logic.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol, runtime_checkable

from autonomy.interfaces import FrameBundle

__all__ = [
    "Footprint",
    "Obstacle",
    "ObstacleClass",
    "ObstacleSource",
    "ObstacleSourceResult",
    "RangeMethod",
    "SourceHealth",
]


def _finite(name: str, value: float) -> None:
    if not isinstance(value, numbers.Real) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number, got {value!r}")


def _interval(name: str, value: Optional[tuple[float, float]]) -> None:
    if value is None:
        return
    lo, hi = value
    _finite(f"{name} lower", lo)
    _finite(f"{name} upper", hi)
    if hi < lo:
        raise ValueError(f"{name} must be (low, high) with low <= high")


class ObstacleClass(str, Enum):
    """Coarse class used by behaviour and metrics."""

    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    CYCLIST = "cyclist"
    UNKNOWN = "unknown"

    @classmethod
    def from_label(cls, label: str) -> ObstacleClass:
        """Map a detector label (COCO names from YOLO) to a coarse class."""
        key = label.strip().lower()
        if key in ("car", "truck", "bus", "motorcycle", "vehicle"):
            return cls.VEHICLE
        if key in ("person", "pedestrian"):
            return cls.PEDESTRIAN
        if key in ("bicycle", "cyclist"):
            return cls.CYCLIST
        return cls.UNKNOWN


class RangeMethod(str, Enum):
    """How the obstacle's distance was measured."""

    DEPTH_MEDIAN = "depth_median"
    LIDAR_CLUSTER = "lidar_cluster"
    GROUND_PROJECTION = "ground_projection"
    KNOWN_WIDTH = "known_width"

    @property
    def is_fallback(self) -> bool:
        """Monocular estimates: flagged, with wider uncertainty."""
        return self in (RangeMethod.GROUND_PROJECTION, RangeMethod.KNOWN_WIDTH)


@dataclass(frozen=True)
class Footprint:
    """Oriented rectangle on the ground plane.

    ``frame`` is ``"vehicle"`` or ``"world"``. The centre and yaw are in that
    frame; length runs along the obstacle's own heading.
    """

    frame: str
    center_x_m: float
    center_y_m: float
    yaw_rad: float
    length_m: float
    width_m: float

    def __post_init__(self) -> None:
        if self.frame not in ("vehicle", "world"):
            raise ValueError("frame must be 'vehicle' or 'world'")
        for name in ("center_x_m", "center_y_m", "yaw_rad", "length_m", "width_m"):
            _finite(name, getattr(self, name))
        if self.length_m <= 0 or self.width_m <= 0:
            raise ValueError("length_m and width_m must be positive")

    def corners(self) -> tuple[tuple[float, float], ...]:
        """Four corners, counter-clockwise from front-left."""
        c, s = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        hl, hw = self.length_m / 2.0, self.width_m / 2.0
        local = ((hl, hw), (-hl, hw), (-hl, -hw), (hl, -hw))
        return tuple((self.center_x_m + c * x - s * y, self.center_y_m + s * x + c * y) for x, y in local)


@dataclass(frozen=True)
class Obstacle:
    """One obstacle observation from one source at one frame.

    * ``source``: which obstacle source produced it, e.g. ``"rgbd"`` or ``"lidar"``.
    * ``track_key``: namespaced identity ``source/session/track`` from the
      detector, or ``None`` when the backend gave no track id.
    * ``footprint_vehicle``: extent in the vehicle frame **at capture time**.
    * ``range_m``: distance from the sensor to the nearest visible surface,
      with one-sigma uncertainty ``range_sigma_m``.
    * ``route_s_m`` / ``route_d_m``: longitudinal and lateral intervals along
      the mission route; filled by the corridor (BL-D-04), else ``None``.
    * ``confirmed``: set by the track filter after N-of-M confirmation.
    """

    source: str
    frame_id: int
    sim_time_s: float
    obstacle_class: ObstacleClass
    label: str
    confidence: float
    footprint_vehicle: Footprint
    range_m: float
    range_sigma_m: float
    range_method: RangeMethod
    track_key: Optional[str] = None
    footprint_world: Optional[Footprint] = None
    route_s_m: Optional[tuple[float, float]] = None
    route_d_m: Optional[tuple[float, float]] = None
    confirmed: bool = False

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("source must be non-empty")
        if not isinstance(self.frame_id, int) or self.frame_id < 0:
            raise ValueError("frame_id must be a nonnegative integer")
        _finite("sim_time_s", self.sim_time_s)
        if not isinstance(self.obstacle_class, ObstacleClass):
            raise TypeError("obstacle_class must be an ObstacleClass")
        _finite("confidence", self.confidence)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.footprint_vehicle.frame != "vehicle":
            raise ValueError("footprint_vehicle must be in the vehicle frame")
        if self.footprint_world is not None and self.footprint_world.frame != "world":
            raise ValueError("footprint_world must be in the world frame")
        _finite("range_m", self.range_m)
        _finite("range_sigma_m", self.range_sigma_m)
        if self.range_m <= 0 or self.range_sigma_m < 0:
            raise ValueError("range_m must be positive and range_sigma_m nonnegative")
        if not isinstance(self.range_method, RangeMethod):
            raise TypeError("range_method must be a RangeMethod")
        _interval("route_s_m", self.route_s_m)
        _interval("route_d_m", self.route_d_m)
        if self.track_key is not None and self.track_key.count("/") < 2:
            raise ValueError("track_key must be namespaced as source/session/track")


class SourceHealth(str, Enum):
    """Could the source observe its coverage area this frame?"""

    OK = "ok"
    DEGRADED = "degraded"
    FAILED = "failed"


@dataclass(frozen=True)
class ObstacleSourceResult:
    """Everything one obstacle source reports for one frame.

    ``coverage_range_m`` is how far ahead the source could actually see this
    frame. An empty ``obstacles`` tuple only means *clear* when the source is
    healthy and its coverage reaches the distance that matters; see
    :meth:`reports_clear`. A failed source reports no obstacles and a reason.
    """

    source: str
    frame_id: int
    health: SourceHealth
    obstacles: tuple[Obstacle, ...]
    coverage_range_m: float
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.health, SourceHealth):
            raise TypeError("health must be a SourceHealth")
        _finite("coverage_range_m", self.coverage_range_m)
        if self.coverage_range_m < 0:
            raise ValueError("coverage_range_m must be nonnegative")
        if self.health is not SourceHealth.OK and not self.reason:
            raise ValueError("a degraded or failed source must give a reason")
        if self.health is SourceHealth.FAILED and self.obstacles:
            raise ValueError("a failed source cannot report obstacles")
        for obstacle in self.obstacles:
            if obstacle.frame_id != self.frame_id:
                raise ValueError("obstacle belongs to a different frame")
            if obstacle.source != self.source:
                raise ValueError("obstacle belongs to a different source")

    def reports_clear(self, required_range_m: float) -> bool:
        """True only for a healthy source that sees far enough and finds nothing.

        A track expiring or a source failing is never evidence of free road.
        """
        return (self.health is SourceHealth.OK and not self.obstacles
                and self.coverage_range_m >= required_range_m)


@runtime_checkable
class ObstacleSource(Protocol):
    """Anything that turns a frame into obstacles: RGB-D association, LiDAR, fusion."""

    source: str

    def reset(self, session_id: str) -> None: ...

    def observe(self, frame: FrameBundle) -> ObstacleSourceResult: ...
