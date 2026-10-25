"""Lane-departure monitor: image-derived lateral offset versus map-based CTE (RV-06).

The A1 demos fed ``perception.lane_path.extract_ego_path`` into
``lane_to_waypoints.build_local_route`` and rebuilt the controller route every
cycle. In A2 the controller follows the frozen map route only. Lane estimates
are repurposed here as an *observer*: the monitor compares the image-derived
lateral offset with the route's signed cross-track error and reports the
difference, a confidence and a validity flag for the HUD and run logs.

Nothing in this module produces waypoints, paths, commands or route updates,
and nothing here imports the simulator. The monitor is pure arithmetic on the
shared contracts and may be driven from recorded frames offline.

Sign convention (``docs/interfaces.md``): ego +Y is left, so a positive
lateral offset or CTE means the rear axle is *left* of the lane/route centre.
``carla_adapter.s0_runner.signed_cross_track`` and
``evaluation.a1_comparison`` use the same convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from autonomy.interfaces import CameraCalibration
from geometry.camera import pixel_to_ground

__all__ = [
    "DepartureReport",
    "LaneDepartureMonitor",
    "LaneEstimate",
    "MonitorConfig",
    "lane_estimate_from_centerline",
]


def _finite(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


@dataclass(frozen=True)
class LaneEstimate:
    """Image-derived ego pose relative to the detected lane centre.

    ``lateral_offset_m``: signed rear-axle offset from the lane centre in
    metres, positive left. ``heading_rad``: ego heading relative to the lane
    tangent in radians, positive counterclockwise (nose pointing left of the
    lane). ``confidence`` is dimensionless in [0, 1]. ``frame_id`` and
    ``sim_time_s`` identify the observation the estimate came from so the
    monitor can reject stale values. Invalid values raise ``ValueError``.
    """

    lateral_offset_m: float
    heading_rad: float
    confidence: float
    frame_id: int
    sim_time_s: float

    def __post_init__(self):
        _finite("lateral_offset_m", self.lateral_offset_m)
        _finite("heading_rad", self.heading_rad)
        _finite("confidence", self.confidence)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must lie in [0, 1]")
        if type(self.frame_id) is not int or self.frame_id < 0:
            raise ValueError("frame_id must be a nonnegative integer")
        _finite("sim_time_s", self.sim_time_s)
        if self.sim_time_s < 0:
            raise ValueError("sim_time_s must be nonnegative")


@dataclass(frozen=True)
class MonitorConfig:
    """Validity and agreement limits for the departure monitor.

    ``max_age_s``: an estimate older than this (simulation seconds) is stale.
    ``min_confidence``: estimates below this are low confidence.
    ``max_abs_offset_m`` / ``max_abs_heading_rad``: estimates outside these
    ranges are physically implausible for a vehicle inside its lane and are
    flagged ``outside_range``. ``disagreement_m``: absolute image-minus-map
    difference above which a valid report is labelled ``disagree``.
    """

    max_age_s: float = 0.5
    min_confidence: float = 0.3
    max_abs_offset_m: float = 2.5
    max_abs_heading_rad: float = math.radians(30.0)
    disagreement_m: float = 0.5

    def __post_init__(self):
        for name in (
            "max_age_s",
            "min_confidence",
            "max_abs_offset_m",
            "max_abs_heading_rad",
            "disagreement_m",
        ):
            value = getattr(self, name)
            _finite(name, value)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.min_confidence > 1.0:
            raise ValueError("min_confidence must not exceed 1")


@dataclass(frozen=True)
class DepartureReport:
    """One monitor output. Never a command, route or waypoint.

    ``difference_m`` is ``image_offset_m - map_cte_m`` (metres, positive when
    the camera places the ego further left than the map does) and is ``None``
    whenever the report is invalid. ``status`` is one of ``agree``,
    ``disagree`` or ``invalid``; ``reasons`` lists the validity failures
    (``no_estimate``, ``stale``, ``low_confidence``, ``outside_range``,
    ``no_map_cte``).
    """

    frame_id: int | None
    sim_time_s: float | None
    image_offset_m: float | None
    image_heading_rad: float | None
    map_cte_m: float | None
    difference_m: float | None
    confidence: float
    valid: bool
    status: str
    reasons: tuple[str, ...] = ()

    def as_record(self) -> dict:
        """Plain JSON/CSV-friendly dictionary for run logs."""
        return {
            "frame_id": self.frame_id,
            "sim_time_s": self.sim_time_s,
            "lane_image_offset_m": self.image_offset_m,
            "lane_image_heading_rad": self.image_heading_rad,
            "lane_map_cte_m": self.map_cte_m,
            "lane_difference_m": self.difference_m,
            "lane_monitor_confidence": self.confidence,
            "lane_monitor_valid": self.valid,
            "lane_monitor_status": self.status,
            "lane_monitor_reasons": ";".join(self.reasons),
        }


def lane_estimate_from_centerline(
    centerline_px,
    calibration: CameraCalibration,
    *,
    frame_id: int,
    sim_time_s: float,
    min_forward_m: float = 2.0,
    max_forward_m: float = 20.0,
    min_points: int = 6,
    expected_points: int = 20,
    residual_scale_m: float = 0.25,
) -> LaneEstimate | None:
    """Project an image-space centreline onto the ground and fit a local line.

    ``centerline_px`` is an Nx2 array of ``(u, v)`` pixel coordinates as
    returned by ``perception.lane_path.extract_ego_path()["centerline"]``.
    Points are projected with ``geometry.camera.pixel_to_ground`` (flat local
    ground assumption) and kept when their forward distance lies in
    ``[min_forward_m, max_forward_m]`` metres. A polynomial fit
    ``y = a + b*x (+ c*x^2)`` in ego coordinates (quadratic once at least eight
    points span three metres, so bends do not bias the intercept) gives the
    lateral offset of the lane centre at the rear axle (``x = 0``); the ego
    offset from the lane is its negation (lane centre to the right means the
    ego is left, positive). Heading is ``-atan(b)``.

    Confidence is the product of point support (``kept / expected_points``,
    capped at 1) and a residual term ``1 / (1 + rms_residual / residual_scale_m)``.
    Returns ``None`` when fewer than ``min_points`` ground points survive, so
    the monitor reports ``no_estimate`` rather than guessing.
    """
    if not isinstance(calibration, CameraCalibration):
        raise TypeError("Expected CameraCalibration")
    if type(min_points) is not int or min_points < 2:
        raise ValueError("min_points must be an integer of at least 2")
    if type(expected_points) is not int or expected_points < min_points:
        raise ValueError("expected_points must be an integer of at least min_points")
    for name, value in (
        ("min_forward_m", min_forward_m),
        ("max_forward_m", max_forward_m),
        ("residual_scale_m", residual_scale_m),
    ):
        _finite(name, value)
        if value <= 0:
            raise ValueError(f"{name} must be positive metres")
    if min_forward_m >= max_forward_m:
        raise ValueError("min_forward_m must be below max_forward_m")
    points = np.asarray(centerline_px, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("centerline must be an Nx2 array of pixel coordinates")
    if points.size and not np.isfinite(points).all():
        raise ValueError("centerline pixels must be finite")
    ground = []
    for u, v in points:
        point = pixel_to_ground(float(u), float(v), calibration)
        if point is None:
            continue
        if min_forward_m <= point[0] <= max_forward_m:
            ground.append((float(point[0]), float(point[1])))
    if len(ground) < min_points:
        return None
    xs = np.array([g[0] for g in ground])
    ys = np.array([g[1] for g in ground])
    if np.ptp(xs) < 1e-6:
        return None
    # A straight-line fit is biased on bends (the far points pull the intercept
    # outward), so fit lane curvature as a quadratic when the support allows
    # and read offset/heading at the rear axle, x = 0.
    order = 2 if len(ground) >= 8 and np.ptp(xs) >= 3.0 else 1
    coefficients = np.polyfit(xs, ys, order)
    slope, intercept = float(coefficients[-2]), float(coefficients[-1])
    residual = ys - np.polyval(coefficients, xs)
    rms = float(math.sqrt(float(np.mean(residual**2))))
    support = min(1.0, len(ground) / expected_points)
    confidence = support / (1.0 + rms / residual_scale_m)
    return LaneEstimate(
        lateral_offset_m=-float(intercept),
        heading_rad=-math.atan(float(slope)),
        confidence=max(0.0, min(1.0, confidence)),
        frame_id=frame_id,
        sim_time_s=sim_time_s,
    )


class LaneDepartureMonitor:
    """Compare image-derived lane offset with map-based CTE; observe only.

    ``update`` consumes the latest ``LaneEstimate`` (or ``None`` when the lane
    extractor found nothing) and the map-based signed CTE in metres at the
    current simulation time. It returns a :class:`DepartureReport`. The
    monitor keeps the last report for the HUD but has no hooks into the
    controller, route manager or stack; callers must not derive commands from it.
    """

    def __init__(self, config: MonitorConfig | None = None):
        self.config = config or MonitorConfig()
        self.last_report: DepartureReport | None = None
        self.update_count = 0
        self.valid_count = 0

    def reset(self) -> None:
        self.last_report = None
        self.update_count = 0
        self.valid_count = 0

    def update(
        self,
        estimate: LaneEstimate | None,
        map_cte_m: float | None,
        *,
        now_sim_time_s: float,
        frame_id: int | None = None,
    ) -> DepartureReport:
        """Produce one report. Stale, low-confidence or out-of-range estimates are invalid.

        ``now_sim_time_s`` is the current simulation time in seconds; staleness
        is ``now_sim_time_s - estimate.sim_time_s`` compared with
        ``config.max_age_s``. A negative age (estimate from the future) is also
        stale. ``map_cte_m`` must be finite or ``None`` (``no_map_cte``).
        """
        _finite("now_sim_time_s", now_sim_time_s)
        reasons = []
        confidence = 0.0
        offset = heading = None
        if estimate is None:
            reasons.append("no_estimate")
        elif not isinstance(estimate, LaneEstimate):
            raise TypeError("Expected LaneEstimate or None")
        else:
            confidence = estimate.confidence
            offset, heading = estimate.lateral_offset_m, estimate.heading_rad
            age = now_sim_time_s - estimate.sim_time_s
            if age < 0 or age > self.config.max_age_s:
                reasons.append("stale")
            if estimate.confidence < self.config.min_confidence:
                reasons.append("low_confidence")
            if (
                abs(estimate.lateral_offset_m) > self.config.max_abs_offset_m
                or abs(estimate.heading_rad) > self.config.max_abs_heading_rad
            ):
                reasons.append("outside_range")
        if map_cte_m is None:
            reasons.append("no_map_cte")
        else:
            _finite("map_cte_m", map_cte_m)
        valid = not reasons
        difference = offset - map_cte_m if valid else None
        if not valid:
            status = "invalid"
        elif abs(difference) > self.config.disagreement_m:
            status = "disagree"
        else:
            status = "agree"
        report = DepartureReport(
            frame_id=estimate.frame_id if estimate is not None else frame_id,
            sim_time_s=estimate.sim_time_s if estimate is not None else now_sim_time_s,
            image_offset_m=offset,
            image_heading_rad=heading,
            map_cte_m=map_cte_m,
            difference_m=difference,
            confidence=confidence,
            valid=valid,
            status=status,
            reasons=tuple(reasons),
        )
        self.last_report = report
        self.update_count += 1
        self.valid_count += int(valid)
        return report
