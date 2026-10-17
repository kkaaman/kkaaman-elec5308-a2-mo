"""RV-06: diagnostic lane-departure monitor and HUD; it never commands the car.

The LaneUNet centreline is projected to the ground, fitted, and evaluated at a
look-ahead point ``REFERENCE_X_M`` ahead of the rear axle, just beyond the nearest
road the camera sees (~7.3 m). The fixed route's map offset is evaluated at the
same point, so neither side extrapolates into the camera's blind zone (that
extrapolation gave 0.5 m errors on curves). Both use the same sign: positive
means the look-ahead point on the ego's heading is LEFT of the centreline/route.
Nothing here sets a path, a speed or a steering angle; the fixed mission route
is the only controller input (RV-06: no route rebuilding from lane estimates).
"""

import math
from dataclasses import dataclass

import numpy as np

MIN_CONFIDENCE = 0.3
SIGN_DEADBAND_M = 0.15
REFERENCE_X_M = 8.0


@dataclass(frozen=True)
class LaneDeparture:
    """Signed lateral offset of the ego from the perceived lane centre.

    ``valid`` is geometric plausibility; ``confidence`` (0..1) combines point
    count, longitudinal coverage, fit residual and extrapolation distance.
    """

    signed_offset_m: float | None
    side: str
    valid: bool
    reason: str
    confidence: float = 0.0
    heading_rad: float | None = None
    residual_m: float | None = None
    points: int = 0
    span_m: float = 0.0


def _unknown(reason, points=0):
    return LaneDeparture(None, "unknown", False, reason, points=points)


def monitor_lane(points_xy, *, ego_x_m=0.0, ego_y_m=0.0, threshold_m=0.4):
    """Fit a perceived centreline (core X-forward/Y-left) and evaluate it at the ego.

    Positive error means ego is left of the perceived centreline. Unknown or
    degenerate geometry remains unknown; it cannot request a steering command.
    """
    if not np.isfinite(threshold_m) or threshold_m <= 0:
        raise ValueError("Positive threshold required")
    try:
        points = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    except (TypeError, ValueError):
        return _unknown("missing_or_invalid_lane")
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 2:
        return _unknown("missing_or_invalid_lane", len(points))
    points = points[np.argsort(points[:, 0])]
    span = float(points[-1, 0] - points[0, 0])
    if span < 1e-3:
        return _unknown("degenerate_lane", len(points))
    degree = 2 if len(points) >= 6 and span >= 6.0 else 1
    coefficients = np.polyfit(points[:, 0], points[:, 1], degree)
    residual = float(np.sqrt(np.mean((np.polyval(coefficients, points[:, 0]) - points[:, 1]) ** 2)))
    centre_y = float(np.polyval(coefficients, ego_x_m))
    slope = float(np.polyval(np.polyder(coefficients), ego_x_m))
    offset = ego_y_m - centre_y
    gap = max(0.0, float(points[0, 0]) - ego_x_m)
    confidence = (
        min(1.0, len(points) / 20)
        * min(1.0, span / 10.0)
        * float(np.clip(1.0 - residual / 0.3, 0.0, 1.0))
        * float(np.clip(1.0 - (gap - 4.0) / 10.0, 0.0, 1.0))
    )
    if not math.isfinite(offset) or abs(offset) > 3.5:
        return LaneDeparture(
            None, "unknown", False, "implausible_offset", 0.0, points=len(points), span_m=span
        )
    if residual > 0.5:
        return LaneDeparture(
            None, "unknown", False, "poor_fit", 0.0, points=len(points), span_m=span
        )
    side = "left" if offset > threshold_m else "right" if offset < -threshold_m else "centred"
    return LaneDeparture(
        offset,
        side,
        True,
        "diagnostic_only",
        confidence,
        math.atan(slope),
        residual,
        len(points),
        span,
    )


def monitor_from_mask(mask, calibration, *, max_range_m=30.0, reference_x_m=REFERENCE_X_M):
    """LaneUNet mask -> image centreline -> ground points -> LaneDeparture.

    Uses a flat-road ground plane at ego Z = 0 and the run's camera calibration
    (rear-axle origin). The offset is evaluated ``reference_x_m`` ahead of the
    rear axle. Returns (departure, image_path_or_None, ground_points).
    """
    from geometry.camera import pixel_to_ground
    from perception.lane_path import extract_ego_path

    path = extract_ego_path(mask)
    if path is None:
        return _unknown("no_lane_pair"), None, ()
    ground = []
    for u, v in path["centerline"]:
        point = pixel_to_ground(float(u), float(v), calibration)
        if point is not None and 0 < point[0] < max_range_m:
            ground.append((float(point[0]), float(point[1])))
    return monitor_lane(ground, ego_x_m=reference_x_m), path, tuple(ground)


def map_lookahead_offset(route, x_m, y_m, yaw_rad, reference_x_m=REFERENCE_X_M):
    """Signed offset (+left) of the ego look-ahead point from the fixed route."""
    px = x_m + reference_x_m * math.cos(yaw_rad)
    py = y_m + reference_x_m * math.sin(yaw_rad)
    return route.project(px, py).lateral_m


def compare_to_map(departure, map_signed_cte_m, *, min_confidence=MIN_CONFIDENCE):
    """Per-frame comparison with the fixed route's offset at the same look-ahead point."""
    usable = bool(
        departure.valid
        and departure.confidence >= min_confidence
        and map_signed_cte_m is not None
        and math.isfinite(map_signed_cte_m)
    )
    if not usable:
        return dict(usable=False, difference_m=None, sign_agree=None)
    difference = departure.signed_offset_m - map_signed_cte_m
    decided = (
        abs(map_signed_cte_m) >= SIGN_DEADBAND_M
        and abs(departure.signed_offset_m) >= SIGN_DEADBAND_M
    )
    return dict(
        usable=True,
        difference_m=difference,
        sign_agree=(
            math.copysign(1, departure.signed_offset_m) == math.copysign(1, map_signed_cte_m)
        )
        if decided
        else None,
    )


def summarize_monitor(records):
    """records: dicts with 'comparison' (from compare_to_map), 'valid', 'curve'."""

    def block(rows):
        used = [r["comparison"] for r in rows if r["comparison"]["usable"]]
        diffs = np.abs([c["difference_m"] for c in used]) if used else np.array([])
        signs = [c["sign_agree"] for c in used if c["sign_agree"] is not None]
        return dict(
            frames=len(rows),
            valid_fraction=(sum(r["valid"] for r in rows) / len(rows)) if rows else None,
            usable_frames=len(used),
            median_abs_difference_m=float(np.median(diffs)) if len(diffs) else None,
            p95_abs_difference_m=float(np.percentile(diffs, 95)) if len(diffs) else None,
            sign_decided_frames=len(signs),
            sign_agreement=(sum(signs) / len(signs)) if signs else None,
        )

    return dict(
        all=block(records),
        straight=block([r for r in records if not r["curve"]]),
        curve=block([r for r in records if r["curve"]]),
        control_coupling=False,
    )


def draw_monitor(rgb, departure, map_signed_cte_m=None):
    """HUD overlay: lane monitor value, confidence/validity and map CTE."""
    import cv2

    result = rgb.copy()
    if departure.valid:
        lane = f"Lane monitor: {departure.signed_offset_m:+.2f} m ({departure.side}) conf {departure.confidence:.2f}"
    else:
        lane = f"Lane monitor: INVALID ({departure.reason})"
    lines = [lane + "  [diagnostic only]"]
    if map_signed_cte_m is not None:
        text = f"Map CTE: {map_signed_cte_m:+.2f} m"
        if departure.valid:
            text += f"  diff {departure.signed_offset_m - map_signed_cte_m:+.2f} m"
        lines.append(text)
    colour = (
        (0, 220, 255)
        if departure.valid and departure.confidence >= MIN_CONFIDENCE
        else (255, 140, 0)
    )
    for i, text in enumerate(lines):
        y = 45 + 18 * i
        cv2.putText(result, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3)
        cv2.putText(result, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, colour, 1)
    return result
