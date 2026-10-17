"""Diagnostic-only lane departure monitor; has no controller or path setter."""

from dataclasses import dataclass

import numpy as np

from planning.corridor import RouteGeometry


@dataclass(frozen=True)
class LaneDeparture:
    signed_offset_m: float | None
    side: str
    valid: bool
    reason: str


def monitor_lane(boundary_points_xy, *, ego_x_m=0.0, ego_y_m=0.0, threshold_m=0.4):
    """Input is a perceived centreline in core X-forward/Y-left coordinates.

    Positive error means ego is left of the perceived centreline. Unknown or
    degenerate geometry remains unknown; it cannot request a steering command.
    """
    if not np.isfinite(threshold_m) or threshold_m <= 0:
        raise ValueError("Positive threshold required")
    try:
        route = RouteGeometry(boundary_points_xy)
        projection = route.project(ego_x_m, ego_y_m)
    except (TypeError, ValueError):
        return LaneDeparture(None, "unknown", False, "missing_or_invalid_lane")
    error = projection.lateral_m
    side = "left" if error > threshold_m else "right" if error < -threshold_m else "centred"
    return LaneDeparture(error, side, True, "diagnostic_only")


def draw_monitor(rgb, departure):
    import cv2

    result = rgb.copy()
    label = f"Lane monitor: {departure.side}"
    if departure.valid:
        label += f" ({departure.signed_offset_m:+.2f} m)"
    cv2.putText(result, label, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 0), 1)
    return result
