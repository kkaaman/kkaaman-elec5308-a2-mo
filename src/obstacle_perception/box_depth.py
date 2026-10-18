"""Robust, explicitly invalid depth measurements for canonical detection boxes."""

import math
from dataclasses import dataclass

import numpy as np

from geometry.camera import pixel_depth_to_ego, pixel_to_ground
from obstacle_perception.boxes import canonical_box

# Measured points lower than this above the ego ground plane are road surface,
# kerb or lane paint, not obstacle surface. A box around a car seen at an angle
# also contains the road beside it; counting those pixels dragged a kerb-side
# parked SUV's footprint ~1 m into the lane on the first live S0 run. 0.4 m
# leaves car bodies (sills ~0.3-0.4 m, mirrors ~1 m) and legs/torsos, and
# absorbs braking pitch and gentle grade changes within the camera's range.
GROUND_CLEARANCE_M = 0.4
# Lateral footprint padding. Every supported pixel is projected at its own
# measured depth, so the depth spread of an oblique surface is geometry, not
# lateral error; padding by it moved roadside cars into the corridor.
LATERAL_TOLERANCE_M = 0.2


@dataclass(frozen=True)
class DepthPolicy:
    inner_fraction: float = 0.5
    min_valid_fraction: float = 0.6
    min_pixels: int = 9
    max_depth_m: float = 100.0
    max_spread_m: float = 1.0
    max_relative_spread: float = 0.1

    def __post_init__(self):
        for x in (self.inner_fraction, self.min_valid_fraction):
            if not math.isfinite(x) or not 0 < x <= 1:
                raise ValueError("ROI fractions must lie in (0, 1]")
        if type(self.min_pixels) is not int or self.min_pixels < 1:
            raise ValueError("min_pixels must be a positive integer")
        for x in (self.max_depth_m, self.max_spread_m, self.max_relative_spread):
            if not math.isfinite(x) or x <= 0:
                raise ValueError("Depth limits must be finite and positive")


@dataclass(frozen=True)
class DepthEstimate:
    valid: bool
    reason: str | None
    axial_depth_m: float | None = None
    point_ego_m: tuple | None = None
    valid_pixels: int = 0
    valid_fraction: float = 0.0
    spread_m: float | None = None
    source: str = "aligned_depth"
    usable_for_clearance: bool = False


def estimate_box_depth(depth_m, box, calibration, policy=None):
    """Use a central ROI; invalid data returns unknown, never clear space.

    Input must already be convention-validated axial depth. A supported surface
    point is not the obstacle centre or its complete extent.
    """
    policy = policy or DepthPolicy()
    box = canonical_box(box)
    if not isinstance(depth_m, np.ndarray) or depth_m.ndim != 2 or depth_m.dtype.kind != "f":
        raise ValueError("Expected 2D floating axial depth")
    h, w = depth_m.shape
    x0, x1 = max(0, box.x_min), min(w, box.x_max)
    y0, y1 = max(0, box.y_min), min(h, box.y_max)
    if x1 <= x0 or y1 <= y0:
        return DepthEstimate(False, "empty_roi")
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = (x1 - x0) * policy.inner_fraction / 2, (y1 - y0) * policy.inner_fraction / 2
    r0, c0 = max(0, math.ceil(cy - dy - 0.5)), max(0, math.ceil(cx - dx - 0.5))
    roi = depth_m[
        r0 : min(h, math.floor(cy + dy - 0.5) + 1),
        c0 : min(w, math.floor(cx + dx - 0.5) + 1),
    ]
    usable = np.isfinite(roi) & (roi > 0) & (roi < policy.max_depth_m)
    values = roi[usable]
    count, fraction = len(values), len(values) / roi.size if roi.size else 0
    if count < policy.min_pixels or fraction < policy.min_valid_fraction:
        return DepthEstimate(
            False, "insufficient_valid_depth", valid_pixels=count, valid_fraction=fraction
        )
    # Road, kerb and paint inside the ROI are not obstacle surface.
    vv, uu = np.nonzero(usable)
    raised = values[_heights(uu + c0, vv + r0, values, calibration) >= GROUND_CLEARANCE_M]
    if len(raised) >= policy.min_pixels:
        values = raised
    source = "aligned_depth"
    median = float(np.median(values))
    q10, q90 = np.quantile(values, [0.1, 0.9])
    spread = float(q90 - q10)
    if spread > max(policy.max_spread_m, policy.max_relative_spread * median):
        # Never a background median (that would manufacture clearance). A thin
        # pedestrian's box is mostly background; its NEAREST substantial surface
        # is the object (or a nearer occluder): conservative either way.
        cluster = _foreground_cluster(values, policy)
        if cluster is None:
            return DepthEstimate(
                False,
                "mixed_or_spread_depth",
                valid_pixels=count,
                valid_fraction=fraction,
                spread_m=spread,
            )
        source = "foreground_cluster"
        median = float(np.median(cluster))
        q10, q90 = np.quantile(cluster, [0.1, 0.9])
        spread = float(q90 - q10)
    point = tuple(float(x) for x in pixel_depth_to_ego(cx, cy, median, calibration))
    return DepthEstimate(True, None, median, point, count, fraction, spread, source)


def _heights(us, vs, depths, calibration):
    """Ego-frame z (height above the ego ground plane) of ROI pixels."""
    optical = np.column_stack(
        (
            (us - calibration.cx_px) * depths / calibration.fx_px,
            (vs - calibration.cy_px) * depths / calibration.fy_px,
            depths,
        )
    )
    matrix = np.asarray(calibration.camera_to_ego, dtype=float)
    return optical @ matrix[2, :3] + matrix[2, 3]


def _foreground_cluster(values, policy, minimum_share=0.3):
    """Pixels within one spread tolerance behind the 10th-percentile surface.

    Accepted only if they form a substantial share of the ROI; otherwise the box
    stays unknown. A small near patch over a far background is not a surface.
    """
    near = float(np.quantile(values, 0.1))
    width = max(policy.max_spread_m, policy.max_relative_spread * near)
    cluster = values[(values >= near - 0.25 * width) & (values <= near + width)]
    if len(cluster) < policy.min_pixels or len(cluster) < minimum_share * len(values):
        return None
    return cluster


def ground_fallback(box, calibration):
    """Explicit assumed-ground result. Never permits clearance by itself."""
    box = canonical_box(box)
    point = pixel_to_ground((box.x_min + box.x_max) / 2, box.y_max, calibration)
    return DepthEstimate(
        False,
        "assumed_ground_not_validated",
        point_ego_m=None if point is None else tuple(point),
        source="ground_fallback",
    )


def pinhole_fallback(box, real_width_m, calibration):
    from traffic_perception.distance import estimate_distance_m

    box = canonical_box(box)
    depth = estimate_distance_m(box.x_max - box.x_min, real_width_m, calibration.fx_px)
    return DepthEstimate(
        False, "assumed_size_not_validated", axial_depth_m=depth, source="pinhole_fallback"
    )


def estimate_batch_depth(frame, detections, policy=None):
    """Associate only detections from the same aligned source observation."""
    metadata = detections.metadata
    if (
        metadata.episode_id != frame.episode_id
        or metadata.frame_id != frame.frame_id
        or abs(metadata.sim_time_s - frame.sim_time_s) > 1e-6
    ):
        raise ValueError("Detection/depth source observation mismatch")
    if not detections.valid or frame.depth_m is None:
        raise ValueError("Valid detections and aligned depth required")
    if frame.rgb.shape[:2] != (detections.image_height, detections.image_width):
        raise ValueError("Detection/depth image dimensions mismatch")
    return tuple(
        estimate_box_depth(frame.depth_m, detection.box, frame.calibration, policy)
        for detection in detections.detections
    )
