"""Robust, explicitly invalid depth measurements for canonical detection boxes."""

import math
from dataclasses import dataclass

import numpy as np

from geometry.camera import pixel_depth_to_ego, pixel_to_ground
from obstacle_perception.boxes import canonical_box


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
    roi = depth_m[
        math.ceil(cy - dy) : math.floor(cy + dy), math.ceil(cx - dx) : math.floor(cx + dx)
    ]
    values = roi[np.isfinite(roi) & (roi > 0) & (roi < policy.max_depth_m)]
    count, fraction = len(values), len(values) / roi.size if roi.size else 0
    if count < policy.min_pixels or fraction < policy.min_valid_fraction:
        return DepthEstimate(
            False, "insufficient_valid_depth", valid_pixels=count, valid_fraction=fraction
        )
    median = float(np.median(values))
    q10, q90 = np.quantile(values, [0.1, 0.9])
    spread = float(q90 - q10)
    # Conservative: reject substantial multiple surfaces rather than selecting
    # a background median and manufacturing clearance.
    if spread > max(policy.max_spread_m, policy.max_relative_spread * median):
        return DepthEstimate(
            False,
            "mixed_or_spread_depth",
            valid_pixels=count,
            valid_fraction=fraction,
            spread_m=spread,
        )
    point = tuple(float(x) for x in pixel_depth_to_ego(cx, cy, median, calibration))
    return DepthEstimate(True, None, median, point, count, fraction, spread)


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
