"""Exact-frame depth association; observed surfaces are never actor centres."""

import math
from dataclasses import dataclass

import numpy as np

from geometry.transforms import ego_to_world_matrix, transform_point
from obstacle_perception.box_depth import (
    GROUND_CLEARANCE_M,
    LATERAL_TOLERANCE_M,
    estimate_batch_depth,
)
from obstacle_perception.types import ObservationMetadata


@dataclass(frozen=True)
class Footprint:
    metadata: ObservationMetadata
    class_name: str
    corners_world_m: tuple[tuple[float, float], ...]
    track_id: int | None
    uncertainty_m: float
    confidence: float
    confirmed: bool = False
    observed_this_frame: bool = True

    def __post_init__(self):
        corners = np.asarray(self.corners_world_m, dtype=float)
        if corners.shape != (4, 2) or not np.isfinite(corners).all():
            raise ValueError("Four finite world XY corners required")
        if not math.isfinite(self.uncertainty_m) or self.uncertainty_m < 0:
            raise ValueError("Uncertainty must be finite and nonnegative")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("Invalid confidence")
        if not isinstance(self.metadata, ObservationMetadata):
            raise TypeError("Observation metadata required")
        if self.track_id is not None:
            self.metadata.track_identity(self.track_id)
        object.__setattr__(
            self, "corners_world_m", tuple(tuple(float(v) for v in p) for p in corners)
        )

    @property
    def identity(self):
        return None if self.track_id is None else self.metadata.track_identity(self.track_id)


@dataclass(frozen=True)
class AssociatedFrame:
    footprints: tuple[Footprint, ...]
    invalid_depth_count: int
    reasons: tuple[str, ...]
    invalid_boxes: tuple = ()
    # Same order as invalid_boxes; keeps class/track identity for confirmation.
    invalid_detections: tuple = ()


def associate(frame, batch, policy=None):
    """Conservative camera-aligned slabs from visible surface and width.

    Occluded length is an explicit class prior, not a measured box. Unknown
    orientation and partial occlusion remain limitations. No ground truth used.
    Supported pixels below ``GROUND_CLEARANCE_M`` (road, kerb, paint) are not
    obstacle surface; obstacles lower than that are outside the baseline scope.
    """
    estimates = estimate_batch_depth(frame, batch, policy)
    state = frame.ego.state
    transform = ego_to_world_matrix(state.x_m, state.y_m, state.yaw_rad, frame.ego.z_m)
    footprints, reasons, invalid_boxes, invalid_detections = [], [], [], []
    for detection, estimate in zip(batch.detections, estimates):
        if detection.class_name not in {"car", "truck", "bus", "person", "bicycle", "motorcycle"}:
            continue
        if not estimate.valid:
            reasons.append(f"{detection.class_name}:{estimate.reason}")
            invalid_boxes.append(detection.box)
            invalid_detections.append(detection)
            continue
        box, depth = detection.box, estimate.axial_depth_m
        uncertainty = max(0.20, (estimate.spread_m or 0) / 2)
        length = {"person": 0.8, "bicycle": 2.0, "motorcycle": 2.3, "truck": 9.0, "bus": 12.0}.get(
            detection.class_name, 5.0
        )
        # Project supported pixels at their OWN depths. Assigning one median
        # depth to the extreme box edges invents in-lane surfaces on roadside
        # cars viewed obliquely (the near and far sides have different depths).
        h, w = frame.depth_m.shape
        x0, y0 = max(0, math.floor(box.x_min)), max(0, math.floor(box.y_min))
        x1, y1 = min(w, math.ceil(box.x_max)), min(h, math.ceil(box.y_max))
        roi = frame.depth_m[y0:y1, x0:x1]
        vv, uu = np.indices(roi.shape)
        support = (
            np.isfinite(roi)
            & (roi >= max(0.05, depth - max(uncertainty, estimate.spread_m or 0)))
            & (roi <= depth + length)
        )
        zs = roi[support][:: max(1, int(support.sum() / 2000))]
        us = (uu[support] + x0)[:: max(1, int(support.sum() / 2000))]
        vs = (vv[support] + y0)[:: max(1, int(support.sum() / 2000))]
        cal = frame.calibration
        optical = np.column_stack(
            ((us - cal.cx_px) * zs / cal.fx_px, (vs - cal.cy_px) * zs / cal.fy_px, zs)
        )
        matrix = np.asarray(cal.camera_to_ego)
        cloud = optical @ matrix[:3, :3].T + matrix[:3, 3]
        # Ego frame: z is height above the ego's ground contact plane.
        cloud = cloud[cloud[:, 2] >= GROUND_CLEARANCE_M]
        if len(cloud) < 3:
            reasons.append(f"{detection.class_name}:insufficient_surface_support")
            invalid_boxes.append(box)
            invalid_detections.append(detection)
            continue
        low, high = np.quantile(cloud, [0.01, 0.99], axis=0)
        low[0] -= uncertainty
        high[0] += uncertainty
        low[1] -= LATERAL_TOLERANCE_M
        high[1] += LATERAL_TOLERANCE_M
        high[0] = max(high[0], low[0] + length)
        points = [
            (low[0], low[1], 0),
            (high[0], low[1], 0),
            (high[0], high[1], 0),
            (low[0], high[1], 0),
        ]
        corners = tuple(tuple(transform_point(p, transform)[:2]) for p in points)
        footprints.append(
            Footprint(
                batch.metadata,
                detection.class_name,
                corners,
                detection.track_id,
                uncertainty,
                detection.confidence,
            )
        )
    return AssociatedFrame(
        tuple(footprints),
        len(reasons),
        tuple(reasons),
        tuple(invalid_boxes),
        tuple(invalid_detections),
    )
