"""Painting LiDAR points with camera classes (plan v2.1, section 9). G2 framework.

Each LiDAR point is projected into every camera that sees it. Points are painted in
depth order: within a pixel bucket only the nearest points take the pixel's class, so a
point hidden behind a nearer one (a kerb behind a parked car) is not painted with what
the camera sees in front of it. Class evidence is probabilistic: a point carries the
segmentation's class probabilities at its pixel, averaged over the cameras that saw it,
and ``painted`` says whether any did.

Frames: LiDAR points in the vehicle frame; ``CameraCalibration.camera_to_ego`` maps the
optical frame (+X right, +Y down, +Z forward) into the vehicle frame.

Time: a sweep is painted only with the segmentation of the images taken with it. The
stack hands ``paint`` a ``SegmentedFrame`` carrying the source frame id and the sweep's
own frame id; a mismatch is refused (review of the G3 framework: the stack segmented
every second tick and painted every tick, projecting the current LiDAR into the previous
image's segmentation, so moving edges smeared by a tick of travel).
"""

from dataclasses import dataclass

import numpy as np

from mapless.road_segmentation import ROAD_CLASSES, Segmentation


@dataclass(frozen=True)
class SegmentedFrame:
    """The segmentations of one sensor frame's images, tagged with where they came from."""

    frame_id: int
    time_s: float
    segmentations: tuple

    def __post_init__(self):
        object.__setattr__(self, "segmentations", tuple(self.segmentations))
        if any(not isinstance(s, Segmentation) for s in self.segmentations):
            raise TypeError("segmentations must be Segmentation objects")

    def __iter__(self):
        return iter(self.segmentations)

    def __len__(self):
        return len(self.segmentations)


@dataclass(frozen=True)
class PaintingConfig:
    bucket_px: int = 4  # pixel bucket for the depth test
    depth_tolerance_m: float = 1.0  # a point this much deeper than the bucket's nearest is hidden ...
    depth_tolerance_rel: float = 0.08  # ... or this fraction deeper
    min_depth_m: float = 0.5
    max_depth_m: float = 80.0

    def __post_init__(self):
        if self.bucket_px < 1 or self.min_depth_m <= 0 or self.max_depth_m <= self.min_depth_m:
            raise ValueError("PaintingConfig: bucket_px >= 1 and 0 < min_depth_m < max_depth_m")


@dataclass(frozen=True)
class PaintedPoints:
    probabilities: np.ndarray  # N x C over ROAD_CLASSES (uniform where unpainted)
    painted: np.ndarray  # N bool
    cameras: np.ndarray  # N int: how many cameras painted the point

    @property
    def labels(self):
        return self.probabilities.argmax(axis=1)


def project(points_ego, calibration, width, height, config):
    """(u, v, depth, visible) of vehicle-frame points in one camera."""
    m = np.asarray(calibration.camera_to_ego, dtype=float)
    ego_to_camera = np.linalg.inv(m)
    p = np.asarray(points_ego, dtype=float)[:, :3]
    cam = p @ ego_to_camera[:3, :3].T + ego_to_camera[:3, 3]
    depth = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = calibration.fx_px * cam[:, 0] / depth + calibration.cx_px
        v = calibration.fy_px * cam[:, 1] / depth + calibration.cy_px
    visible = (depth >= config.min_depth_m) & (depth <= config.max_depth_m) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u, v, depth, visible


def nearest_in_bucket(u, v, depth, visible, width, height, config):
    """Which visible points are the nearest (within tolerance) in their pixel bucket."""
    b = config.bucket_px
    cols, rows = (width + b - 1) // b, (height + b - 1) // b
    keep = np.zeros(len(u), dtype=bool)
    idx = np.flatnonzero(visible)
    if idx.size == 0:
        return keep
    bucket = (v[idx].astype(int) // b) * cols + (u[idx].astype(int) // b)
    nearest = np.full(cols * rows, np.inf)
    np.minimum.at(nearest, bucket, depth[idx])
    limit = nearest[bucket] + np.maximum(config.depth_tolerance_m, config.depth_tolerance_rel * nearest[bucket])
    keep[idx] = depth[idx] <= limit
    return keep


def paint(points_ego, cameras, segmentations, config=None, frame_id=None):
    """Paint vehicle-frame LiDAR points from ``cameras`` (CameraImages) and their ``segmentations``.

    ``frame_id`` is the sweep's frame; when ``segmentations`` is a ``SegmentedFrame`` from
    another frame the call is refused, never painted with stale pixels.
    """
    c = config or PaintingConfig()
    if isinstance(segmentations, SegmentedFrame) and frame_id is not None and segmentations.frame_id != frame_id:
        raise ValueError(f"segmentation of frame {segmentations.frame_id} cannot paint the sweep of frame {frame_id}")
    n = len(points_ego)
    probs = np.zeros((n, len(ROAD_CLASSES)), dtype=np.float64)
    count = np.zeros(n, dtype=int)
    by_name = {s.camera: s for s in segmentations}
    for image in cameras:
        seg = by_name.get(image.name)
        if seg is None:
            continue
        if not isinstance(seg, Segmentation):
            raise TypeError("segmentations must be Segmentation objects")
        h, w = image.rgb.shape[:2]
        u, v, depth, visible = project(points_ego, image.calibration, w, h, c)
        keep = nearest_in_bucket(u, v, depth, visible, w, h, c)
        idx = np.flatnonzero(keep)
        if idx.size == 0:
            continue
        sh, sw = seg.probabilities.shape[:2]
        if (sh, sw) == (h, w):
            pix = seg.probabilities[v[idx].astype(int), u[idx].astype(int)]
        else:
            # Probabilities at a lower resolution than the image: the cell covering the pixel.
            pix = seg.probabilities[np.minimum((v[idx] * sh / h).astype(int), sh - 1), np.minimum((u[idx] * sw / w).astype(int), sw - 1)]
        probs[idx] += pix
        count[idx] += 1
    painted = count > 0
    out = np.full((n, len(ROAD_CLASSES)), 1.0 / len(ROAD_CLASSES))
    out[painted] = probs[painted] / count[painted, None]
    return PaintedPoints(out.astype(np.float32), painted, count)
