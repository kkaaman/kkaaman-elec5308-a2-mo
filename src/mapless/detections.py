"""Camera detections joined to LiDAR clusters (plan v2.1, section 9). G2 framework.

- **Clusters.** Obstacle-class LiDAR points (``mapless.terrain`` OBSTACLE, i.e. above
  kerb height) are clustered with a range-tuned link distance: the LiDAR's rings spread
  with range, so the link grows with distance and a small pedestrian cluster at 30 m
  is not dropped. Each cluster is a ``LidarMeasurement`` (centre, extent, point count)
  for the tracker, class unknown: this is the unclassified-obstacle channel, which
  works without any detector.
- **Boxes.** A ``Detector`` returns image boxes with a class and confidence per camera
  (``YoloDetector`` wraps the A1 ``perception.object_detection.ObjectDetector``;
  ``NoDetector`` returns nothing). Each box is given its *foreground* cluster: of the
  clusters whose projected points fall in the box, the nearest one that covers enough
  of the box, never the median depth of every point in the box (a kerb behind a car
  would pull it back). The box's class becomes the cluster's class probabilities;
  boxes with no cluster are passed on as camera-only class evidence (no range).
"""

import math
from dataclasses import dataclass

import numpy as np

from mapless.contracts import CLASSES
from mapless.painting import PaintingConfig, project
from mapless.terrain import PointClass
from mapless.tracking import LidarMeasurement

DETECTOR_CLASS_TO_TRACK = {
    "person": "pedestrian",
    "bicycle": "bike",
    "motorcycle": "bike",
    "car": "car",
    "bus": "truck",
    "truck": "truck",
}


@dataclass(frozen=True)
class DetectionsConfig:
    link_m: float = 0.6  # link distance at 0 m ...
    link_per_m: float = 0.03  # ... growing this much per metre of range
    min_points: int = 3
    max_cluster_m: float = 12.0  # the reported extent is clamped to this (a wall, a row of cars)
    box_cover: float = 0.3  # a cluster must cover this share of the box's width to be its foreground
    class_confidence_floor: float = 0.1  # mass kept on the other classes

    def __post_init__(self):
        if self.link_m <= 0 or self.min_points < 1 or not 0 < self.box_cover <= 1:
            raise ValueError("DetectionsConfig: positive link, min_points >= 1, box_cover in (0, 1]")


@dataclass(frozen=True)
class ImageBox:
    camera: str
    class_name: str  # detector vocabulary ("car", "person", ...)
    confidence: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class Detector:
    """Interface: ``detect(image) -> [ImageBox]``."""

    def detect(self, image):
        raise NotImplementedError


class NoDetector(Detector):
    def detect(self, image):
        return []


class YoloDetector(Detector):
    """The A1 detector (ultralytics), one call per camera image."""

    def __init__(self, config=None):
        from perception.object_detection import ObjectDetector  # noqa: PLC0415

        self._detector = ObjectDetector(config)

    def detect(self, image):
        out = []
        for d in self._detector.detect(np.ascontiguousarray(image.rgb)):
            b = d.bounding_box
            out.append(ImageBox(image.name, d.class_name, float(d.confidence), b.x_min, b.y_min, b.x_max, b.y_max))
        return out


def class_probabilities(class_name, confidence, floor=0.1):
    name = DETECTOR_CLASS_TO_TRACK.get(class_name, "unknown")
    probs = {k: floor / (len(CLASSES) - 1) for k in CLASSES}
    probs[name] = 1.0 - floor
    total = sum(probs.values())
    return {k: v / total for k, v in probs.items()}


def cluster_obstacles(sweep, config=None):
    """Range-tuned single-linkage clustering of the sweep's OBSTACLE points (vehicle frame).

    Returns a list of index arrays into ``sweep.xyz``. Points are linked when within each
    other's link distance (``link_m + link_per_m * range``, the larger of the pair), found
    on a coarse grid of the smallest link distance so the cost stays near linear. One
    pass over all points: a car straddling a range boundary stays one cluster.
    """
    c = config or DetectionsConfig()
    idx = np.flatnonzero(sweep.cls == PointClass.OBSTACLE)
    if idx.size == 0:
        return []
    xy = sweep.xyz[idx, :2]
    r = np.hypot(xy[:, 0], xy[:, 1])
    link = c.link_m + c.link_per_m * r
    cell = float(link.min())
    keys = np.floor(xy / cell).astype(int)
    lookup = {}
    for k, key in enumerate(map(tuple, keys)):
        lookup.setdefault(key, []).append(k)
    parent = np.arange(len(idx))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # Points in a cell and its neighbours within the pair's link distance are linked; the
    # neighbourhood radius in cells follows the larger link of the cell's points.
    for key, members in lookup.items():
        reach = int(math.ceil(link[members].max() / cell))
        for di in range(-reach, reach + 1):
            for dj in range(-reach, reach + 1):
                others = lookup.get((key[0] + di, key[1] + dj))
                if not others:
                    continue
                a, b = np.array(members), np.array(others)
                d = np.hypot(xy[a][:, None, 0] - xy[b][None, :, 0], xy[a][:, None, 1] - xy[b][None, :, 1])
                allowed = np.maximum(link[a][:, None], link[b][None, :])
                for i, j in zip(*np.nonzero(d <= allowed)):
                    parent[find(a[i])] = find(b[j])
    roots = np.array([find(i) for i in range(len(idx))])
    out = []
    for root in np.unique(roots):
        members = np.flatnonzero(roots == root)
        if len(members) >= c.min_points:
            out.append(idx[members])
    return out


def cluster_measurement(sweep, members, class_probs=None, config=None):
    """A LidarMeasurement from one cluster: centre, plan-view extent along its long axis."""
    c = config or DetectionsConfig()
    p = sweep.xyz[members, :2]
    centre = p.mean(axis=0)
    extent = None
    if len(p) >= 3:
        centred = p - centre
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        along, across = centred @ vt[0], centred @ vt[1]
        length, width = max(float(np.ptp(along)), 0.3), max(float(np.ptp(across)), 0.3)
        extent = (min(length, c.max_cluster_m), min(width, c.max_cluster_m), float(np.arctan2(vt[0][1], vt[0][0])))
    sigma = 0.3 + 0.01 * float(np.hypot(*centre))
    return LidarMeasurement(centre, extent, class_probs, int(len(members)), sigma)


def join(sweep, clusters, cameras, boxes_by_camera, config=None, painting=None):
    """Give each image box its foreground cluster; return (measurements, camera_only).

    ``measurements``: a LidarMeasurement per cluster, classed where a box claimed it.
    ``camera_only``: (xy_hint, class_probabilities) for boxes no cluster matched, with
    the hint at the box's bearing 15 m out (class evidence only; the tracker never
    creates a track from it).
    """
    c = config or DetectionsConfig()
    pc = painting or PaintingConfig()
    classes = {}
    camera_only = []
    cams = {image.name: image for image in cameras}
    for name, boxes in boxes_by_camera.items():
        image = cams.get(name)
        if image is None or not boxes:
            continue
        h, w = image.rgb.shape[:2]
        projected = []
        for k, members in enumerate(clusters):
            u, v, depth, visible = project(sweep.xyz[members], image.calibration, w, h, pc)
            projected.append((u[visible], v[visible], depth[visible]))
        for box in boxes:
            best = None
            for k, (u, v, depth) in enumerate(projected):
                inside = (u >= box.x_min) & (u <= box.x_max) & (v >= box.y_min) & (v <= box.y_max)
                if not inside.any():
                    continue
                cover = float(np.ptp(u[inside])) / max(box.x_max - box.x_min, 1.0)
                if cover < c.box_cover and inside.sum() < 5:
                    continue
                near = float(np.percentile(depth[inside], 20))
                if best is None or near < best[0]:
                    best = (near, k)
            probs = class_probabilities(box.class_name, box.confidence, c.class_confidence_floor)
            if best is None:
                # No range: a hint along the box's bearing, for class evidence only.
                m = np.asarray(image.calibration.camera_to_ego, dtype=float)
                cx = (box.x_min + box.x_max) / 2
                ray_cam = np.array([(cx - image.calibration.cx_px) / image.calibration.fx_px, 0.0, 1.0])
                ray = m[:3, :3] @ ray_cam
                hint = m[:3, 3][:2] + 15.0 * ray[:2] / max(np.hypot(*ray[:2]), 1e-6)
                camera_only.append((hint, probs))
            else:
                classes.setdefault(best[1], []).append((box.confidence, probs))
    measurements = []
    for k, members in enumerate(clusters):
        probs = None
        if k in classes:
            conf, probs = max(classes[k], key=lambda item: item[0])
        measurements.append(cluster_measurement(sweep, members, probs, c))
    return measurements, camera_only
