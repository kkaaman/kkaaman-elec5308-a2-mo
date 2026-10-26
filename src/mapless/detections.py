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


def _components(n, i, j):
    """Connected-component labels of ``n`` nodes joined by edges ``(i, j)``.

    Hook and compress: each edge hooks the larger of its two roots onto the smaller,
    then every node jumps to its root; repeated until no edge joins two roots. All in
    numpy (the per-pair union-find in Python took 200 ms a sweep live, 26 Oct).
    """
    labels = np.arange(n)
    if len(i) == 0:
        return labels
    while True:
        li, lj = labels[i], labels[j]
        m = np.minimum(li, lj)
        new = labels.copy()
        np.minimum.at(new, li, m)
        np.minimum.at(new, lj, m)
        while True:
            nxt = new[new]
            if np.array_equal(nxt, new):
                break
            new = nxt
        if np.array_equal(new, labels):
            return labels
        labels = new


def _link_pairs(xy, link, block=256):
    """Index pairs ``i < j`` closer than the larger of their two link distances.

    Points are sorted by x and compared a block at a time with every point within the
    largest link distance in x, so the cost follows the points' spread, not n squared.
    """
    n = len(xy)
    order = np.argsort(xy[:, 0], kind="stable")
    xs = xy[order, 0]
    reach = float(link.max())
    out_i, out_j = [], []
    for start in range(0, n, block):
        stop = min(start + block, n)
        a = order[start:stop]
        lo = int(np.searchsorted(xs, xs[start] - reach, "left"))
        hi = int(np.searchsorted(xs, xs[stop - 1] + reach, "right"))
        b = order[lo:hi]
        d = np.hypot(xy[a, 0][:, None] - xy[b, 0][None, :], xy[a, 1][:, None] - xy[b, 1][None, :])
        ii, jj = np.nonzero(d <= np.maximum(link[a][:, None], link[b][None, :]))
        ai, bj = a[ii], b[jj]
        keep = ai < bj
        out_i.append(ai[keep])
        out_j.append(bj[keep])
    return np.concatenate(out_i), np.concatenate(out_j)


def cluster_obstacles(sweep, config=None):
    """Range-tuned single-linkage clustering of the sweep's OBSTACLE points (vehicle frame).

    Returns a list of index arrays into ``sweep.xyz``. Points are linked when within each
    other's link distance (``link_m + link_per_m * range``, the larger of the pair). One
    pass over all points: a car straddling a range boundary stays one cluster.
    """
    c = config or DetectionsConfig()
    idx = np.flatnonzero(sweep.cls == PointClass.OBSTACLE)
    if idx.size == 0:
        return []
    xy = sweep.xyz[idx, :2].astype(float)
    link = c.link_m + c.link_per_m * np.hypot(xy[:, 0], xy[:, 1])
    labels = _components(len(idx), *_link_pairs(xy, link))
    order = np.argsort(labels, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(labels[order])) + 1)
    return [idx[np.sort(g)] for g in groups if len(g) >= c.min_points]


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
