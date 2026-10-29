"""Camera detections joined to LiDAR clusters (plan v2.1, section 9). G2 framework.

- **Clusters.** Obstacle-class LiDAR points (``mapless.terrain`` OBSTACLE, i.e. above
  kerb height) are clustered with a range-tuned link distance: the LiDAR's rings spread
  with range, so the link grows with distance and a small pedestrian cluster at 30 m
  is not dropped. Each cluster is a ``LidarMeasurement`` (centre, extent, point count)
  for the tracker, class unknown: this is the unclassified-obstacle channel, which
  works without any detector.
- **Objects, not scenery.** Single linkage chains: on a Town05 grade (26 Oct) both kerb
  railings and the road rising ahead and behind joined into one 63 x 26 m cluster whose
  mean sat 2.8 m ahead of the bumper, tracked as a 12 x 12 m "vehicle" the car followed
  at a negative gap. A cluster larger than any road user (``max_object_length_m`` by
  ``max_object_width_m`` in plan) is re-clustered at a tight link; the object-sized
  pieces (a car beside a railing) are measured as usual, and what is still too big, or
  a thin line of it, is *structure*: never a track, still OBSTACLE points, so the driver's LiDAR obstacle stop
  and the corridor still see it.
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
    # Plan-view size of the largest road user (a bus is 12 m; an L-shaped view of a car or
    # bus at an angle stays under 4 m across its long axis). Larger clusters are split.
    max_object_length_m: float = 12.0
    max_object_width_m: float = 4.0
    split_link_m: float = 0.3  # the tight link an oversized cluster is re-clustered with
    split_link_per_m: float = 0.01
    # A piece of a split cluster is a stretch of the structure (a railing, a wall's scan
    # line), not a road user beside it, when it is a line: narrower than split_min_width_m
    # and at least split_line_min_length_m long and split_line_aspect times its width
    # (4.9 m x 0.3 m pieces of the Town05 railings slid along at up to 11 m/s and were
    # predicted into the path), or a thin piece continuing such a line along its own
    # direction (within split_line_angle_deg, split_line_offset_m of it and
    # split_line_gap_m of its end: a railing broken by its posts). Narrow alone was the
    # rule, and a pedestrian 0.7 m from a wall, a cyclist by a fence and a car's rear face
    # beside a railing all became structure and were never tracked (review of 3e9083a, H1).
    # Known limit: a car seen side-on within the link distance of a barrier (~0.6 m) is a
    # line too; requiring a collinear second stretch kept it, but turned seven more Town05
    # railing and road-ring stretches back into objects. Its points stay OBSTACLE points.
    split_min_width_m: float = 0.5
    split_line_min_length_m: float = 2.5
    split_line_aspect: float = 3.0
    split_line_angle_deg: float = 20.0
    split_line_offset_m: float = 0.3
    split_line_gap_m: float = 2.0
    # ... and a thin piece of it spanning less than split_sliver_height_m of height is one
    # scan ring's sliver of scenery (on the Town05 grade, 36 of them at the LiDAR's own
    # height, hedge and wall tops, became tracks and conflicts when narrow stopped being
    # the rule, live on e93154f); a road user beside a structure spans several rings.
    split_sliver_height_m: float = 0.3
    split_sliver_width_m: float = 0.08  # ... or narrower than this across: points on one scan line
    # A piece wholly above split_max_floor_m (its lowest point over the road) is a wall or
    # hedge top, a branch or a sign, not a road user, who stands on the road (seven of the
    # Town05 pieces sat 1.5-1.9 m up and above).
    split_max_floor_m: float = 1.4
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


def _cell_pairs(xy, link, cell):
    """Pairs ``i < j`` within the larger of their link distances, hashing the points into
    ``cell``-sized squares (``cell`` no shorter than any link) and comparing each square with
    itself and four neighbours (the other four see it from their side). All in numpy."""
    empty = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    if len(xy) < 2:
        return empty
    cx = np.floor(xy[:, 0] / cell).astype(np.int64)
    cy = np.floor(xy[:, 1] / cell).astype(np.int64)
    cy -= cy.min() - 1
    width = int(cy.max()) + 2
    key = cx * width + cy
    order = np.argsort(key, kind="stable")
    skey = key[order]
    ukey, ustart, ucount = np.unique(skey, return_index=True, return_counts=True)
    out_i, out_j = [], []
    for dx, dy in ((0, 0), (0, 1), (1, -1), (1, 0), (1, 1)):
        target = skey + dx * width + dy
        pos = np.minimum(np.searchsorted(ukey, target), len(ukey) - 1)
        p = np.flatnonzero(ukey[pos] == target)
        if not len(p):
            continue
        cnt, start = ucount[pos[p]], ustart[pos[p]]
        rep = np.repeat(p, cnt)
        q = np.repeat(start, cnt) + (np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        if dx == 0 and dy == 0:
            same = rep < q
            rep, q = rep[same], q[same]
        a, b = order[rep], order[q]
        near = np.hypot(xy[a, 0] - xy[b, 0], xy[a, 1] - xy[b, 1]) <= np.maximum(link[a], link[b])
        out_i.append(np.minimum(a[near], b[near]))
        out_j.append(np.maximum(a[near], b[near]))
    if not out_i:
        return empty
    return np.concatenate(out_i), np.concatenate(out_j)


def _link_pairs(xy, link, bands=(5.0, 10.0, 15.0, 20.0, 30.0, 45.0)):
    """Index pairs ``i < j`` closer than the larger of their two link distances.

    By range band: each band's points, with the farther points that could reach into it,
    are hashed into squares as big as the largest link among them (``_cell_pairs``), and a
    pair is kept by the band holding its nearer point. The link grows with range, so near
    the car, where the points are densest, the squares stay small. On Town05's recorded
    sweeps (~4900 OBSTACLE points) the pairs are identical to the x-sorted block search this
    replaced, at 26 ms p50 and 35 ms p95 against 80 and 112 ms (fourth review: clustering
    was the largest stage).
    """
    empty = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    if len(xy) < 2:
        return empty
    rng = np.hypot(xy[:, 0], xy[:, 1])
    edges = (0.0, *bands, np.inf)
    out_i, out_j = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        own = (rng >= lo) & (rng < hi)
        if not own.any():
            continue
        # A farther point reaches into the band only from within its own link of the edge.
        part = np.flatnonzero(own | ((rng >= hi) & (rng - link < hi)))
        i, j = _cell_pairs(xy[part], link[part], float(link[part].max()))
        keep = own[part[i]] | own[part[j]]
        out_i.append(part[i[keep]])
        out_j.append(part[j[keep]])
    if not out_i:
        return empty
    return np.concatenate(out_i), np.concatenate(out_j)


_SURE_OFFSETS = ((0, 1), (1, -1), (1, 0), (1, 1))  # the half of the 8-neighbourhood each cell looks at


def _band_edges(xy, link):
    """Edges (i, j) among these points with the components of the link graph (``i`` and
    ``j`` joined when within the larger of their link distances), each one a link of it.

    The points are hashed into squares of side ``g`` = the shortest link / (2 sqrt 2):
    any two points in one square, or in two squares side by side or corner to corner,
    are within the shortest link of each other, so those are joined without measuring
    (a star per square, one edge per neighbouring pair of squares). Only squares farther
    apart, still within the longest link, and not joined that way already, have their
    points' distances measured. Dense walls and car sides are then joined square by
    square instead of pair by pair: 130 000 linked pairs a sweep within 10 m of Town05's
    walls (fifth review, F5-05)."""
    n = len(xy)
    empty = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    if n < 2:
        return empty
    g = float(link.min()) / (2.0 * math.sqrt(2.0)) * (1.0 - 1e-9)
    reach = float(link.max())
    m = int(math.ceil(reach / g)) + 1
    cx = np.floor(xy[:, 0] / g).astype(np.int64)
    cy = np.floor(xy[:, 1] / g).astype(np.int64)
    cy -= cy.min() - (m + 1)
    width = int(cy.max()) + m + 2
    key = cx * width + cy
    order = np.argsort(key, kind="stable")
    ukey, ustart, ucount = np.unique(key[order], return_index=True, return_counts=True)
    ncell = len(ukey)
    first = order[ustart]
    cell_of = np.empty(n, dtype=np.int64)
    cell_of[order] = np.repeat(np.arange(ncell), ucount)
    cells = np.arange(ncell)

    def neighbours(dx, dy):
        target = ukey + dx * width + dy
        pos = np.minimum(np.searchsorted(ukey, target), ncell - 1)
        hit = ukey[pos] == target
        return cells[hit], pos[hit]

    out_i, out_j = [np.arange(n)], [first[cell_of]]
    sure_a, sure_b = [], []
    for dx, dy in _SURE_OFFSETS:
        a, b = neighbours(dx, dy)
        sure_a.append(a)
        sure_b.append(b)
    sure_a, sure_b = np.concatenate(sure_a), np.concatenate(sure_b)
    out_i.append(first[sure_a])
    out_j.append(first[sure_b])
    comp = _components(ncell, sure_a, sure_b)
    # Every farther offset within the longest link, looked up at once (one search, not one
    # per offset), offset by offset in the same order as a loop would.
    offsets = np.array(
        [
            dx * width + dy
            for dx in range(0, m + 1)
            for dy in range(-m, m + 1)
            if not ((dx == 0 and dy <= 0) or (abs(dx) <= 1 and abs(dy) <= 1)) and g * math.hypot(max(abs(dx) - 1, 0), max(abs(dy) - 1, 0)) <= reach
        ],
        dtype=np.int64,
    )
    if len(offsets):
        target = (offsets[:, None] + ukey[None, :]).ravel()
        pos = np.minimum(np.searchsorted(ukey, target), ncell - 1)
        hit = ukey[pos] == target
        a, b = np.tile(cells, len(offsets))[hit], pos[hit]
        apart = comp[a] != comp[b]
        a, b = a[apart], b[apart]
        if len(a):
            na, nb = ucount[a], ucount[b]
            size = na * nb
            k = np.repeat(np.arange(len(a)), size)
            local = np.arange(int(size.sum())) - np.repeat(np.cumsum(size) - size, size)
            pa = order[ustart[a][k] + local // nb[k]]
            pb = order[ustart[b][k] + local % nb[k]]
            near = np.hypot(xy[pa, 0] - xy[pb, 0], xy[pa, 1] - xy[pb, 1]) <= np.maximum(link[pa], link[pb])
            out_i.append(pa[near])
            out_j.append(pb[near])
    return np.concatenate(out_i), np.concatenate(out_j)


def _link_components(xy, link, bands=(5.0, 10.0, 15.0, 20.0, 30.0, 45.0)):
    """Connected-component labels of the link graph (points joined when within the larger
    of their link distances): those of ``_link_pairs``' pairs, from ``_band_edges`` by the
    same range bands, without listing every linked pair."""
    n = len(xy)
    if n < 2:
        return np.arange(n)
    rng = np.hypot(xy[:, 0], xy[:, 1])
    edges = (0.0, *bands, np.inf)
    out_i, out_j = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        own = (rng >= lo) & (rng < hi)
        if not own.any():
            continue
        part = np.flatnonzero(own | ((rng >= hi) & (rng - link < hi)))
        i, j = _band_edges(xy[part], link[part])
        out_i.append(part[i])
        out_j.append(part[j])
    if not out_i:
        return np.arange(n)
    return _components(n, np.concatenate(out_i), np.concatenate(out_j))


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
    labels = _link_components(xy, link)
    order = np.argsort(labels, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(labels[order])) + 1)
    return [idx[np.sort(g)] for g in groups if len(g) >= c.min_points]


def plan_extent(xy):
    """(length, width) of points in plan, along and across their principal axis."""
    if len(xy) < 2:
        return 0.0, 0.0
    centred = xy - xy.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return float(np.ptp(centred @ vt[0])), float(np.ptp(centred @ vt[1]))


def oversized(xy, config):
    length, width = plan_extent(xy)
    return length > config.max_object_length_m or width > config.max_object_width_m


def _piece_shape(xy):
    """(length, width, unit direction, centre) of points in plan, along their principal axis."""
    centre = xy.mean(axis=0)
    if len(xy) < 2:
        return 0.0, 0.0, np.array([1.0, 0.0]), centre
    _, _, vt = np.linalg.svd(xy - centre, full_matrices=False)
    return float(np.ptp((xy - centre) @ vt[0])), float(np.ptp((xy - centre) @ vt[1])), vt[0], centre


def _structure_pieces(xy, pieces, config, height=None):
    """Which pieces of an oversized cluster (index arrays into ``xy``) are structure: still
    oversized, a line, a thin piece continuing a line, or a thin one-ring sliver (by
    ``height``, the points' heights, when given; ``DetectionsConfig``)."""
    c = config
    shapes = [_piece_shape(xy[piece]) for piece in pieces]

    def thin(shape):
        length, width = shape[0], shape[1]
        return width < c.split_min_width_m and length >= c.split_line_aspect * max(width, 0.05)

    lines = [k for k, sh in enumerate(shapes) if thin(sh) and sh[0] >= c.split_line_min_length_m]
    cos_max = np.cos(np.radians(c.split_line_angle_deg))

    def continues(k):
        length, _, direction, centre = shapes[k]
        for m in lines:
            l_len, _, l_dir, l_centre = shapes[m]
            if abs(float(direction @ l_dir)) < cos_max:
                continue
            rel = centre - l_centre
            normal = np.array([-l_dir[1], l_dir[0]])
            if abs(float(rel @ normal)) > c.split_line_offset_m:
                continue
            if abs(float(rel @ l_dir)) - 0.5 * (l_len + length) <= c.split_line_gap_m:
                return True
        return False

    out = []
    for k, piece in enumerate(pieces):
        sliver = shapes[k][1] < c.split_sliver_width_m or (height is not None and shapes[k][1] < c.split_min_width_m and float(np.ptp(height[piece])) < c.split_sliver_height_m)
        sliver = sliver or (height is not None and float(height[piece].min()) > c.split_max_floor_m)
        # Still oversized, from the shape already measured (the same arithmetic as ``oversized``).
        too_big = shapes[k][0] > c.max_object_length_m or shapes[k][1] > c.max_object_width_m
        structure = too_big or k in lines or sliver or (thin(shapes[k]) and continues(k))
        out.append(structure)
    return out


def split_oversized(sweep, clusters, config=None):
    """``(objects, structures)``: clusters no larger than a road user, and the rest.

    An oversized cluster is re-clustered at the tight link; its object-sized pieces join
    ``objects`` and what is still oversized, or a line (``_structure_pieces``), goes to
    ``structures`` (index arrays both). Pieces under ``min_points`` are dropped, as in
    ``cluster_obstacles``.
    """
    c = config or DetectionsConfig()
    objects, structures = [], []
    for members in clusters:
        xy = sweep.xyz[members, :2].astype(float)
        if not oversized(xy, c):
            objects.append(members)
            continue
        link = c.split_link_m + c.split_link_per_m * np.hypot(xy[:, 0], xy[:, 1])
        labels = _link_components(xy, link)
        order = np.argsort(labels, kind="stable")
        pieces = [np.sort(g) for g in np.split(order, np.flatnonzero(np.diff(labels[order])) + 1) if len(g) >= c.min_points]
        for piece, structure in zip(pieces, _structure_pieces(xy, pieces, c, sweep.height_m[members])):
            (structures if structure else objects).append(members[piece])
    return objects, structures


def cluster_and_split(sweep, config=None):
    """``split_oversized(sweep, cluster_obstacles(sweep))``, with the tight link's pieces
    found once for the points of every oversized cluster together: a tight link implies a
    range-tuned one, so no piece spans two clusters (``_link_components`` for both; it no
    longer lists the linked pairs a second pass could filter)."""
    c = config or DetectionsConfig()
    idx = np.flatnonzero(sweep.cls == PointClass.OBSTACLE)
    if idx.size == 0:
        return [], []
    xy = sweep.xyz[idx, :2].astype(float)
    rng = np.hypot(xy[:, 0], xy[:, 1])
    labels = _link_components(xy, c.link_m + c.link_per_m * rng)
    order = np.argsort(labels, kind="stable")
    groups = [np.sort(g) for g in np.split(order, np.flatnonzero(np.diff(labels[order])) + 1) if len(g) >= c.min_points]
    big = [g for g in groups if oversized(xy[g], c)]
    tight = None
    if big:
        # Tight components over the oversized clusters' points, labelled by the lowest
        # sweep index in each (as ``_components`` labels the whole sweep).
        pts = np.concatenate(big)
        local = _link_components(xy[pts], c.split_link_m + c.split_link_per_m * rng[pts])
        tight = np.full(len(idx), -1, dtype=np.int64)
        tight[pts] = pts[local]
    big_ids = {id(g) for g in big}
    objects, structures = [], []
    for g in groups:
        if id(g) not in big_ids:
            objects.append(idx[g])
            continue
        sub = tight[g]
        o = np.argsort(sub, kind="stable")
        pieces = [g[np.sort(piece)] for piece in np.split(o, np.flatnonzero(np.diff(sub[o])) + 1) if len(piece) >= c.min_points]
        for members, structure in zip(pieces, _structure_pieces(xy, pieces, c, sweep.height_m[idx])):
            (structures if structure else objects).append(idx[members])
    return objects, structures


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
    return LidarMeasurement(centre, extent, class_probs, int(len(members)), sigma, plan_xy=p)


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
