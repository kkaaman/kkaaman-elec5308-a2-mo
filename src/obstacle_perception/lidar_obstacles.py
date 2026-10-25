"""LiDAR obstacles: ground removal, corridor crop, bounded clustering, tracking (E2-03/E2-04).

Every function here consumes a validated :class:`~obstacle_perception.lidar.LidarSweep`
in rear-axle ego coordinates (X forward, Y left, Z up, metres) and the shared
:class:`~planning.corridor.RouteGeometry`. The output is a source-namespaced set of
clusters and tracks whose corridor gap is computed by the same swept-corridor code
the RGB-D path uses, so a governor receives one kind of number from both sensors.

Validity rules (all conservative):

* A sweep with too few returns, too small a density fraction, or no accepted road
  band is **invalid**; the batch carries a reason and no clusters. Invalid never
  means clear.
* Only points classified as obstacle height (``obstacle_min_m`` to ``overhead_m``
  above the estimated road) form clusters. Kerb-height returns are reported as
  raised counts, not obstacles, unless they stand inside the corridor.
* A cluster is in the corridor when at least one of its points lies within the
  swept corridor; its gap is the front-bumper gap of those corridor points.
* ``usable_for_clearance`` is always ``False``. Not seeing an obstacle is not the
  same as seeing the road clear.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from statistics import median

import cv2
import numpy as np

from obstacle_perception.lidar import LidarSweep
from obstacle_perception.tracking import TrackPolicy
from obstacle_perception.types import ObservationMetadata, Obstacle, TrackIdentity
from planning.corridor import RouteGeometry

LIDAR_SOURCE_ID = "lidar"
LIDAR_ALGORITHM_VERSION = "lidar-obstacles-v1"

CLASS_IGNORED = -1
CLASS_ROAD = 0
CLASS_RAISED = 1
CLASS_OBSTACLE = 2


def _positive(name, value):
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class EgoBody:
    """Ego box in the rear-axle frame used to drop returns from the vehicle itself.

    ``front_m`` is the distance from the rear axle to the front bumper (positive X),
    ``rear_m`` the rear overhang (negative X), ``half_width_m`` half the body width,
    ``height_m`` the roof height above the ground, all in metres. ``margin_m`` is
    added around the box.
    """

    front_m: float = 3.8
    rear_m: float = -1.0
    half_width_m: float = 1.05
    height_m: float = 1.6
    margin_m: float = 0.3

    def __post_init__(self):
        for name in ("front_m", "half_width_m", "height_m"):
            _positive(name, getattr(self, name))
        if not math.isfinite(self.rear_m) or self.rear_m >= 0:
            raise ValueError("rear_m must be negative (behind the rear axle)")
        if not math.isfinite(self.margin_m) or self.margin_m < 0:
            raise ValueError("margin_m must be nonnegative")


@dataclass(frozen=True)
class LidarObstaclePolicy:
    """Thresholds for sweep validity, ground removal, clustering and the corridor crop.

    Heights are metres above the estimated road surface; the ego frame's Z = 0 is the
    nominal ground under the rear axle. ``min_points`` and ``min_density_fraction``
    bound sweep validity; ``max_range_m`` is the planar range from the rear axle
    used for all processing; ``band_length_m``/``road_window_m``/``path_window_m``
    shape the road height profile along X (see :func:`road_profile`); ``max_grade``
    is metres of rise per metre; ``cell_m`` is the clustering grid cell and
    ``cluster_merge_m`` the footprint gap below which clusters merge; ``cell_m``
    is the clustering grid cell; ``corridor_half_width_m`` and ``corridor_ahead_m``
    define the default swept corridor used when the caller supplies none.
    ``clear_bin_m``/``clear_min_points``/``clear_gap_m``/``clear_start_m`` govern
    the positive seen-clear distance: road returns must appear in consecutive
    bins along the corridor, starting within ``clear_start_m`` of the bumper, with
    unseen gaps no longer than ``clear_gap_m + clear_gap_per_m * distance`` (one
    sweep sees the road in rings that spread with range), and in the centre strip
    plus at least ``clear_min_strips`` of ``clear_strips`` lateral strips.
    """

    min_points: int = 500
    min_density_fraction: float = 0.1
    max_range_m: float = 40.0
    band_length_m: float = 2.0
    road_window_m: float = 3.0
    path_window_m: float = 1.5
    surface_depth_m: float = 0.05
    band_tolerance_m: float = 0.07
    band_tolerance_per_m: float = 0.015
    max_grade: float = 0.12
    max_band_gap_m: float = 4.0
    ground_min_m: float = -0.4
    ground_max_m: float = 0.10
    obstacle_min_m: float = 0.30
    overhead_m: float = 2.4
    min_band_points: int = 3
    cell_m: float = 0.5
    cluster_merge_m: float = 1.5
    min_cluster_points: int = 6
    max_clusters: int = 32
    corridor_half_width_m: float = 1.5
    corridor_ahead_m: float = 40.0
    raised_corridor_min_points: int = 12
    clear_bin_m: float = 0.5
    clear_min_points: int = 2
    clear_gap_m: float = 1.0
    clear_gap_per_m: float = 0.3
    clear_start_m: float = 8.0
    clear_strips: int = 3
    clear_min_strips: int = 2

    def __post_init__(self):
        for name in (
            "min_points",
            "min_band_points",
            "min_cluster_points",
            "max_clusters",
            "raised_corridor_min_points",
            "clear_min_points",
            "clear_strips",
            "clear_min_strips",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "max_range_m",
            "band_length_m",
            "road_window_m",
            "path_window_m",
            "band_tolerance_m",
            "max_grade",
            "max_band_gap_m",
            "surface_depth_m",
            "cluster_merge_m",
            "cell_m",
            "corridor_half_width_m",
            "corridor_ahead_m",
            "clear_bin_m",
            "clear_gap_m",
            "clear_start_m",
        ):
            _positive(name, getattr(self, name))
        for name in ("clear_gap_per_m", "band_tolerance_per_m"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.path_window_m > self.road_window_m:
            raise ValueError("path_window_m cannot exceed road_window_m")
        if self.clear_min_strips > self.clear_strips:
            raise ValueError("clear_min_strips exceeds clear_strips")
        if not 0 < self.min_density_fraction <= 1:
            raise ValueError("min_density_fraction must lie in (0, 1]")
        if not self.ground_min_m < 0 <= self.ground_max_m < self.obstacle_min_m < self.overhead_m:
            raise ValueError(
                "Height bands must satisfy ground_min < 0 <= ground_max < obstacle_min < overhead"
            )


@dataclass(frozen=True)
class GroundSegmentation:
    """Per-point ground classification of one sweep.

    ``points_xyz`` are the retained ego-frame points (metres) after body and range
    removal; ``height_m`` is each point's height above the estimated road;
    ``classes`` holds ``CLASS_*`` labels; ``band_x_m``/``band_z_m`` are the road
    height profile along X (NaN where a band was rejected).
    """

    points_xyz: np.ndarray
    height_m: np.ndarray
    classes: np.ndarray
    band_x_m: np.ndarray
    band_z_m: np.ndarray
    dropped_body: int
    dropped_range: int

    def of(self, label):
        return self.points_xyz[self.classes == label]

    @property
    def accepted_bands(self) -> int:
        return int(np.isfinite(self.band_z_m).sum())


def road_profile(xyz, policy):
    """Road height (metres) in ``band_length_m`` bands along X, grown outward from Z = 0.

    The band containing the rear axle is anchored at the ego frame ground. Each next
    band is estimated from its lowest returns (the median of points within
    ``surface_depth_m`` of the ``min_band_points``-th lowest) under the ego's own
    path (``|y| < path_window_m``) when enough are seen there, otherwise from the
    wider ``road_window_m``. The estimate is accepted only if it continues the grade of
    the previous accepted bands within ``band_tolerance_m + band_tolerance_per_m *
    distance`` (the first band may additionally differ by ``max_grade * distance``
    because the grade is unknown), and only within ``max_band_gap_m`` of the last
    accepted band. A vehicle filling a band, or pavement where the road is in its
    shadow, therefore cannot become the road.
    """
    c = policy
    n = int(math.ceil(c.max_range_m / c.band_length_m))
    centres = np.arange(-n, n + 1) * c.band_length_m
    heights = np.full(centres.shape, np.nan)
    heights[n] = 0.0
    index = np.floor((xyz[:, 0] + c.band_length_m / 2) / c.band_length_m).astype(int) + n
    path = np.abs(xyz[:, 1]) < c.path_window_m
    wide = np.abs(xyz[:, 1]) < c.road_window_m
    z = xyz[:, 2]
    for direction in (1, -1):
        accepted = [(0.0, 0.0)]
        k = n + direction
        while 0 <= k < centres.size:
            x1, z1 = accepted[-1]
            if abs(centres[k] - x1) > c.max_band_gap_m:
                break
            in_band = index == k
            band = z[in_band & path]
            if band.size < c.min_band_points:
                band = z[in_band & wide]
            if band.size >= c.min_band_points:
                band = np.sort(band)
                low = band[min(c.min_band_points - 1, band.size - 1)]
                estimate = float(np.median(band[band <= low + c.surface_depth_m]))
                distance = abs(centres[k] - x1)
                if len(accepted) >= 2:
                    x0, z0 = accepted[-2]
                    slope = float(np.clip((z1 - z0) / (x1 - x0), -c.max_grade, c.max_grade))
                    tolerance = c.band_tolerance_m + c.band_tolerance_per_m * distance
                else:
                    slope = 0.0
                    tolerance = (
                        c.band_tolerance_m + (c.band_tolerance_per_m + c.max_grade) * distance
                    )
                if abs(estimate - (z1 + slope * (centres[k] - x1))) <= tolerance:
                    heights[k] = estimate
                    accepted.append((float(centres[k]), estimate))
            k += direction
    return centres, heights


def interpolate_profile(x_m, band_x_m, band_z_m, max_grade):
    """Road height at ``x_m``: linear between accepted bands, last accepted grade beyond them.

    Extrapolation uses the grade of the two outermost accepted bands on that side,
    clipped to ``max_grade`` (metres per metre), so a sloped road keeps reading as
    road past the last ring of returns.
    """
    ok = np.isfinite(band_z_m)
    bx, bz = band_x_m[ok], band_z_m[ok]
    road = np.interp(x_m, bx, bz)
    for side, order in ((x_m > bx[-1], slice(-2, None)), (x_m < bx[0], slice(0, 2))):
        if side.any() and len(bx) >= 2:
            x_pair, z_pair = bx[order], bz[order]
            slope = float(
                np.clip((z_pair[1] - z_pair[0]) / (x_pair[1] - x_pair[0]), -max_grade, max_grade)
            )
            anchor_x, anchor_z = (
                (x_pair[1], z_pair[1]) if order.start == -2 else (x_pair[0], z_pair[0])
            )
            road = np.where(side, anchor_z + slope * (x_m - anchor_x), road)
    return road


def remove_ground(sweep, policy=None, body=None):
    """Classify a sweep's returns as road, raised (kerb), obstacle or ignored.

    Returns a :class:`GroundSegmentation`. Points inside the ego body box plus
    margin and beyond ``max_range_m`` planar range from the rear axle are dropped
    first. Heights are metres above the interpolated road profile.
    """
    if not isinstance(sweep, LidarSweep):
        raise TypeError("remove_ground expects a LidarSweep")
    c = policy or LidarObstaclePolicy()
    body = body or EgoBody()
    xyz = sweep.points_xyzi[:, :3].astype(float)
    m = body.margin_m
    on_body = (
        (xyz[:, 0] > body.rear_m - m)
        & (xyz[:, 0] < body.front_m + m)
        & (np.abs(xyz[:, 1]) < body.half_width_m + m)
        & (xyz[:, 2] < body.height_m + 2 * m)
    )
    in_range = np.hypot(xyz[:, 0], xyz[:, 1]) <= c.max_range_m
    keep = ~on_body & in_range
    xyz = xyz[keep]
    centres, heights = road_profile(xyz, c)
    road_z = interpolate_profile(xyz[:, 0], centres, heights, c.max_grade)
    h = xyz[:, 2] - road_z
    classes = np.full(len(xyz), CLASS_IGNORED, dtype=np.int8)
    classes[(h >= c.ground_min_m) & (h < c.ground_max_m)] = CLASS_ROAD
    classes[(h >= c.ground_max_m) & (h < c.obstacle_min_m)] = CLASS_RAISED
    classes[(h >= c.obstacle_min_m) & (h < c.overhead_m)] = CLASS_OBSTACLE
    return GroundSegmentation(
        xyz, h, classes, centres, heights, int(on_body.sum()), int((~in_range & ~on_body).sum())
    )


@dataclass(frozen=True)
class EgoRoutePose:
    """Rear-axle pose (metres, radians) in the frame the route polyline uses.

    The default is the identity: the route is already expressed in the ego frame.
    """

    x_m: float = 0.0
    y_m: float = 0.0
    yaw_rad: float = 0.0

    def __post_init__(self):
        for name in ("x_m", "y_m", "yaw_rad"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")

    def to_route(self, xy_ego_m):
        """Ego-frame XY points (N-by-2, metres) expressed in the route frame."""
        p = np.asarray(xy_ego_m, dtype=float).reshape(-1, 2)
        c, s = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        rotation = np.array([[c, -s], [s, c]])
        return p @ rotation.T + np.array([self.x_m, self.y_m])


def project_points(route, xy_m):
    """Vectorised :meth:`RouteGeometry.project` for N-by-2 points (metres).

    Returns ``(s_m, lateral_m, distance_m)`` arrays; lateral is positive left of
    travel and distance is the Euclidean distance to the clamped nearest segment
    point, so points past the route end are far unless near the endpoint.
    """
    if not isinstance(route, RouteGeometry):
        raise TypeError("route must be RouteGeometry")
    p = np.asarray(xy_m, dtype=float).reshape(-1, 2)
    if p.size == 0:
        empty = np.zeros(0)
        return empty, empty, empty
    relative = p[:, None, :] - route.points[None, :-1, :]
    along = np.einsum("nij,ij->ni", relative, route.tangents)
    along = np.clip(along, 0.0, route.lengths[None, :])
    residual = relative - along[..., None] * route.tangents[None]
    distance = np.linalg.norm(residual, axis=2)
    i = np.argmin(distance, axis=1)
    rows = np.arange(len(p))
    lateral = np.einsum("nj,nj->n", residual[rows, i], route.normals[i])
    return route.arc[i] + along[rows, i], lateral, distance[rows, i]


def corridor_mask(route, xy_ego_m, *, ego_pose=None, ego_s_m=0.0, half_width_m, ahead_m):
    """Boolean mask of ego-frame XY points inside the swept corridor ahead of the ego.

    A point is inside when its distance to the route is at most ``half_width_m``
    and its arc coordinate lies in ``(ego_s_m, ego_s_m + ahead_m]`` (metres).
    """
    _positive("half_width_m", half_width_m)
    _positive("ahead_m", ahead_m)
    pose = ego_pose or EgoRoutePose()
    s, _, distance = project_points(route, pose.to_route(xy_ego_m))
    return (distance <= half_width_m) & (s > ego_s_m) & (s <= ego_s_m + ahead_m)


def cluster_points(xy_m, *, cell_m, min_points, merge_gap_m=None):
    """Group XY points (metres) by 8-connected occupancy cells of ``cell_m``.

    Clusters whose axis-aligned footprints come within ``merge_gap_m`` (default
    ``cell_m``) of each other are merged, so a sparsely sampled roof and the face
    below it stay one object. Returns a list of index arrays, one per cluster with
    at least ``min_points`` points. Deterministic and bounded by the number of
    occupied cells.
    """
    _positive("cell_m", cell_m)
    p = np.asarray(xy_m, dtype=float).reshape(-1, 2)
    if len(p) == 0:
        return []
    cells = np.floor(p / cell_m).astype(np.int64)
    origin = cells.min(axis=0)
    cells -= origin
    shape = cells.max(axis=0) + 3
    grid = np.zeros((int(shape[0]), int(shape[1])), np.uint8)
    grid[cells[:, 0] + 1, cells[:, 1] + 1] = 1
    count, labels = cv2.connectedComponents(grid, connectivity=8)
    point_labels = labels[cells[:, 0] + 1, cells[:, 1] + 1]
    groups = [np.flatnonzero(point_labels == label) for label in range(1, count)]
    groups = [g for g in groups if len(g)]
    margin = cell_m if merge_gap_m is None else merge_gap_m
    boxes = [(p[g].min(axis=0), p[g].max(axis=0)) for g in groups]
    parent = list(range(len(groups)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if (boxes[i][0] - margin <= boxes[j][1]).all() and (
                boxes[j][0] <= boxes[i][1] + margin
            ).all():
                parent[root(j)] = root(i)
    merged = {}
    for i, g in enumerate(groups):
        merged.setdefault(root(i), []).append(g)
    clusters = []
    for key in sorted(merged):
        members = np.sort(np.concatenate(merged[key]))
        if len(members) >= min_points:
            clusters.append(members)
    return clusters


def seen_clear_distance(
    road_xy_ego_m,
    route,
    *,
    policy,
    body,
    ego_pose=None,
    ego_s_m=0.0,
    half_width_m=None,
    ahead_m=None,
):
    """Front-bumper distance (metres) to which the corridor road surface is positively seen.

    Road-class points inside the corridor are binned along the route arc from the
    front bumper and across ``clear_strips`` lateral strips. A bin counts as seen
    when its centre strip and at least ``clear_min_strips`` strips hold
    ``clear_min_points`` returns. The result is the end of the last seen bin
    reached without an unseen stretch longer than the allowed gap; coverage must
    begin within ``clear_start_m``. Returns 0.0 when nothing qualifies. This is positive
    evidence of visible road, independent of whether any obstacle was found.
    """
    c = policy
    pose = ego_pose or EgoRoutePose()
    half_width = c.corridor_half_width_m if half_width_m is None else half_width_m
    ahead = c.corridor_ahead_m if ahead_m is None else ahead_m
    xy = np.asarray(road_xy_ego_m, dtype=float).reshape(-1, 2)
    if len(xy) == 0:
        return 0.0
    s, lateral, distance = project_points(route, pose.to_route(xy))
    start = ego_s_m + body.front_m
    inside = (distance <= half_width) & (s > start) & (s <= start + ahead)
    if not inside.any():
        return 0.0
    bins = np.floor((s[inside] - start) / c.clear_bin_m).astype(int)
    # Lateral strips across the corridor: the centre strip and most strips must
    # be seen, so road beside an unobserved middle is not clearance.
    strips = np.clip(
        np.floor((lateral[inside] + half_width) / (2 * half_width) * c.clear_strips).astype(int),
        0,
        c.clear_strips - 1,
    )
    counts = np.zeros((bins.max() + 1, c.clear_strips), dtype=int)
    np.add.at(counts, (bins, strips), 1)
    seen = counts >= c.clear_min_points
    covered = seen[:, c.clear_strips // 2] & (seen.sum(axis=1) >= c.clear_min_strips)
    first = int(np.argmax(covered)) if covered.any() else -1
    if first < 0 or first * c.clear_bin_m > c.clear_start_m:
        return 0.0
    last = first
    for k in range(first + 1, len(covered)):
        if covered[k]:
            last = k
        elif (k - last) * c.clear_bin_m > c.clear_gap_m + c.clear_gap_per_m * (
            last + 1
        ) * c.clear_bin_m:
            break
    return float((last + 1) * c.clear_bin_m)


@dataclass(frozen=True)
class LidarCluster:
    """One obstacle-height cluster in the rear-axle ego frame (metres).

    ``footprint_ego_m`` is the axis-aligned XY rectangle enclosing all cluster
    points (four ordered corners). ``nearest_point_ego_m`` is the cluster point
    with the smallest planar range from the rear axle and ``nearest_surface_m``
    that range. ``corridor_points`` counts points inside the swept corridor and
    ``corridor_gap_m`` is the front-bumper gap along the route for those points
    (``None`` when the cluster does not touch the corridor). ``raised`` marks a
    kerb-height cluster admitted only because it stands inside the corridor.
    """

    point_count: int
    center_ego_m: tuple[float, float, float]
    extent_m: tuple[float, float, float]
    footprint_ego_m: tuple[tuple[float, float], ...]
    nearest_point_ego_m: tuple[float, float, float]
    nearest_surface_m: float
    corridor_points: int
    corridor_gap_m: float | None
    raised: bool = False

    @property
    def in_corridor(self) -> bool:
        return self.corridor_gap_m is not None

    def to_obstacle(self, metadata, track_id=None, *, min_cluster_points=6):
        """Typed :class:`Obstacle`; confidence grows with point support, capped at 1."""
        return Obstacle(
            metadata,
            "lidar_raised" if self.raised else "lidar_cluster",
            min(1.0, self.point_count / (4.0 * min_cluster_points)),
            self.center_ego_m,
            self.extent_m,
            0.0,
            track_id,
        )

    def to_dict(self):
        return dict(
            point_count=self.point_count,
            center_ego_m=list(self.center_ego_m),
            extent_m=list(self.extent_m),
            footprint_ego_m=[list(c) for c in self.footprint_ego_m],
            nearest_point_ego_m=list(self.nearest_point_ego_m),
            nearest_surface_m=self.nearest_surface_m,
            corridor_points=self.corridor_points,
            corridor_gap_m=self.corridor_gap_m,
            raised=self.raised,
        )


@dataclass(frozen=True)
class _Footprint:
    corners_world_m: tuple


@dataclass(frozen=True)
class LidarObstacleBatch:
    """Clusters extracted from one sweep, or an explicit invalid observation.

    ``nearest_gap_m`` is the smallest corridor front-bumper gap over in-corridor
    clusters (metres) or ``None``. ``seen_clear_m`` is the front-bumper distance to
    which corridor road returns were positively observed (metres; ``None`` when the
    batch is invalid). ``visible_distance_m`` is the smaller of the two.
    ``truncated`` reports that more than ``max_clusters`` clusters existed and only
    the nearest were kept. ``usable_for_clearance`` is always False: a consumer must
    combine ``seen_clear_m`` with its own hysteresis before treating the corridor as
    clear.
    """

    metadata: ObservationMetadata
    valid: bool
    invalid_reason: str | None
    clusters: tuple[LidarCluster, ...]
    point_count: int
    density_fraction: float
    road_points: int
    raised_points: int
    obstacle_points: int
    accepted_bands: int
    seen_clear_m: float | None = None
    truncated: bool = False
    usable_for_clearance: bool = False

    def __post_init__(self):
        if not isinstance(self.metadata, ObservationMetadata):
            raise TypeError("metadata must be ObservationMetadata")
        if type(self.valid) is not bool:
            raise ValueError("valid must be bool")
        clusters = tuple(self.clusters)
        if not self.valid:
            if not self.invalid_reason:
                raise ValueError("invalid batches require a reason")
            if clusters:
                raise ValueError("invalid batches cannot contain clusters")
        elif self.invalid_reason is not None:
            raise ValueError("valid batches cannot carry an invalid reason")
        if self.usable_for_clearance:
            raise ValueError("LiDAR clusters never authorise clearance")
        if self.valid and (
            self.seen_clear_m is None
            or not math.isfinite(self.seen_clear_m)
            or self.seen_clear_m < 0
        ):
            raise ValueError("valid batches require a nonnegative seen_clear_m")
        if not self.valid and self.seen_clear_m is not None:
            raise ValueError("invalid batches cannot claim a seen-clear distance")
        object.__setattr__(self, "clusters", clusters)

    @property
    def nearest_gap_m(self) -> float | None:
        gaps = [c.corridor_gap_m for c in self.clusters if c.corridor_gap_m is not None]
        return min(gaps) if gaps else None

    @property
    def visible_distance_m(self) -> float | None:
        """Metres from the bumper to the nearer of the nearest obstacle and the seen-clear end."""
        if not self.valid:
            return None
        gap = self.nearest_gap_m
        return float(self.seen_clear_m) if gap is None else float(min(gap, self.seen_clear_m))

    @property
    def nearest_cluster(self) -> LidarCluster | None:
        candidates = [c for c in self.clusters if c.corridor_gap_m is not None]
        return min(candidates, key=lambda c: c.corridor_gap_m) if candidates else None

    def to_dict(self):
        return dict(
            frame_id=self.metadata.frame_id,
            sim_time_s=self.metadata.sim_time_s,
            valid=self.valid,
            invalid_reason=self.invalid_reason,
            point_count=self.point_count,
            density_fraction=self.density_fraction,
            road_points=self.road_points,
            raised_points=self.raised_points,
            obstacle_points=self.obstacle_points,
            accepted_bands=self.accepted_bands,
            truncated=self.truncated,
            nearest_gap_m=self.nearest_gap_m,
            seen_clear_m=self.seen_clear_m,
            visible_distance_m=self.visible_distance_m,
            clusters=[c.to_dict() for c in self.clusters],
        )


def lidar_metadata(sweep, *, source_id=LIDAR_SOURCE_ID, version=LIDAR_ALGORITHM_VERSION):
    """Source-namespaced observation identity for a sweep."""
    return ObservationMetadata(
        sweep.episode_id, sweep.frame_id, sweep.sim_time_s, source_id, version
    )


def footprints_overlap(a, b, margin_m=0.0):
    """True when two axis-aligned XY footprints (metres) intersect after growing by ``margin_m``."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a_low, a_high = a.min(axis=0) - margin_m, a.max(axis=0) + margin_m
    b_low, b_high = b.min(axis=0), b.max(axis=0)
    return bool((a_low <= b_high).all() and (b_low <= a_high).all())


def _build_cluster(points, mask, route, ego_pose, ego_s_m, front_m, half_width_m, raised):
    xyz = points
    low, high = xyz.min(axis=0), xyz.max(axis=0)
    center = (low + high) / 2
    extent = np.maximum(high - low, 1e-3)
    footprint = ((low[0], low[1]), (high[0], low[1]), (high[0], high[1]), (low[0], high[1]))
    ranges = np.hypot(xyz[:, 0], xyz[:, 1])
    nearest = int(np.argmin(ranges))
    gap = None
    inside = int(mask.sum())
    if inside:
        corridor_xy = ego_pose.to_route(xyz[mask][:, :2])
        lo, hi = corridor_xy.min(axis=0), corridor_xy.max(axis=0)
        corners = ((lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1]))
        gap = route.gap(_Footprint(corners), ego_s_m, front_m, half_width_m)
        # The crop proved a point lies inside the swept corridor; a degenerate
        # footprint that the polygon clip misses still reports its nearest gap.
        if gap is None:
            s, _, _ = project_points(route, corridor_xy)
            gap = float(s.min() - ego_s_m - front_m)
    return LidarCluster(
        int(len(xyz)),
        tuple(float(v) for v in center),
        tuple(float(v) for v in extent),
        tuple((float(a), float(b)) for a, b in footprint),
        tuple(float(v) for v in xyz[nearest]),
        float(ranges[nearest]),
        inside,
        None if gap is None else float(gap),
        raised,
    )


def extract_obstacles(
    sweep,
    route,
    *,
    policy=None,
    body=None,
    ego_pose=None,
    ego_s_m=0.0,
    half_width_m=None,
    ahead_m=None,
):
    """Ground removal, corridor crop and bounded clustering for one sweep.

    ``route`` is the :class:`RouteGeometry` (metres) in the frame given by
    ``ego_pose`` (default: the ego frame itself); ``ego_s_m`` is the rear axle's
    arc coordinate on that route. Corridor gaps are front-bumper gaps using
    ``body.front_m`` exactly once. Returns a :class:`LidarObstacleBatch`, invalid
    when the sweep is too sparse or the road could not be established.
    """
    if not isinstance(sweep, LidarSweep):
        raise TypeError("extract_obstacles expects a LidarSweep")
    if not isinstance(route, RouteGeometry):
        raise TypeError("route must be RouteGeometry")
    c = policy or LidarObstaclePolicy()
    body = body or EgoBody()
    pose = ego_pose or EgoRoutePose()
    half_width = c.corridor_half_width_m if half_width_m is None else half_width_m
    ahead = c.corridor_ahead_m if ahead_m is None else ahead_m
    metadata = lidar_metadata(sweep)
    density = sweep.density_fraction

    def invalid(reason, segmentation=None):
        return LidarObstacleBatch(
            metadata,
            False,
            reason,
            (),
            sweep.point_count,
            density,
            0 if segmentation is None else int((segmentation.classes == CLASS_ROAD).sum()),
            0 if segmentation is None else int((segmentation.classes == CLASS_RAISED).sum()),
            0 if segmentation is None else int((segmentation.classes == CLASS_OBSTACLE).sum()),
            0 if segmentation is None else segmentation.accepted_bands,
        )

    if sweep.point_count < c.min_points:
        return invalid("sparse_sweep_points")
    if density < c.min_density_fraction:
        return invalid("sparse_sweep_density")
    segmentation = remove_ground(sweep, c, body)
    if segmentation.accepted_bands < 2:
        return invalid("road_not_established", segmentation)
    clusters = []
    truncated = False
    for label, raised in ((CLASS_OBSTACLE, False), (CLASS_RAISED, True)):
        points = segmentation.of(label)
        if len(points) == 0:
            continue
        mask = corridor_mask(
            route,
            points[:, :2],
            ego_pose=pose,
            ego_s_m=ego_s_m,
            half_width_m=half_width,
            ahead_m=ahead,
        )
        if raised:
            # Kerb-height returns are the road edge unless they stand in the lane.
            points, mask = points[mask], mask[mask]
            minimum = c.raised_corridor_min_points
        else:
            minimum = c.min_cluster_points
        if len(points) == 0:
            continue
        for members in cluster_points(
            points[:, :2], cell_m=c.cell_m, min_points=minimum, merge_gap_m=c.cluster_merge_m
        ):
            cluster = _build_cluster(
                points[members],
                mask[members],
                route,
                pose,
                ego_s_m,
                body.front_m,
                half_width,
                raised,
            )
            # A kerb-height cluster under an obstacle cluster is that obstacle's
            # sills and wheels: keep one cluster so one object yields one track.
            if raised and any(
                footprints_overlap(cluster.footprint_ego_m, other.footprint_ego_m, c.cell_m)
                for other in clusters
                if not other.raised
            ):
                continue
            clusters.append(cluster)
    clusters.sort(
        key=lambda k: (k.corridor_gap_m is None, k.corridor_gap_m or 0.0, k.nearest_surface_m)
    )
    if len(clusters) > c.max_clusters:
        clusters, truncated = clusters[: c.max_clusters], True
    seen_clear = seen_clear_distance(
        segmentation.of(CLASS_ROAD)[:, :2],
        route,
        policy=c,
        body=body,
        ego_pose=pose,
        ego_s_m=ego_s_m,
        half_width_m=half_width,
        ahead_m=ahead,
    )
    return LidarObstacleBatch(
        metadata,
        True,
        None,
        tuple(clusters),
        sweep.point_count,
        density,
        int((segmentation.classes == CLASS_ROAD).sum()),
        int((segmentation.classes == CLASS_RAISED).sum()),
        int((segmentation.classes == CLASS_OBSTACLE).sum()),
        segmentation.accepted_bands,
        seen_clear,
        truncated,
    )


@dataclass(frozen=True)
class LidarTrackObservation:
    """One LiDAR track; ``gap_m`` is a front-bumper corridor gap in metres.

    ``gap_valid`` requires a confirmed, current, in-corridor cluster from a valid
    sweep. ``robust_gap_m`` is the median of recent valid gaps (descriptive, never
    permission to ignore a smaller current gap).
    """

    identity: TrackIdentity
    metadata: ObservationMetadata
    cluster: LidarCluster
    confirmed: bool
    current: bool
    gap_valid: bool
    gap_m: float | None
    robust_gap_m: float | None
    invalid_reason: str | None

    def to_dict(self):
        return dict(
            track_id=self.identity.track_id,
            source_id=self.identity.source_id,
            confirmed=self.confirmed,
            current=self.current,
            gap_valid=self.gap_valid,
            gap_m=self.gap_m,
            robust_gap_m=self.robust_gap_m,
            invalid_reason=self.invalid_reason,
            nearest_surface_m=self.cluster.nearest_surface_m,
        )


@dataclass(frozen=True)
class LidarTrackBatch:
    """Tracks after one sweep. ``valid`` mirrors the sweep batch; clearance is never usable."""

    metadata: ObservationMetadata
    tracks: tuple[LidarTrackObservation, ...]
    expired: tuple[TrackIdentity, ...]
    valid: bool
    invalid_reason: str | None
    seen_clear_m: float | None = None
    usable_for_clearance: bool = False

    @property
    def nearest_gap_m(self) -> float | None:
        gaps = [t.gap_m for t in self.tracks if t.gap_valid and t.gap_m is not None]
        return min(gaps) if gaps else None

    @property
    def visible_distance_m(self) -> float | None:
        """Metres from the bumper to the nearer of the nearest confirmed gap and seen-clear end."""
        if not self.valid or self.seen_clear_m is None:
            return None
        gap = self.nearest_gap_m
        return float(self.seen_clear_m) if gap is None else float(min(gap, self.seen_clear_m))

    @property
    def unknown_tracks(self) -> bool:
        """True when a corridor-relevant track lacks a valid current gap.

        Unconfirmed or lost clusters whose last footprint touched the corridor are
        unknown; clusters that only ever stood outside the corridor are not.
        """
        return any(t.cluster.in_corridor and not t.gap_valid for t in self.tracks)

    def to_dict(self):
        return dict(
            frame_id=self.metadata.frame_id,
            sim_time_s=self.metadata.sim_time_s,
            valid=self.valid,
            invalid_reason=self.invalid_reason,
            nearest_gap_m=self.nearest_gap_m,
            seen_clear_m=self.seen_clear_m,
            visible_distance_m=self.visible_distance_m,
            unknown_tracks=self.unknown_tracks,
            expired=[e.track_id for e in self.expired],
            tracks=[t.to_dict() for t in self.tracks],
        )


class LidarTrackFilter:
    """N-of-M confirmation of LiDAR clusters with source-namespaced IDs.

    Clusters are associated across sweeps by footprint centre within ``gate_m +
    closing_speed_mps * elapsed`` metres or by overlapping footprints grown by half
    that gate (greedy, nearest centre first). IDs are integers scoped by
    :class:`TrackIdentity` to the episode, ``source_id`` and algorithm version, so
    they can never collide with RGB-D tracker IDs. Invalid sweeps count as misses
    for every track; expiry removes tracks unseen for longer than ``expiry_s``.
    """

    def __init__(
        self,
        *,
        source_id=LIDAR_SOURCE_ID,
        version=LIDAR_ALGORITHM_VERSION,
        policy=None,
        gate_m=1.5,
        closing_speed_mps=10.0,
    ):
        ObservationMetadata("validation", 0, 0, source_id, version)
        _positive("gate_m", gate_m)
        if not math.isfinite(closing_speed_mps) or closing_speed_mps < 0:
            raise ValueError("closing_speed_mps must be nonnegative")
        self.source_id, self.version = source_id, version
        self.policy = policy or TrackPolicy()
        self.gate_m = gate_m
        self.closing_speed_mps = closing_speed_mps
        self._episode = None
        self._states = {}
        self._next_id = 0
        self._last = None

    def reset(self, *, episode_id):
        ObservationMetadata(episode_id, 0, 0, self.source_id, self.version)
        self._episode = episode_id
        self._states.clear()
        self._next_id = 0
        self._last = None

    def update(self, batch):
        if not isinstance(batch, LidarObstacleBatch):
            raise TypeError("update expects a LidarObstacleBatch")
        m = batch.metadata
        if (m.episode_id, m.source_id, m.model_version) != (
            self._episode,
            self.source_id,
            self.version,
        ):
            raise ValueError("Reset filter for matching episode/source/version")
        if self._last is not None and (
            m.frame_id <= self._last[0] or m.sim_time_s <= self._last[1]
        ):
            raise ValueError("Sweeps must increase in frame and time")
        self._last = (m.frame_id, m.sim_time_s)
        expired = tuple(
            identity
            for identity, state in self._states.items()
            if m.sim_time_s - state["last_seen"] > self.policy.expiry_s
        )
        for identity in expired:
            del self._states[identity]
        assigned = {}
        if batch.valid:
            pairs = []
            for index, cluster in enumerate(batch.clusters):
                centre = np.array(cluster.center_ego_m[:2])
                for identity, state in self._states.items():
                    previous = state["cluster"]
                    distance = float(np.linalg.norm(centre - np.array(previous.center_ego_m[:2])))
                    # The gate grows with the time since the track was last seen,
                    # so a dropped sweep does not split one approaching object into
                    # two IDs. Large footprints (walls, buses) shift their centre as
                    # their visible part changes; overlapping footprints still associate.
                    gate = self.gate_m + self.closing_speed_mps * (
                        m.sim_time_s - state["last_seen"]
                    )
                    if distance <= gate or footprints_overlap(
                        cluster.footprint_ego_m, previous.footprint_ego_m, gate / 2
                    ):
                        pairs.append((distance, index, identity))
            used_clusters, used_tracks = set(), set()
            for _, index, identity in sorted(pairs, key=lambda p: (p[0], p[1], p[2].track_id)):
                if index in used_clusters or identity in used_tracks:
                    continue
                used_clusters.add(index)
                used_tracks.add(identity)
                assigned[identity] = batch.clusters[index]
            for index, cluster in enumerate(batch.clusters):
                if index not in used_clusters:
                    identity = m.track_identity(self._next_id)
                    self._next_id += 1
                    self._states[identity] = dict(
                        hits=deque(maxlen=self.policy.window),
                        gaps=deque(maxlen=self.policy.distance_samples),
                        last_seen=m.sim_time_s,
                        cluster=cluster,
                    )
                    assigned[identity] = cluster
        rows = []
        for identity, state in self._states.items():
            present = identity in assigned
            if present:
                state["cluster"] = assigned[identity]
                state["last_seen"] = m.sim_time_s
            state["hits"].append(present)
            confirmed = sum(state["hits"]) >= self.policy.hits
            cluster = state["cluster"]
            gap = cluster.corridor_gap_m if present and batch.valid else None
            valid = bool(confirmed and present and batch.valid and gap is not None)
            gaps = state["gaps"]
            while gaps and m.sim_time_s - gaps[0][0] > self.policy.expiry_s:
                gaps.popleft()
            if valid:
                gaps.append((m.sim_time_s, gap))
            reason = (
                None
                if valid
                else "invalid_sweep"
                if not batch.valid
                else "unconfirmed"
                if not confirmed
                else "not_visible"
                if not present
                else "outside_corridor"
            )
            rows.append(
                LidarTrackObservation(
                    identity,
                    m,
                    cluster,
                    confirmed,
                    present,
                    valid,
                    gap if valid else None,
                    median(g[1] for g in gaps) if valid and gaps else None,
                    reason,
                )
            )
        return LidarTrackBatch(
            m, tuple(rows), expired, batch.valid, batch.invalid_reason, batch.seen_clear_m
        )


def track_obstacles(tracks, *, min_cluster_points=6):
    """Typed :class:`Obstacle` objects for the confirmed current tracks of a batch."""
    return tuple(
        t.cluster.to_obstacle(
            t.metadata, t.identity.track_id, min_cluster_points=min_cluster_points
        )
        for t in tracks.tracks
        if t.confirmed and t.current
    )


__all__ = [
    "CLASS_IGNORED",
    "CLASS_OBSTACLE",
    "CLASS_RAISED",
    "CLASS_ROAD",
    "EgoBody",
    "EgoRoutePose",
    "GroundSegmentation",
    "LIDAR_ALGORITHM_VERSION",
    "LIDAR_SOURCE_ID",
    "LidarCluster",
    "LidarObstacleBatch",
    "LidarObstaclePolicy",
    "LidarTrackBatch",
    "LidarTrackFilter",
    "LidarTrackObservation",
    "cluster_points",
    "corridor_mask",
    "extract_obstacles",
    "footprints_overlap",
    "interpolate_profile",
    "lidar_metadata",
    "project_points",
    "remove_ground",
    "road_profile",
    "seen_clear_distance",
    "track_obstacles",
]
