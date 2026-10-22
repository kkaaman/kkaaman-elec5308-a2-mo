"""LiDAR terrain: road surface, raised kerbs and obstacles, accumulated in the odometry frame.

**Per sweep (vehicle frame).** The road is the lowest extensive surface in a town:
pavements sit on kerbs about 0.15 m above it. The road height is estimated in 2 m bands
along x, outwards from the car, where the road is at z = 0 (the vehicle frame's origin
is on the ground). The first band on each side may differ from z = 0 by up to
``max_grade`` per metre; after that each band must continue the grade of the two before
it within a tight tolerance, so a car filling a band, with the road behind it in
shadow, cannot become "the road".
Each point is then labelled by its height above the road at its x:

- ROAD below ``raised_min_m``;
- RAISED (kerb, pavement) up to ``obstacle_min_m``;
- OBSTACLE up to ``overhead_m`` (cars, poles, walls, people);
- ignored above that (canopies, signs) or well below the road.

Points on the car's own body are dropped first (the roof LiDAR sees the roof).

**Accumulated (odometry frame).** A fixed grid follows the car. Each sweep adds one unit
of evidence per class to every cell it hit, and evidence decays with time: slowly for
the road and kerbs, which do not move, and quickly for obstacles, which may. One sweep
covers the road only in rings, about 1.5 m apart at 12 m and 6 m apart at 25 m, so the
corridor needs the accumulation.

LiDAR intensity in the simulator depends on range only, not on paint, so lane markings
are invisible here. That is why the corridor follows the kerb.
"""

import math
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from mapless.ego_motion import Pose2D


class PointClass(IntEnum):
    IGNORED = -1
    ROAD = 0
    RAISED = 1
    OBSTACLE = 2


class CellLabel(IntEnum):
    UNKNOWN = -1
    ROAD = 0
    BARRIER = 1  # kerb, pavement or obstacle: the corridor ends here


@dataclass(frozen=True)
class VehicleBody:
    """The car's box in the vehicle frame (rear-axle origin)."""

    front_m: float = 3.78
    rear_m: float = -1.01
    half_width_m: float = 1.05
    height_m: float = 1.50

    def __post_init__(self):
        if not (self.front_m > 0 > self.rear_m) or self.half_width_m <= 0 or self.height_m <= 0:
            raise ValueError("VehicleBody must enclose the rear axle and have a positive size")


@dataclass(frozen=True)
class TerrainConfig:
    cell_m: float = 0.25
    size_m: float = 120.0
    recentre_m: float = 10.0
    band_m: float = 2.0
    road_window_m: float = 4.0  # |y| used for the road-height estimate
    road_percentile: float = 10.0
    band_tolerance_m: float = 0.07  # road height may differ this much from the extrapolation,
    band_tolerance_per_m: float = 0.015  # plus this per metre from the last accepted band
    max_grade: float = 0.12
    raised_min_m: float = 0.10  # kerbs are about 0.15 m; road camber and LiDAR noise stay below
    obstacle_min_m: float = 0.30
    overhead_m: float = 2.4
    below_m: float = -0.4
    max_range_m: float = 45.0
    body_margin_m: float = 0.3
    static_tau_s: float = 3.0
    obstacle_tau_s: float = 0.5
    barrier_min_evidence: float = 1.0
    road_min_evidence: float = 1.0

    def __post_init__(self):
        for name in ("cell_m", "size_m", "band_m", "static_tau_s", "obstacle_tau_s", "max_range_m"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.raised_min_m < self.obstacle_min_m < self.overhead_m:
            raise ValueError("Height thresholds must increase: raised < obstacle < overhead")
        if self.recentre_m + self.max_range_m > self.size_m / 2:
            raise ValueError("The grid must hold max_range_m around the car wherever it recentres")


@dataclass(frozen=True)
class ClassifiedSweep:
    xyz: np.ndarray  # N x 3, vehicle frame, body points removed
    height_m: np.ndarray  # height above the estimated road at each point's x
    cls: np.ndarray  # PointClass per point
    band_x_m: np.ndarray  # road-height profile: band centres
    band_z_m: np.ndarray  # and accepted heights (NaN where rejected)

    def of(self, point_class):
        return self.xyz[self.cls == point_class]


def road_profile(xyz, config):
    """Road height in bands along x, grown outwards from the car's own band at z = 0."""
    c = config
    n_ahead = int(math.ceil(c.max_range_m / c.band_m))
    centres = np.arange(-n_ahead, n_ahead + 1) * c.band_m
    heights = np.full(centres.shape, np.nan)
    near = np.abs(xyz[:, 1]) < c.road_window_m
    x, z = xyz[near, 0], xyz[near, 2]
    index = np.floor((x + c.band_m / 2) / c.band_m).astype(int) + n_ahead
    valid = (index >= 0) & (index < centres.size)
    order = np.lexsort((z[valid], index[valid]))  # by band, then height within the band
    sorted_index, sorted_z = index[valid][order], z[valid][order]
    bounds = np.searchsorted(sorted_index, np.arange(centres.size + 1))
    zero = n_ahead
    heights[zero] = 0.0
    for direction in (1, -1):
        accepted = [(0.0, 0.0)]
        k = zero + direction
        while 0 <= k < centres.size:
            band = sorted_z[bounds[k] : bounds[k + 1]]  # ascending
            if band.size >= 3:
                low = band[int(c.road_percentile / 100.0 * (band.size - 1))]
                top = int(np.searchsorted(band, low + 0.05, side="right"))
                estimate = float(band[top // 2]) if top else float(low)
                (x1, z1) = accepted[-1]
                if len(accepted) >= 2:
                    (x0, z0) = accepted[-2]
                    slope = float(np.clip((z1 - z0) / (x1 - x0), -c.max_grade, c.max_grade))
                else:
                    slope = 0.0
                predicted = z1 + slope * (centres[k] - x1)
                tolerance = c.band_tolerance_m + c.band_tolerance_per_m * abs(centres[k] - x1)
                if len(accepted) == 1:
                    tolerance += c.max_grade * abs(centres[k] - x1)  # the grade is not known yet
                if abs(estimate - predicted) <= tolerance:
                    heights[k] = estimate
                    accepted.append((float(centres[k]), estimate))
            k += direction
    return centres, heights


def classify_sweep(xyz, body, config):
    """Label one sweep's points (vehicle frame, metres)."""
    c = config
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    m = c.body_margin_m
    on_body = (
        (xyz[:, 0] > body.rear_m - m)
        & (xyz[:, 0] < body.front_m + m)
        & (np.abs(xyz[:, 1]) < body.half_width_m + m)
        & (xyz[:, 2] < body.height_m + 0.6)
    )
    in_range = np.hypot(xyz[:, 0], xyz[:, 1]) <= c.max_range_m
    xyz = xyz[~on_body & in_range]
    centres, heights = road_profile(xyz, c)
    ok = np.isfinite(heights)
    road_z = np.interp(xyz[:, 0], centres[ok], heights[ok]) if ok.any() else np.zeros(len(xyz))
    h = xyz[:, 2] - road_z
    cls = np.full(len(xyz), PointClass.IGNORED, dtype=np.int8)
    cls[(h >= c.below_m) & (h < c.raised_min_m)] = PointClass.ROAD
    cls[(h >= c.raised_min_m) & (h < c.obstacle_min_m)] = PointClass.RAISED
    cls[(h >= c.obstacle_min_m) & (h < c.overhead_m)] = PointClass.OBSTACLE
    return ClassifiedSweep(xyz, h, cls, centres, heights)


class TerrainGrid:
    """Decaying per-class evidence on a grid in the odometry frame that follows the car."""

    def __init__(self, config=None):
        self.config = config or TerrainConfig()
        c = self.config
        self.n = int(round(c.size_m / c.cell_m))
        self.origin = np.array([-c.size_m / 2, -c.size_m / 2])  # odometry coordinates of cell (0, 0)
        self.road = np.zeros((self.n, self.n), dtype=np.float32)
        self.raised = np.zeros_like(self.road)
        self.obstacle = np.zeros_like(self.road)
        self.time_s = None

    def _centre(self):
        return self.origin + self.config.size_m / 2

    def _recentre(self, x, y):
        c = self.config
        shift = np.round((np.array([x, y]) - self._centre()) / c.cell_m).astype(int)
        for name in ("road", "raised", "obstacle"):
            old = getattr(self, name)
            new = np.zeros_like(old)
            di, dj = shift
            src_i = slice(max(0, di), min(self.n, self.n + di))
            dst_i = slice(max(0, -di), min(self.n, self.n - di))
            src_j = slice(max(0, dj), min(self.n, self.n + dj))
            dst_j = slice(max(0, -dj), min(self.n, self.n - dj))
            new[dst_i, dst_j] = old[src_i, src_j]
            setattr(self, name, new)
        self.origin = self.origin + shift * c.cell_m

    def cells(self, xy):
        """Integer cell indices for odometry points, and which fall inside the grid."""
        ij = np.floor((np.asarray(xy, dtype=float).reshape(-1, 2) - self.origin) / self.config.cell_m).astype(int)
        inside = (ij[:, 0] >= 0) & (ij[:, 0] < self.n) & (ij[:, 1] >= 0) & (ij[:, 1] < self.n)
        return ij, inside

    def add(self, sweep, pose, time_s):
        """Decay to ``time_s``, then add a classified sweep seen from odometry ``pose``."""
        if not isinstance(sweep, ClassifiedSweep) or not isinstance(pose, Pose2D):
            raise TypeError("add(ClassifiedSweep, Pose2D, time_s)")
        c = self.config
        if self.time_s is not None:
            dt = max(0.0, time_s - self.time_s)
            self.road *= math.exp(-dt / c.static_tau_s)
            self.raised *= math.exp(-dt / c.static_tau_s)
            self.obstacle *= math.exp(-dt / c.obstacle_tau_s)
        self.time_s = time_s
        if math.hypot(*(np.array([pose.x_m, pose.y_m]) - self._centre())) > c.recentre_m:
            self._recentre(pose.x_m, pose.y_m)
        xy = pose.to_world(sweep.xyz[:, :2])
        ij, inside = self.cells(xy)
        flat = ij[:, 0] * self.n + ij[:, 1]
        for point_class, grid in (
            (PointClass.ROAD, self.road),
            (PointClass.RAISED, self.raised),
            (PointClass.OBSTACLE, self.obstacle),
        ):
            hit = np.unique(flat[inside & (sweep.cls == point_class)])
            grid.ravel()[hit] += 1.0

    def labels(self, xy):
        """CellLabel for odometry points: BARRIER, ROAD or UNKNOWN (outside or unseen)."""
        c = self.config
        ij, inside = self.cells(xy)
        out = np.full(len(ij), CellLabel.UNKNOWN, dtype=np.int8)
        i, j = ij[inside, 0], ij[inside, 1]
        road, raised, obstacle = self.road[i, j], self.raised[i, j], self.obstacle[i, j]
        barrier = raised + obstacle
        is_barrier = (barrier >= c.barrier_min_evidence) & (barrier >= 0.5 * road)
        is_road = ~is_barrier & (road >= c.road_min_evidence)
        result = np.full(len(i), CellLabel.UNKNOWN, dtype=np.int8)
        result[is_road] = CellLabel.ROAD
        result[is_barrier] = CellLabel.BARRIER
        out[inside] = result
        return out

    def obstacle_evidence(self, xy):
        ij, inside = self.cells(xy)
        out = np.zeros(len(ij), dtype=np.float32)
        out[inside] = self.obstacle[ij[inside, 0], ij[inside, 1]]
        return out
