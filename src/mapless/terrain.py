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

import copy
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
    BARRIER = 1  # kerb or pavement: the road ends here
    OCCUPIED = 2  # something standing on the road (car, pole, person); not a road edge


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
    path_window_m: float = 1.5  # |y| of the car's own path, preferred for the estimate
    road_percentile: float = 10.0
    band_tolerance_m: float = 0.07  # road height may differ this much from the extrapolation,
    band_tolerance_per_m: float = 0.015  # plus this per metre from the last accepted band
    max_grade: float = 0.12
    raised_min_m: float = 0.10  # kerbs are about 0.15 m; road camber and LiDAR noise stay below
    obstacle_min_m: float = 0.30
    overhead_m: float = 2.4
    below_m: float = -0.4
    max_range_m: float = 45.0
    body_margin_m: float = 0.3  # beside and behind the body (mirrors, the rear radars) ...
    body_front_margin_m: float = 0.05  # ... and ahead of the front bumper, where nothing of the car is
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


def _banded(xyz, window_m, centres, n_ahead, band_m):
    """Heights of points with |y| < window_m, sorted by band then height, with band bounds."""
    near = np.abs(xyz[:, 1]) < window_m
    x, z = xyz[near, 0], xyz[near, 2]
    index = np.floor((x + band_m / 2) / band_m).astype(int) + n_ahead
    valid = (index >= 0) & (index < centres.size)
    order = np.lexsort((z[valid], index[valid]))
    sorted_index, sorted_z = index[valid][order], z[valid][order]
    return sorted_z, np.searchsorted(sorted_index, np.arange(centres.size + 1))


def _surface(band, percentile):
    low = band[int(percentile / 100.0 * (band.size - 1))]
    top = int(np.searchsorted(band, low + 0.05, side="right"))
    return float(band[top // 2]) if top else float(low)


def road_profile(xyz, config):
    """Road height in bands along x, grown outwards from the car's own band at z = 0.

    Each band's estimate comes from the car's own path (|y| < ``path_window_m``) when the
    LiDAR sees it there, otherwise from the wider ``road_window_m``. Only a path estimate
    may set the grade on the first band: beside the car the LiDAR sees only beyond its
    3.8 m blind ring, which on a narrow road is pavement (the first narrow-road test took
    the pavement for the road).
    """
    c = config
    n_ahead = int(math.ceil(c.max_range_m / c.band_m))
    centres = np.arange(-n_ahead, n_ahead + 1) * c.band_m
    heights = np.full(centres.shape, np.nan)
    path_z, path_bounds = _banded(xyz, c.path_window_m, centres, n_ahead, c.band_m)
    wide_z, wide_bounds = _banded(xyz, c.road_window_m, centres, n_ahead, c.band_m)
    zero = n_ahead
    heights[zero] = 0.0
    for direction in (1, -1):
        accepted = [(0.0, 0.0)]
        k = zero + direction
        while 0 <= k < centres.size:
            path = path_z[path_bounds[k] : path_bounds[k + 1]]
            wide = wide_z[wide_bounds[k] : wide_bounds[k + 1]]
            from_path = path.size >= 3
            band = path if from_path else wide
            first = len(accepted) == 1
            if band.size >= 3 and (from_path or not first):
                estimate = _surface(band, c.road_percentile)
                (x1, z1) = accepted[-1]
                if len(accepted) >= 2:
                    (x0, z0) = accepted[-2]
                    slope = float(np.clip((z1 - z0) / (x1 - x0), -c.max_grade, c.max_grade))
                else:
                    slope = 0.0
                predicted = z1 + slope * (centres[k] - x1)
                tolerance = c.band_tolerance_m + c.band_tolerance_per_m * abs(centres[k] - x1)
                if first:
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
    # The 0.3 m margin ahead of the bumper hid a person standing just in front of it (0 of
    # 272 points left, full review of 73ae759): nothing of the car is there to mask.
    on_body = (
        (xyz[:, 0] > body.rear_m - m)
        & (xyz[:, 0] < body.front_m + c.body_front_margin_m)
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
        self.tall = np.zeros_like(self.road)  # obstacle hits remembered as long as kerbs are
        self.seen = np.full((self.n, self.n), -np.inf)  # simulation time each cell was last hit
        # ... and last hit by each class, so a fresh road return cannot vouch for an old kerb
        # (review of 2b7e14d): "fresh kerb" means a recent raised hit in that cell.
        self.seen_by_class = {k: np.full((self.n, self.n), -np.inf) for k in (PointClass.ROAD, PointClass.RAISED, PointClass.OBSTACLE)}
        self.time_s = None

    def reset(self):
        """Forget every observation: the odometry frame they were placed in is no longer sound."""
        for name in ("road", "raised", "obstacle", "tall"):
            getattr(self, name).fill(0.0)
        self.seen.fill(-np.inf)
        for grid in self.seen_by_class.values():
            grid.fill(-np.inf)
        self.origin = np.array([-self.config.size_m / 2, -self.config.size_m / 2])
        self.time_s = None

    def _centre(self):
        return self.origin + self.config.size_m / 2

    def _recentre(self, x, y):
        c = self.config
        shift = np.round((np.array([x, y]) - self._centre()) / c.cell_m).astype(int)
        for name in ("road", "raised", "obstacle", "tall", "seen", *self.seen_by_class):
            old = getattr(self, name) if not isinstance(name, PointClass) else self.seen_by_class[name]
            new = np.full_like(old, -np.inf) if name == "seen" or isinstance(name, PointClass) else np.zeros_like(old)
            di, dj = shift
            src_i = slice(max(0, di), min(self.n, self.n + di))
            dst_i = slice(max(0, -di), min(self.n, self.n - di))
            src_j = slice(max(0, dj), min(self.n, self.n + dj))
            dst_j = slice(max(0, -dj), min(self.n, self.n - dj))
            new[dst_i, dst_j] = old[src_i, src_j]
            if isinstance(name, PointClass):
                self.seen_by_class[name] = new
            else:
                setattr(self, name, new)
        self.origin = self.origin + shift * c.cell_m

    def view(self):
        """The grid's placement now, without its arrays: a TerrainGrid whose ``origin`` is a
        copy and whose evidence arrays are this grid's own (read none of them through it).
        What ``RoadModelBuilder.add`` needs from another thread (sixth review, F6-01)."""
        out = copy.copy(self)
        out.origin = self.origin.copy()
        return out

    def snapshot(self):
        """A copy of the grid now, arrays and all, for reading from another thread while
        this one is updated (``RoadModelBuilder.build`` on the semantic worker, F6-01)."""
        out = copy.copy(self)
        out.origin = self.origin.copy()
        for name, value in vars(self).items():
            if isinstance(value, np.ndarray):
                setattr(out, name, value.copy())
            elif isinstance(value, dict) and value and all(isinstance(v, np.ndarray) for v in value.values()):
                setattr(out, name, {k: v.copy() for k, v in value.items()})
        return out

    def cells(self, xy):
        """Integer cell indices for odometry points, and which fall inside the grid."""
        ij = np.floor((np.asarray(xy, dtype=float).reshape(-1, 2) - self.origin) / self.config.cell_m).astype(int)
        inside = (ij[:, 0] >= 0) & (ij[:, 0] < self.n) & (ij[:, 1] >= 0) & (ij[:, 1] < self.n)
        return ij, inside

    def add(self, sweep, pose, time_s, stationary=False):
        """Decay to ``time_s``, then add a classified sweep seen from odometry ``pose``.

        ``stationary``: the car stands still. The LiDAR's rings then hit the same cells
        every sweep and never the cells between them, which were seen while the car moved
        and cannot have been seen differently since. Their static evidence (road, kerb,
        tall) is kept as it was instead of decaying: after 5-10 s at a junction mouth the
        road ahead had decayed to unknown, the verified distance fell from 22 m to 0 and
        the car never moved off (full review of 73ae759, S7). Cells this sweep covers decay
        as usual, so their evidence stays bounded; obstacle evidence always decays.
        """
        if not isinstance(sweep, ClassifiedSweep) or not isinstance(pose, Pose2D):
            raise TypeError("add(ClassifiedSweep, Pose2D, time_s)")
        c = self.config
        if math.hypot(*(np.array([pose.x_m, pose.y_m]) - self._centre())) > c.recentre_m:
            self._recentre(pose.x_m, pose.y_m)
        xy = pose.to_world(sweep.xyz[:, :2])
        ij, inside = self.cells(xy)
        flat = ij[:, 0] * self.n + ij[:, 1]
        covered = np.unique(flat[inside & (sweep.cls != PointClass.IGNORED)])
        if self.time_s is not None:
            dt = max(0.0, time_s - self.time_s)
            static = math.exp(-dt / c.static_tau_s)
            for grid in (self.road, self.raised, self.tall):
                if stationary:
                    grid.ravel()[covered] *= static
                else:
                    grid *= static
            self.obstacle *= math.exp(-dt / c.obstacle_tau_s)
        self.time_s = time_s
        self.seen.ravel()[covered] = time_s
        for point_class, grid in (
            (PointClass.ROAD, self.road),
            (PointClass.RAISED, self.raised),
            (PointClass.OBSTACLE, self.obstacle),
            (PointClass.OBSTACLE, self.tall),
        ):
            hit = np.unique(flat[inside & (sweep.cls == point_class)])
            grid.ravel()[hit] += 1.0
            self.seen_by_class[point_class].ravel()[hit] = time_s

    def seen_at(self, xy, point_class=None):
        """Simulation time each odometry point's cell was last hit, by ``point_class`` if given (-inf if never or outside)."""
        ij, inside = self.cells(xy)
        out = np.full(len(ij), -np.inf)
        source = self.seen if point_class is None else self.seen_by_class[PointClass(point_class)]
        out[inside] = source[ij[inside, 0], ij[inside, 1]]
        return out

    def labels(self, xy, max_age_s=None):
        """CellLabel for odometry points: BARRIER, OCCUPIED, ROAD or UNKNOWN (outside or unseen).

        With ``max_age_s``, a cell not hit by any point within that long is UNKNOWN whatever
        evidence it still holds: evidence decays slowly (a cell seen 60 times takes 12 s
        to fall below one), so without this an empty or dead LiDAR would leave old road
        looking current (review of 94b5caa).

        Only raised ground (kerb, pavement) is a BARRIER. Obstacles are OCCUPIED, so a row of
        parked cars does not become the road edge: on Town10HD's shoulders the first live
        G1b run took parked cars and kerb in turn as the edge.
        """
        c = self.config
        ij, inside = self.cells(xy)
        out = np.full(len(ij), CellLabel.UNKNOWN, dtype=np.int8)
        i, j = ij[inside, 0], ij[inside, 1]
        road, raised, obstacle = self.road[i, j], self.raised[i, j], self.obstacle[i, j]
        tall = self.tall[i, j]
        # A kerb is raised and nothing more. The low band of a car's side (sills, wheels) is
        # also 0.1-0.3 m high, but its cells also see the taller body. ``tall`` decays like
        # ``raised``, so the two compare fairly (``obstacle`` fades six times faster).
        is_barrier = (raised >= c.barrier_min_evidence) & (raised >= 0.5 * road) & (tall < 0.5 * raised)
        is_occupied = ~is_barrier & (obstacle >= c.barrier_min_evidence)
        is_road = ~is_barrier & ~is_occupied & (road >= c.road_min_evidence)
        result = np.full(len(i), CellLabel.UNKNOWN, dtype=np.int8)
        result[is_road] = CellLabel.ROAD
        result[is_barrier] = CellLabel.BARRIER
        result[is_occupied] = CellLabel.OCCUPIED
        if max_age_s is not None and self.time_s is not None:
            stale = self.seen[i, j] < self.time_s - max_age_s
            # A label is as fresh as the class that made it, not any hit in the cell.
            stale |= is_barrier & (self.seen_by_class[PointClass.RAISED][i, j] < self.time_s - max_age_s)
            stale |= is_road & ~is_barrier & ~is_occupied & (self.seen_by_class[PointClass.ROAD][i, j] < self.time_s - max_age_s)
            result[stale] = CellLabel.UNKNOWN
        out[inside] = result
        return out

    def labels_block(self, i0, i1, j0, j1, max_age_s=None):
        """``labels`` for the cell block ``[i0:i1, j0:j1]`` (an (i1 - i0) x (j1 - j0) array),
        on array slices rather than gathered points: the same rule, a quarter of the time
        for the road model's 45 m window (fifth review, F5-05)."""
        c = self.config
        road, raised, obstacle, tall = (a[i0:i1, j0:j1] for a in (self.road, self.raised, self.obstacle, self.tall))
        is_barrier = (raised >= c.barrier_min_evidence) & (raised >= 0.5 * road) & (tall < 0.5 * raised)
        is_occupied = ~is_barrier & (obstacle >= c.barrier_min_evidence)
        is_road = ~is_barrier & ~is_occupied & (road >= c.road_min_evidence)
        result = np.full(road.shape, CellLabel.UNKNOWN, dtype=np.int8)
        result[is_road] = CellLabel.ROAD
        result[is_barrier] = CellLabel.BARRIER
        result[is_occupied] = CellLabel.OCCUPIED
        if max_age_s is not None and self.time_s is not None:
            limit = self.time_s - max_age_s
            stale = self.seen[i0:i1, j0:j1] < limit
            stale |= is_barrier & (self.seen_by_class[PointClass.RAISED][i0:i1, j0:j1] < limit)
            stale |= is_road & ~is_barrier & ~is_occupied & (self.seen_by_class[PointClass.ROAD][i0:i1, j0:j1] < limit)
            result[stale] = CellLabel.UNKNOWN
        return result

    def tall_evidence(self, xy, max_age_s=None):
        """Obstacle evidence kept as long as kerbs (``tall``) per odometry point, zero outside
        the grid or with no obstacle hit within ``max_age_s``: high where something has stood
        a while (a wall, a railing), low where something only passed through."""
        ij, inside = self.cells(xy)
        out = np.zeros(len(ij), dtype=np.float32)
        i, j = ij[inside, 0], ij[inside, 1]
        value = self.tall[i, j]
        if max_age_s is not None and self.time_s is not None:
            value = np.where(self.seen_by_class[PointClass.OBSTACLE][i, j] < self.time_s - max_age_s, 0.0, value)
        out[inside] = value
        return out

    def raised_evidence(self, xy, max_age_s=None):
        """Raised (kerb, pavement) evidence per odometry point, zero where no raised hit is
        within ``max_age_s`` or outside the grid: the kerb face itself, wherever a cell's road
        returns outvote it."""
        ij, inside = self.cells(xy)
        out = np.zeros(len(ij), dtype=np.float32)
        i, j = ij[inside, 0], ij[inside, 1]
        value = self.raised[i, j]
        # The same veto as ``labels``: a cell that also sees a tall body (a parked car's sill
        # under its side) is not a kerb (fresh-eyes review of 58adaf3: without it a car
        # parked flush with the exit kerb became the kerb, 1.9 m into the road).
        value = np.where(self.tall[i, j] >= 0.5 * value, 0.0, value)
        if max_age_s is not None and self.time_s is not None:
            value = np.where(self.seen_by_class[PointClass.RAISED][i, j] < self.time_s - max_age_s, 0.0, value)
        out[inside] = value
        return out

    def obstacle_evidence(self, xy):
        ij, inside = self.cells(xy)
        out = np.zeros(len(ij), dtype=np.float32)
        out[inside] = self.obstacle[ij[inside, 0], ij[inside, 1]]
        return out
