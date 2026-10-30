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

**Far to the side, along the path.** The bands along x describe the road straight ahead;
a point far to one side has only its x in common with them. In a turn the road straight
ahead is not the road the car turns into: the profile followed a street dipping 0.1-0.7 m
beyond a Town03 junction, and the exit road beside it, level with the car, read as kerb
and as obstacles; the exit line was read 1.2 m into the road and the latched turn re-planned
twice (PC 1, r11_left_parked_lidargap, 84fb071). Given the path the car is driving (the
last tick's corridor, ``path_xy``), points ``path_profile_min_range_m`` or more away and
outside the bands' window are referenced to a profile grown the same way along the path
(its arc length and distance from it). On a straight road the two are the same; the near
field, where the corridor reads its kerbs, is always the bands'.

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
    # Far points referenced along the path (module docstring): from this far from the car,
    # outside road_window_m; the path resampled every path_profile_step_m and looked up on a
    # path_profile_cell_m raster; used when it leaves the bands' window by path_profile_min_offset_m.
    path_profile: bool = True
    path_profile_min_range_m: float = 10.0
    path_profile_step_m: float = 2.0
    path_profile_cell_m: float = 2.0
    path_profile_min_offset_m: float = 1.0

    def __post_init__(self):
        for name in ("cell_m", "size_m", "band_m", "static_tau_s", "obstacle_tau_s", "max_range_m"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.raised_min_m < self.obstacle_min_m < self.overhead_m:
            raise ValueError("Height thresholds must increase: raised < obstacle < overhead")
        if self.recentre_m + self.max_range_m > self.size_m / 2:
            raise ValueError("The grid must hold max_range_m around the car wherever it recentres")
        if self.path_profile_step_m <= 0 or self.path_profile_cell_m <= 0:
            raise ValueError("path_profile_step_m and path_profile_cell_m must be positive")


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
    the pavement for the road). ``xyz``'s first two columns may be any (along, across)
    coordinates: ``path_reference`` grows the same profile along a path.
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


def _path_stations(path_xy, config):
    """The path (vehicle frame) every ``path_profile_step_m``, straight back from its first
    point and on from its last as far as the sensor reaches, with arc lengths from the
    station nearest the rear axle (s = 0 there); None for a path of fewer than two points."""
    c = config
    p = np.asarray(path_xy, dtype=float).reshape(-1, 2)
    seg = np.hypot(*np.diff(p, axis=0).T) if len(p) >= 2 else np.zeros(0)
    keep = np.concatenate(([True], seg > 1e-6)) if len(p) >= 2 else np.ones(len(p), dtype=bool)
    p = p[keep]
    if len(p) < 2:
        return None
    arc = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))))
    step = c.path_profile_step_m
    along = np.arange(0.0, arc[-1] + 1e-9, step)
    pts = np.column_stack((np.interp(along, arc, p[:, 0]), np.interp(along, arc, p[:, 1])))
    reach = c.max_range_m + c.road_window_m
    back_dir = (p[0] - p[1]) / np.hypot(*(p[0] - p[1]))
    on_dir = (p[-1] - p[-2]) / np.hypot(*(p[-1] - p[-2]))
    extra = np.arange(step, 2.0 * reach + 1e-9, step)
    back = p[0] + extra[:, None] * back_dir
    on = pts[-1] + extra[:, None] * on_dir
    xy = np.vstack((back[::-1], pts, on))
    s_all = np.concatenate((-extra[::-1], along, along[-1] + extra))
    near = np.hypot(xy[:, 0], xy[:, 1]) <= reach + step
    if near.sum() < 2:
        return None
    xy, s_all = xy[near], s_all[near]
    return xy, s_all - s_all[int(np.argmin(np.hypot(xy[:, 0], xy[:, 1])))]


_RASTERS = {}


def _raster_centres(config):
    """The ``path_profile_cell_m`` raster's cell centres over the sensor's square (cached)."""
    c = config
    key = (c.path_profile_cell_m, c.max_range_m)
    if key not in _RASTERS:
        n = int(math.ceil(2.0 * c.max_range_m / c.path_profile_cell_m))
        centres = ((np.arange(n) + 0.5) * c.path_profile_cell_m - c.max_range_m).astype(np.float32)
        gx, gy = np.meshgrid(centres, centres, indexing="ij")
        _RASTERS[key] = (n, gx.ravel(), gy.ravel())
    return _RASTERS[key]


def path_reference(xyz, path_xy, config):
    """Road height for each point along the path ``path_xy`` (vehicle frame): ``road_profile``
    grown over (arc length along the path, distance from it), read at each point's arc
    length; with the arc lengths and distances. None when the path stays within
    ``path_profile_min_offset_m`` of the x axis out to the sensor's reach (the bands along x
    are then the same profile) or grows no band beyond the car's.

    Each point takes the station nearest its ``path_profile_cell_m`` raster cell, then the
    nearer of the two path segments either side of that station."""
    c = config
    built = _path_stations(path_xy, c)
    if built is None:
        return None
    q, s_q = built
    ahead = (q[:, 0] >= 0.0) & (q[:, 0] <= c.max_range_m)
    if not ahead.any() or float(np.abs(q[ahead, 1]).max()) < c.path_profile_min_offset_m:
        return None
    n, gx, gy = _raster_centres(c)
    qx, qy = q[:, 0].astype(np.float32), q[:, 1].astype(np.float32)
    d2 = (gx[:, None] - qx[None, :]) ** 2 + (gy[:, None] - qy[None, :]) ** 2
    nearest = np.argmin(d2, axis=1)
    px, py = xyz[:, 0].astype(np.float32), xyz[:, 1].astype(np.float32)
    i = np.clip(((px + c.max_range_m) / c.path_profile_cell_m).astype(np.int64), 0, n - 1)
    j = np.clip(((py + c.max_range_m) / c.path_profile_cell_m).astype(np.int64), 0, n - 1)
    k = nearest[i * n + j]
    sx, sy = np.diff(qx), np.diff(qy)
    length2 = np.maximum(sx * sx + sy * sy, 1e-9)
    ds = np.diff(s_q).astype(np.float32)
    best = np.full(len(px), np.inf, dtype=np.float32)
    s_out = np.zeros(len(px), dtype=np.float32)
    for lo in (k - 1, k):
        a = np.clip(lo, 0, len(q) - 2)
        ax, ay, ux, uy = qx[a], qy[a], sx[a], sy[a]
        f = np.clip(((px - ax) * ux + (py - ay) * uy) / length2[a], 0.0, 1.0)
        ex, ey = px - ax - f * ux, py - ay - f * uy
        e2 = ex * ex + ey * ey
        better = e2 < best
        best = np.where(better, e2, best)
        s_out = np.where(better, s_q[a].astype(np.float32) + f * ds[a], s_out)
    d_out = np.sqrt(best)
    centres_s, heights = road_profile(np.column_stack((s_out, d_out, xyz[:, 2])), c)
    ok = np.isfinite(heights)
    if ok.sum() < 3:
        return None
    return np.interp(s_out, centres_s[ok], heights[ok]), s_out, d_out


def classify_sweep(xyz, body, config, path_xy=None):
    """Label one sweep's points (vehicle frame, metres). ``path_xy``: the path the car is
    driving (vehicle frame), for the points far to the side (module docstring)."""
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
    if path_xy is not None and c.path_profile and len(xyz):
        along = path_reference(xyz, path_xy, c)
        if along is not None:
            far = (np.abs(xyz[:, 1]) >= c.road_window_m) & (np.hypot(xyz[:, 0], xyz[:, 1]) >= c.path_profile_min_range_m)
            road_z = np.where(far, along[0], road_z)
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
        self._still_since = None  # the last sweep's time before the current standstill began
        self._seen_before = None  # per class, the cells hit before it (a standstill's scenery)

    def reset(self):
        """Forget every observation: the odometry frame they were placed in is no longer sound."""
        for name in ("road", "raised", "obstacle", "tall"):
            getattr(self, name).fill(0.0)
        self.seen.fill(-np.inf)
        for grid in self.seen_by_class.values():
            grid.fill(-np.inf)
        self.origin = np.array([-self.config.size_m / 2, -self.config.size_m / 2])
        self.time_s = None
        self._still_since = None
        self._seen_before = None

    def _centre(self):
        return self.origin + self.config.size_m / 2

    def _recentre(self, x, y):
        c = self.config
        self._seen_before = None  # not shifted with the arrays: a standstill's transient evidence keeps, as before
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
        as usual, so their evidence stays bounded; obstacle evidence always decays. So does
        evidence of a class first added *during* the standstill to a cell already observed
        (it had none of that class before) that this sweep does not cover: nothing static
        appears and vanishes while the car stands, something moved through. A cell first
        observed during the standstill (the whole grid at start-up) keeps, as before. An oncoming car passing a car
        waiting to turn left left raised hits from its wheels in the opposing lane, kept as
        kerb for good, and no turn ever cleared them (PC Temp, Town03, dd9344c: held 25 s).
        Scenery seen before the standstill and hit now and then since (a ring's edge) keeps.
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
        if not stationary:
            self._still_since = self._seen_before = None
        elif self._still_since is None:
            self._still_since = self.time_s if self.time_s is not None else time_s
            self._seen_before = {k: np.isfinite(v) for k, v in self.seen_by_class.items()}
            self._seen_before["any"] = np.isfinite(self.seen)
        if self.time_s is not None:
            dt = max(0.0, time_s - self.time_s)
            static = math.exp(-dt / c.static_tau_s)
            for point_class, grid in ((PointClass.ROAD, self.road), (PointClass.RAISED, self.raised), (PointClass.OBSTACLE, self.tall)):
                if stationary and self._seen_before is not None:
                    decaying = (self.seen_by_class[point_class] > self._still_since) & ~self._seen_before[point_class] & self._seen_before["any"]
                    decaying.ravel()[covered] = True
                    grid[decaying] *= static
                elif stationary:
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
