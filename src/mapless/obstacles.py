"""Obstacles in the car's path, from this tick's LiDAR obstacle points.

The path is the corridor centreline; its tube is the car's half-width plus ``margin_m``
each side. Each obstacle point is projected onto the centreline, giving its distance
along the path from the rear axle and its lateral offset. The nearest place along the
path where ``min_points`` tube points fall within ``cluster_window_m`` is the obstacle;
fewer points than that are treated as noise.

Only the current sweep is used, never accumulated evidence, so an object that has moved
leaves no trail. Points beyond the end of the centreline are not judged: the corridor
does not say where the path goes there, and the planner never drives past its end.

``observe_path`` also says how far along the tube the road is *seen clear*: ground
points in 0.5 m slices, without an obstacle, from near the car outwards (unseen gaps
up to ``gap_m`` or ``gap_per_m`` of the distance allowed). Not seeing an obstacle is not
the same as seeing the road clear: an occluded or unlit obstacle produces neither, and a
stop is released only on positive clearance (review of 94b5caa; run e crept 0.6 m
towards a parked car between flickering detections).
"""

import math
from dataclasses import dataclass, replace

import numpy as np

from mapless.corridor import Corridor
from mapless.terrain import ClassifiedSweep, PointClass, VehicleBody


@dataclass(frozen=True)
class ObstacleConfig:
    margin_m: float = 0.35
    min_points: int = 6
    cluster_window_m: float = 1.0
    clear_bin_m: float = 0.5
    clear_min_points: int = 2  # ground points that make a lateral strip of a slice seen ...
    clear_strips: int = 3  # ... in the centre strip and at least clear_min_strips of this many across the tube
    clear_min_strips: int = 2
    raised_min_points: int = 12  # raised (kerb-high) points in the tube that count as an obstacle, up to raised_ref_m ...
    # ... then scaled with (raised_ref_m / distance)^2, the way the rings spread, but never
    # below raised_min_points_far. On the 32-channel roof LiDAR a 0.4 m, 0.25 m high brick
    # gives 17 raised points at 6 m from the axle, 5 at 9 m, 3 at 11 m and none past 20 m:
    # it is seen in time to stop at 2 m/s, not at 4 m/s.
    raised_ref_m: float = 5.0
    raised_min_points_far: int = 3
    raised_margin_m: float = 0.0  # the raised check's tube: the body's full width (the wheels run at its edges)
    clear_start_m: float = 8.0  # seen ground must begin within this distance of the rear axle
    gap_m: float = 1.0
    gap_per_m: float = 0.3

    def __post_init__(self):
        if not math.isfinite(self.margin_m) or self.margin_m < -0.5:
            raise ValueError("margin_m must be finite and not below -0.5")
        if self.min_points < 1 or self.cluster_window_m <= 0:
            raise ValueError("min_points and cluster_window_m must be positive")


@dataclass(frozen=True)
class PathObstacle:
    distance_m: float  # along the centreline, from the rear axle to the obstacle's nearest point
    lateral_m: float  # of that point from the centreline, +left
    points: int  # tube points in its cluster
    # The lateral span of the tube points within cluster_window_m of the nearest point:
    # the region a held stop must see clear before it is released.
    lateral_min_m: float | None = None
    lateral_max_m: float | None = None


def project_onto_polyline(points, polyline):
    """Arc length and signed lateral offset (+left) of each point on its nearest segment."""
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    line = np.asarray(polyline, dtype=float)
    a, b = line[:-1], line[1:]
    d = b - a
    length2 = np.maximum((d**2).sum(axis=1), 1e-12)
    rel = p[:, None, :] - a[None, :, :]
    t = np.clip((rel * d[None]).sum(axis=2) / length2[None], 0.0, 1.0)
    foot = a[None] + t[..., None] * d[None]
    dist2 = ((p[:, None, :] - foot) ** 2).sum(axis=2)
    k = np.argmin(dist2, axis=1)
    rows = np.arange(len(p))
    arcs = np.concatenate(([0.0], np.cumsum(np.sqrt(length2))))
    s = arcs[k] + t[rows, k] * np.sqrt(length2[k])
    cross = d[k, 0] * rel[rows, k, 1] - d[k, 1] * rel[rows, k, 0]
    lateral = np.sign(cross) * np.sqrt(dist2[rows, k])
    beyond = (k == len(a) - 1) & (t[rows, k] >= 1.0)
    before = (k == 0) & (t[rows, k] <= 0.0)
    return s, lateral, beyond | before


def _arcs(line):
    return np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(line, axis=0).T))))


def _curvature(line):
    """Signed curvature (1/m, +left) at each vertex of a polyline."""
    if len(line) < 3:
        return np.zeros(len(line))
    dx, dy = np.gradient(line[:, 0]), np.gradient(line[:, 1])
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    return (dx * ddy - dy * ddx) / np.maximum((dx**2 + dy**2) ** 1.5, 1e-9)


def tube_bounds(line, s, half_m, body):
    """Lateral bounds (right, left; +left) of the obstacle tube at arc lengths ``s`` on ``line``.

    On a bend the outer front corner sweeps body.front_m^2 * |k| / 2 outside the rear
    axle's path (0.59 m at 12 m radius): the tube widens by that on the outside, up to 1 m
    (full review of 73ae759: the corner ran 0.13-0.4 m outside the fixed tube).
    """
    kappa = np.interp(s, _arcs(line), _curvature(line))
    outer = np.minimum(body.front_m**2 * np.abs(kappa) / 2.0, 1.0)
    return -half_m - np.where(kappa > 0, outer, 0.0), half_m + np.where(kappa < 0, outer, 0.0)


def path_obstacle(obstacle_points_body, corridor, body, config=None, ref_m=None, floor=None):
    """Nearest obstacle in the corridor tube ahead of the rear axle, or None.

    ``ref_m``/``floor``: the cluster threshold is ``min_points * (ref_m / distance)^2``, at
    least ``floor`` (fewer points reach an object further away), instead of ``min_points``.
    """
    if not isinstance(corridor, Corridor) or not isinstance(body, VehicleBody):
        raise TypeError("path_obstacle(points, Corridor, VehicleBody, config)")
    c = config or ObstacleConfig()
    if not corridor.ok or len(obstacle_points_body) == 0:
        return None
    pts = np.asarray(obstacle_points_body, dtype=float)[:, :2]
    line = corridor.centre_body
    half = body.half_width_m + c.margin_m
    lo, hi = line.min(axis=0) - half - 1.0, line.max(axis=0) + half + 1.0
    near = np.all((pts >= lo) & (pts <= hi), axis=1)
    pts = pts[near]
    if len(pts) == 0:
        return None
    s, lateral, outside = project_onto_polyline(pts, line)
    s_car, _, _ = project_onto_polyline(np.zeros((1, 2)), line)
    ahead = s - s_car[0]
    right, left = tube_bounds(line, s, half, body)
    tube = (lateral >= right) & (lateral <= left) & ~outside & (ahead > 0)
    if tube.sum() < (c.min_points if floor is None else floor):
        return None
    order = np.argsort(ahead[tube])
    along, side = ahead[tube][order], lateral[tube][order]
    counts = np.searchsorted(along, along + c.cluster_window_m, side="right") - np.arange(len(along))
    need = c.min_points if ref_m is None else np.maximum(floor, np.minimum(c.min_points, c.min_points * (ref_m / np.maximum(along, 1.0)) ** 2))
    dense = np.flatnonzero(counts >= need)
    if dense.size == 0:
        return None
    i = int(dense[0])
    span = side[i : i + int(counts[i])]
    return PathObstacle(float(along[i]), float(side[i]), int(counts[i]), float(span.min()), float(span.max()))


@dataclass(frozen=True)
class PathObservation:
    obstacle: PathObstacle | None
    clear_to_m: float  # along the path from the rear axle, how far the tube is seen clear
    # This sweep's ground points in the tube (widened on bends, as ``tube_bounds``) as
    # (along from the rear axle, lateral), or None: the planner checks a held obstacle's
    # own region against them.
    ground: np.ndarray | None = None


def observe_path(sweep, corridor, body, config=None):
    """This sweep's view of the path: the nearest obstacle, and how far it is seen clear.

    Obstacles are OBSTACLE points (0.3 m up), and also RAISED points (kerb-high) when
    enough of them stand in the tube: a 20 cm object in the lane is not driven over. Kerbs
    alongside are outside the tube by construction (review of 2b7e14d).

    Seen clear needs ground in the centre strip and in at least ``clear_min_strips`` of
    ``clear_strips`` strips across the tube per slice: road seen beside an unobserved
    middle is not clearance (review of 697d53c: both outer strips certified an unseen centre).
    """
    if not isinstance(sweep, ClassifiedSweep):
        raise TypeError("sweep must be a ClassifiedSweep")
    c = config or ObstacleConfig()
    obstacle = path_obstacle(sweep.of(PointClass.OBSTACLE), corridor, body, c)
    # Raised points count only under the car's own body (no margin): a kerb beside the path
    # is the road's edge, not an obstacle; a brick under a wheel is. The band was the inner
    # 0.85 m and 12 points at any range: a 0.4 m brick under the left wheel was never seen
    # and was hit at 2 m/s (full review of 73ae759). Now the full body width, and a
    # threshold that falls with range as the rings spread.
    low = path_obstacle(
        sweep.of(PointClass.RAISED), corridor, body, replace(c, min_points=c.raised_min_points, margin_m=c.raised_margin_m),
        ref_m=c.raised_ref_m, floor=c.raised_min_points_far,
    )
    if low is not None and (obstacle is None or low.distance_m < obstacle.distance_m):
        obstacle = low
    if not corridor.ok:
        return PathObservation(obstacle, 0.0)
    line = corridor.centre_body
    ground = sweep.of(PointClass.ROAD)[:, :2]
    half = body.half_width_m + c.margin_m
    lo, hi = line.min(axis=0) - half - 1.0, line.max(axis=0) + half + 1.0
    ground = ground[np.all((ground >= lo) & (ground <= hi), axis=1)]
    clear = 0.0
    in_tube = np.zeros((0, 2))
    if len(ground):
        s, lateral, outside = project_onto_polyline(ground, line)
        s_car, _, _ = project_onto_polyline(np.zeros((1, 2)), line)
        right, left = tube_bounds(line, s, half, body)
        wide = (lateral >= right) & (lateral <= left) & ~outside & (s >= s_car[0])
        in_tube = np.column_stack((s[wide] - s_car[0], lateral[wide]))
        inside = (np.abs(lateral) <= half) & ~outside
        along, lateral = (s - s_car[0])[inside], lateral[inside]
        limit = obstacle.distance_m if obstacle is not None else math.inf
        keep = (along >= 0) & (along < limit)
        along, lateral = along[keep], lateral[keep]
        if len(along):
            bins = np.floor(along / c.clear_bin_m).astype(int)
            strips = np.clip(((lateral + half) / (2 * half) * c.clear_strips).astype(int), 0, c.clear_strips - 1)
            counts = np.zeros((bins.max() + 1, c.clear_strips), dtype=int)
            np.add.at(counts, (bins, strips), 1)
            strip_seen = counts >= c.clear_min_points
            centre = strip_seen[:, c.clear_strips // 2]  # the car's own track must be seen
            seen = np.flatnonzero(centre & (strip_seen.sum(axis=1) >= c.clear_min_strips)) * c.clear_bin_m
            if seen.size and seen[0] <= c.clear_start_m:
                clear = float(seen[0])
                for x in seen[1:]:
                    if x - clear > max(c.gap_m, c.gap_per_m * x):
                        break
                    clear = float(x)
                clear = min(clear + c.clear_bin_m, limit)
    return PathObservation(obstacle, clear, in_tube)
