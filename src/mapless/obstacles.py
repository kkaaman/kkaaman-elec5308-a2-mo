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
from dataclasses import dataclass

import numpy as np

from mapless.corridor import Corridor
from mapless.terrain import ClassifiedSweep, PointClass, VehicleBody


@dataclass(frozen=True)
class ObstacleConfig:
    margin_m: float = 0.35
    min_points: int = 6
    cluster_window_m: float = 1.0
    clear_bin_m: float = 0.5
    clear_min_points: int = 2  # ground points that make a slice of the tube seen
    clear_start_m: float = 8.0  # seen ground must begin within this distance of the rear axle
    gap_m: float = 1.0
    gap_per_m: float = 0.3

    def __post_init__(self):
        if not math.isfinite(self.margin_m) or self.margin_m < 0:
            raise ValueError("margin_m must be non-negative")
        if self.min_points < 1 or self.cluster_window_m <= 0:
            raise ValueError("min_points and cluster_window_m must be positive")


@dataclass(frozen=True)
class PathObstacle:
    distance_m: float  # along the centreline, from the rear axle to the obstacle's nearest point
    lateral_m: float  # of that point from the centreline, +left
    points: int  # tube points in its cluster


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


def path_obstacle(obstacle_points_body, corridor, body, config=None):
    """Nearest obstacle in the corridor tube ahead of the rear axle, or None."""
    if not isinstance(corridor, Corridor) or not isinstance(body, VehicleBody):
        raise TypeError("path_obstacle(points, Corridor, VehicleBody, config)")
    c = config or ObstacleConfig()
    if not corridor.ok or len(obstacle_points_body) == 0:
        return None
    pts = np.asarray(obstacle_points_body, dtype=float)[:, :2]
    line = corridor.centre_body
    half = body.half_width_m + c.margin_m
    lo, hi = line.min(axis=0) - half, line.max(axis=0) + half
    near = np.all((pts >= lo) & (pts <= hi), axis=1)
    pts = pts[near]
    if len(pts) == 0:
        return None
    s, lateral, outside = project_onto_polyline(pts, line)
    s_car, _, _ = project_onto_polyline(np.zeros((1, 2)), line)
    ahead = s - s_car[0]
    tube = (np.abs(lateral) <= half) & ~outside & (ahead > 0)
    if tube.sum() < c.min_points:
        return None
    order = np.argsort(ahead[tube])
    along, side = ahead[tube][order], lateral[tube][order]
    counts = np.searchsorted(along, along + c.cluster_window_m, side="right") - np.arange(len(along))
    dense = np.flatnonzero(counts >= c.min_points)
    if dense.size == 0:
        return None
    i = int(dense[0])
    return PathObstacle(float(along[i]), float(side[i]), int(counts[i]))


@dataclass(frozen=True)
class PathObservation:
    obstacle: PathObstacle | None
    clear_to_m: float  # along the path from the rear axle, how far the tube is seen clear


def observe_path(sweep, corridor, body, config=None):
    """This sweep's view of the path: the nearest obstacle, and how far it is seen clear."""
    if not isinstance(sweep, ClassifiedSweep):
        raise TypeError("sweep must be a ClassifiedSweep")
    c = config or ObstacleConfig()
    obstacle = path_obstacle(sweep.of(PointClass.OBSTACLE), corridor, body, c)
    if not corridor.ok:
        return PathObservation(obstacle, 0.0)
    line = corridor.centre_body
    ground = sweep.of(PointClass.ROAD)[:, :2]
    half = body.half_width_m + c.margin_m
    lo, hi = line.min(axis=0) - half, line.max(axis=0) + half
    ground = ground[np.all((ground >= lo) & (ground <= hi), axis=1)]
    clear = 0.0
    if len(ground):
        s, lateral, outside = project_onto_polyline(ground, line)
        s_car, _, _ = project_onto_polyline(np.zeros((1, 2)), line)
        along = (s - s_car[0])[(np.abs(lateral) <= half) & ~outside]
        limit = obstacle.distance_m if obstacle is not None else math.inf
        along = along[(along >= 0) & (along < limit)]
        if len(along):
            bins = np.floor(along / c.clear_bin_m).astype(int)
            counts = np.bincount(bins)
            seen = np.flatnonzero(counts >= c.clear_min_points) * c.clear_bin_m
            if seen.size and seen[0] <= c.clear_start_m:
                clear = float(seen[0])
                for x in seen[1:]:
                    if x - clear > max(c.gap_m, c.gap_per_m * x):
                        break
                    clear = float(x)
                clear = min(clear + c.clear_bin_m, limit)
    return PathObservation(obstacle, clear)
