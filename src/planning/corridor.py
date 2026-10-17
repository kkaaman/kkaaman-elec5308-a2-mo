"""Rear-axle route coordinates and conservative swept-corridor intersection."""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RouteProjection:
    s_m: float
    lateral_m: float
    distance_m: float
    segment: int


def _clip(poly, axis, bound, sign):
    """Clip a convex polygon by sign * (coordinate - bound) >= 0."""
    result = []
    for a, b in zip(poly, poly[1:] + poly[:1]):
        da, db = sign * (a[axis] - bound), sign * (b[axis] - bound)
        if da >= -1e-9:
            result.append(a)
        if (da < 0) != (db < 0):
            t = da / (da - db)
            result.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return result


class RouteGeometry:
    """Copied open polyline. Positive lateral distance is left of travel.

    Obstacles are intersected with each swept segment, rather than projecting
    their centres onto a single infinite tangent. That matters after a bend.
    """

    def __init__(self, points_xy_m):
        p = np.asarray(points_xy_m, dtype=float)
        if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or not np.isfinite(p).all():
            raise ValueError("Route requires finite Nx2 points")
        delta = np.diff(p, axis=0)
        lengths = np.linalg.norm(delta, axis=1)
        if (lengths <= 1e-6).any() or np.linalg.norm(p[-1] - p[0]) < 1e-6:
            raise ValueError("Route must be open with distinct consecutive points")
        self.points = p.copy()
        self.lengths = lengths
        self.tangents = delta / lengths[:, None]
        self.normals = np.column_stack((-self.tangents[:, 1], self.tangents[:, 0]))
        self.arc = np.r_[0.0, np.cumsum(lengths)]
        self.length_m = float(self.arc[-1])
        for value in (self.points, self.lengths, self.tangents, self.normals, self.arc):
            value.setflags(write=False)

    def project(self, x, y, *, minimum_s=0.0, maximum_s=None):
        maximum_s = self.length_m if maximum_s is None else maximum_s
        if not all(math.isfinite(v) for v in (x, y, minimum_s, maximum_s)):
            raise ValueError("Projection values must be finite")
        if minimum_s > maximum_s or minimum_s > self.length_m or maximum_s < 0:
            raise ValueError("Empty projection interval")
        relative = np.array([x, y]) - self.points[:-1]
        along = np.einsum("ij,ij->i", relative, self.tangents)
        lower = np.maximum(0, minimum_s - self.arc[:-1])
        upper = np.minimum(self.lengths, maximum_s - self.arc[:-1])
        valid = lower <= upper
        along = np.clip(along, lower, np.maximum(lower, upper))
        residual = relative - along[:, None] * self.tangents
        distance = np.linalg.norm(residual, axis=1)
        distance[~valid] = np.inf
        i = int(np.argmin(distance))
        return RouteProjection(
            float(self.arc[i] + along[i]),
            float(residual[i] @ self.normals[i]),
            float(distance[i]),
            i,
        )

    def at(self, s_m, lateral_m=0.0):
        if not math.isfinite(s_m) or not math.isfinite(lateral_m):
            raise ValueError("Route coordinates must be finite")
        s = float(np.clip(s_m, 0, self.length_m))
        i = min(int(np.searchsorted(self.arc, s, side="right")) - 1, len(self.lengths) - 1)
        p = self.points[i] + (s - self.arc[i]) * self.tangents[i] + lateral_m * self.normals[i]
        return float(p[0]), float(p[1]), math.atan2(self.tangents[i, 1], self.tangents[i, 0])

    def curvature_ahead(self, s_m, horizon_m):
        """(distance, absolute curvature) at vertices; straight segments need no division."""
        if not math.isfinite(horizon_m) or horizon_m <= 0:
            raise ValueError("Positive horizon required")
        result = []
        for i in range(1, len(self.tangents)):
            distance = float(self.arc[i] - s_m)
            if -2 <= distance <= horizon_m:
                a, b = self.tangents[i - 1], self.tangents[i]
                turn = math.atan2(a[0] * b[1] - a[1] * b[0], float(a @ b))
                result.append(
                    (max(0.0, distance), abs(turn) / ((self.lengths[i - 1] + self.lengths[i]) / 2))
                )
        return tuple(result)

    def footprint_interval(self, corners_xy_m, *, half_width_m, minimum_s=0.0, maximum_s=None):
        """Arc interval overlapping the corridor, or None. No bumper offset here."""
        corners = np.asarray(corners_xy_m, dtype=float)
        if corners.shape != (4, 2) or not np.isfinite(corners).all():
            raise ValueError("Footprint requires four ordered finite XY corners")
        if not math.isfinite(half_width_m) or half_width_m <= 0:
            raise ValueError("Corridor half-width must be positive")
        maximum_s = self.length_m if maximum_s is None else maximum_s
        intervals = []
        for i, length in enumerate(self.lengths):
            lo, hi = max(0.0, minimum_s - self.arc[i]), min(length, maximum_s - self.arc[i])
            if lo > hi:
                continue
            relative = corners - self.points[i]
            poly = list(
                map(
                    tuple,
                    np.column_stack((relative @ self.tangents[i], relative @ self.normals[i])),
                )
            )
            for axis, bound, sign in (
                (0, lo, 1),
                (0, hi, -1),
                (1, -half_width_m, 1),
                (1, half_width_m, -1),
            ):
                if poly:
                    poly = _clip(poly, axis, bound, sign)
            if poly:
                intervals.append(
                    (
                        float(self.arc[i] + min(p[0] for p in poly)),
                        float(self.arc[i] + max(p[0] for p in poly)),
                    )
                )
        return (
            None if not intervals else (min(p[0] for p in intervals), max(p[1] for p in intervals))
        )

    def gap(self, footprint, ego_s_m, front_bumper_from_rear_axle_m, half_width_m):
        """Front-bumper gap; the rear-axle offset is applied exactly once."""
        interval = self.footprint_interval(
            footprint.corners_world_m, half_width_m=half_width_m, minimum_s=max(0.0, ego_s_m - 2.0)
        )
        if interval is None or interval[1] < ego_s_m:
            return None
        return interval[0] - ego_s_m - front_bumper_from_rear_axle_m
