"""Polyline validation, projection and arc-length interpolation."""

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Iterable

from ._validation import clamp, finite, positive
from .types import Waypoint


@dataclass(frozen=True)
class Projection:
    """Nearest point on a polyline, its arc length and Euclidean distance."""

    progress_m: float
    point: Waypoint
    distance_m: float


class ReferencePath:
    """Copied forward-only open polyline; consecutive duplicates are removed.

    Smoothness and absence of self-intersections are caller responsibilities.
    Exactly closed paths are rejected. Linear scans favour clarity for short
    course routes over the complexity of a spatial index.
    """

    def __init__(self, waypoints: Iterable[Waypoint]) -> None:
        points = []
        for point in waypoints:
            if not isinstance(point, Waypoint):
                raise TypeError("path entries must be Waypoint instances")
            if not points or point != points[-1]:
                points.append(point)
        if len(points) < 2:
            raise ValueError("path must contain at least two different points")
        if points[0] == points[-1]:
            raise ValueError("closed paths are not supported by this MVP")
        lengths = [0.0]
        for a, b in zip(points, points[1:]):
            segment = math.hypot(b.x_m - a.x_m, b.y_m - a.y_m)
            positive("path segment length", segment)
            total = lengths[-1] + segment
            finite("path length", total)
            if total <= lengths[-1]:
                raise ValueError("path segment too small relative to total path length")
            lengths.append(total)
        self.points = tuple(points)
        self.arc_lengths = tuple(lengths)
        self.length_m = lengths[-1]

    def point_at(self, progress_m: float) -> Waypoint:
        """Interpolate at a finite arc length, clamping beyond either endpoint."""
        finite("progress_m", progress_m)
        s = clamp(progress_m, 0.0, self.length_m)
        index = min(bisect_right(self.arc_lengths, s) - 1, len(self.points) - 2)
        a, b = self.points[index : index + 2]
        length = self.arc_lengths[index + 1] - self.arc_lengths[index]
        ratio = (s - self.arc_lengths[index]) / length
        return Waypoint(a.x_m + ratio * (b.x_m - a.x_m), a.y_m + ratio * (b.y_m - a.y_m))

    def project(self, x_m: float, y_m: float) -> Projection:
        """Project onto all segments; equal-distance ties favour earlier progress."""
        finite("x_m", x_m)
        finite("y_m", y_m)
        best = None
        for index, (a, b) in enumerate(zip(self.points, self.points[1:])):
            length = self.arc_lengths[index + 1] - self.arc_lengths[index]
            ux, uy = (b.x_m - a.x_m) / length, (b.y_m - a.y_m) / length
            along = (x_m - a.x_m) * ux + (y_m - a.y_m) * uy
            finite("path projection", along)
            along = clamp(along, 0.0, length)
            point = Waypoint(a.x_m + along * ux, a.y_m + along * uy)
            distance = math.hypot(x_m - point.x_m, y_m - point.y_m)
            finite("distance to path", distance)
            if best is None or distance < best.distance_m:
                best = Projection(self.arc_lengths[index] + along, point, distance)
        assert best is not None
        return best
