"""Evaluation-only polygon distances; never imported by the autonomy stack."""

import math

import numpy as np


def point_segment_distance(p, a, b):
    p, a, b = map(lambda x: np.asarray(x, dtype=float), (p, a, b))
    delta = b - a
    denom = float(delta @ delta)
    t = 0.0 if denom == 0 else float(np.clip((p - a) @ delta / denom, 0, 1))
    return float(np.linalg.norm(p - (a + t * delta)))


def segment_distance(a, b, c, d):
    def cross(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    if cross(a, b, c) * cross(a, b, d) < 0 and cross(c, d, a) * cross(c, d, b) < 0:
        return 0.0
    return min(
        point_segment_distance(a, c, d),
        point_segment_distance(b, c, d),
        point_segment_distance(c, a, b),
        point_segment_distance(d, a, b),
    )


def front_edge_gap(ego_front_edge, obstacle_polygon):
    """Euclidean front-bumper-edge to bounding-box perimeter distance (metres)."""
    if len(ego_front_edge) != 2 or len(obstacle_polygon) < 3:
        raise ValueError("Expected edge and polygon")
    if not all(math.isfinite(v) for p in (*ego_front_edge, *obstacle_polygon) for v in p):
        raise ValueError("Nonfinite geometry")
    return min(
        segment_distance(*ego_front_edge, a, b)
        for a, b in zip(obstacle_polygon, obstacle_polygon[1:] + obstacle_polygon[:1])
    )
