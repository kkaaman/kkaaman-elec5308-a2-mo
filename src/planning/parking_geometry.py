"""Small convex geometry helpers for bounded P1 full-body sweeps."""

import math

import numpy as np


def polygon(value):
    """Copy and validate a strictly convex CCW polygon (no repeated vertices)."""
    p = np.asarray(value, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all():
        raise ValueError("Expected finite convex Nx2 polygon")
    edges = np.roll(p, -1, axis=0) - p
    # Every other vertex must lie strictly to the left of each directed edge.
    for i, (a, e) in enumerate(zip(p, edges)):
        rel = p - a
        cross = e[0] * rel[:, 1] - e[1] * rel[:, 0]
        other = [j for j in range(len(p)) if j not in (i, (i + 1) % len(p))]
        if np.any(cross[other] <= 1e-9):
            raise ValueError("Polygon must be strictly convex and counterclockwise")
    return tuple(map(tuple, p))


def rectangle(x, y, yaw, front, rear, half_width, margin=0.0):
    points = np.array([
        (-rear - margin, -half_width - margin),
        (front + margin, -half_width - margin),
        (front + margin, half_width + margin),
        (-rear - margin, half_width + margin),
    ])
    c, s = math.cos(yaw), math.sin(yaw)
    return tuple(map(tuple, points @ np.array([[c, s], [-s, c]]) + (x, y)))


def footprint(state, vehicle, margin=0.0):
    return rectangle(
        state.x_m, state.y_m, state.yaw_rad,
        vehicle.front_m, vehicle.rear_m, vehicle.half_width_m, margin,
    )


def bay_polygon(bay):
    return rectangle(bay.x_m, bay.y_m, bay.yaw_rad, bay.length_m / 2,
                     bay.length_m / 2, bay.width_m / 2)


def contains(outer, inner):
    p, q = np.asarray(outer), np.asarray(inner)
    for a, b in zip(p, np.roll(p, -1, axis=0)):
        edge, rel = b - a, q - a
        if np.any(edge[0] * rel[:, 1] - edge[1] * rel[:, 0] < -1e-8):
            return False
    return True


def intersects(a, b):
    """Separating-axis test; touching counts as contact."""
    a, b = np.asarray(a), np.asarray(b)
    for p in (a, b):
        edges = np.roll(p, -1, axis=0) - p
        for e in edges:
            axis = np.array([-e[1], e[0]])
            ap, bp = a @ axis, b @ axis
            if ap.max() < bp.min() - 1e-9 or bp.max() < ap.min() - 1e-9:
                return False
    return True


def swept_rectangle(control_points, vehicle, padding):
    """Enclose the complete Bezier body sweep, not only endpoint rectangles.

    The curve lies in its control hull and its derivative in the derivative
    hull. Positive derivative projection bounds tangent deviation from the
    chord. A rotating corner moves <=2*r*sin(deviation/2), used on every side.
    """
    p = np.asarray(control_points)
    chord = p[-1] - p[0]
    yaw = math.atan2(chord[1], chord[0])
    c, s = math.cos(yaw), math.sin(yaw)
    local = (p - p[0]) @ np.array([[c, -s], [s, c]])
    derivatives = np.diff(local, axis=0)
    if np.any(derivatives[:, 0] <= 0):
        raise ValueError("Cannot bound a reversing/stationary tangent")
    deviation = max(abs(np.arctan2(derivatives[:, 1], derivatives[:, 0])))
    pad = padding + 2 * vehicle.corner_radius_m * math.sin(deviation / 2)
    low, high = local.min(axis=0), local.max(axis=0)
    centre_y = (low[1] + high[1]) / 2
    origin = p[0] + np.array([-s, c]) * centre_y
    return rectangle(
        *origin, yaw, high[0] + vehicle.front_m,
        vehicle.rear_m - low[0], (high[1] - low[1]) / 2 + vehicle.half_width_m, pad,
    )
