"""Radar returns as measurements for the tracker (plan v2.1, section 10). G3 framework.

A return is ``(range_m, azimuth_rad, elevation_rad, range_rate_mps)`` in the sensor's
polar frame, range rate positive when the range grows (the sign CARLA 0.9.16 reports,
measured by the rig check). ``mapless.frame.RadarReturns.points_ego`` places returns
in the vehicle frame; this module turns them into what a track can be updated with:

- **Ego-motion compensation.** A static object's range rate is the ego's own velocity
  at the sensor, projected on the line of sight. The sensor's velocity includes the
  lever arm: ``v_sensor = v_ego + omega x r`` (section 5, case 4). What is left after
  subtracting it is the object's own radial speed.
- **Clustering.** Returns within ``cluster_m`` of each other and within
  ``cluster_rate_mps`` in compensated radial speed are one object. Clustering is
  conservative: a wide cluster is split rather than merged, and a cluster never adds
  LiDAR-like extent (a radar return has no size).

Nothing here decides anything: an empty radar frame is reported as empty, never as
"lane clear".
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RadarConfig:
    cluster_m: float = 1.5  # returns this close in the plane are one object ...
    cluster_rate_mps: float = 1.0  # ... when their radial speeds agree this well
    min_returns: int = 1  # returns a cluster needs (1: every return counts, conservatively)
    max_cluster_span_m: float = 6.0  # wider clusters are split along their long axis
    bearing_sigma_rad: float = math.radians(2.0)  # the tracker's measurement noise is TrackerConfig.radar_*_sigma
    # Returns below this height in the vehicle frame (z up from the actor origin, at the
    # road) are the road itself: a bumper radar with a 15 degree vertical field hits the
    # tarmac from 5 m out, and on the first full live run (25 Oct) those returns became a
    # "lead" 5 m ahead that held the car to 0.3-1 m/s for the whole route.
    min_height_m: float = 0.3
    max_height_m: float = 3.0  # above: gantries, trees, overhead signs

    def __post_init__(self):
        if self.cluster_m <= 0 or self.cluster_rate_mps <= 0 or self.min_returns < 1:
            raise ValueError("RadarConfig: positive cluster sizes and min_returns >= 1")
        if self.max_height_m <= self.min_height_m:
            raise ValueError("RadarConfig: max_height_m must exceed min_height_m")


@dataclass(frozen=True)
class RadarMeasurement:
    """One clustered radar object in the vehicle frame."""

    sensor: str
    xy: np.ndarray  # vehicle frame
    range_m: float
    bearing_rad: float  # from the sensor, in the vehicle frame (+ left)
    radial_speed_mps: float  # the object's own, ego motion removed (+ away from the sensor)
    returns: int
    sensor_xy: np.ndarray  # the sensor's position in the vehicle frame


def sensor_pose(sensor_to_ego):
    """(x, y, yaw) of a sensor from its 4x4 sensor_to_ego transform."""
    m = np.asarray(sensor_to_ego, dtype=float)
    return float(m[0, 3]), float(m[1, 3]), float(math.atan2(m[1, 0], m[0, 0]))


def ego_radial_speed(points_ego, sensor_xy, ego_speed_mps, yaw_rate_rps):
    """The range rate a static object at each point would show, from the ego's own motion.

    ``v_sensor = v_ego + omega x r`` with ``v_ego = (speed, 0)`` at the rear axle and
    ``r`` the sensor's lever arm; the range rate of a static point is ``-v_sensor . u``
    with ``u`` the unit line of sight from the sensor (the point approaches at the
    speed the sensor moves towards it).
    """
    sx, sy = float(sensor_xy[0]), float(sensor_xy[1])
    v_sensor = np.array([ego_speed_mps - yaw_rate_rps * sy, yaw_rate_rps * sx])
    p = np.asarray(points_ego, dtype=float).reshape(-1, 2) - (sx, sy)
    r = np.hypot(p[:, 0], p[:, 1])
    u = p / np.maximum(r, 1e-6)[:, None]
    return -(u @ v_sensor)


def compensate(radar, ego_speed_mps, yaw_rate_rps, config=None):
    """Returns in the vehicle frame with the object's own radial speed: N x 3 (x, y, rate).

    Returns from the road surface (below ``min_height_m``) and from overhead structure
    (above ``max_height_m``) are dropped: neither is a road user.
    """
    c = config or RadarConfig()
    xyz = radar.points_ego()
    keep = (xyz[:, 2] >= c.min_height_m) & (xyz[:, 2] <= c.max_height_m)
    pts = xyz[keep, :2]
    sx, sy, _ = sensor_pose(radar.sensor_to_ego)
    own = ego_radial_speed(pts, (sx, sy), ego_speed_mps, yaw_rate_rps)
    rate = np.asarray(radar.returns[keep, 3], dtype=float) - own
    return np.column_stack((pts, rate)), np.array([sx, sy])


def _split_wide(members, xy, max_span):
    """Split a cluster along its long axis until every part spans at most ``max_span``."""
    stack, out = [members], []
    while stack:
        idx = stack.pop()
        p = xy[idx]
        if len(idx) < 2:
            out.append(idx)
            continue
        centred = p - p.mean(axis=0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        along = centred @ vt[0]
        if np.ptp(along) <= max_span:
            out.append(idx)
            continue
        cut = float(np.median(along))
        a, b = idx[along <= cut], idx[along > cut]
        if not len(a) or not len(b):
            out.append(idx)
            continue
        stack.extend([a, b])
    return out


def cluster_returns(compensated, sensor_xy, name, config=None):
    """Clusters of compensated returns (N x 3: x, y, radial speed) as RadarMeasurements.

    Single-linkage in the plane within ``cluster_m`` and in radial speed within
    ``cluster_rate_mps``, then wide clusters split (two cars side by side are two).
    """
    c = config or RadarConfig()
    data = np.asarray(compensated, dtype=float).reshape(-1, 3)
    n = len(data)
    if n == 0:
        return []
    xy, rate = data[:, :2], data[:, 2]
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    close = (d <= c.cluster_m) & (np.abs(rate[:, None] - rate[None, :]) <= c.cluster_rate_mps)
    for i, j in zip(*np.nonzero(np.triu(close, 1))):
        parent[find(i)] = find(j)
    roots = np.array([find(i) for i in range(n)])
    out = []
    for root in np.unique(roots):
        members = np.flatnonzero(roots == root)
        for part in _split_wide(members, xy, c.max_cluster_span_m):
            if len(part) < c.min_returns:
                continue
            centre = xy[part].mean(axis=0)
            rel = centre - sensor_xy
            out.append(
                RadarMeasurement(
                    sensor=name,
                    xy=centre,
                    range_m=float(np.hypot(*rel)),
                    bearing_rad=float(math.atan2(rel[1], rel[0])),
                    radial_speed_mps=float(np.median(rate[part])),
                    returns=int(len(part)),
                    sensor_xy=np.asarray(sensor_xy, dtype=float),
                )
            )
    return out


def measure(radar, ego_speed_mps, yaw_rate_rps, config=None):
    """RadarMeasurements from one ``RadarReturns``; an empty frame gives an empty list."""
    compensated, sensor_xy = compensate(radar, ego_speed_mps, yaw_rate_rps, config)
    return cluster_returns(compensated, sensor_xy, radar.name, config)
