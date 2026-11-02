"""The painted stop line across the car's lane, from the front camera (G4 M07, 2 Nov).

The car holds for a red light at the stop line, and the only place a mapless car knows
for the junction is the sat-nav's entry point: with the route's 2 m noise it was 3.9 m
early or late on the G2 seeds at Town10HD 468 (r19s), and the painted line there sits at
the junction's entry. So the line is looked for in the image while a red or yellow is in
force.

The ground ahead in the car's lane (``lane_half_m`` either side of its centre line,
``range_m`` ahead of the rear axle) is sampled through the camera's calibration onto a
grid in the car's frame (an inverse perspective view of flat ground). A stop line is a
bright band across the lane: rows whose brightness stands ``contrast`` above the road's
median over at least ``cover`` of the lane's width, ``thickness_m`` deep along it. The
lane's own markings run along it and cover a sliver of each row; a zebra crossing's bars
cover about half; an arrow, a fraction. The nearest such band is the line; its distance
is measured to its near edge.

``StopLineTracker`` keeps the detections in the odometry frame (the car moves between
them) and locks the line once several agree while the car has moved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StopLineConfig:
    range_m: tuple = (6.0, 25.0)  # ahead of the rear axle (the far rows are a few pixels deep)
    step_m: float = 0.1
    lane_half_m: float = 1.2  # across the lane, either side of the car's centre line
    lateral_samples: int = 25
    contrast: float = 45.0  # grey levels above the road's median
    cover: float = 0.75  # share of the lane's width a row must be bright across
    thickness_m: tuple = (0.3, 1.1)  # a band's depth along the lane (a row each side added)


def _grey(rgb):
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def detect(rgb, calibration, config=None):
    """Distance (m, ahead of the rear axle) to the near edge of a stop line across the
    lane in ``rgb`` (H x W x 3, uint8), or None. ``calibration`` carries ``fx_px``,
    ``fy_px``, ``cx_px``, ``cy_px`` and ``camera_to_ego`` (camera: x right, y down, z
    forward; ego: x forward, y left, z up, ground at z 0)."""
    c = config or StopLineConfig()
    xs = np.arange(c.range_m[0], c.range_m[1], c.step_m)
    ys = np.linspace(-c.lane_half_m, c.lane_half_m, c.lateral_samples)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    ego = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size), np.ones(gx.size)])
    m = np.asarray(calibration.camera_to_ego, dtype=float)
    cam = np.linalg.inv(m) @ ego
    z = cam[2]
    ok = z > 0.5
    u = np.full(z.shape, -1.0)
    v = np.full(z.shape, -1.0)
    u[ok] = calibration.fx_px * cam[0, ok] / z[ok] + calibration.cx_px
    v[ok] = calibration.fy_px * cam[1, ok] / z[ok] + calibration.cy_px
    h, w = rgb.shape[:2]
    inside = ok & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
    grey = np.full(z.shape, np.nan)
    grey[inside] = _grey(rgb[v[inside].astype(int), u[inside].astype(int)].astype(float))
    grid = grey.reshape(gx.shape)
    if np.isnan(grid).mean() > 0.5:
        return None
    road = np.nanmedian(grid)
    bright = np.where(np.isnan(grid), False, grid > road + c.contrast)
    # Each column bright within a row either side: a line a few degrees off square to the
    # car, or crossed by a crack, still covers its rows (a1rec at 6-10 m: 0.44-0.6 single-row).
    near = bright.copy()
    near[1:] |= bright[:-1]
    near[:-1] |= bright[1:]
    seen = ~np.isnan(grid)
    cover = (near & seen).sum(axis=1) / np.maximum(seen.sum(axis=1), 1)
    rows = np.flatnonzero(cover >= c.cover)
    if rows.size == 0:
        return None
    # Bands of consecutive rows; the nearest one of a stop line's depth.
    breaks = np.flatnonzero(np.diff(rows) > 1)
    starts = np.concatenate(([rows[0]], rows[breaks + 1]))
    ends = np.concatenate((rows[breaks], [rows[-1]]))
    for a, b in zip(starts, ends):
        depth = (b - a + 1) * c.step_m
        if c.thickness_m[0] <= depth <= c.thickness_m[1]:
            return float(xs[a])
    return None


class StopLineTracker:
    """Stop-line detections kept in the odometry frame, locked once they agree.

    A line is confirmed by ``min_hits`` detections within ``agree_m`` of their median in
    the odometry frame, taken while the car moved at least ``travel_m``: a painted line
    stays put as the car approaches, an artefact at a fixed distance in the image (a
    shadow edge, the sampled band's end) moves with the car (a1rec, 2 Nov: 6.5 m ahead
    for five frames at s 86). Once locked, the line is kept until ``reset`` (the car has
    held at it and gone, or the light no longer asks for a stop): it leaves the camera's
    view as the car draws up to it."""

    def __init__(self, config=None, min_hits=3, agree_m=0.4, travel_m=1.5, window_s=4.0):
        self.config = config or StopLineConfig()
        self.min_hits, self.agree_m, self.travel_m, self.window_s = min_hits, agree_m, travel_m, window_s
        self.hits = []  # (time_s, line x, line y, car x, car y) in the odometry frame
        self.locked = None

    def update(self, rgb, calibration, pose, time_s):
        """Look in this image (taken at ``pose``, the car's odometry pose); returns the
        detection's distance ahead, or None."""
        d = detect(rgb, calibration, self.config)
        if d is not None and self.locked is None:
            line = pose.to_world(np.array([[d, 0.0], [0.0, 0.0]]))
            self.hits = [h for h in self.hits if time_s - h[0] <= self.window_s] + [(float(time_s), *map(float, line[0]), *map(float, line[1]))]
            pts = np.array([[h[1], h[2]] for h in self.hits])
            centre = np.median(pts, axis=0)
            agree = [h for h, p in zip(self.hits, pts) if np.hypot(*(p - centre)) <= self.agree_m]
            if len(agree) >= self.min_hits:
                travel = np.hypot(agree[-1][3] - agree[0][3], agree[-1][4] - agree[0][4])
                if travel >= self.travel_m:
                    self.locked = np.median(np.array([[h[1], h[2]] for h in agree]), axis=0)
        return d

    def distance_m(self, pose):
        """The locked line's distance ahead of the rear axle, or None."""
        if self.locked is None:
            return None
        return float(pose.to_body(self.locked[None])[0, 0])

    def reset(self):
        self.hits, self.locked = [], None
