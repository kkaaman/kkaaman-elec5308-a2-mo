"""Drivable corridor and lane centre from the LiDAR terrain grid (G1b, kerb following).

Each tick:

1. **Reference.** Last tick's centreline, moved into the current vehicle frame and
   extended straight past its end. On the first tick it is the line straight ahead.
2. **Lateral profiles.** At stations every ``step_m`` along the reference, cells are
   sampled along the normal out to ``lateral_search_m`` each side. From the reference
   point, the first BARRIER cell to the right is the right road edge, and likewise on the
   left. A station whose reference point is itself a barrier is *blocked* (something
   stands in the lane) and gives no lane evidence.
3. **Openings.** A right edge that jumps outwards by more than ``max_edge_jump_m``
   between stations is a side road, driveway or junction mouth, not the lane bending
   away. It gives no evidence, and the edge before the gap is kept.
4. **Lane centre.** Traffic keeps right: the lane centre is ``keep_right_m`` left of the
   right edge. A corridor narrower than ``two_lane_min_m`` is one lane, and the centre is
   its middle.
5. **Fit.** A polynomial y(x) of degree 1 to 3 (by how far the evidence reaches) is fitted
   to these centre points and to last tick's centreline, which gets ``prior_weight``. This
   bridges openings and blind stretches and keeps the path still between ticks. Points
   further than ``outlier_m`` from a first fit are dropped before the second.

The centreline is valid up to ``extrapolate_m`` past the furthest evidence, or to what is
left of last tick's corridor, whichever reaches further, and never beyond a heading of
``max_heading_rad`` from the car's. Junction turns need a different geometry, which is
G1c. Without kerbs (open country roads) this corridor has no evidence; that needs the
camera lane estimate.
"""

import math
from dataclasses import dataclass

import numpy as np

from mapless.ego_motion import Pose2D
from mapless.terrain import CellLabel, TerrainGrid


@dataclass(frozen=True)
class CorridorConfig:
    step_m: float = 1.0
    start_m: float = -2.0
    horizon_m: float = 40.0
    lateral_search_m: float = 8.0
    lateral_step_m: float = 0.25
    keep_right_m: float = 1.75  # half a 3.5 m lane; measure it live (the drive log reports it)
    two_lane_min_m: float = 5.5  # a narrower corridor is one lane: drive in its middle
    max_unseen_m: float = 0.3  # an edge needs road seen within this distance before it
    min_width_m: float = 2.8
    max_edge_jump_m: float = 1.0
    prior_weight: float = 0.5
    outlier_m: float = 0.6
    min_evidence: int = 5
    extrapolate_m: float = 6.0
    max_heading_rad: float = 0.9
    min_valid_m: float = 6.0

    def __post_init__(self):
        for name in ("step_m", "horizon_m", "lateral_search_m", "lateral_step_m", "keep_right_m", "min_width_m"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.start_m < 0 < self.horizon_m:
            raise ValueError("The corridor must start behind the rear axle and reach ahead")
        if not 0 <= self.prior_weight <= 1:
            raise ValueError("prior_weight must be in [0, 1]")


@dataclass(frozen=True)
class Corridor:
    time_s: float
    pose: Pose2D  # odometry pose the body-frame arrays refer to
    centre_body: np.ndarray  # M x 2, vehicle frame, every step_m in x from start_m to valid_length_m
    centre_odometry: np.ndarray  # the same points in the odometry frame
    valid_length_m: float  # x up to which the centreline is supported
    coefficients: tuple  # y(x) polynomial, highest power first
    station_x_m: np.ndarray
    reference_y_m: np.ndarray  # the reference line the edges were searched from
    right_edge_m: np.ndarray  # lateral offset of the right edge from the reference (NaN if none)
    left_edge_m: np.ndarray
    blocked: np.ndarray  # reference point was a barrier
    evidence_body: np.ndarray  # K x 2 lane-centre points used in the fit
    reason: str = ""

    @property
    def ok(self):
        return len(self.centre_body) >= 2 and not self.reason

    def lateral_offset_at(self, x_m):
        """y of the fitted centreline at body-frame x."""
        return float(np.polyval(self.coefficients, x_m))

    def right_edge_body(self, x_m, within_m=2.0):
        """(x, y) in the vehicle frame of the right edge at the station nearest x (None if none)."""
        near = np.flatnonzero(np.isfinite(self.right_edge_m) & (np.abs(self.station_x_m - x_m) <= within_m))
        if near.size == 0 or not self.ok:
            return None
        i = near[np.argmin(np.abs(self.station_x_m[near] - x_m))]
        return float(self.station_x_m[i]), float(self.reference_y_m[i] + self.right_edge_m[i])

    def right_edge_from_centre_m(self, x_m, within_m=2.0):
        """Right edge's lateral offset from the fitted centre at the station nearest x (None if none)."""
        near = np.flatnonzero(np.isfinite(self.right_edge_m) & (np.abs(self.station_x_m - x_m) <= within_m))
        if near.size == 0 or not self.ok:
            return None
        i = near[np.argmin(np.abs(self.station_x_m[near] - x_m))]
        edge_y = self.reference_y_m[i] + self.right_edge_m[i]
        return float(edge_y - self.lateral_offset_at(self.station_x_m[i]))


def _edges(labels, j0, max_gap_cells):
    """First barrier outwards from column ``j0`` on each row, and whether it is a seen edge.

    Returns (index, seen) for the right (towards column 0) and the left (towards the last
    column). A seen edge has road within ``max_gap_cells`` before it. A barrier found only
    across a longer unseen stretch (the LiDAR's blind ring beside the car, or between
    sparse far rings) may be the far side of a pavement rather than the kerb, so it is
    not trusted.
    """
    out = []
    for sub, to_column in ((labels[:, j0::-1], lambda k: j0 - k), (labels[:, j0:], lambda k: j0 + k)):
        steps = np.arange(sub.shape[1])
        barrier = sub == CellLabel.BARRIER
        has = barrier.any(axis=1)
        first = np.where(has, barrier.argmax(axis=1), sub.shape[1])
        road_before = (sub == CellLabel.ROAD) & (steps[None, :] < first[:, None])
        last_road = np.where(road_before, steps[None, :], -1).max(axis=1)
        seen = has & (last_road >= 0) & (first - last_road <= max_gap_cells + 1)
        out.append((np.where(has, to_column(first), -1), seen, road_before.any(axis=1)))
    return out


def _fit(x, y, w, degree, outlier_m, raw_count):
    coefficients = np.polyfit(x, y, degree, w=np.sqrt(w))
    residual = np.abs(np.polyval(coefficients, x) - y)
    keep = np.ones(len(x), dtype=bool)
    keep[:raw_count] = residual[:raw_count] <= outlier_m  # only evidence can be an outlier
    if keep.sum() > degree + 1 and not keep.all():
        coefficients = np.polyfit(x[keep], y[keep], degree, w=np.sqrt(w[keep]))
    return coefficients, keep


class CorridorEstimator:
    def __init__(self, config=None):
        self.config = config or CorridorConfig()
        self.previous = None

    def reset(self):
        self.previous = None

    def _reference(self, stations, pose):
        """Last centreline in the current vehicle frame, extended straight past its end."""
        if self.previous is None:
            return np.zeros_like(stations), 0.0
        body = pose.to_body(self.previous.centre_odometry)
        x, y = body[:, 0], body[:, 1]
        if len(x) < 2 or np.any(np.diff(x) <= 0):
            return np.zeros_like(stations), 0.0
        slope_end = (y[-1] - y[-2]) / (x[-1] - x[-2])
        slope_start = (y[1] - y[0]) / (x[1] - x[0])
        ref = np.interp(stations, x, y)
        ref = np.where(stations > x[-1], y[-1] + slope_end * (stations - x[-1]), ref)
        ref = np.where(stations < x[0], y[0] + slope_start * (stations - x[0]), ref)
        return ref, float(x[-1])

    def update(self, grid, pose, time_s):
        if not isinstance(grid, TerrainGrid) or not isinstance(pose, Pose2D):
            raise TypeError("update(TerrainGrid, Pose2D, time_s)")
        c = self.config
        stations = np.arange(c.start_m, c.horizon_m + 1e-9, c.step_m)
        ref, prior_end = self._reference(stations, pose)
        heading = np.arctan(np.gradient(ref, stations))
        offsets = np.arange(-c.lateral_search_m, c.lateral_search_m + 1e-9, c.lateral_step_m)
        j0 = int(np.argmin(np.abs(offsets)))
        nx, ny = -np.sin(heading), np.cos(heading)
        px = stations[:, None] + offsets[None, :] * nx[:, None]
        py = ref[:, None] + offsets[None, :] * ny[:, None]
        body = np.column_stack((px.ravel(), py.ravel()))
        labels = grid.labels(pose.to_world(body)).reshape(px.shape)

        n = len(stations)
        gap = int(round(c.max_unseen_m / c.lateral_step_m))
        half_cell = grid.config.cell_m / 2
        right = np.full(n, np.nan)
        left = np.full(n, np.nan)
        (jr, right_seen, road_right), (jl, left_seen, road_left) = _edges(labels, j0, gap)
        blocked = labels[:, j0] == CellLabel.BARRIER
        usable = ~blocked & (road_right | road_left | (labels[:, j0] == CellLabel.ROAD))
        # The first barrier sample lies on average half a lateral step past the barrier
        # cell's inner side, and the kerb on average half a cell past that side.
        take = usable & right_seen
        right[take] = offsets[jr[take]] + c.lateral_step_m / 2 - half_cell
        take = usable & left_seen
        left[take] = offsets[jl[take]] - c.lateral_step_m / 2 + half_cell

        # Openings: the right edge may not jump outwards; seed with the median of the nearest edges.
        found = np.flatnonzero(np.isfinite(right) & (stations > 0))
        accepted_right = np.full(n, np.nan)
        if found.size:
            last = float(np.median(right[found[:5]]))
            for i in range(n):
                if np.isfinite(right[i]) and right[i] >= last - c.max_edge_jump_m:
                    accepted_right[i] = right[i]
                    last = right[i]

        centre = np.full(n, np.nan)
        for i in range(n):
            r, lt = accepted_right[i], left[i]
            if not np.isfinite(r):
                continue
            if np.isfinite(lt):
                width = lt - r
                if width < c.min_width_m:
                    blocked[i] = True
                    continue
                centre[i] = (r + lt) / 2 if width < c.two_lane_min_m else r + c.keep_right_m
            else:
                centre[i] = r + c.keep_right_m
        has = np.isfinite(centre)
        ex = stations[has] - centre[has] * np.sin(heading[has])
        ey = ref[has] + centre[has] * np.cos(heading[has])
        evidence = np.column_stack((ex, ey))

        x_parts, y_parts, w_parts = [ex], [ey], [np.ones(len(ex))]
        if self.previous is not None and c.prior_weight > 0:
            prior = pose.to_body(self.previous.centre_odometry)
            ahead = prior[:, 0] >= c.start_m
            x_parts.append(prior[ahead, 0])
            y_parts.append(prior[ahead, 1])
            w_parts.append(np.full(int(ahead.sum()), c.prior_weight))
        x_all, y_all, w_all = np.concatenate(x_parts), np.concatenate(y_parts), np.concatenate(w_parts)

        reach = float(ex.max()) + c.extrapolate_m if len(ex) else -math.inf
        valid = min(c.horizon_m, max(reach, prior_end))
        reason = ""
        if len(ex) < c.min_evidence and self.previous is None:
            reason = f"only {len(ex)} lane-centre points and no previous corridor"
        elif len(x_all) < 2 or valid < c.min_valid_m:
            reason = f"corridor reaches only {valid:.1f} m"
        coefficients = (0.0, 0.0)
        if not reason:
            span = float(np.ptp(x_all))
            degree = 1 if span < 10 else 2 if span < 20 else 3
            degree = min(degree, len(x_all) - 1)
            coefficients, keep = _fit(x_all, y_all, w_all, degree, c.outlier_m, len(ex))
            evidence = evidence[keep[: len(ex)]]
            xs = np.arange(c.start_m, valid + 1e-9, c.step_m)
            slope = np.polyval(np.polyder(coefficients), xs)
            too_steep = np.flatnonzero(np.abs(np.arctan(slope)) > c.max_heading_rad)
            if too_steep.size:
                xs = xs[: too_steep[0]]
                valid = float(xs[-1]) if len(xs) else c.start_m
            if valid < c.min_valid_m:
                reason = f"corridor turns more than {math.degrees(c.max_heading_rad):.0f} deg within {valid:.1f} m"
        if reason:
            xs = np.zeros(0)
        centre_body = np.column_stack((xs, np.polyval(coefficients, xs))) if len(xs) else np.zeros((0, 2))
        corridor = Corridor(
            time_s=time_s,
            pose=pose,
            centre_body=centre_body,
            centre_odometry=pose.to_world(centre_body) if len(centre_body) else np.zeros((0, 2)),
            valid_length_m=float(valid) if not reason else 0.0,
            coefficients=tuple(float(v) for v in coefficients),
            station_x_m=stations,
            reference_y_m=ref,
            right_edge_m=accepted_right,
            left_edge_m=left,
            blocked=blocked,
            evidence_body=evidence,
            reason=reason,
        )
        if corridor.ok:
            self.previous = corridor
        return corridor
