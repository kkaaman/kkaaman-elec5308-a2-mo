"""Drivable corridor and lane centre from the LiDAR terrain grid (G1b, kerb following).

Each tick:

1. **Reference.** Last tick's centreline, moved into the current vehicle frame and
   extended straight past its end. On the first tick it is the line straight ahead.
2. **Lateral profiles.** At stations every ``step_m`` along the reference, cells are
   sampled along the normal out to ``lateral_search_m`` each side. From the reference
   point, the first BARRIER cell to the right is the right road edge, and likewise on the
   left. A station whose reference point is itself a barrier is *blocked* (something
   stands in the lane) and gives no lane evidence.
3. **Continuity.** A kerb does not jump. A right edge more than ``max_edge_jump_m`` from
   the last accepted one gives no evidence: outwards it is a side road, driveway or
   junction mouth; inwards it is something on the road that looks like a kerb. The chain
   starts from the median of the nearest edges.
4. **Lane centre.** Traffic keeps right: the lane centre is ``keep_right_m`` left of the
   right edge. By default that distance is measured on a calibration drive: the first
   ``calibrate_m`` along the start line, where the car starts aligned in its lane, using
   this same detector. A standstill estimate (``observe_start``) is the fallback. A corridor narrower than ``two_lane_min_m`` is one lane,
   and the centre is its middle. Only kerbs are edges: parked cars and other obstacles
   are OCCUPIED, never the road edge.
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
from mapless.terrain import CellLabel, PointClass, TerrainGrid


@dataclass(frozen=True)
class CorridorConfig:
    step_m: float = 1.0
    start_m: float = -2.0
    horizon_m: float = 40.0
    lateral_search_m: float = 8.0
    # Finer than the grid's cells, so the first kerb sample sits at most 5 cm past the kerb
    # cell's inner side whatever line the profile is drawn from. With 0.25 m samples the
    # calibration drive (profiles from the start line, on cell boundaries) and later driving
    # (profiles from the fitted line) quantised the same kerb differently, 0.25 m apart.
    lateral_step_m: float = 0.05
    # Lane centre's distance left of the kerb. None: measured at start-up, where the car
    # stands in its lane (Town10HD has shoulders: 4.4 m, not half a lane). The measurement
    # falls back to ``default_keep_right_m`` when no kerb is seen while standing.
    keep_right_m: float | None = None
    default_keep_right_m: float = 1.75
    calibrate_m: float = 8.0  # with keep_right_m None: drive this far on the start line, measuring
    calibration_min_edges: int = 30
    two_lane_min_m: float = 5.5  # a narrower corridor is one lane: drive in its middle
    max_unseen_m: float = 0.3  # an edge needs road seen within this distance before it
    min_width_m: float = 2.8
    max_edge_jump_m: float = 1.0
    edge_gate_m: float = 0.75  # kerb evidence must lie within this of keep_right from the reference
    reference_min_radius_m: float = 15.0  # tightest bend the reference may extrapolate
    evidence_range_m: float = 20.0  # evidence weight halves at this distance ahead
    prior_weight: float = 0.5
    outlier_m: float = 0.6
    min_evidence: int = 5
    extrapolate_m: float = 6.0
    max_heading_rad: float = 0.9
    min_valid_m: float = 6.0

    def __post_init__(self):
        for name in ("step_m", "horizon_m", "lateral_search_m", "lateral_step_m", "default_keep_right_m", "min_width_m"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.keep_right_m is not None and not (math.isfinite(self.keep_right_m) and self.keep_right_m > 0):
            raise ValueError("keep_right_m must be positive, or None to measure it at start-up")
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
    blocked: np.ndarray  # reference point was a kerb or occupied
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


def _carried(previous, pose, valid_length_m):
    body = pose.to_body(previous.centre_odometry)
    keep = body[:, 0] <= valid_length_m
    return body[keep]


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
        self.keep_right_m = self.config.keep_right_m
        self.keep_right_source = "configured" if self.keep_right_m is not None else "pending"
        self._start_kerbs = []
        self._calibration_origin = None
        self._calibration_edges = []

    def reset(self):
        self.previous = None

    def _profiles(self, grid, pose, stations, ref):
        """Cell labels across the reference at each station, the lateral offsets, and the centre column."""
        c = self.config
        heading = np.arctan(np.gradient(ref, stations))
        offsets = np.arange(-c.lateral_search_m, c.lateral_search_m + 1e-9, c.lateral_step_m)
        j0 = int(np.argmin(np.abs(offsets)))
        nx, ny = -np.sin(heading), np.cos(heading)
        px = stations[:, None] + offsets[None, :] * nx[:, None]
        py = ref[:, None] + offsets[None, :] * ny[:, None]
        body = np.column_stack((px.ravel(), py.ravel()))
        labels = grid.labels(pose.to_world(body)).reshape(px.shape)
        return labels, offsets, j0, heading

    def measure_kerb(self, grid, pose, near_m=3.0, far_m=15.0, min_stations=4):
        """Distance (m) from the car's centreline to the kerb on its right, 3-15 m ahead; None if unseen."""
        c = self.config
        stations = np.arange(near_m, far_m + 1e-9, c.step_m)
        labels, offsets, j0, _ = self._profiles(grid, pose, stations, np.zeros_like(stations))
        gap = int(round(c.max_unseen_m / c.lateral_step_m))
        (jr, seen, _), _ = _edges(labels, j0, gap)
        seen &= labels[:, j0] == CellLabel.ROAD
        if seen.sum() < min_stations:
            return None
        edges = offsets[jr[seen]] + c.lateral_step_m / 2 - grid.config.cell_m / 2
        # The median assumes the car stands parallel to its lane, as it does when placed in
        # it. A line fit (to correct a skewed start) extrapolated quantised edges back to the
        # axle and was off by up to 0.6 m on the synthetic road.
        return float(-np.median(edges))

    def observe_start(self, sweep, near_m=3.0, far_m=15.0, min_bins=4):
        """Call with each classified sweep while standing in the lane at start-up.

        Collects the kerb's distance for ``keep_right_m`` from the points themselves, not
        the grid: at a standstill the grid's cells sit in one fixed phase against the kerb,
        which biased the measurement by up to a cell. Per 1 m slice ahead, the kerb is the
        innermost raised point on the right, provided road was seen between it and the car
        and nothing tall stands within 1 m of it (the sills of a parked car are kerb-high).
        """
        if self.keep_right_source != "pending":
            return
        xyz, cls = sweep.xyz, sweep.cls
        right = (xyz[:, 1] < 0) & (xyz[:, 1] > -self.config.lateral_search_m) & (xyz[:, 0] >= near_m) & (xyz[:, 0] < far_m)
        distances = []
        for x0 in np.arange(near_m, far_m, 1.0):
            in_slice = right & (xyz[:, 0] >= x0) & (xyz[:, 0] < x0 + 1.0)
            raised = xyz[in_slice & (cls == PointClass.RAISED), 1]
            if raised.size == 0:
                continue
            kerb_y = raised.max()
            road = xyz[in_slice & (cls == PointClass.ROAD), 1]
            tall = xyz[in_slice & (cls == PointClass.OBSTACLE), 1]
            if not np.any(road > kerb_y) or np.any(np.abs(tall - kerb_y) < 1.0):
                continue
            distances.append(-kerb_y)
        if len(distances) >= min_bins:
            self._start_kerbs.append(float(np.median(distances)))

    def _resolve_keep_right(self):
        """Fallback when the calibration drive saw too little kerb: the standstill estimate, else the default."""
        if len(self._start_kerbs) >= 3:
            self.keep_right_m = float(np.median(self._start_kerbs))
            self.keep_right_source = f"standstill estimate ({len(self._start_kerbs)} sweeps; calibration drive saw too little kerb)"
        else:
            self.keep_right_m = self.config.default_keep_right_m
            self.keep_right_source = "default (no kerb seen at start-up or on the calibration drive)"

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
        # Past its end, continue with the line's end curvature (bounded), so a bend that has
        # started keeps bending in the reference and its kerb stays inside the edge gate.
        tail = min(len(x), 6)
        bend = 0.0
        if tail >= 3:
            bend = float(np.polyfit(x[-tail:], y[-tail:], 2)[0])  # y'' / 2
            limit = 0.5 / self.config.reference_min_radius_m
            bend = float(np.clip(bend, -limit, limit))
        d = stations - x[-1]
        ref = np.where(stations > x[-1], y[-1] + slope_end * d + bend * d**2, ref)
        ref = np.where(stations < x[0], y[0] + slope_start * (stations - x[0]), ref)
        return ref, float(x[-1])

    def _edges_along(self, grid, pose, stations, ref):
        """Right and left kerb offsets from the reference at each station, blocked stations, headings."""
        c = self.config
        labels, offsets, j0, heading = self._profiles(grid, pose, stations, ref)
        n = len(stations)
        gap = int(round(c.max_unseen_m / c.lateral_step_m))
        half_cell = grid.config.cell_m / 2
        right = np.full(n, np.nan)
        left = np.full(n, np.nan)
        (jr, right_seen, road_right), (jl, left_seen, road_left) = _edges(labels, j0, gap)
        blocked = np.isin(labels[:, j0], (CellLabel.BARRIER, CellLabel.OCCUPIED))
        usable = ~blocked & (road_right | road_left | (labels[:, j0] == CellLabel.ROAD))
        # The first barrier sample lies on average half a lateral step past the barrier
        # cell's inner side, and the kerb on average half a cell past that side.
        take = usable & right_seen
        right[take] = offsets[jr[take]] + c.lateral_step_m / 2 - half_cell
        take = usable & left_seen
        left[take] = offsets[jl[take]] - c.lateral_step_m / 2 + half_cell
        # Continuity: no jumps in either direction; seed with the median of the nearest edges.
        found = np.flatnonzero(np.isfinite(right) & (stations > 0))
        accepted_right = np.full(n, np.nan)
        if found.size:
            last = float(np.median(right[found[:5]]))
            for i in range(n):
                if np.isfinite(right[i]) and abs(right[i] - last) <= c.max_edge_jump_m:
                    accepted_right[i] = right[i]
                    last = right[i]
        return accepted_right, left, blocked, heading

    def _calibrate(self, grid, pose, time_s, stations):
        """Drive the first ``calibrate_m`` along the start line, measuring the kerb from it.

        The car starts in its lane, aligned with it, so the start line is the lane's line
        and the kerb's distance from it is ``keep_right_m``. Measuring with the driving
        detector itself, over many sweeps and cell phases, keeps the two consistent: the
        first live run measured the kerb from single points at a standstill, 0.44 m closer
        than the driving detector found it, and drove 0.45 m off centre.

        Returns the start-line corridor while calibrating, or None once done.
        """
        c = self.config
        origin = self._calibration_origin
        a = origin.to_body(pose.to_world(np.array([[0.0, 0.0]])))[0]
        travelled = float(a[0])
        line = pose.to_body(origin.to_world(np.array([[0.0, 0.0], [100.0, 0.0]])))
        slope = (line[1, 1] - line[0, 1]) / (line[1, 0] - line[0, 0])
        intercept = line[0, 1] - slope * line[0, 0]
        ref = intercept + slope * stations
        right, _, _, _ = self._edges_along(grid, pose, stations, ref)
        near = np.isfinite(right) & (stations >= 3.0) & (stations <= 20.0)
        self._calibration_edges.extend((-right[near]).tolist())  # offsets are already normal to the line
        if travelled >= c.calibrate_m:
            if len(self._calibration_edges) >= c.calibration_min_edges:
                self.keep_right_m = float(np.median(self._calibration_edges))
                self.keep_right_source = f"measured over the first {c.calibrate_m:g} m ({len(self._calibration_edges)} edges)"
            else:
                self._resolve_keep_right()
            self.previous = None
            return None
        xs = np.arange(c.start_m, c.horizon_m + 1e-9, c.step_m)
        centre_body = np.column_stack((xs, intercept + slope * xs))
        return Corridor(
            time_s=time_s,
            pose=pose,
            centre_body=centre_body,
            centre_odometry=pose.to_world(centre_body),
            valid_length_m=float(c.horizon_m),
            coefficients=(float(slope), float(intercept)),
            station_x_m=stations,
            reference_y_m=ref,
            right_edge_m=right,
            left_edge_m=np.full(len(stations), np.nan),
            blocked=np.zeros(len(stations), dtype=bool),
            evidence_body=np.zeros((0, 2)),
        )

    def _carry_forward(self, pose, time_s, stations, ref, right, left, blocked, evidence, prior_end):
        previous = self.previous
        centre_body = _carried(previous, pose, prior_end)
        coefficients = previous.coefficients
        if len(centre_body) >= 2:
            degree = min(len(previous.coefficients) - 1, len(centre_body) - 1)
            coefficients = tuple(float(v) for v in np.polyfit(centre_body[:, 0], centre_body[:, 1], degree))
        corridor = Corridor(
            time_s=time_s,
            pose=pose,
            centre_body=centre_body,
            centre_odometry=pose.to_world(centre_body) if len(centre_body) else np.zeros((0, 2)),
            valid_length_m=float(prior_end),
            coefficients=coefficients,
            station_x_m=stations,
            reference_y_m=ref,
            right_edge_m=right,
            left_edge_m=left,
            blocked=blocked,
            evidence_body=evidence,
            reason="" if len(centre_body) >= 2 else "previous corridor used up",
        )
        if corridor.ok:
            self.previous = corridor
        return corridor

    def update(self, grid, pose, time_s):
        if not isinstance(grid, TerrainGrid) or not isinstance(pose, Pose2D):
            raise TypeError("update(TerrainGrid, Pose2D, time_s)")
        c = self.config
        if self.keep_right_source == "pending":
            self.keep_right_source = "calibrating"
            self._calibration_origin = pose
        stations = np.arange(c.start_m, c.horizon_m + 1e-9, c.step_m)
        if self.keep_right_source == "calibrating":
            calibration = self._calibrate(grid, pose, time_s, stations)
            if calibration is not None:
                return calibration
        keep_right = self.keep_right_m
        ref, prior_end = self._reference(stations, pose)
        accepted_right, left, blocked, heading = self._edges_along(grid, pose, stations, ref)
        # The reference is last tick's centreline, so a true kerb lies about keep_right to its
        # right everywhere. An edge far from that is something else: on the shoulder-parking
        # road the chain walked out 1.5 m behind a parked car at 26 m, the cubic bent to
        # follow it, and the bent path ran into the next parked car.
        stray = np.abs(accepted_right + keep_right) > c.edge_gate_m
        accepted_right = np.where(stray, np.nan, accepted_right)
        n = len(stations)

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
                centre[i] = (r + lt) / 2 if width < c.two_lane_min_m else r + keep_right
            else:
                centre[i] = r + keep_right
        has = np.isfinite(centre)
        ex = stations[has] - centre[has] * np.sin(heading[has])
        ey = ref[has] + centre[has] * np.cos(heading[has])
        evidence = np.column_stack((ex, ey))

        if len(ex) < c.min_evidence and self.previous is not None and prior_end >= c.min_valid_m:
            # Too little to refit: keep last tick's line as it is. Refitting the prior on its
            # own extrapolates it a little further each tick and drifts.
            return self._carry_forward(pose, time_s, stations, ref, accepted_right, left, blocked, evidence, prior_end)
        # Near evidence is denser and better placed than far evidence.
        x_parts, y_parts, w_parts = [ex], [ey], [1.0 / (1.0 + (np.maximum(ex, 0.0) / c.evidence_range_m) ** 2)]
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
            # The degree follows how far the evidence itself reaches, not the prior: a cubic
            # fitted to 12 m of kerb plus last tick's line bent 2.5 m by 35 m on the
            # shoulder-parking road.
            span = float(np.ptp(ex)) if len(ex) >= 2 else 0.0
            degree = 1 if span < 8 else 2 if span < 20 else 3
            degree = min(degree, len(x_all) - 1)
            coefficients, keep = _fit(x_all, y_all, w_all, degree, c.outlier_m, len(ex))
            evidence = evidence[keep[: len(ex)]]
            xs = np.arange(c.start_m, valid + 1e-9, c.step_m)
            ys = np.polyval(coefficients, xs)
            # Past the evidence, do not extrapolate the polynomial. Continue along its tangent,
            # bending with its end curvature only when the evidence was long enough to show
            # a bend (degree 2 or more), and never tighter than reference_min_radius_m.
            end = float(ex.max()) if len(ex) else 0.0
            tangent = float(np.polyval(np.polyder(coefficients), end))
            half_curvature = 0.0
            if degree >= 2:
                limit = 0.5 / c.reference_min_radius_m
                half_curvature = float(np.clip(np.polyval(np.polyder(coefficients, 2), end) / 2, -limit, limit))
            beyond = xs > end
            d = xs[beyond] - end
            ys[beyond] = float(np.polyval(coefficients, end)) + tangent * d + half_curvature * d**2
            slope = np.polyval(np.polyder(coefficients), xs)
            slope[beyond] = tangent + 2 * half_curvature * d
            too_steep = np.flatnonzero(np.abs(np.arctan(slope)) > c.max_heading_rad)
            if too_steep.size:
                xs = xs[: too_steep[0]]
                valid = float(xs[-1]) if len(xs) else c.start_m
            if valid < c.min_valid_m:
                reason = f"corridor turns more than {math.degrees(c.max_heading_rad):.0f} deg within {valid:.1f} m"
        if reason:
            xs = np.zeros(0)
        centre_body = np.column_stack((xs, ys[: len(xs)])) if len(xs) else np.zeros((0, 2))
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
