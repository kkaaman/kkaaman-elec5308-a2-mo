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
   right edge. By default that distance is measured at a standstill before the car moves
   (``calibrate``): the kerb's raw inner contour from 10 m behind to 15 m ahead is fitted
   as a curve, and the rear axle's distance to it is the lane centre's. It is never
   defaulted. A corridor narrower than ``two_lane_min_m`` is one lane,
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
from mapless.terrain import CellLabel, ClassifiedSweep, PointClass, TerrainGrid


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
    # Lane centre's distance left of the kerb. None: measured at a standstill before moving
    # (``CorridorEstimator.calibrate``); Town10HD has shoulders, so it is 4.4 m, not half a
    # lane. A measurement that does not settle fails; there is no default.
    keep_right_m: float | None = None
    calibration_behind_m: float = 10.0  # kerb stations from this far behind the rear axle ...
    calibration_far_m: float = 15.0  # ... to this far ahead
    calibration_station_m: float = 0.5
    calibration_min_stations: int = 8  # kerb seen at this many stations ...
    calibration_min_span_m: float = 6.0  # ... spread over at least this length
    calibration_max_mad_m: float = 0.15  # median distance of the stations from the fitted kerb
    calibration_max_heading_deg: float = 10.0  # the car must stand roughly along its lane
    calibration_min_fits: int = 10  # sweeps with a clean fit ...
    calibration_max_spread_m: float = 0.15  # ... agreeing within this (p90 - p10)
    calibration_timeout_s: float = 4.0  # not settled by then: failed
    # Freshness (review of 94b5caa): evidence must come from recent sweeps, not from what
    # the grid still remembers.
    edge_max_age_s: float = 3.0  # kerb cells must have been hit within this long
    support_max_age_s: float = 1.0  # road ahead must have been hit within this long
    support_start_m: float = 8.0  # observed road must begin within this distance ahead
    support_gap_m: float = 1.0  # an unseen gap along the road ahead may be this long ...
    support_gap_per_m: float = 0.3  # ... or this fraction of its distance (LiDAR rings spread out)
    evidence_max_age_s: float = 5.0  # a corridor whose kerb evidence is older is not valid
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
        for name in ("step_m", "horizon_m", "lateral_search_m", "lateral_step_m", "min_width_m", "calibration_station_m"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.keep_right_m is not None and not (math.isfinite(self.keep_right_m) and self.keep_right_m > 0):
            raise ValueError("keep_right_m must be positive, or None to measure it at a standstill")
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
    evidence_time_s: float = -math.inf  # when the kerb evidence behind this line was last fresh
    support_m: float = 0.0  # how far along the line the road is currently observed
    calibrating: bool = False

    @property
    def evidence_age_s(self):
        return self.time_s - self.evidence_time_s

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


def _rounded(fit):
    out = {}
    for k, v in fit.items():
        if isinstance(v, bool | int):
            out[k] = v
        elif isinstance(v, tuple):
            out[k] = [round(x, 5) for x in v]
        else:
            out[k] = round(v, 3)
    return out


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
        self._fits = []
        self._last_fit = None
        self._edge_seen_s = np.zeros(0)
        self._calibration_began_s = None
        self.calibration_summary = None

    def reset(self):
        self.previous = None

    def _profiles(self, grid, pose, stations, ref, max_age_s=None):
        """Cell labels across the reference at each station, the lateral offsets, and the centre column."""
        c = self.config
        max_age_s = c.edge_max_age_s if max_age_s is None else max_age_s
        heading = np.arctan(np.gradient(ref, stations))
        offsets = np.arange(-c.lateral_search_m, c.lateral_search_m + 1e-9, c.lateral_step_m)
        j0 = int(np.argmin(np.abs(offsets)))
        nx, ny = -np.sin(heading), np.cos(heading)
        px = stations[:, None] + offsets[None, :] * nx[:, None]
        py = ref[:, None] + offsets[None, :] * ny[:, None]
        body = np.column_stack((px.ravel(), py.ravel()))
        labels = grid.labels(pose.to_world(body), max_age_s=max_age_s).reshape(px.shape)
        return labels, offsets, j0, heading

    def _support(self, grid, pose, centre_body):
        """How far along the line (x, m) the road is currently observed.

        Samples a 1 m band along the line every 0.5 m, counting road or occupied cells hit
        within ``support_max_age_s``. Observation must begin within ``support_start_m`` (the
        LiDAR cannot see the ground under and right beside the car), and may skip unseen
        gaps up to ``support_gap_m`` or ``support_gap_per_m`` of the distance. A kerb in the
        band ends it. Obstacles do not: stopping for them is the obstacle check's job.
        """
        c = self.config
        if len(centre_body) < 2:
            return 0.0
        xs = np.arange(max(0.0, float(centre_body[0, 0])), float(centre_body[-1, 0]) + 1e-9, 0.5)
        if len(xs) == 0:
            return 0.0
        ys = np.interp(xs, centre_body[:, 0], centre_body[:, 1])
        band = np.array([-0.5, 0.0, 0.5])
        pts = np.column_stack((np.repeat(xs, len(band)), (ys[:, None] + band[None, :]).ravel()))
        labels = grid.labels(pose.to_world(pts), max_age_s=c.support_max_age_s).reshape(len(xs), len(band))
        observed = np.any((labels == CellLabel.ROAD) | (labels == CellLabel.OCCUPIED), axis=1)
        kerb = np.any(labels == CellLabel.BARRIER, axis=1)
        first = np.flatnonzero(observed & (xs <= c.support_start_m))
        if first.size == 0:
            return 0.0
        last = float(xs[first[0]])
        for x, seen, blocked in zip(xs[first[0] :], observed[first[0] :], kerb[first[0] :]):
            if blocked:
                break
            if seen:
                last = float(x)
            elif x - last > max(c.support_gap_m, c.support_gap_per_m * x):
                break
        return last

    @property
    def calibrating(self):
        return self.keep_right_m is None

    def calibrate(self, sweep, pose, time_s, stationary):
        """Measure ``keep_right_m`` while standing still; call every tick until it settles.

        The car stands centred and aligned in its lane. Each sweep, the right kerb's inner
        contour is taken from the raw classified points (``_kerb_contour``): at stations
        from ``calibration_behind_m`` behind to ``calibration_far_m`` ahead, the innermost
        RAISED point with road just inside it and nothing tall beside it. The contour is
        fitted as a quadratic when both sides of the axle are seen, else a line
        (``_fit_kerb``), and the axle's nearest distance to the fit is the lane centre's
        distance from the kerb.

        Why raw points and a curve: the live "straight" site on Town10HD bends at 58 m
        radius, and a line through 25 m of its kerb read the chord, 0.4 m inside the kerb
        beside the car (run with 2b7e14d). The grid's cell-quantised edges were too coarse
        for a curve (a quadratic on them read 1.55 m for a 1.75 m straight kerb); the raw
        contour has hundreds of points per station and reads within 0.07 m on straights,
        skewed starts and bends down to 50 m radius offline. Not measured while driving
        the start heading either: run e drifted 0.39 m off a lane 2.5 degrees off it.

        Accepted once ``calibration_min_fits`` sweeps give clean fits (scatter, span,
        heading) that agree within ``calibration_max_spread_m``. Not settled within
        ``calibration_timeout_s``: failed, with the reason, and never a default. The car does
        not move while calibrating, so an empty LiDAR cannot move it (review of 94b5caa).

        Returns "configured", "calibrating", "accepted" or "failed: <why>".
        """
        c = self.config
        if self.keep_right_source == "configured":
            return "configured"
        if self.keep_right_m is not None:
            return "accepted"
        if self.keep_right_source.startswith("calibration failed"):
            return "failed: " + self.keep_right_source.split(": ", 1)[1]
        self.keep_right_source = "calibrating"
        if self._calibration_began_s is None:
            self._calibration_began_s = time_s
        if sweep is not None and not isinstance(sweep, ClassifiedSweep):
            raise TypeError("sweep must be a ClassifiedSweep or None")
        if stationary and sweep is not None:
            stations, contour = self._kerb_contour(sweep)
            fit = self._fit_kerb(stations, contour)
            if fit is not None and fit["clean"]:
                self._fits.append(fit)
            self._last_fit = fit
        summary = self._summarise_calibration(time_s)
        self.calibration_summary = summary
        if summary["good_fits"] >= c.calibration_min_fits and summary["spread_m"] <= c.calibration_max_spread_m:
            self.keep_right_m = summary["distance_m"]
            self.keep_right_source = f"measured at a standstill ({summary['good_fits']} sweeps)"
            summary["status"] = "accepted"
            last = self._fits[-1]
            # The first prior: the fitted kerb moved keep_right along its normal, to the left.
            kerb = np.poly1d(last["coefficients"])
            xs = np.arange(c.start_m, c.calibration_far_m + c.extrapolate_m + 1e-9, c.step_m)
            slope = kerb.deriv()(xs)
            norm = np.hypot(1.0, slope)
            px, py = xs - self.keep_right_m * slope / norm, kerb(xs) + self.keep_right_m / norm
            ys = np.interp(xs, px, py)
            centre_body = np.column_stack((xs, ys))
            self.previous = Corridor(
                time_s=time_s,
                pose=pose,
                centre_body=centre_body,
                centre_odometry=pose.to_world(centre_body),
                valid_length_m=float(xs[-1]),
                coefficients=tuple(float(v) for v in np.polyfit(xs, ys, len(last["coefficients"]) - 1)),
                station_x_m=xs,
                reference_y_m=ys,
                right_edge_m=np.full(len(xs), np.nan),
                left_edge_m=np.full(len(xs), np.nan),
                blocked=np.zeros(len(xs), dtype=bool),
                evidence_body=np.zeros((0, 2)),
                evidence_time_s=time_s,
            )
            return "accepted"
        if time_s - self._calibration_began_s > c.calibration_timeout_s:
            why = self._calibration_problem(summary)
            self.keep_right_source = f"calibration failed: {why}"
            summary["status"] = self.keep_right_source
            return "failed: " + why
        return "calibrating"

    def _kerb_contour(self, sweep):
        """Per station, y of the innermost raised point on the right (NaN where no kerb)."""
        c = self.config
        xyz, cls = sweep.xyz, sweep.cls
        step = c.calibration_station_m
        right = (xyz[:, 1] < -1.0) & (xyz[:, 1] > -c.lateral_search_m)
        stations = np.arange(-c.calibration_behind_m, c.calibration_far_m + 1e-9, step)
        contour = np.full(len(stations), np.nan)
        for i, s in enumerate(stations):
            in_slice = right & (xyz[:, 0] >= s - step / 2) & (xyz[:, 0] < s + step / 2)
            raised = xyz[in_slice & (cls == PointClass.RAISED), 1]
            if raised.size < 3:
                continue
            y = float(raised.max())
            road = xyz[in_slice & (cls == PointClass.ROAD), 1]
            tall = xyz[in_slice & (cls == PointClass.OBSTACLE), 1]
            # Road must be seen just inside it (a pavement's far side is not the kerb) and
            # nothing tall within 1 m (a parked car's sills are kerb-high).
            if not np.any((road > y) & (road < y + 2 * c.max_unseen_m)) or np.any(np.abs(tall - y) < 1.0):
                continue
            contour[i] = y
        return stations, contour

    def _fit_kerb(self, stations, right):
        """Fit one sweep's kerb contour; the rear axle's nearest distance to it.

        A quadratic when the kerb is seen well on both sides of the axle (the axle is then
        bracketed, and a bend measured, not chorded), otherwise a line. Stations more than
        0.3 m from a first fit are dropped before the second.
        """
        c = self.config
        ok = np.isfinite(right)
        if ok.sum() < c.calibration_min_stations or np.ptp(stations[ok]) < c.calibration_min_span_m:
            return None
        x, y = stations[ok], right[ok]
        n_side = c.calibration_min_stations - 2
        degree = 2 if ok.sum() >= 20 and (x < -2).sum() >= n_side and (x > 2).sum() >= n_side else 1
        coefficients = np.polyfit(x, y, degree)
        keep = np.abs(y - np.polyval(coefficients, x)) <= 0.3
        if keep.sum() >= c.calibration_min_stations and not keep.all():
            coefficients = np.polyfit(x[keep], y[keep], degree)
        elif keep.sum() < c.calibration_min_stations:
            keep = np.ones(len(x), dtype=bool)
        residual = np.abs(y[keep] - np.polyval(coefficients, x[keep]))
        xs = np.linspace(float(x[keep].min()), float(x[keep].max()), 400)
        distance = float(np.min(np.hypot(xs, np.polyval(coefficients, xs))))
        slope = float(np.polyval(np.polyder(coefficients), 0.0))
        heading = math.degrees(math.atan(slope))
        span = float(np.ptp(x[keep]))
        fit = dict(
            distance=distance,
            heading_deg=heading,
            curvature=2.0 * float(coefficients[0]) if degree == 2 else 0.0,
            mad=float(np.median(residual)),
            stations=int(keep.sum()),
            behind=int((x[keep] < 0).sum()),
            span=span,
            degree=degree,
            coefficients=tuple(float(v) for v in coefficients),
        )
        fit["clean"] = bool(
            fit["mad"] <= c.calibration_max_mad_m
            and abs(heading) <= c.calibration_max_heading_deg
            and abs(fit["curvature"]) <= 1.0 / c.reference_min_radius_m
            and span >= c.calibration_min_span_m
        )
        return fit

    def _summarise_calibration(self, time_s):
        fits = self._fits
        last = self._last_fit
        summary = dict(
            status="calibrating",
            seconds=round(time_s - self._calibration_began_s, 2),
            good_fits=len(fits),
            last_fit=None if last is None else _rounded(last),
            spread_m=math.inf,
            distance_m=None,
        )
        if fits:
            d = np.array([f["distance"] for f in fits])
            summary.update(
                distance_m=round(float(np.median(d)), 3),
                spread_m=round(float(np.percentile(d, 90) - np.percentile(d, 10)), 3),
                heading_deg=round(float(np.median([f["heading_deg"] for f in fits])), 2),
                curvature=round(float(np.median([f["curvature"] for f in fits])), 4),
                quadratic_share=round(float(np.mean([f["degree"] == 2 for f in fits])), 2),
                mad_m=round(float(np.median([f["mad"] for f in fits])), 3),
                stations=int(np.median([f["stations"] for f in fits])),
            )
        return summary

    def _calibration_problem(self, summary):
        c = self.config
        last = summary["last_fit"]
        if summary["good_fits"] == 0:
            if last is None:
                return f"kerb not seen at {c.calibration_min_stations}+ stations over {c.calibration_min_span_m:g} m"
            return (
                f"kerb fit not clean (scatter {last['mad']} m, heading {last['heading_deg']} deg,"
                f" {last['stations']} stations over {last['span']} m)"
            )
        if summary["good_fits"] < c.calibration_min_fits:
            return f"only {summary['good_fits']} clean kerb fits"
        return f"kerb distance not stable (spread {summary['spread_m']} m)"

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
        # When each right kerb cell was last hit: the evidence is as fresh as the sweeps
        # behind it, not as the tick that reads the grid (review of 94b5caa).
        self._edge_seen_s = np.full(n, -np.inf)
        if take.any():
            rows = np.flatnonzero(take)
            kerb = np.column_stack((
                stations[rows] - offsets[jr[rows]] * np.sin(heading[rows]),
                ref[rows] + offsets[jr[rows]] * np.cos(heading[rows]),
            ))
            self._edge_seen_s[rows] = grid.seen_at(pose.to_world(kerb), PointClass.RAISED)
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

    def _carry_forward(self, grid, pose, time_s, stations, ref, right, left, blocked, evidence, prior_end):
        c = self.config
        previous = self.previous
        centre_body = _carried(previous, pose, prior_end)
        support = self._support(grid, pose, centre_body)
        valid = min(float(prior_end), support)
        centre_body = centre_body[centre_body[:, 0] <= valid] if len(centre_body) else centre_body
        reason = "" if len(centre_body) >= 2 else "previous corridor used up"
        if not reason and valid < c.min_valid_m:
            reason = f"road ahead observed only to {valid:.1f} m"
        # Carrying the line does not renew its evidence; the kerb hits that did arrive do,
        # even when too few to refit (at a standstill only the LiDAR's ring crossings hit the
        # kerb: 4 edges on the live stop run, and the car was faulted for stale evidence
        # while a fresh kerb stood beside it).
        seen = self._edge_seen_s[np.isfinite(self._edge_seen_s)]
        evidence_time = max(previous.evidence_time_s, float(seen.max()) if len(seen) else -math.inf)
        age = time_s - evidence_time
        if not reason and age > c.evidence_max_age_s:
            reason = f"kerb evidence {age:.1f} s old"
        if reason:
            centre_body = np.zeros((0, 2))
        coefficients = previous.coefficients
        if len(centre_body) >= 2:
            degree = min(len(previous.coefficients) - 1, len(centre_body) - 1)
            coefficients = tuple(float(v) for v in np.polyfit(centre_body[:, 0], centre_body[:, 1], degree))
        corridor = Corridor(
            time_s=time_s,
            pose=pose,
            centre_body=centre_body,
            centre_odometry=pose.to_world(centre_body) if len(centre_body) else np.zeros((0, 2)),
            valid_length_m=valid if not reason else 0.0,
            coefficients=coefficients,
            station_x_m=stations,
            reference_y_m=ref,
            right_edge_m=right,
            left_edge_m=left,
            blocked=blocked,
            evidence_body=evidence,
            reason=reason,
            evidence_time_s=evidence_time,
            support_m=float(support),
        )
        if corridor.ok:
            self.previous = corridor
        return corridor

    def update(self, grid, pose, time_s):
        if not isinstance(grid, TerrainGrid) or not isinstance(pose, Pose2D):
            raise TypeError("update(TerrainGrid, Pose2D, time_s)")
        c = self.config
        stations = np.arange(c.start_m, c.horizon_m + 1e-9, c.step_m)
        if self.keep_right_m is None:
            reason = self.keep_right_source if self.keep_right_source.startswith("calibration failed") else "calibrating"
            return Corridor(
                time_s=time_s,
                pose=pose,
                centre_body=np.zeros((0, 2)),
                centre_odometry=np.zeros((0, 2)),
                valid_length_m=0.0,
                coefficients=(0.0, 0.0),
                station_x_m=stations,
                reference_y_m=np.zeros(len(stations)),
                right_edge_m=np.full(len(stations), np.nan),
                left_edge_m=np.full(len(stations), np.nan),
                blocked=np.zeros(len(stations), dtype=bool),
                evidence_body=np.zeros((0, 2)),
                reason=reason,
                calibrating=True,
            )
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
            return self._carry_forward(grid, pose, time_s, stations, ref, accepted_right, left, blocked, evidence, prior_end)
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
        support = 0.0
        if not reason:
            support = self._support(grid, pose, np.column_stack((xs, ys[: len(xs)])))
            if support < valid:
                valid = support
                xs = xs[xs <= valid]
            if valid < c.min_valid_m:
                reason = f"road ahead observed only to {valid:.1f} m"
        # Fresh when enough edges were used, and then as old as the newest kerb hit among them.
        seen = self._edge_seen_s[has][np.isfinite(self._edge_seen_s[has])]
        evidence_time = float(seen.max()) if len(ex) >= c.min_evidence and len(seen) else -math.inf
        if self.previous is not None:
            evidence_time = max(evidence_time, self.previous.evidence_time_s)
        if not reason and time_s - evidence_time > c.evidence_max_age_s:
            reason = f"kerb evidence {time_s - evidence_time:.1f} s old"
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
            evidence_time_s=evidence_time,
            support_m=float(support),
        )
        if corridor.ok:
            self.previous = corridor
        return corridor
