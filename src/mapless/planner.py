"""Lane-following trajectory with stops for obstacles and the route goal (G1b).

The trajectory follows the corridor centreline in the odometry frame. Its speed is the
lowest of:

- the cruise speed;
- the curve limit sqrt(lateral_accel / |curvature|);
- a braking ramp sqrt(2 a d) to a stop point, which is ``stop_gap_m`` short of an
  obstacle's nearest point for the front bumper, or the goal for the rear axle.

Past a stop point every speed is zero. A goal stop is sent as a final-goal trajectory
that ends at the goal, so the controller latches FINISHED there.

**Stop hold.** One sweep can miss an obstacle that the previous ones saw (occlusion,
drop-off). The last obstacle stop point is kept in the odometry frame and holds in one
of three states:

- BLOCKED: an obstacle is seen; the stop point follows it.
- CLEAR: the tube is seen clear (ground observed, nothing standing on it) to
  ``clear_beyond_m`` past where the obstacle was. Only ``resume_clear_s`` of continuous
  CLEAR releases the stop.
- UNKNOWN: neither. The stop stays. A missed or occluded obstacle is not clearance
  (review of 94b5caa; live run e released after 1 s of missed detections and crept 0.6 m).

Each held obstacle keeps its own hold, and the stop is the nearest of them: an
unresolved hold is never made less restrictive because something else was seen (full
review of 73ae759: one sweep that missed the held obstacle 7 m ahead but saw another
22 m ahead moved the stop to 15.7 m and forgot the first). A detection within
``hold_match_m`` along the path of a held obstacle, and no more than
``hold_lateral_match_m`` beside its lateral span, is that obstacle (its stop follows it as
it moves); another beside it is held on its own (third review, F2). Each hold has an id,
and ``Plan.holds`` reports every hold's release prerequisites for the log.
A hold keeps the obstacle's own position (its near face, across its lateral span), and
its stop is worked out from that on each tick's path; on a bend the tube and the region
widen on the outside as ``obstacles.tube_bounds``.

Clear also needs the held obstacle's own region seen: ground in every
``clear_region_bin_m`` lateral bin across the obstacle's lateral span (plus
``clear_region_pad_m``), in the ``clear_region_depth_m`` just behind its near face. Where
it stood the ground was hidden (under it, or in its shadow); once it has gone, that ground
is in view. The road in front of the face proves nothing. "Seen" is judged on the ground
points of every sweep since the obstacle was last detected, gathered on the hold in the
odometry frame (``PathObservation.ground_body``): one sweep's alone reached a 0.5 m strip
only within about 6 m (rings 3 m apart at 18 m), and a walker who had crossed out of the
lane, or a lead that had driven on, held the car almost to a stop (live stack runs on
e93154f). Points, not grid cells: 0.25 m cells leaked the ground beside and in front of a
0.5 m box into its bins. Ground beside a
narrow or off-centre obstacle is not its clearance (same review: a 0.5 m box at the lane
centre, and a car over the right half of the footprint, each "cleared" while still there,
and the car drove into them).
"""

import math
from dataclasses import dataclass

import numpy as np

from mapless.corridor import Corridor
from mapless.obstacles import PathObservation, project_onto_polyline, tube_bounds
from mapless.terrain import VehicleBody
from navigation_control.local import LocalTrajectory, TrajectorySample


@dataclass(frozen=True)
class PlannerConfig:
    cruise_mps: float = 2.0
    lateral_accel_mps2: float = 1.5
    decel_mps2: float = 1.0
    stop_gap_m: float = 2.5
    validity_s: float = 0.5
    resume_clear_s: float = 1.0
    # A held obstacle whose region cannot show ground (the LiDAR's near blind zone: its
    # lowest unblocked ring meets the road beyond the bumper, so no sweep from where the car
    # stands sees the road there) is released once the stack has seen it leave
    # (``mark_departed``: every track on its region a vehicle that has driven clear of it)
    # and it has gone unseen for departed_release_s, the tube beyond seen clear. A car that
    # crossed 1 m in front of the bumper held the car at a left turn for 130 s, its region
    # "0 ground points" for good (PC1, Town03, r9_left_oncoming, b4280cd). Its height is no
    # proof: a person who crouches there drops below the rings.
    departed_release_s: float = 0.5
    # Observability-limited speed (eighth review, P1): the car never drives faster than it
    # can stop stop_gap_m short of the end of the tube it sees clear (``clear_to_m``),
    # reacting after sight_response_s (the measured brake lag, 0.1-0.25 s, BL-E-03) and
    # braking at decel_mps2: v * t + v^2 / (2 a) <= clear - front - stop_gap. Something it
    # has not seen is treated as standing where its view of the road ends. On every live
    # run recorded (r8, r9: 15 runs) this bound lay below the car's speed only with an
    # obstacle in view, where the stop for it binds as well; it is the guarantee, not a
    # new behaviour, at the 2 m/s cruise. ``sight_governed`` False restores the old plan.
    sight_governed: bool = True
    sight_response_s: float = 0.25
    clear_beyond_m: float = 2.0
    clear_gap_s: float = 0.15  # clear observations further apart than this are not continuous
    hold_match_m: float = 2.0  # a detection this close along the path to a held obstacle is that obstacle ...
    hold_lateral_match_m: float = 0.5  # ... unless its lateral span lies further than this beside the hold's
    clear_region_pad_m: float = 0.3  # beyond the held obstacle's lateral span
    clear_region_start_m: float = 0.1  # the region begins this far behind the near face ...
    clear_region_depth_m: float = 0.5  # ... and is this deep: where it stood, not the road in front of it
    clear_region_bin_m: float = 0.3
    clear_region_min_points: int = 1  # ground points in each bin of the region
    clear_region_move_m: float = 0.25  # a sweep joins the gathered ground only from this far on (a new viewpoint)

    def __post_init__(self):
        for name in ("cruise_mps", "lateral_accel_mps2", "decel_mps2", "validity_s"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("stop_gap_m", "resume_clear_s"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class Plan:
    trajectory: LocalTrajectory | None
    stop_reason: str  # "", "obstacle", "obstacle_hold", "goal"
    stop_distance_m: float  # rear-axle distance along the path to the stop point (inf if none)
    curve_speed_mps: float  # lowest curve limit on the trajectory
    hold_state: str = ""  # "", "blocked", "clear", "unknown"
    # Why the binding hold is not clear yet, for the log: the region's lateral bins with
    # ground (seen/all), the ground points gathered since it was last seen, and the
    # clearance along the path against what it needs.
    hold_note: str = ""
    # Every hold this tick, nearest first (``LanePlanner.hold_reports``): its id, state,
    # place and each release prerequisite (third review, F2).
    holds: tuple = ()
    # The observability bound on this plan's speed (``sight_speed``), inf when not applied.
    sight_speed_mps: float = math.inf


def sight_speed(distance_m, decel_mps2, response_s, margin_m):
    """The highest speed from which the car stops ``margin_m`` short of something
    ``distance_m`` ahead of its front, reacting after ``response_s`` and braking at
    ``decel_mps2`` (as ``evaluation.observability.supported_speed``)."""
    room = float(distance_m) - float(margin_m)
    if room <= 0.0:
        return 0.0
    a, t = float(decel_mps2), float(response_s)
    return -a * t + math.sqrt(a * a * t * t + 2.0 * a * room)


def curvature(coefficients, x):
    d1 = np.polyval(np.polyder(coefficients), x)
    d2 = np.polyval(np.polyder(coefficients, 2), x) if len(coefficients) > 2 else np.zeros_like(x)
    return d2 / (1.0 + d1**2) ** 1.5


def polyline_curvature(points):
    """Signed curvature at each vertex of a polyline (finite differences, 1/m)."""
    if len(points) < 3:
        return np.zeros(len(points))
    dx, dy = np.gradient(points[:, 0]), np.gradient(points[:, 1])
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    return (dx * ddy - dy * ddx) / np.maximum((dx**2 + dy**2) ** 1.5, 1e-9)


@dataclass(eq=False)  # compared by identity: the fields are arrays
class _Hold:
    """One held obstacle, in the odometry frame."""

    # 2 x 2: the obstacle's lateral extremes at its near face. The obstacle itself, not a
    # point on the path: the stop is worked out from it on each tick's path, so a path
    # that changes shape does not move the stop (a stop point kept on the bend's old
    # centreline projected 2.4 m behind the car on a straight refit, and released).
    span_xy: np.ndarray
    hold_id: int = 0  # persistent while it is held, for the log
    created_s: float = -math.inf
    seen_s: float = -math.inf  # when it was last detected
    detections: int = 0  # detections matched to it
    ground_odo: np.ndarray | None = None  # ground points (odometry) on its region since then, one sweep per viewpoint
    ground_now: np.ndarray | None = None  # ... and this sweep's, from where the car stands
    gathered_at: np.ndarray | None = None  # where the car was (odometry) for the last sweep added to ground_odo
    note: str = ""  # why it is not clear (see Plan.hold_note)
    # This tick's release prerequisites: the region's lateral bins with ground (seen, all),
    # the tube seen clear to (m) against what it needs, None when not judged this tick.
    region_bins: tuple | None = None
    clear_to: tuple | None = None
    clear_since_s: float | None = None
    clear_last_s: float | None = None  # the last clear observation: continuity is judged against it
    state: str = "blocked"
    departed_s: float | None = None  # when the stack last saw the tracks on its region drive clear of it (``mark_departed``)


class LanePlanner:
    def __init__(self, config, body, tube_half_m=None):
        if not isinstance(config, PlannerConfig) or not isinstance(body, VehicleBody):
            raise TypeError("LanePlanner(PlannerConfig, VehicleBody)")
        self.config = config
        self.body = body
        # The obstacle tube's half-width: a held region is checked only inside it.
        self.tube_half_m = body.half_width_m + 0.35 if tube_half_m is None else float(tube_half_m)
        self._holds = []
        self._next_id = 1

    @property
    def holding(self):
        return bool(self._holds)

    @property
    def hold_count(self):
        return len(self._holds)

    def held_regions(self):
        """Each hold as the stack judges departures on it: (id, the near face's lateral
        extremes in odometry (2 x 2, a copy), when it was last detected)."""
        return tuple((h.hold_id, h.span_xy.copy(), h.seen_s) for h in self._holds)

    def mark_departed(self, hold_id, time_s):
        """The stack saw what stood on hold ``hold_id``'s region drive clear of it at
        ``time_s``: the hold may be released without ground on its region (see
        ``departed_release_s``) once the tube beyond is seen clear for resume_clear_s."""
        for hold in self._holds:
            if hold.hold_id == hold_id:
                hold.departed_s = float(time_s)

    def reset(self, keep_holds=False):
        """Forget the plan state. ``keep_holds``: keep the obstacle holds (the car stood still
        across a frame rebuild, so their odometry positions are still where it is), with
        their clearance counts restarted."""
        if keep_holds:
            for hold in self._holds:
                hold.clear_since_s = hold.clear_last_s = None
                hold.ground_odo = hold.ground_now = hold.gathered_at = None
                hold.state = "unknown"
        else:
            self._holds = []

    def _along(self, corridor, line, xy):
        """Arc length on ``line`` and lateral offset of odometry points, the line continued
        straight past its end. A projection clamps to the end point there and its "lateral"
        is the distance to it: a missed parked car 27 m out on a 24 m line (live stop run,
        27 Oct) put its stop at the line's end, 3.9 m short, and an obstacle to one side of
        the line's continuation could be dropped as off the path."""
        body = corridor.pose.to_body(np.asarray(xy, dtype=float).reshape(-1, 2))
        s, lateral, outside = project_onto_polyline(body, line)
        d = line[-1] - line[-2]
        u = d / max(float(np.hypot(*d)), 1e-9)
        rel = body - line[-1]
        ahead = rel @ u
        past = outside & (ahead > 0)
        s = np.where(past, s + ahead, s)
        lateral = np.where(past, u[0] * rel[:, 1] - u[1] * rel[:, 0], lateral)
        return s, lateral

    def _gather(self, hold, observation, corridor, line, near, lat_lo, lat_hi):
        """Add this sweep's in-tube ground points on the hold's region (with a margin) to it."""
        c = self.config
        if observation is None or observation.ground is None or observation.ground_body is None or not len(observation.ground):
            return
        s_car = float(project_onto_polyline(np.zeros((1, 2)), line)[0][0])
        along, lateral = observation.ground[:, 0] + s_car, observation.ground[:, 1]
        keep = (along >= near - 0.5) & (along <= near + c.clear_region_start_m + c.clear_region_depth_m + 0.5)
        keep &= (lateral >= lat_lo - c.clear_region_pad_m - 0.5) & (lateral <= lat_hi + c.clear_region_pad_m + 0.5)
        if not keep.any():
            return
        pts = corridor.pose.to_world(observation.ground_body[keep])
        # Standing still, every sweep repeats the same rays: their noise, gathered over
        # seconds, put a point in the strip behind a missed pedestrian's feet and released
        # the hold (Stage 3). Only a new viewpoint adds; from where the car stands, the
        # current sweep counts alone.
        here = np.array([corridor.pose.x_m, corridor.pose.y_m])
        if hold.gathered_at is not None and float(np.hypot(*(here - hold.gathered_at))) < c.clear_region_move_m:
            hold.ground_now = pts
            return
        hold.gathered_at, hold.ground_now = here, None
        hold.ground_odo = pts if hold.ground_odo is None else np.vstack((hold.ground_odo, pts))[-4000:]

    def _near(self, corridor, line, hold):
        """A hold's near face along ``line`` and its lateral span there."""
        s, lateral = self._along(corridor, line, hold.span_xy)
        return float(s.min()), float(lateral.min()), float(lateral.max())

    def _span_at(self, corridor, line, arcs, s, lat_lo, lat_hi):
        """Odometry points ``lat_lo`` and ``lat_hi`` (+left) off the line at arc ``s``."""
        k = int(np.clip(np.searchsorted(arcs, s, side="right") - 1, 0, len(line) - 2))
        d = line[k + 1] - line[k]
        normal = np.array([-d[1], d[0]]) / max(float(np.hypot(*d)), 1e-9)
        foot = np.array([np.interp(s, arcs, line[:, 0]), np.interp(s, arcs, line[:, 1])])
        if s > arcs[-1]:
            # Past the line's end, straight on (as ``_along`` reads it back): interpolation
            # clamped the span to the end, and a hold beyond it was placed short of itself.
            foot = line[-1] + (s - arcs[-1]) * np.array([normal[1], -normal[0]])
        return corridor.pose.to_world(np.array([foot + lat_lo * normal, foot + lat_hi * normal]))

    def _region_seen(self, observation, corridor, line, s_car, hold):
        """Ground in every lateral bin of the held obstacle's region (see the module notes)."""
        c = self.config
        if observation is None:
            return False
        near, lat_lo, lat_hi = self._near(corridor, line, hold)
        s_near = near - s_car
        right, left = tube_bounds(line, np.array([near]), self.tube_half_m, self.body)
        lo = max(lat_lo - c.clear_region_pad_m, float(right[0]))
        hi = min(lat_hi + c.clear_region_pad_m, float(left[0]))
        if hi <= lo:
            return True  # the region lies outside the tube: nothing of it is on the path
        bins = max(1, int(math.ceil((hi - lo) / c.clear_region_bin_m)))
        gathered = hold.ground_odo if hold.ground_now is None or hold.ground_odo is None else np.vstack((hold.ground_odo, hold.ground_now))
        if gathered is not None and len(gathered):
            # Every sweep's ground since it was last seen, as points (exact, not grid cells:
            # 0.25 m cells leaked the ground beside and in front of a 0.5 m box into its
            # bins, and the car drove into it). At 18 m the rings are 3 m apart, and one
            # sweep alone put ground in the 0.5 m strip only within about 6 m: a walker who
            # had crossed out of the lane and a lead that had driven on held the car almost
            # to a stop (live stack runs on e93154f). As the car drives, the rings sweep the
            # strip; an obstacle still there keeps it in shadow.
            along, lateral = self._along(corridor, line, gathered)
            along = along - s_car
        else:
            if observation.ground is None or not len(observation.ground):
                return False
            along, lateral = observation.ground[:, 0], observation.ground[:, 1]
        start = s_near + c.clear_region_start_m
        near = (along >= start) & (along <= start + c.clear_region_depth_m) & (lateral >= lo) & (lateral <= hi)
        counts = np.bincount(np.clip(((lateral[near] - lo) / (hi - lo) * bins).astype(int), 0, bins - 1), minlength=bins)
        hold.region_bins = (int((counts >= c.clear_region_min_points).sum()), bins)
        hold.note = f"region bins {hold.region_bins[0]}/{bins}, {int(near.sum())} ground points"
        return bool(np.all(counts >= c.clear_region_min_points))

    def plan(self, corridor, observation, goal_body, now_s, hold_at_m=None, speed_cap_mps=None):
        """``observation``: this sweep's ``PathObservation`` (None without a sweep this tick).

        ``goal_body``: the route goal in the vehicle frame, when it is within reach.
        ``hold_at_m``: stop the rear axle this far along the path (a junction the car may
        not enter yet); reported as stop reason "junction".
        ``speed_cap_mps``: a lower cruise speed for this plan (lead-car following).
        """
        if not isinstance(corridor, Corridor):
            raise TypeError("corridor must be a Corridor")
        if observation is not None and not isinstance(observation, PathObservation):
            raise TypeError("observation must be a PathObservation or None")
        c = self.config
        if not corridor.ok:
            for hold in self._holds:  # no corridor: clearance is not being observed
                hold.clear_since_s = hold.clear_last_s = None
                hold.state = "unknown"
            return Plan(None, "", math.inf, math.inf, "unknown" if self._holds else "")
        line = corridor.centre_body
        arcs = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(line, axis=0).T))))
        s_car = float(project_onto_polyline(np.zeros((1, 2)), line)[0][0])

        stop_s, reason, state = math.inf, "", ""
        obstacle = None if observation is None else observation.obstacle
        seen = None
        if obstacle is not None:
            near_s = s_car + obstacle.distance_m
            lat_lo = obstacle.lateral_m if obstacle.lateral_min_m is None else obstacle.lateral_min_m
            lat_hi = obstacle.lateral_m if obstacle.lateral_max_m is None else obstacle.lateral_max_m
            # The same obstacle as a hold already kept: within hold_match_m along the path,
            # and beside it across the path no further than hold_lateral_match_m (third
            # review, F2: by distance along alone, a second obstacle at a similar distance
            # was merged into the first and their clearance evidence with it). A walker
            # crossing stays one hold: each sweep's span overlaps the one before.
            for hold in self._holds:
                near_h, lo_h, hi_h = self._near(corridor, line, hold)
                gap = abs(near_h - near_s)
                beside = max(0.0, lat_lo - hi_h, lo_h - lat_hi)
                if gap <= c.hold_match_m and beside <= c.hold_lateral_match_m and (seen is None or gap < seen[0]):
                    seen = (gap, hold)
            if seen is not None:
                # While it is seen, its region is all it has covered (a walker crossing widens it).
                _, held_lat = self._along(corridor, line, seen[1].span_xy)
                lat_lo, lat_hi = min(lat_lo, float(held_lat.min())), max(lat_hi, float(held_lat.max()))
            span = self._span_at(corridor, line, arcs, near_s, lat_lo, lat_hi)
            if seen is None:
                hold = _Hold(span, hold_id=self._next_id, created_s=now_s)
                self._next_id += 1
                self._holds.append(hold)
            else:
                hold = seen[1]
                hold.span_xy = span
            hold.seen_s = now_s
            hold.detections += 1
            hold.region_bins = hold.clear_to = None
            hold.ground_odo = hold.ground_now = hold.gathered_at = None
            hold.clear_since_s = hold.clear_last_s = None
            hold.state = "blocked"
            seen = hold
        released = False
        for hold in list(self._holds):
            if hold is seen:
                continue
            near, lat_lo, lat_hi = self._near(corridor, line, hold)
            right, left = tube_bounds(line, np.array([near]), self.tube_half_m, self.body)
            if near < s_car or lat_lo > float(left[0]) + 1.0 or lat_hi < float(right[0]) - 1.0:
                # Behind the rear axle, or off this path (the line moved on: a turn, a
                # re-acquired corridor): no longer an obstacle the car can drive into.
                self._holds.remove(hold)
                continue
            # Clear to clear_beyond_m past the held region, but never past the line's end:
            # beyond it the path is not known yet. The car cannot drive past the line; when it
            # grows, its obstacle check looks again. The held region itself must still show
            # ground (or its object be seen to leave, below). A hold 0.8 m past a 6.8 m corridor
            # could never be cleared (live stack turn on ff1910b); nor, while the clamp applied
            # only to holds past the end, could one at 6.5 m on a 7.1 m corridor: it needed
            # 8.5 m seen clear, the corridor showed 7.5, and the car stood for 115 s with the
            # region's ground all seen (PC1, Town03, r9_left_parked, b4280cd).
            line_end = float(np.hypot(*np.diff(line, axis=0).T).sum()) - s_car
            needed = min(near - s_car + c.clear_beyond_m, line_end - 0.5)
            self._gather(hold, observation, corridor, line, near, lat_lo, lat_hi)
            hold.region_bins = None
            region = self._region_seen(observation, corridor, line, s_car, hold)
            hold.clear_to = None if observation is None else (observation.clear_to_m, needed)
            if observation is not None:
                hold.note += f"; clear to {observation.clear_to_m:.1f} m of {needed:.1f}"
            departed = not region and hold.departed_s is not None and hold.departed_s > hold.seen_s and now_s - hold.seen_s >= c.departed_release_s
            if departed:
                hold.note += "; its object seen to leave"
            if observation is not None and observation.clear_to_m >= needed and (region or departed):
                hold.state = "clear"
                # Continuous means every tick: a gap (a fault, a missing sweep, the planner
                # not called) restarts the count (review of 2b7e14d: one clear observation,
                # 3 s of nothing, one more, and the stop released).
                if hold.clear_since_s is None or now_s - hold.clear_last_s > c.clear_gap_s:
                    hold.clear_since_s = now_s
                hold.clear_last_s = now_s
                if now_s - hold.clear_since_s >= c.resume_clear_s:
                    self._holds.remove(hold)
                    released = True
            else:
                hold.state = "unknown"
                hold.clear_since_s = hold.clear_last_s = None
        binding = None
        for hold in self._holds:
            held_stop = self._near(corridor, line, hold)[0] - self.body.front_m - c.stop_gap_m
            if binding is None or held_stop < binding[0]:
                binding = (held_stop, hold)
        note = ""
        if binding is not None:
            stop_s, state = binding[0], binding[1].state
            reason = "obstacle" if binding[1] is seen else "obstacle_hold"
            note = "" if binding[1] is seen else binding[1].note
        elif released:
            state = "clear"

        if hold_at_m is not None and math.isfinite(hold_at_m) and s_car + hold_at_m < stop_s:
            stop_s, reason = s_car + hold_at_m, "junction"

        final_goal = False
        end = len(line)
        if goal_body is not None:
            goal_s, _, outside = project_onto_polyline(np.asarray(goal_body, dtype=float)[None], line)
            if not outside[0] and goal_s[0] < stop_s:
                stop_s, reason, final_goal = float(goal_s[0]), "goal", True
                end = int(np.searchsorted(arcs, stop_s)) + 1

        k = np.abs(polyline_curvature(line))
        curve = np.sqrt(c.lateral_accel_mps2 / np.maximum(k, 1e-6))
        speed = np.minimum(c.cruise_mps if speed_cap_mps is None else min(c.cruise_mps, max(0.0, speed_cap_mps)), curve)
        speed = np.minimum(speed, np.sqrt(2.0 * c.decel_mps2 * np.maximum(0.0, stop_s - arcs)))
        sight = math.inf
        if c.sight_governed and observation is not None:
            # From where the car is: the plan is made again every tick as the view grows.
            sight = sight_speed(observation.clear_to_m - self.body.front_m, c.decel_mps2, c.sight_response_s, c.stop_gap_m)
            speed = np.minimum(speed, sight)
        speed[arcs >= stop_s] = 0.0
        points, speeds = line[:end], speed[:end]
        if final_goal:
            goal_xy = (np.interp(stop_s, arcs, line[:, 0]), np.interp(stop_s, arcs, line[:, 1]))
            keep = arcs[:end] < stop_s - 0.05
            points = np.vstack((points[keep], goal_xy))
            speeds = np.append(speeds[keep], 0.0)
            if len(points) < 2:
                points = np.vstack((line[:1], goal_xy))
                speeds = np.array([0.0, 0.0])
        odometry = corridor.pose.to_world(points)
        samples = tuple(TrajectorySample(float(x), float(y), float(v)) for (x, y), v in zip(odometry, speeds))
        trajectory = LocalTrajectory(
            samples=samples,
            created_s=now_s,
            valid_until_s=now_s + c.validity_s,
            manoeuvre_id="goal_stop" if final_goal else "lane_follow",
            final_goal=final_goal,
        )
        return Plan(
            trajectory,
            reason,
            stop_s - s_car if math.isfinite(stop_s) else math.inf,
            float(curve.min()),
            state,
            note,
            self.hold_reports(corridor, line, s_car, now_s, seen),
            sight,
        )

    def hold_reports(self, corridor, line, s_car, now_s, seen=None, limit=4):
        """Each hold, nearest first, with why it is still held (third review, F2): id,
        state, near face from the rear axle and lateral span (m), seconds since it was last
        detected and detections matched to it, and the release prerequisites as judged this
        tick: ``region`` (lateral bins of its region with ground, seen/all), ``clear_to``
        (tube seen clear to, of what it needs, m) and ``clear_for_s`` (continuous clear
        time, of ``resume_clear_s``). ``waiting_for`` names what is missing."""
        c = self.config
        out = []
        for hold in self._holds:
            near, lo, hi = self._near(corridor, line, hold)
            if hold is seen:
                waiting = "detected this tick"
            elif hold.clear_to is None:
                waiting = "no sweep this tick"
            elif hold.clear_to[0] < hold.clear_to[1]:
                waiting = "the tube seen clear far enough"
            elif hold.region_bins is not None and hold.region_bins[0] < hold.region_bins[1]:
                waiting = "ground across its region"
            elif hold.region_bins is None and hold.state != "clear":
                waiting = "ground on its region"
            else:
                waiting = "continuous clearance"
            out.append(
                dict(
                    id=hold.hold_id,
                    state=hold.state,
                    near_m=round(near - s_car, 2),
                    lateral_m=[round(lo, 2), round(hi, 2)],
                    seen_ago_s=round(now_s - hold.seen_s, 2),
                    detections=hold.detections,
                    region=None if hold.region_bins is None else f"{hold.region_bins[0]}/{hold.region_bins[1]}",
                    clear_to=None if hold.clear_to is None else [round(hold.clear_to[0], 1), round(hold.clear_to[1], 1)],
                    clear_for_s=None if hold.clear_since_s is None else round(now_s - hold.clear_since_s, 2),
                    resume_after_s=c.resume_clear_s,
                    waiting_for=waiting,
                )
            )
        out.sort(key=lambda r: r["near_m"])
        return tuple(out[:limit])
