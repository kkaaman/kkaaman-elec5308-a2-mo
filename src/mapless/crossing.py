"""Straight through a junction (a STRAIGHT command on the sat-nav route; M13, the mixed route,
and the baseline route's crossing of its one junction).

Kerb following has no kerb across a junction. Worse, it has the wrong one: the right kerb
curves away into the side road's fillet, each station within the continuity limit of the
last, and the fitted centreline follows it into the mouth (synthetic crossroads, 14 m wide:
the car drifted a metre right and stopped for the side road's hedge). The turn code latches
the lane's heading before the mouth for the same reason.

So does this. On the approach, while the entry point is still ``freeze_m`` or more ahead,
the lane line (the corridor's centre at the car, along the kerb evidence beside and
behind it) is latched in the odometry frame each tick. From ``enter_m`` before the entry
the car follows that line instead of the kerb fit: a corridor with no edges, valid as far
as the road along it is observed (``support``), so the obstacle stop and the planner's ramp
work on it as on any corridor. The line is the path's first part: once the route's exit
point and the points after it are in view the path is latched in the odometry frame as
the line to the route's exit point, then an arc tangent to it whose curvature fits the
route points after the exit (``_build_path``; the exit lane need not continue the entry
lane's: Town10HD junction 468, 43 m long, its exit lane curving 50 degrees over 30 m).
From ``handback_from_entry_m`` past the entry the kerb corridor is seeded from the path
and beside it; what it sees of the kerb moves the path sideways (the odometry drifts a
metre over a kerbless box) and, agreeing with the path at the car and running on towards
the route, is handed the car — the kerb corridor follows a curving exit as no line or
sat-nav arc does. Past ``give_up_m`` beyond the exit with no kerb, the driver's ordinary
"no corridor" fault stands.

What the car does not do here is look for traffic: a junction crossed on a STRAIGHT is a
give-way or signalled one, and the lights (M07) and crossing traffic are the stack's and
later work's.
"""

import math
from dataclasses import dataclass

import numpy as np

from mapless.corridor import Corridor


@dataclass(frozen=True)
class CrossingConfig:
    enabled: bool = True  # False: STRAIGHT keeps kerb following (the failure the crossing is for, in its test)
    latch_from_m: float = 30.0  # latch the lane line while the entry is within this ...
    freeze_m: float = 15.0  # ... and at least this far ahead (the kerb fit tilts into the fillet from ~16 m)
    history_m: float = 40.0  # the line's direction is the car's own track over this far behind it ...
    history_min_m: float = 20.0  # ... when at least this much of it is known (else the kerb evidence beside the car)
    enter_m: float = 12.0  # follow the latched line from this far before the entry (the kerb fit tilts into the fillet from ~16 m)
    reacquire_m: float = 8.0  # try the kerb corridor again from this far past the exit (the exit kerb begins past the fillet)
    handback_offset_m: float = 0.75  # the kerb corridor must lie within this of the path at the car ...
    handback_heading_rad: float = 0.18  # ... and in direction, to be handed to
    # The dead-reckoned line drifts (r12, Town10HD: 0.7-1.5 m over a 43 m junction, the
    # odometry's heading with nothing to correct it). Into the junction the kerb corridor is
    # seeded from the path at these lateral offsets (m, left positive) and the first fit
    # with kerb evidence re-anchors the line to it, by this share of the difference a tick.
    seed_offsets_m: tuple = (0.0, -0.75, 0.75, -1.5, 1.5, -2.25, 2.25, -3.0, 3.0, -3.75, 3.75)  # the exit line fitted 2.5 m left (route noise), the kerb 4.3 m right of the path: the near seeds missed it
    seed_every_ticks: int = 4  # the search runs every this many ticks (its cost: one kerb fit per seed)
    reanchor_gain: float = 0.5
    # The crossing's exit is a curve on Town10HD 468 (50 degrees over 30 m from the exit
    # point, r13): no exit *line* fits it, the kerb corridor does. Hand back as soon as a
    # kerb fit agrees with the path at the car and runs to within this of the route past
    # the exit (a side road's fillet runs away from it).
    # The latched line lives in the odometry frame, which turns with the gyro's bias
    # residual (the stationary estimate is 1 s long; r12/r13: 0.4-1.5 m off by the end of a
    # 45 m kerbless box, a 2-sigma seed 2.6 m). The compass does not drift. The line's
    # heading follows the smoothed odometry-minus-compass yaw since the latch, rotating
    # about the car's foot on it. (The route EKF's yaw was tried as the reference: it
    # integrates the raw gyro between compass updates and drifts more than the odometry.)
    heading_hold: bool = True
    heading_hold_alpha: float = 0.01  # EMA weight a tick on the yaw gap (5 s at 20 Hz: compass noise 0.03 rad -> ~0.002)
    handback_route_m: float = 5.0
    reanchor_past_exit_m: float = 8.0  # kerb evidence up to this far past the arc's start (it has barely begun) still moves the path sideways
    handback_span_m: float = 6.0  # the fit's kerb evidence spans at least this ...
    handback_min_evidence: int = 7  # ... with at least this many lane-centre points (the far kerb is sparse: 5-8 stations at 20-30 m)
    reanchor_max_m: float = 4.0  # a fit further off than this is some other kerb (the exit line's own error is 2-3 m at times: route noise, seed 5314 of the bend test's sweep)
    give_up_m: float = 25.0  # past the exit without a kerb: the crossing is over, the fault stands
    look_ahead_m: float = 40.0
    step_m: float = 1.0
    start_m: float = -2.0
    max_route_offset_m: float = 6.0  # the entry point must lie this near the lane line, laterally (3 sigma of a 2 m route noise; a 3.8 m point read as 4.4 m from the car's own position error, and the crossing never engaged)
    # The exit lane need not continue the entry lane's line (Town10HD junction 468: 43 m
    # long, its exit lane a curve of 50 degrees over 30 m, r13). The path is the line to
    # the route's exit point, then an arc whose curvature fits the route points after it.
    exit_curvature_min: float = 1.0 / 400.0  # a gentler arc is driven straight on the line
    exit_curvature_max: float = 1.0 / 15.0  # a STRAIGHT's exit is no tighter than this
    exit_lead_m: float = 10.0  # the arc begins this far before the route's exit point (468: the lane bends 7 m before it; the synthetic bend, 14 m)
    # The crossing's exit is a curve on Town10HD 468: no exit *line* fits it, the kerb
    # corridor does. Hand back as soon as a kerb fit agrees with the path at the car and
    # runs to within this of the route past the exit (a side road's fillet runs away from it).
    # The latched line lives in the odometry frame, which turns with the gyro's bias
    # residual (the stationary estimate is 1 s long; r12/r13: 0.4-1.5 m off by the end of a
    # 45 m kerbless box, a 2-sigma seed 2.6 m). The compass does not drift. The line's
    # heading follows the smoothed odometry-minus-compass yaw since the latch, rotating
    # about the car's foot on it. (The route EKF's yaw was tried as the reference: it
    # integrates the raw gyro between compass updates and drifts more than the odometry.)
    heading_hold: bool = True
    heading_hold_alpha: float = 0.01  # EMA weight a tick on the yaw gap (5 s at 20 Hz: compass noise 0.03 rad -> ~0.002)
    handback_route_m: float = 5.0
    reanchor_past_exit_m: float = 8.0  # kerb evidence up to this far past the arc's start (it has barely begun) still moves the path sideways
    handback_span_m: float = 6.0  # the fit's kerb evidence spans at least this ...
    handback_min_evidence: int = 7  # ... with at least this many lane-centre points (the far kerb is sparse: 5-8 stations at 20-30 m)
    handback_from_entry_m: float = 10.0  # the kerb corridor may take over from this far past the entry

    def __post_init__(self):
        if not 0.0 < self.freeze_m < self.latch_from_m:
            raise ValueError("CrossingConfig: 0 < freeze_m < latch_from_m")
        if self.enter_m < 0.0 or self.reacquire_m < 0.0 or self.give_up_m <= self.reacquire_m:
            raise ValueError("CrossingConfig: enter_m, reacquire_m >= 0 and give_up_m > reacquire_m")
        if self.look_ahead_m <= 5.0 or self.step_m <= 0.0:
            raise ValueError("CrossingConfig: look_ahead_m > 5 and step_m > 0")
        if not 0.0 < self.history_min_m <= self.history_m:
            raise ValueError("CrossingConfig: 0 < history_min_m <= history_m")


def _wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _fit_arc_curvature(points, kappa_max=0.1, steps=401):
    """Curvature of the arc tangent to the x axis at the origin that best fits ``points``
    (N x 2: along and across the line, metres, along > 0), by a grid search; 0 when no
    point fits any arc (all beyond the arc's quarter turn)."""
    best, best_err = 0.0, math.inf
    for kappa in np.linspace(-kappa_max, kappa_max, steps):
        if abs(kappa) < 1e-6:
            err = float(np.sum(points[:, 1] ** 2))
        else:
            sin_phi = kappa * points[:, 0]
            if np.any(np.abs(sin_phi) >= 1.0):
                continue
            across = (1.0 - np.cos(np.arcsin(sin_phi))) / kappa
            err = float(np.sum((points[:, 1] - across) ** 2))
        if err < best_err:
            best, best_err = float(kappa), err
    return best


class StraightCrossing:
    """The latched lane line and the crossing's state; one instance per driver."""

    def __init__(self, config=None):
        self.config = config or CrossingConfig()
        self._yaw_gap = None  # smoothed odometry yaw minus compass yaw, over the whole drive
        self.reset()

    def reset(self):
        self.line = None  # (x, y, heading) in the odometry frame
        self.active = False
        self.entry_i = None
        self.note = ""
        self._track = []  # the rear axle's odometry positions, oldest first (the lane's direction)
        self.path_odo = None  # the latched path (line, arc) in the odometry frame, once the exit is known
        self._geometry = None
        self.bend_rad = 0.0
        self._yaw_gap_ref = None  # the smoothed odometry-minus-compass yaw at the latch ...
        self._hold_applied = 0.0  # ... and how far the line has been turned to follow it since


    def observe(self, pose, moving, compass_yaw_rad=None):
        """Each tick: the car's own track, for the lane's direction, and the smoothed gap
        between the odometry's yaw and the compass's, for the heading hold."""
        if compass_yaw_rad is not None:
            gap = _wrap_angle(pose.yaw_rad - compass_yaw_rad)
            self._yaw_gap = gap if self._yaw_gap is None else _wrap_angle(self._yaw_gap + self.config.heading_hold_alpha * _wrap_angle(gap - self._yaw_gap))
        if not moving:
            return
        here = (pose.x_m, pose.y_m)
        if self._track and math.dist(self._track[-1], here) < 0.25:
            return
        self._track.append(here)
        while len(self._track) > 2 and math.dist(self._track[0], here) > self.config.history_m:
            self._track.pop(0)

    def _track_heading(self):
        """Direction of the car's track over the last history_m, or None when too short."""
        if len(self._track) < 4 or math.dist(self._track[0], self._track[-1]) < self.config.history_min_m:
            return None
        xy = np.asarray(self._track, dtype=float)
        d = xy - xy.mean(axis=0)
        _, _, vt = np.linalg.svd(d, full_matrices=False)
        direction = vt[0]
        if np.dot(direction, xy[-1] - xy[0]) < 0:
            direction = -direction
        return float(math.atan2(direction[1], direction[0]))

    def latch(self, corridor, pose, entry_body):
        """On the approach: remember the lane line while the kerb beside the car is good.
        ``entry_body`` is the route's entry point in the vehicle frame. The direction is
        the car's own track (it has been following the lane centre: the kerb fit's slope
        at the car is tilted by the fillet 15 m on, 1.5 degrees on the synthetic road); the
        lateral position is the corridor's at the car."""
        c = self.config
        if not c.enabled or self.active or not corridor.ok:
            return
        ahead = float(entry_body[0])
        if not c.freeze_m <= ahead <= c.latch_from_m or abs(float(entry_body[1])) > c.max_route_offset_m:
            return
        track = self._track_heading()
        heading = pose.yaw_rad + corridor.heading_near() if track is None else track
        centre = pose.to_world(np.array([[0.0, corridor.lateral_offset_at(0.0)]]))[0]
        self.line = (float(centre[0]), float(centre[1]), float(heading))
        self._yaw_gap_ref, self._hold_applied = self._yaw_gap, 0.0
        self.note = "lane line latched" + (" (kerb heading)" if track is None else "")

    def should_enter(self, entry_body):
        return self.line is not None and not self.active and float(entry_body[0]) <= self.config.enter_m

    def enter(self, entry_i):
        self.active, self.entry_i = True, entry_i
        self.note = "crossing on the latched lane line"

    def hold_heading(self, pose):
        """Turn the line (and the path) about the car's foot on it by what the odometry's
        yaw has drifted from the compass's since the latch; returns the turn applied so far
        (radians, left positive)."""
        c = self.config
        if not c.heading_hold or self.line is None or self._yaw_gap is None or self._yaw_gap_ref is None:
            return 0.0
        drift = _wrap_angle(self._yaw_gap - self._yaw_gap_ref)  # the odometry's yaw high by this: the line's odometry heading rises with it
        turn = drift - self._hold_applied
        if abs(turn) < 1e-4:
            return self._hold_applied
        x0, y0, h = self.line
        u = np.array([math.cos(h), math.sin(h)])
        origin = np.array([x0, y0])
        car = np.array([pose.x_m, pose.y_m])
        foot = origin + float(np.dot(car - origin, u)) * u
        h2 = h + turn
        self.line = (float(foot[0]), float(foot[1]), float(h2))
        self._hold_applied = drift
        if self._geometry is not None:
            g = self._geometry
            rot = np.array([[math.cos(turn), -math.sin(turn)], [math.sin(turn), math.cos(turn)]])
            g["e"] = foot + rot @ (g["e"] - foot)
            g["foot"], g["u"] = foot, np.array([math.cos(h2), math.sin(h2)])
            self._assemble()
        return self._hold_applied

    def line_body(self, pose, length_m=None):
        """The latched line in the vehicle frame: N x 2 points every step_m from start_m."""
        c = self.config
        x0, y0, h = self.line
        length = c.look_ahead_m if length_m is None else length_m
        here = pose.to_body(np.array([[x0, y0]]))[0]
        rel = h - pose.yaw_rad
        # Project the car onto the line, then march along it.
        along = -(here[0] * math.cos(rel) + here[1] * math.sin(rel))
        foot = here + along * np.array([math.cos(rel), math.sin(rel)])
        s = np.arange(c.start_m, length + 1e-9, c.step_m)
        return np.column_stack((foot[0] + s * math.cos(rel), foot[1] + s * math.sin(rel)))

    def path_body(self, pose, exit_body=None, after_body=None):
        """The crossing path in the vehicle frame, from 2 m behind the car on. Latched in
        the odometry frame the first time the exit direction is known (``_build_path``):
        the line, a blend, the exit line; re-anchoring shifts it. Before that, the line."""
        if self.path_odo is None and exit_body is not None and after_body is not None:
            self._build_path(pose, exit_body, after_body)
        if self.path_odo is None:
            return self.line_body(pose)
        body = pose.to_body(self.path_odo)
        here = int(np.argmin(np.hypot(body[:, 0], body[:, 1])))
        first = max(0, here - int(round(-self.config.start_m / self.config.step_m)))
        out = body[first:]
        folds = np.flatnonzero(np.diff(out[:, 0]) <= 0.0)  # the arc turned past the car's x axis
        if len(folds):
            out = out[: folds[0] + 1]
        if len(out) < 4:
            return self.line_body(pose)
        return out

    def _build_path(self, pose, exit_body, after_body):
        """The latched line to the route's exit point, then an arc tangent to it whose
        curvature fits the route points after the exit (``after_body``, k x 2: within 60 m
        of the exit), as a polyline in the odometry frame (``_assemble``). Only the
        points' positions along and across the line are used, never the exit point's own
        position across it: the sat-nav's 2 m of noise across the line bent the path on
        the synthetic crossroads by 1.5 m, and a line fitted through three points on a
        curving exit (Town10HD 468, r13: 50 degrees over 30 m) sat 3 m to the wrong side
        and 40 degrees off. The arc is the model of such an exit; the kerb corridor, handed
        the car as soon as it sees the exit kerb, is its measurement."""
        c = self.config
        x0, y0, h = self.line
        u = np.array([math.cos(h), math.sin(h)])
        left = np.array([-u[1], u[0]])
        pts = pose.to_world(np.vstack([np.asarray(exit_body, dtype=float)[None, :], np.asarray(after_body, dtype=float).reshape(-1, 2)]))
        car = np.array([pose.x_m, pose.y_m])
        foot = np.array([x0, y0]) + float(np.dot(car - np.array([x0, y0]), u)) * u  # the car's projection onto the line
        rel = pts - foot
        along, across = rel @ u, rel @ left
        along_e = float(along[0]) - c.exit_lead_m  # where the arc begins
        after = np.array([(a - along_e, x) for a, x in zip(along[1:], across[1:]) if a > along_e + 2.0])
        kappa = _fit_arc_curvature(after) if len(after) else 0.0
        if abs(kappa) > c.exit_curvature_max:
            kappa = math.copysign(c.exit_curvature_max, kappa)
        straight = abs(kappa) < c.exit_curvature_min or along_e < 5.0
        self._geometry = {"foot": foot, "u": u, "e": foot + along_e * u, "along_e": along_e, "kappa": 0.0 if straight else float(kappa), "straight": straight}
        self.bend_rad = 0.0 if straight else float(kappa * min(30.0, 60.0))  # the turn over the first 30 m of exit, for the record
        self._assemble()

    def _assemble(self):
        """``path_odo`` from the geometry: the line to the exit point, then the arc."""
        c = self.config
        g = self._geometry
        foot, u, kappa = g["foot"], g["u"], g["kappa"]
        along_e = g["along_e"] = float(np.dot(g["e"] - foot, u))
        left = np.array([-u[1], u[0]])
        if g["straight"]:
            n = int((max(c.look_ahead_m, along_e + 60.0) - c.start_m) / c.step_m) + 1
            self.path_odo = np.array([foot + (c.start_m + k * c.step_m) * u for k in range(n)])
            return
        before = np.array([foot + (c.start_m + k * c.step_m) * u for k in range(max(0, int((along_e - c.start_m) / c.step_m) + 1))]).reshape(-1, 2)
        e = foot + along_e * u
        arc = []
        for k in range(1, int(60.0 / c.step_m) + 1):
            phi = kappa * k * c.step_m
            if abs(phi) > math.radians(70):
                break  # further round, the path would fold back on the car's x axis (the kerb corridor's stations are x)
            arc.append(e + (math.sin(phi) / kappa) * u + ((1.0 - math.cos(phi)) / kappa) * left)
        self.path_odo = np.vstack([before, np.array(arc).reshape(-1, 2)])

    def past_exit_point_m(self, pose, point_body):
        """How far a point of the path (vehicle frame) lies past the arc's start (the route's
        exit point less ``exit_lead_m``) along the latched line (negative: short of it);
        None before the path is built."""
        g = getattr(self, "_geometry", None)
        if g is None:
            return None
        p = pose.to_world(np.asarray(point_body, dtype=float)[None, :])[0]
        return float(np.dot(p - g["foot"], g["u"])) - g["along_e"]

    def corridor(self, grid, pose, time_s, support, exit_body=None, after_body=None):
        """A corridor along the crossing path (``path_body``), valid as far as the road
        along it is observed (``support(grid, pose, centre_body)``: the kerb corridor's rule)."""
        points = self.path_body(pose, exit_body, after_body)
        seen = float(support(grid, pose, points))
        valid = max(float(points[1, 0]), seen)
        n = len(points)
        near = points[points[:, 0] <= 30.0]
        degree = 3 if len(near) >= 8 else 1
        coefficients = tuple(float(v) for v in np.polyfit(near[:, 0], near[:, 1], degree)) if len(near) >= 2 else (0.0, float(points[0, 1]))
        return Corridor(
            time_s=time_s,
            pose=pose,
            centre_body=points,
            centre_odometry=pose.to_world(points),
            valid_length_m=valid,
            coefficients=coefficients,
            station_x_m=points[:, 0],
            reference_y_m=points[:, 1],
            right_edge_m=np.full(n, np.nan),
            left_edge_m=np.full(n, np.nan),
            blocked=np.zeros(n, dtype=bool),
            evidence_body=np.zeros((0, 2)),
            evidence_time_s=time_s,
            support_m=seen,
        )

    def past_exit_m(self, exit_body):
        """How far the rear axle is past the route's exit point (negative: short of it)."""
        return -float(exit_body[0])

    def agrees(self, corridor, pose, path=None):
        """Whether a kerb corridor matches the crossing path at the car: the fit past the
        exit grabs the far fillet first (synthetic crossroads: a 2.6 m jump at the hand-back)."""
        c = self.config
        if not corridor.ok:
            return False
        path = self.line_body(pose, length_m=10.0) if path is None else path
        ahead = path[path[:, 0] >= -0.5]
        if len(ahead) < 2:
            return False
        at_car = float(np.interp(0.0, path[:, 0], path[:, 1]))
        heading = math.atan2(ahead[min(5, len(ahead) - 1), 1] - ahead[0, 1], ahead[min(5, len(ahead) - 1), 0] - ahead[0, 0])
        return abs(corridor.lateral_offset_at(0.0) - at_car) <= c.handback_offset_m and abs(corridor.heading_at(0.0) - heading) <= c.handback_heading_rad

    def reanchor(self, pose, lateral_m, heading_rad=0.0):
        """Move the path sideways towards a kerb fit: ``lateral_m`` left of the path where
        the fit's evidence is (at the car, in the junction, the fit is its seed). Half the
        difference a time; a fit further off than ``reanchor_max_m`` is some other kerb."""
        c = self.config
        if self.line is None or abs(lateral_m) > c.reanchor_max_m:
            return False
        x0, y0, h = self.line
        shift = c.reanchor_gain * lateral_m
        n = np.array([-math.sin(pose.yaw_rad), math.cos(pose.yaw_rad)])  # the car's left, odometry frame
        self.line = (float(x0 + shift * n[0]), float(y0 + shift * n[1]), float(h))
        if self._geometry is not None:
            self._geometry["foot"] = self._geometry["foot"] + shift * n
            self._geometry["e"] = self._geometry["e"] + shift * n
            self._assemble()  # the path moves sideways with the line
        self.note = "crossing on the lane line, re-anchored to a kerb"
        return True

    def may_try_kerb(self, entry_body):
        """Past handback_from_entry_m beyond the entry, the kerb corridor is tried every tick."""
        return self.active and -float(entry_body[0]) >= self.config.handback_from_entry_m

    def may_reacquire(self, exit_body):
        return self.active and self.past_exit_m(exit_body) >= self.config.reacquire_m

    def given_up(self, exit_body):
        return self.active and self.past_exit_m(exit_body) >= self.config.give_up_m
