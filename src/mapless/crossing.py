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
the line to ``transition_m`` before the exit, a cubic blend onto the exit lane's line (the
least-squares line through those points) and that line (``_build_path``; the exit lane
need not continue the entry lane's: Town10HD junction 468, 43 m long, bends 22 degrees
right in its last 10 m). From ``handback_from_entry_m`` past the entry the kerb corridor
is seeded from the path and beside it; what it sees of the kerb re-anchors the path (the
exit line in position and direction, the latched line sideways: the odometry drifts a
metre over a kerbless box, the sat-nav's exit line is a metre or two and several degrees
out) and, agreeing with the path at the car, is handed the car. Past ``give_up_m`` beyond
the exit with no kerb, the driver's ordinary "no corridor" fault stands.

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
    exit_turn_max_rad: float = 0.45  # kerb evidence along the exit line may turn it by up to this (its direction is the sat-nav's)
    reanchor_max_m: float = 4.0  # a fit further off than this is some other kerb (the exit line's own error is 2-3 m at times: route noise, seed 5314 of the bend test's sweep)
    give_up_m: float = 25.0  # past the exit without a kerb: the crossing is over, the fault stands
    look_ahead_m: float = 40.0
    step_m: float = 1.0
    start_m: float = -2.0
    max_route_offset_m: float = 6.0  # the entry point must lie this near the lane line, laterally (3 sigma of a 2 m route noise; a 3.8 m point read as 4.4 m from the car's own position error, and the crossing never engaged)
    # The exit lane need not continue the entry lane's line (Town10HD junction 468: 43 m long,
    # its last 10 m bending 25 degrees right into the exit road). The path is the line, a
    # cubic blend from transition_m before the route's exit point to that point in the
    # direction of the point after it, then that exit line.
    bend_min_rad: float = 0.05  # a smaller deflection between the line and the exit line is driven straight on the line
    transition_m: float = 15.0  # the blend from the line to the exit point begins this far before it
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


class StraightCrossing:
    """The latched lane line and the crossing's state; one instance per driver."""

    def __init__(self, config=None):
        self.config = config or CrossingConfig()
        self.reset()

    def reset(self):
        self.line = None  # (x, y, heading) in the odometry frame
        self.active = False
        self.entry_i = None
        self.note = ""
        self._track = []  # the rear axle's odometry positions, oldest first (the lane's direction)
        self.path_odo = None  # the latched path (line, blend, exit line) in the odometry frame, once the exit is known
        self._geometry = None
        self.bend_rad = 0.0

    def observe(self, pose, moving):
        """Each tick: the car's own track, for the lane's direction."""
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
        self.note = "lane line latched" + (" (kerb heading)" if track is None else "")

    def should_enter(self, entry_body):
        return self.line is not None and not self.active and float(entry_body[0]) <= self.config.enter_m

    def enter(self, entry_i):
        self.active, self.entry_i = True, entry_i
        self.note = "crossing on the latched lane line"

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
        if len(out) < 4:
            return self.line_body(pose)
        return out

    def _build_path(self, pose, exit_body, after_body):
        """The line to ``transition_m`` before the route's exit point, a cubic blend from
        there onto the exit lane's line, then that line: as a polyline in the odometry
        frame (``_assemble``). The exit line is the least-squares line through the route's
        exit point and the points after it (``after_body``, k x 2: within 60 m of the exit),
        in their order; the blend ends at the exit point's projection onto it. (An arc
        tangent to both lines was tried first: with a shallow bend the lines' intersection
        moves tens of metres with the exit direction's sat-nav noise, and the car turned
        off 25 m early. The exit point on its own was tried next: its 2 m of noise across
        the line bent the path on the synthetic crossroads by 1.5 m.)"""
        c = self.config
        x0, y0, h = self.line
        u = np.array([math.cos(h), math.sin(h)])
        pts = pose.to_world(np.vstack([np.asarray(exit_body, dtype=float)[None, :], np.asarray(after_body, dtype=float).reshape(-1, 2)]))
        e = pts[0]
        if np.hypot(*(pts[-1] - e)) < 2.0:
            return
        if len(pts) == 2:
            v, centre = pts[1] - e, e
        else:
            centre = pts.mean(axis=0)
            _, _, vt = np.linalg.svd(pts - centre, full_matrices=False)
            v = vt[0]
            if np.dot(v, pts[-1] - e) < 0:
                v = -v
        v = v / np.hypot(*v)
        theta = math.atan2(u[0] * v[1] - u[1] * v[0], float(np.dot(u, v)))  # +: the exit line bends left
        e = centre + float(np.dot(e - centre, v)) * v  # the exit point on the fitted exit line
        car = np.array([pose.x_m, pose.y_m])
        foot = np.array([x0, y0]) + float(np.dot(car - np.array([x0, y0]), u)) * u  # the car's projection onto the line
        along_e = float(np.dot(e - foot, u))  # the exit point along the latched line
        straight = abs(theta) < c.bend_min_rad or abs(theta) > math.radians(80) or along_e < c.transition_m + 5.0
        self._geometry = {"foot": foot, "u": u, "v": v, "e": e, "along_e": along_e, "straight": straight}
        self.bend_rad = 0.0 if straight else float(theta)
        self._assemble()

    def _assemble(self):
        """``path_odo`` from the geometry: the line alone, or line, blend and exit line."""
        c = self.config
        g = self._geometry
        foot, u, v, e, along_e = g["foot"], g["u"], g["v"], g["e"], g["along_e"]
        if g["straight"]:
            n = int((max(c.look_ahead_m, along_e + 60.0) - c.start_m) / c.step_m) + 1
            self.path_odo = np.array([foot + (c.start_m + k * c.step_m) * u for k in range(n)])
            return
        p0 = foot + (along_e - c.transition_m) * u
        before = np.array([foot + (c.start_m + k * c.step_m) * u for k in range(int((along_e - c.transition_m - c.start_m) / c.step_m) + 1)])
        length = float(np.hypot(*(e - p0)))
        m0, m1 = length * u, length * v
        n = max(4, int(math.ceil(length / c.step_m)))
        ts = np.linspace(0.0, 1.0, n + 1)[1:]
        blend = np.array([(2 * t**3 - 3 * t**2 + 1) * p0 + (t**3 - 2 * t**2 + t) * m0 + (-2 * t**3 + 3 * t**2) * e + (t**3 - t**2) * m1 for t in ts])
        after = np.array([e + k * c.step_m * v for k in range(1, int(60.0 / c.step_m) + 1)])
        self.path_odo = np.vstack([before, blend, after]) if len(before) else np.vstack([blend, after])

    def on_exit_line(self, pose, point_body):
        """Whether a point of the path (vehicle frame) lies past the blend's start: kerb
        evidence there is the exit lane's, and moves the exit line rather than the latched one."""
        g = getattr(self, "_geometry", None)
        if g is None or g["straight"]:
            return False
        p = pose.to_world(np.asarray(point_body, dtype=float)[None, :])[0]
        return float(np.dot(p - g["foot"], g["u"])) >= g["along_e"] - self.config.transition_m

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

    def reanchor(self, pose, lateral_m, heading_rad):
        """Move the latched line towards a kerb fit seen from the car: ``lateral_m`` left of
        the path at the car, ``heading_rad`` its direction in the vehicle frame."""
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

    def reanchor_exit(self, lateral_m, turn_rad=0.0):
        """Move the exit line towards the kerb fit seen along it: ``lateral_m`` left of the
        path where the evidence is, ``turn_rad`` the fit's direction left of the path's
        there. The latched line and the blend's start stay; the blend is rebuilt to the
        moved line (the exit line's position and direction are the sat-nav's, a metre or
        two and several degrees out; the kerb beside it is seen 20 m before the car gets
        there)."""
        c = self.config
        g = self._geometry
        if g is None or g["straight"] or abs(lateral_m) > c.reanchor_max_m:
            return False
        v = g["v"]
        g["e"] = g["e"] + c.reanchor_gain * lateral_m * np.array([-v[1], v[0]])
        a = c.reanchor_gain * turn_rad
        g["v"] = np.array([v[0] * math.cos(a) - v[1] * math.sin(a), v[0] * math.sin(a) + v[1] * math.cos(a)])
        u = g["u"]
        self.bend_rad = math.atan2(u[0] * g["v"][1] - u[1] * g["v"][0], float(np.dot(u, g["v"])))
        self._assemble()
        self.note = "crossing on the lane line, exit line re-anchored to a kerb"
        return True

    def may_try_kerb(self, entry_body):
        """Past handback_from_entry_m beyond the entry, the kerb corridor is tried every tick."""
        return self.active and -float(entry_body[0]) >= self.config.handback_from_entry_m

    def may_reacquire(self, exit_body):
        return self.active and self.past_exit_m(exit_body) >= self.config.reacquire_m

    def given_up(self, exit_body):
        return self.active and self.past_exit_m(exit_body) >= self.config.give_up_m
