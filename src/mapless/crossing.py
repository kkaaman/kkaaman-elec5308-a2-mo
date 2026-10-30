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
the car follows that line instead of the kerb fit: a straight corridor with no edges,
valid as far as the road along it is observed (``support``), so the obstacle stop and the
planner's ramp work on it as on any corridor. Once the rear axle is ``reacquire_m`` past
the exit point the kerb corridor is seeded with the line and asked for its own fit; when it
has one the crossing hands back. Past ``give_up_m`` beyond the exit with no kerb, the
driver's ordinary "no corridor" fault stands.

The line is a straight: over a 20–30 m junction the odometry drifts centimetres, and a
junction the route calls STRAIGHT is crossed straight. What the car does not do here is
look for traffic: a junction crossed on a STRAIGHT is a give-way or signalled one, and
the lights (M07) and crossing traffic are the stack's and later work's.
"""

import math
from dataclasses import dataclass

import numpy as np

from mapless.corridor import Corridor


@dataclass(frozen=True)
class CrossingConfig:
    latch_from_m: float = 30.0  # latch the lane line while the entry is within this ...
    freeze_m: float = 15.0  # ... and at least this far ahead (the kerb fit tilts into the fillet from ~16 m)
    history_m: float = 40.0  # the line's direction is the car's own track over this far behind it ...
    history_min_m: float = 20.0  # ... when at least this much of it is known (else the kerb evidence beside the car)
    enter_m: float = 12.0  # follow the latched line from this far before the entry (the kerb fit tilts into the fillet from ~16 m)
    reacquire_m: float = 8.0  # try the kerb corridor again from this far past the exit (the exit kerb begins past the fillet)
    handback_offset_m: float = 1.5  # the kerb corridor must lie within this of the line at the car (the line drifts; the kerb is right) ...
    handback_heading_rad: float = 0.12  # ... and in direction, to be handed to
    give_up_m: float = 25.0  # past the exit without a kerb: the crossing is over, the fault stands
    look_ahead_m: float = 40.0
    step_m: float = 1.0
    start_m: float = -2.0
    max_route_offset_m: float = 4.0  # the entry point must lie this near the lane line, laterally

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
        if self.active or not corridor.ok:
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

    def corridor(self, grid, pose, time_s, support):
        """A straight corridor along the latched line, valid as far as the road along it is
        observed (``support(grid, pose, centre_body)``: the kerb corridor's rule)."""
        points = self.line_body(pose)
        seen = float(support(grid, pose, points))
        valid = max(float(points[1, 0]), seen)
        n = len(points)
        slope = (points[-1, 1] - points[0, 1]) / (points[-1, 0] - points[0, 0]) if points[-1, 0] != points[0, 0] else 0.0
        return Corridor(
            time_s=time_s,
            pose=pose,
            centre_body=points,
            centre_odometry=pose.to_world(points),
            valid_length_m=valid,
            coefficients=(float(slope), float(points[0, 1] - slope * points[0, 0])),
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

    def agrees(self, corridor, pose):
        """Whether a kerb corridor matches the latched line at the car: the fit past the
        exit grabs the far fillet first (synthetic crossroads: a 2.6 m jump at the hand-back)."""
        c = self.config
        if not corridor.ok:
            return False
        line = self.line_body(pose, length_m=10.0)
        at_car = float(np.interp(0.0, line[:, 0], line[:, 1]))
        heading = math.atan2(line[-1, 1] - line[0, 1], line[-1, 0] - line[0, 0])
        return abs(corridor.lateral_offset_at(0.0) - at_car) <= c.handback_offset_m and abs(corridor.heading_at(0.0) - heading) <= c.handback_heading_rad

    def may_reacquire(self, exit_body):
        return self.active and self.past_exit_m(exit_body) >= self.config.reacquire_m

    def given_up(self, exit_body):
        return self.active and self.past_exit_m(exit_body) >= self.config.give_up_m
