"""Passing a vehicle stopped in the car's lane (3 Nov).

The driver holds behind an obstacle in its lane as before (``planner``). Held at a
standstill for ``wait_s`` behind the same obstacle, one at least ``min_width_m`` across (a
vehicle, not a walker), with no route junction near and nothing vetoing it (the stack's
moving tracks on the far side), the car may pass it:

1. **Plan.** The obstacle's edges come from this sweep's OBSTACLE points about its near
   face (the tube clips the planner's view of it). With overtaking on, the planner holds
   ``hold_gap_m`` short of an obstacle (not ``stop_gap_m``): room to steer out. The path is shifted left by that
   edge, the body's half width and ``margin_m``, at most ``max_offset_m``; the shifted
   path must be seen clear (``observe_path`` on it) past the obstacle's far side.
   The side: the right when the corridor's lane centre is far enough from the right kerb
   for a lane between (traffic keeps right: that lane runs the car's way), else the left
   (the oncoming lane of a two-lane road).
2. **Out.** The shift ramps in (a half-cosine) from where the car stands, over ``ramp_m``
   or less (fully out by the obstacle's near face, ``min_ramp_m`` at least), indicator
   left, at ``speed_mps`` at most.
3. **Past.** Held at the full shift until the original lane is seen clear from the rear
   of the car to ``return_clear_m`` ahead and the car has gone ``min_pass_m`` beyond the
   obstacle's near face.
4. **Back.** The shift ramps out over ``ramp_m``, indicator right; then done.

The shift is a function of the distance travelled since the start (odometry), so it stays
put on the road as the corridor is re-fitted every tick; the corridor estimator keeps
following the right kerb throughout. Everything else still binds on the shifted path: the
obstacle check, the planner's holds and speed limits, the stack's caps.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from mapless.terrain import PointClass


@dataclass(frozen=True)
class OvertakeConfig:
    wait_s: float = 3.0  # held at a standstill behind the same obstacle this long first
    min_width_m: float = 1.2  # an obstacle at least this wide across the path (a vehicle)
    max_hold_distance_m: float = 12.0  # its near face no further than this ahead of the rear axle
    margin_m: float = 0.8  # between the body's side and the obstacle's left edge
    max_offset_m: float = 4.0  # never more than a lane's width or so
    ramp_m: float = 12.0  # out and back each over this (out: over less when the obstacle is nearer, fully out by its near face)
    min_ramp_m: float = 8.0  # never out over less than this (the car's turning circle)
    speed_mps: float = 2.0  # at most this throughout
    assumed_length_m: float = 5.5  # the obstacle's length until the lane beside it is seen clear
    min_pass_m: float = 7.0  # past its near face by at least this (rear axle) before returning
    return_clear_m: float = 8.0  # the original lane seen clear this far ahead of the front to return
    clear_beyond_m: float = 3.0  # the shifted path seen clear this far past the obstacle (assumed length) to start ...
    confirm_s: float = 1.0  # ... and nothing on it in any sweep for this long (one sweep can miss a car)
    edge_window_m: float = 2.5  # OBSTACLE points this far beyond its near face give its edges
    hold_gap_m: float = 6.5  # the planner holds the front this far short of an obstacle: room to steer out
    # The side a pass goes: "auto" is the right when there is a lane between the car and the
    # right kerb (traffic keeps right: a lane there runs the car's way), else the left (the
    # oncoming lane of a two-lane road); or a tuple of sides in order of preference.
    sides: object = "auto"
    min_shift_m: float = 0.0  # the shift at least this (0: the obstacle's edge, half the body and the margin)
    kerb_min_m: float = 1.6  # a pass on the right keeps the path this far from the right kerb (a lane there)


def ramp(s, start, length):
    """0 before ``start``, 1 after ``start + length``, a half-cosine between."""
    u = np.clip((np.asarray(s, dtype=float) - start) / max(length, 1e-6), 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(math.pi * u)


def shift_line(line, offsets):
    """``line`` (M x 2) moved by ``offsets`` (M) along its left normal."""
    line = np.asarray(line, dtype=float)
    if len(line) < 2:
        return line.copy()
    d = np.gradient(line, axis=0)
    n = np.hypot(d[:, 0], d[:, 1])
    n[n < 1e-9] = 1.0
    normal = np.column_stack((-d[:, 1] / n, d[:, 0] / n))
    return line + normal * np.asarray(offsets, dtype=float)[:, None]


def extend(corridor, to_m, step_m=0.5):
    """``corridor`` with its centre line continued straight past its end to body x
    ``to_m``. Beside a stopped vehicle the kerb beyond it is hidden and the line ends at it;
    the way past is judged (and driven) on the continuation, which the obstacle check still
    has to see clear: ground in the tube, nothing standing in it."""
    line = corridor.centre_body
    if len(line) < 2 or line[-1, 0] >= to_m:
        return corridor
    d = line[-1] - line[-2]
    u = d / max(float(np.hypot(*d)), 1e-9)
    n = int(math.ceil((to_m - line[-1, 0]) / max(u[0] * step_m, 1e-3)))
    extra = line[-1] + np.outer(np.arange(1, n + 1) * step_m, u)
    longer = np.vstack((line, extra))
    return dataclasses.replace(corridor, centre_body=longer, centre_odometry=corridor.pose.to_world(longer))


def extent(sweep, near_m, lateral_m, config, reach_m=4.0):
    """The obstacle's right and left edges (body y) from this sweep's OBSTACLE points
    between its near face and ``edge_window_m`` beyond: the run of them, without a gap of
    0.6 m, that holds its nearest point (the tube clips the planner's view of it, and one
    sweep sees only some of it)."""
    pts = sweep.xyz[sweep.cls == PointClass.OBSTACLE]
    sel = pts[(pts[:, 0] >= near_m - 0.3) & (pts[:, 0] <= near_m + config.edge_window_m) & (np.abs(pts[:, 1] - lateral_m) <= reach_m)]
    if len(sel) == 0:
        return None
    ys = np.sort(np.append(sel[:, 1], lateral_m))
    k = int(np.searchsorted(ys, lateral_m))
    lo, hi = k, k
    while lo > 0 and ys[lo] - ys[lo - 1] <= 0.6:
        lo -= 1
    while hi < len(ys) - 1 and ys[hi + 1] - ys[hi] <= 0.6:
        hi += 1
    return float(ys[lo]), float(ys[hi])


SIDES = ("left", "right")


class Overtake:
    """The manoeuvre's state across ticks: ``idle`` -> ``waiting`` -> ``out`` -> ``past``
    -> ``back`` -> ``idle``."""

    def __init__(self, config=None):
        self.config = config or OvertakeConfig()
        self.state = "idle"
        self.since_s = None
        self.offset_m = 0.0
        self.ramp_out_m = self.config.ramp_m
        self.travelled_m = 0.0
        self.obstacle_s_m = None  # its near face, in travelled distance from the start
        self.back_from_m = None  # where the ramp back begins, in travelled distance
        self._last_odo = None
        self.passes = 0
        self.note = ""
        self.started = False  # this tick began a pass (the driver drops the planner's hold on it)
        self._seen = None  # the obstacle last seen while held, and the odometry it was seen from
        self._clear = {}  # by side: the way past seen with nothing on it since, and clear this far at best since
        self.side = None  # the side being passed on
        self.passing_box = None  # what is being passed: its footprint (4 x 2, odometry), assumed_length_m long

    def _advance(self, odometry):
        if self._last_odo is not None:
            self.travelled_m += math.hypot(odometry.x_m - self._last_odo[0], odometry.y_m - self._last_odo[1])
        self._last_odo = (odometry.x_m, odometry.y_m)

    def offsets(self, line):
        """The shift at each point of ``line`` (body frame, from the rear axle on)."""
        c = self.config
        line = np.asarray(line, dtype=float)
        arcs = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(line, axis=0).T)))) if len(line) else np.zeros(0)
        start = 0.0
        if len(line):
            start = float(arcs[int(np.argmin(np.abs(line[:, 0])))])  # the point beside the rear axle
        s = self.travelled_m + (arcs - start)
        out = ramp(s, 0.0, self.ramp_out_m)
        if self.back_from_m is None:
            return self.offset_m * out
        return self.offset_m * out * (1.0 - ramp(s, self.back_from_m, c.ramp_m))

    def step(self, t, odometry, standing, corridor, observation, plan_reason, sweep, observe, body, veto=False, junction_ahead_m=None, keep_right_m=None):
        """One tick. Returns (the corridor to drive, the indicator or None, the speed cap or
        None). ``observe(corridor)``: the obstacle check on a candidate path;
        ``keep_right_m``: the lane centre's distance from the right kerb (the corridor's)."""
        c = self.config
        self.started = False
        if self.state == "idle":
            obstacle = None if observation is None else observation.obstacle
            if obstacle is not None:
                self._seen = (obstacle, (odometry.x_m, odometry.y_m))
            holding = plan_reason in ("obstacle", "obstacle_hold") and standing
            if holding and self._seen is not None:
                # Seen or not this sweep, the hold is the same: the wait runs on. Standing
                # behind it, the line can end at its near face (the kerb beyond it hidden),
                # and an obstacle past the line's end is not in the tube.
                held, at = self._seen
                moved = math.hypot(odometry.x_m - at[0], odometry.y_m - at[1])
                self.since_s = t if self.since_s is None else self.since_s
                if t - self.since_s >= c.wait_s and moved < 0.5 and held.distance_m <= c.max_hold_distance_m:
                    self._plan(t, odometry, corridor, held, sweep, observe, body, veto, junction_ahead_m, keep_right_m)
            elif not holding:
                self.since_s = None
                self._clear = {}
                self.note = ""
                if obstacle is None:
                    self._seen = None
            if self.state == "idle":
                return corridor, None, None
        self._advance(odometry)
        corridor = extend(corridor, self._reach_m(body))
        if self.state in ("out", "past") and self.back_from_m is None:
            original = observe(corridor)
            ahead_clear = original is None or original.obstacle is None or original.obstacle.distance_m - body.front_m >= c.return_clear_m
            if self.travelled_m >= self.obstacle_s_m + c.min_pass_m and ahead_clear:
                self.back_from_m = self.travelled_m
                self.state = "back"
            elif self.travelled_m >= self.ramp_out_m:
                self.state = "past"
        if self.state == "back" and self.travelled_m >= self.back_from_m + c.ramp_m:
            self.state, self.passes = "idle", self.passes + 1
            self.offset_m, self.back_from_m, self.since_s, self._last_odo, self.passing_box = 0.0, None, None, None, None
            self.note = "passed"
            return corridor, None, None
        shifted = dataclasses.replace(corridor, centre_body=shift_line(corridor.centre_body, self.offsets(corridor.centre_body)))
        shifted = dataclasses.replace(shifted, centre_odometry=corridor.pose.to_world(shifted.centre_body))
        back = "right" if self.side == "left" else "left"
        return shifted, (back if self.state == "back" else self.side), c.speed_mps

    def _plan(self, t, odometry, corridor, obstacle, sweep, observe, body, veto, junction_ahead_m, keep_right_m=None):
        c = self.config
        vetoed = set(veto) if isinstance(veto, (set, frozenset, tuple, list)) else (set(SIDES) if veto else set())
        if sweep is None or not corridor.ok:
            self.note = "waiting: no sweep or corridor"
            return
        span = extent(sweep, obstacle.distance_m, obstacle.lateral_m, c)
        if span is None:
            self.note = "waiting: the obstacle's edges not seen"
            return
        if span[1] - span[0] < c.min_width_m:
            self.note = f"waiting: {span[1] - span[0]:.1f} m across, not a vehicle"
            return
        end_m = obstacle.distance_m + c.assumed_length_m + c.ramp_m + c.clear_beyond_m
        if junction_ahead_m is not None and junction_ahead_m < end_m:
            self.note = "refused: a junction within the manoeuvre"
            return
        needed = obstacle.distance_m + c.assumed_length_m + c.clear_beyond_m
        line = extend(corridor, needed + 2.0).centre_body
        ramp_out = float(np.clip(obstacle.distance_m - 0.5, c.min_ramp_m, c.ramp_m))
        shifts = dict(
            left=max(span[1] + body.half_width_m + c.margin_m, c.min_shift_m),
            right=min(span[0] - body.half_width_m - c.margin_m, -c.min_shift_m),
        )
        # A lane on the right: the shifted path still kerb_min_m from the right kerb. Live
        # (S4_5308, Town10HD, 7ba365b): the lane centre 7.8 m from the kerb, the left the
        # oncoming lane, and the pass went there.
        right_lane = keep_right_m is not None and keep_right_m + shifts["right"] >= c.kerb_min_m
        sides = (("right",) if right_lane else ("left",)) if c.sides == "auto" else tuple(c.sides)
        notes = []
        for side in sides:
            # Each side in turn, with its own clear count.
            if side in vetoed:
                self._clear.pop(side, None)
                notes.append(f"{side}: traffic on that side")
                continue
            offset = shifts[side]
            if abs(offset) > c.max_offset_m:
                notes.append(f"{side}: {abs(offset):.1f} m to the side needed")
                continue
            self.offset_m, self.ramp_out_m, self.travelled_m, self._last_odo, self.back_from_m = offset, ramp_out, 0.0, None, None
            candidate = dataclasses.replace(corridor, centre_body=shift_line(line, self.offsets(line)))
            self.offset_m = 0.0
            seen = observe(candidate)
            if seen is None or seen.obstacle is not None:
                self._clear.pop(side, None)
                notes.append(f"{side}: no sweep" if seen is None else f"{side}: something {seen.obstacle.distance_m:.1f} m along the way past")
                continue
            since, best = self._clear.get(side, (t, 0.0))
            best = max(best, seen.clear_to_m)
            self._clear[side] = (since, best)
            if t - since < c.confirm_s or best < needed:
                notes.append(f"{side}: seen clear to {best:.1f} m of {needed:.1f} for {t - since:.1f} s")
                continue
            self.offset_m, self.side, self.obstacle_s_m = offset, side, obstacle.distance_m
            near, far = obstacle.distance_m, obstacle.distance_m + c.assumed_length_m
            self.passing_box = corridor.pose.to_world(np.array([[near, span[0]], [far, span[0]], [far, span[1]], [near, span[1]]]))
            self.state, self.started, self._clear = "out", True, {}
            self._last_odo = (odometry.x_m, odometry.y_m)
            self.note = f"passing: {abs(offset):.2f} m to the {side}"
            return
        self.note = "waiting: " + "; ".join(notes)

    def passing(self, xy, pad_m=1.0):
        """Whether odometry point ``xy`` lies on what is being passed (its box grown by
        ``pad_m``): a fragment of it the tracker sees as a separate, uncertain object is not
        traffic (live S4_5309/5310: a 0.3 m track on the stopped car's corner, "motion
        uncertain", predicted onto the path, stopped the car mid-pass for 0.3 s)."""
        if self.state == "idle" or self.passing_box is None:
            return False
        box = self.passing_box
        u, v = box[1] - box[0], box[3] - box[0]
        lu, lv = float(np.hypot(*u)), float(np.hypot(*v))
        d = np.asarray(xy, dtype=float) - box[0]
        a, b = float(d @ u) / max(lu, 1e-9), float(d @ v) / max(lv, 1e-9)
        return -pad_m <= a <= lu + pad_m and -pad_m <= b <= lv + pad_m

    def _reach_m(self, body):
        """How far ahead (body x) the line is continued while passing: past the end of the
        way back."""
        c = self.config
        ahead = (self.obstacle_s_m or 0.0) + c.assumed_length_m + c.min_pass_m + c.ramp_m - self.travelled_m
        return max(body.front_m + c.return_clear_m, ahead) + 2.0

    def record(self):
        return dict(state=self.state, side=self.side, offset_m=round(self.offset_m, 2), travelled_m=round(self.travelled_m, 1), back_from_m=None if self.back_from_m is None else round(self.back_from_m, 1), note=self.note, passes=self.passes)
