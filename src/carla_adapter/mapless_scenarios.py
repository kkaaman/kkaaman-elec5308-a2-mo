"""Scripted road users for the G3 scenarios M04–M06 (plan v2.1, section 13). Framework.

Each scenario places one scripted actor on the site's lane and drives it tick by tick
with plain controls; nothing here reaches the car, which sees the actor only through
its sensors. The evaluator reads the actor's true box for the gates.

- **M04 crossing pedestrian** (``CrossingPedestrian``): a walker steps off the right
  pavement ``trigger_m`` ahead of the car and crosses the lane at ``walk_mps``, once
  the car is within ``trigger_m`` of the crossing point (late emergence is a separate,
  tagged variant with a smaller trigger distance).
- **M05 braking lead car** (``BrakingLeadCar``): a car ahead holds ``lead_mps`` and
  brakes at ``brake_mps2`` to a stop ``brake_at_m`` along the lane.
- **M06 cut-in** (``CutInCar``): a car in the left lane ahead holds ``approach_mps``
  (slower than the ego) until the ego has closed to ``cut_gap_m`` behind it (actor
  centre to ego rear axle, along the lane), then moves into the ego's lane over
  ``cut_in_s`` while speeding up to ``lead_mps``.

The actors are kept physically plausible: speeds and decelerations are what the
vehicle blueprint can do, and a walker moves at a walking pace. Scenario parameters are
recorded in the run report so a run is reproducible from it.

Status: written against the CARLA 0.9.16 API from the capture and drive harnesses;
not yet run live. Hook: ``MaplessDrive`` creates the scenario from ``--scenario`` and
calls ``spawn`` after the site is chosen and ``step`` every tick.
"""

import math
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import numpy as np

LEAD_CAR = "vehicle.audi.tt"
WALKER = "walker.pedestrian.0001"


def _box_truth(actor, core_xy):
    """The actor's plan-view box in the evaluator's core frame: box centre, yaw, extent.

    The box centre is the bounding box's local ``location`` carried by the actor
    transform (CARLA vehicles' boxes sit ahead of the actor origin), not the origin: the
    external review found the separation measured from the origin and the smallest half
    extent, which hid an overlap of the ego and lead boxes.
    """
    t = actor.get_transform()
    bb = actor.bounding_box
    yaw = math.radians(t.rotation.yaw)
    # CARLA frame: forward (cos, sin), right (-sin, cos); the box offset lies along the actor's axes.
    bx, by = bb.location.x, bb.location.y
    centre = SimpleNamespace(x=t.location.x + bx * math.cos(yaw) - by * math.sin(yaw), y=t.location.y + bx * math.sin(yaw) + by * math.cos(yaw), z=t.location.z)
    v = actor.get_velocity()
    return dict(
        origin_xy=list(core_xy(t.location)),
        centre_xy=list(core_xy(centre)),
        box_offset_m=[bx, by],
        yaw_rad=-yaw,
        extent_m=[bb.extent.x, bb.extent.y],
        speed_mps=math.hypot(v.x, v.y),
    )


def box_vertices(centre_xy, yaw_rad, half_lengths):
    """Corners of an oriented box: ``half_lengths`` along and across the heading."""
    cx, cy = centre_xy
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    a, b = half_lengths
    return [(cx + c * u - s * v, cy + s * u + c * v) for u, v in ((a, b), (a, -b), (-a, -b), (-a, b))]


def ego_vertices(truth, body):
    """The ego body box from the logged rear-axle pose (``body`` already carries the box offset)."""
    x, y = truth["rear_axle_xy"]
    c, s = math.cos(truth["yaw_rad"]), math.sin(truth["yaw_rad"])
    b = body
    return [(x + c * u - s * v, y + s * u + c * v) for u, v in ((b.front_m, b.half_width_m), (b.front_m, -b.half_width_m), (b.rear_m, -b.half_width_m), (b.rear_m, b.half_width_m))]


def _segment_distance(p, a, b):
    ax, ay, bx, by = *a, *b
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / length2))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


def polygon_distance(a, b):
    """Plan-view distance between two convex polygons; 0 when they touch or overlap.

    Separating-axis test on the edge normals of both, then the nearest vertex-to-edge
    pair. Used instead of centre distance less a half extent so a long vehicle is
    measured from its side and an overlap reads 0.
    """
    for poly, other in ((a, b), (b, a)):
        for i in range(len(poly)):
            (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % len(poly)]
            nx, ny = y1 - y0, x0 - x1
            pa = [nx * x + ny * y for x, y in poly]
            pb = [nx * x + ny * y for x, y in other]
            if max(pa) < min(pb) or max(pb) < min(pa):
                break
        else:
            continue
        break
    else:
        return 0.0
    best = math.inf
    for poly, other in ((a, b), (b, a)):
        for p in poly:
            for i in range(len(other)):
                best = min(best, _segment_distance(p, other[i], other[(i + 1) % len(other)]))
    return best


def separations(log, actor_truths, body):
    """Per-tick ego-box to actor-box distance (``None`` where the actor was not recorded)."""
    out = []
    for row, actor in zip(log, actor_truths):
        if actor is None:
            out.append(None)
            continue
        out.append(polygon_distance(ego_vertices(row["truth"], body), box_vertices(actor["centre_xy"], actor["yaw_rad"], actor["extent_m"])))
    return out


def _ego_frame(truth, point_xy):
    """A world point in the ego's rear-axle frame (x forward, y left)."""
    x, y = truth["rear_axle_xy"]
    c, s = math.cos(truth["yaw_rad"]), math.sin(truth["yaw_rad"])
    px, py = point_xy[0] - x, point_xy[1] - y
    return c * px + s * py, -s * px + c * py


@dataclass
class ScenarioParams:
    name: str
    trigger_m: float = 25.0
    walk_mps: float = 1.2
    lead_mps: float = 3.0
    brake_mps2: float = 3.0
    brake_at_m: float = 45.0
    lead_start_m: float = 20.0
    cut_in_s: float = 4.0
    # Cut-in: the first live run's car held lead_mps (faster than the ego) and waited for
    # the ego to draw level, so it never cut in. It now approaches slower than the ego and
    # cuts in once the ego has closed to cut_gap_m (about 8 m bumper to bumper).
    approach_mps: float = 1.0
    cut_gap_m: float = 14.0

    def record(self):
        return asdict(self)


class ScriptedActor:
    """Base: one actor on the site lane, controlled per tick."""

    def __init__(self, carla, params, core_xy):
        self.carla, self.params, self.core_xy = carla, params, core_xy
        self.actor = None
        self.started = False

    def spawn(self, owner, library, waypoints):
        raise NotImplementedError

    def step(self, tick, ego_truth):
        raise NotImplementedError

    def truth(self):
        return None if self.actor is None else _box_truth(self.actor, self.core_xy)

    def hold_ends_run(self):
        """Whether a sustained hold may end the run now (review: holding is not a result
        for every scenario; a car that waits out a crossing must be allowed to resume)."""
        return True

    def destroy(self):
        if self.actor is not None and self.actor.is_alive:
            self.actor.destroy()
        self.actor = None


class BrakingLeadCar(ScriptedActor):
    """M05: a lead car that brakes to a stop at ``brake_at_m``."""

    def spawn(self, owner, library, waypoints):
        p = self.params
        wp = waypoints[min(len(waypoints) - 1, int(round(p.lead_start_m)))]
        t = wp.transform
        t.location.z += 0.3
        self.actor = owner.try_spawn(library.find(LEAD_CAR), t)
        if self.actor is None:
            raise RuntimeError("Could not place the lead car")
        self.waypoints = waypoints
        self.braking = False
        return self.actor

    def step(self, tick, ego_truth):
        carla, p = self.carla, self.params
        t = self.actor.get_transform()
        x, y = self.core_xy(t.location)
        lane = np.array([self.core_xy(w.transform.location) for w in self.waypoints])
        s = float(np.argmin(np.hypot(lane[:, 0] - x, lane[:, 1] - y)))
        v = self.actor.get_velocity()
        speed = math.hypot(v.x, v.y)
        if s >= p.brake_at_m:
            self.braking = True
        if self.braking:
            self.actor.apply_control(carla.VehicleControl(throttle=0.0, brake=min(1.0, p.brake_mps2 / 8.0)))
        else:
            # A plain speed loop along the lane: throttle to hold lead_mps, steer to the lane.
            throttle = float(np.clip(0.35 + 0.3 * (p.lead_mps - speed), 0.0, 0.7))
            ahead = self.waypoints[min(len(self.waypoints) - 1, int(s) + 4)]
            ax, ay = self.core_xy(ahead.transform.location)
            yaw = -math.radians(t.rotation.yaw)
            bearing = math.atan2(ay - y, ax - x) - yaw
            bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
            self.actor.apply_control(carla.VehicleControl(throttle=throttle, steer=float(np.clip(-bearing / math.radians(35), -1, 1))))


class CutInCar(BrakingLeadCar):
    """M06: a car in the left lane that moves into the ego's lane over ``cut_in_s``."""

    def spawn(self, owner, library, waypoints):
        p = self.params
        wp = waypoints[min(len(waypoints) - 1, int(round(p.lead_start_m)))]
        left = wp.get_left_lane()
        # On a two-way road the lane to the left of the rightmost lane is the opposing one
        # (opposite lane_id sign): the cut-in needs a same-direction lane.
        if left is None or left.lane_type != self.carla.LaneType.Driving or left.lane_id * wp.lane_id < 0:
            raise RuntimeError("The cut-in scenario needs a same-direction Driving lane on the left of the site")
        t = left.transform
        t.location.z += 0.3
        self.actor = owner.try_spawn(library.find(LEAD_CAR), t)
        if self.actor is None:
            raise RuntimeError("Could not place the cut-in car")
        self.waypoints = waypoints
        self.left_waypoints = [(lw if (lw := w.get_left_lane()) is not None and lw.lane_id * w.lane_id > 0 else w) for w in waypoints]
        self.braking = False
        self.cut_start = None
        return self.actor

    def step(self, tick, ego_truth):
        p = self.params
        # Hold the left lane at approach_mps until the ego has closed to cut_gap_m behind,
        # then blend the target line into the ego lane and speed up to lead_mps.
        if self.cut_start is None and ego_truth is not None:
            lane = np.array([self.core_xy(w.transform.location) for w in self.waypoints])
            x, y = self.core_xy(self.actor.get_transform().location)
            ex, ey = ego_truth["rear_axle_xy"]
            s_actor = int(np.argmin(np.hypot(lane[:, 0] - x, lane[:, 1] - y)))
            s_ego = int(np.argmin(np.hypot(lane[:, 0] - ex, lane[:, 1] - ey)))
            if s_actor - s_ego <= p.cut_gap_m:
                self.cut_start = tick
        blend = 1.0 if self.cut_start is None else min(1.0, (tick - self.cut_start) * 0.05 / p.cut_in_s)
        if self.cut_start is None:
            blend = 0.0
        lane = []
        for w, lw in zip(self.waypoints, self.left_waypoints):
            a, b = self.core_xy(w.transform.location), self.core_xy(lw.transform.location)
            lane.append(((1 - blend) * b[0] + blend * a[0], (1 - blend) * b[1] + blend * a[1]))
        self._lane_override = np.array(lane)
        self.braking = False
        carla = self.carla
        t = self.actor.get_transform()
        x, y = self.core_xy(t.location)
        s = int(np.argmin(np.hypot(self._lane_override[:, 0] - x, self._lane_override[:, 1] - y)))
        ax, ay = self._lane_override[min(len(lane) - 1, s + 4)]
        yaw = -math.radians(t.rotation.yaw)
        bearing = (math.atan2(ay - y, ax - x) - yaw + math.pi) % (2 * math.pi) - math.pi
        v = self.actor.get_velocity()
        speed, target = math.hypot(v.x, v.y), (p.approach_mps if self.cut_start is None else p.lead_mps)
        throttle = float(np.clip(0.35 + 0.3 * (target - speed), 0.0, 0.7)) if speed < target + 0.3 else 0.0
        brake = 0.0 if speed < target + 0.5 else 0.3
        self.actor.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=float(np.clip(-bearing / math.radians(35), -1, 1))))


class CrossingPedestrian(ScriptedActor):
    """M04: a walker crossing from the right pavement once the ego is within ``trigger_m``."""

    def spawn(self, owner, library, waypoints):
        p = self.params
        wp = waypoints[min(len(waypoints) - 1, int(round(p.brake_at_m)))]
        self.cross_wp = wp
        self.lane_width_m = float(getattr(wp, "lane_width", 3.5))
        t = wp.transform
        yaw = math.radians(t.rotation.yaw)
        # Start 4 m right of the lane centre (on the pavement), facing left across the lane.
        # CARLA frame (y to the right, yaw clockwise from above): forward (cos, sin), right (-sin, cos).
        right = (-math.sin(yaw), math.cos(yaw))
        self.cross_xy = self.core_xy(t.location)
        rx, ry = self.core_xy(SimpleNamespace(x=t.location.x + right[0], y=t.location.y + right[1], z=t.location.z))
        self.right_core = (rx - self.cross_xy[0], ry - self.cross_xy[1])
        t.location.x += 4.0 * right[0]
        t.location.y += 4.0 * right[1]
        t.location.z += 1.0
        t.rotation.yaw = t.rotation.yaw - 90.0
        self.actor = owner.try_spawn(library.find(WALKER), t)
        if self.actor is None:
            raise RuntimeError("Could not place the walker")
        self.direction = (-right[0], -right[1], 0.0)
        return self.actor

    def step(self, tick, ego_truth):
        carla, p = self.carla, self.params
        if not self.started and ego_truth is not None:
            cx, cy = self.core_xy(self.cross_wp.transform.location)
            ex, ey = ego_truth["rear_axle_xy"]
            if math.dist((cx, cy), (ex, ey)) <= p.trigger_m:
                self.started = True
        speed = p.walk_mps if self.started else 0.0
        self.actor.apply_control(carla.WalkerControl(direction=carla.Vector3D(*self.direction), speed=speed))

    def cleared(self):
        """The walker is a metre past the lane's left edge: the car has nothing to wait for."""
        if self.actor is None or not self.started:
            return False
        x, y = self.core_xy(self.actor.get_transform().location)
        offset = (x - self.cross_xy[0]) * self.right_core[0] + (y - self.cross_xy[1]) * self.right_core[1]
        return offset <= -(self.lane_width_m / 2 + 1.0)

    def hold_ends_run(self):
        # Holding while the walker crosses is the expected yield, not an endpoint: only a
        # hold that outlasts the crossing ends the run (and then fails completion).
        return self.cleared()


SCENARIOS = {
    "crossing_pedestrian": (CrossingPedestrian, "M04"),
    "braking_lead": (BrakingLeadCar, "M05"),
    "cut_in": (CutInCar, "M06"),
}


def make_scenario(carla, name, core_xy, **overrides):
    cls, _ = SCENARIOS[name]
    return cls(carla, ScenarioParams(name, **overrides), core_xy)


def score_lead_tracking(log, actor_truths, body, dt_s=0.05):
    """What the stack knew about the lead, against the evaluator's truth (G3: radar must
    earn its place). Reported, not gated:

    - ``first_lead_s`` / ``first_lead_truth_gap_m``: when a lead was first selected and the
      true bumper gap then;
    - ``speed_error_rmse_mps`` / ``speed_error_p95_mps``: the lead's speed along the path,
      as the stack had it, against the actor's, over ticks with a lead;
    - ``gap_error_median_m``: the stack's gap minus the true one (negative: conservative);
    - ``brake_s``: when the actor began to brake (the first tick below its cruise, of a
      drop past 0.3 m/s);
      ``brake_seen_s``: the first tick after it with the stack's lead speed 0.5 m/s below
      the actor's cruise, and ``brake_delay_s`` between them;
    - ``stopped_before_lead_s``: time the ego stood still while the lead still moved
      (false stops), and ``support``: ticks of each sensor set behind the lead.
    """
    gaps = separations(log, actor_truths, body)
    errors, gap_errors, support = [], [], {}
    first = None
    speeds = [a["speed_mps"] if a else None for a in actor_truths]
    cruise = max((v for v in speeds if v is not None), default=None)
    brake_tick = seen_tick = at_cruise = None
    stopped_ticks = 0
    for k, (row, actor) in enumerate(zip(log, actor_truths)):
        if actor is None:
            continue
        lead = (row.get("driver") or {}).get("lead")
        if cruise is not None and brake_tick is None:
            if actor["speed_mps"] >= cruise - 0.1:
                at_cruise = k
            elif at_cruise is not None and actor["speed_mps"] < cruise - 0.3:
                brake_tick = at_cruise + 1  # the first tick below cruise
        if row["truth"]["speed_mps"] < 0.05 and actor["speed_mps"] > 0.5 and row["mode"] != "startup":
            stopped_ticks += 1
        if not lead or lead.get("crossing"):
            continue
        if first is None:
            first = (row["time_s"], gaps[k])
        errors.append(lead["speed_mps"] - actor["speed_mps"])
        if gaps[k] is not None and lead.get("gap_m") is not None:
            gap_errors.append(lead["gap_m"] - gaps[k])
        key = "+".join(lead.get("support") or ()) or "none"
        support[key] = support.get(key, 0) + 1
        if brake_tick is not None and seen_tick is None and k >= brake_tick and lead["speed_mps"] < cruise - 0.5:
            seen_tick = k
    e = np.abs(np.array(errors)) if errors else None
    t = lambda k: None if k is None else round(log[k]["time_s"], 2)  # noqa: E731
    return dict(
        first_lead_s=None if first is None else round(first[0], 2),
        first_lead_truth_gap_m=None if first is None or first[1] is None else round(first[1], 2),
        speed_error_rmse_mps=None if e is None else round(float(np.sqrt(np.mean(e**2))), 3),
        speed_error_p95_mps=None if e is None else round(float(np.percentile(e, 95)), 3),
        gap_error_median_m=round(float(np.median(gap_errors)), 2) if gap_errors else None,
        brake_s=t(brake_tick),
        brake_seen_s=t(seen_tick),
        brake_delay_s=None if brake_tick is None or seen_tick is None else round((seen_tick - brake_tick) * dt_s, 2),
        stopped_before_lead_s=round(stopped_ticks * dt_s, 2),
        support=support,
        ticks=len(errors),
    )


def score_moving(log, actor_truths, body, min_separation_m=1.0):
    """Separation gate for M04–M06 from the drive log and the actor's truth per tick.

    The ego-box to actor-box distance (``polygon_distance``) never fell under
    ``min_separation_m``; an overlap reads 0 and fails. ``separation_ok`` is ``None``
    when no actor was recorded: missing evidence is INCOMPLETE, not a pass (review).
    """
    measured = [d for d in separations(log, actor_truths, body) if d is not None]
    worst = min(measured) if measured else None
    return dict(min_separation_m=worst, separation_ok=None if worst is None else worst >= min_separation_m, ticks=len(measured))


COMPLETION_REQUIRED = {
    "crossing_pedestrian": "yield while the walker is in the path, then resume and reach the goal",
    "braking_lead": "hold behind the stopped lead for --hold-s with the gap at or above the separation threshold, or reach the goal",
    "cut_in": "a gap at or above the separation threshold once the cut-in is complete, then reach the goal",
}


def _ahead_in_band(truth, actor, body, margin_m):
    """The actor's box centre is not behind the ego and within ``margin_m`` of the body's sides."""
    dx, dy = _ego_frame(truth, actor["centre_xy"])
    return dx >= body.rear_m and abs(dy) <= body.half_width_m + margin_m, dx, dy


def score_completion(name, log, actor_truths, body, end, held_s, min_separation_m=1.0, goal_tolerance_m=3.0, settle_s=2.0):
    """Per-scenario completion (review: "held at the obstacle" was accepted for every scenario).

    Returns ``required`` (what the scenario asks of the car), ``passed`` (``True``,
    ``False``, or ``None`` when the scenario did not play out far enough to judge) and
    the evidence behind it.
    """
    out = dict(required=COMPLETION_REQUIRED[name], end=end, goal_distance_m=log[-1]["truth"]["goal_distance_m"] if log else None)
    reached = end == "finished" and out["goal_distance_m"] is not None and out["goal_distance_m"] <= goal_tolerance_m
    gaps = separations(log, actor_truths, body)
    rows = [(row, actor, gap) for row, actor, gap in zip(log, actor_truths, gaps) if actor is not None]
    if not rows:
        return out | dict(passed=None, note="no actor measurements")
    if name == "crossing_pedestrian":
        in_path = [i for i, (row, actor, _) in enumerate(rows) if _ahead_in_band(row["truth"], actor, body, 1.0)[0]]
        if not in_path:
            return out | dict(passed=None, note="the walker never entered the path")
        first = in_path[0]
        speeds = [rows[i][0]["truth"]["speed_mps"] for i in in_path]
        # The speed the car was making before it had to react: the fastest in the 3 s
        # before the walker entered the path (live M04 B6, 26 Oct: the car had yielded to
        # 0.78 m/s before the walker entered, and the entry speed read as the reference).
        t_entry = rows[first][0]["time_s"]
        before = [row["truth"]["speed_mps"] for row, _, _ in rows[:first + 1] if row["time_s"] >= t_entry - 3.0]
        entry_speed = max(before)
        yield_speed = min(speeds)
        yielded = yield_speed <= max(0.5, 0.5 * entry_speed)
        # Cleared: left of the body's band after crossing, or already behind the ego.
        cleared = None
        for i in range(first, len(rows)):
            inside, dx, dy = _ahead_in_band(rows[i][0]["truth"], rows[i][1], body, 1.0)
            if not inside and (dy > body.half_width_m + 1.0 or dx < body.rear_m):
                cleared = i
                break
        resumed = cleared is not None and max(r["truth"]["speed_mps"] for r, _, _ in rows[cleared:]) >= 0.5
        out |= dict(entry_speed_mps=entry_speed, yield_speed_mps=yield_speed, yielded=yielded, walker_cleared=cleared is not None, resumed=resumed)
        if cleared is None and not yielded:
            return out | dict(passed=False)
        if cleared is None:
            return out | dict(passed=None, note="the walker had not cleared the path when the run ended")
        return out | dict(passed=yielded and resumed and reached)
    if name == "cut_in":
        cut = next((i for i, (row, actor, _) in enumerate(rows) if _ahead_in_band(row["truth"], actor, body, 0.5)[0]), None)
        if cut is None:
            return out | dict(passed=None, note="the cut-in car never entered the ego's lane")
        t_cut = rows[cut][0]["time_s"]
        settled = [gap for row, _, gap in rows[cut:] if row["time_s"] >= t_cut + settle_s]
        out |= dict(cut_in_time_s=t_cut, settled_ticks=len(settled), min_gap_after_cut_in_m=min(settled) if settled else None)
        if not settled:
            return out | dict(passed=None, note=f"the run ended within {settle_s:g} s of the cut-in")
        return out | dict(passed=min(settled) >= min_separation_m and reached)
    if name == "braking_lead":
        _, last_actor, final_gap = rows[-1]
        lead_stopped = last_actor["speed_mps"] < 0.05
        out |= dict(lead_stopped=lead_stopped, final_gap_m=final_gap, held_s=held_s)
        if end == "held at the obstacle":
            return out | dict(passed=lead_stopped and final_gap is not None and final_gap >= min_separation_m)
        return out | dict(passed=reached)
    raise KeyError(name)
