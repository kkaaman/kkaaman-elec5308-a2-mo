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
    # ... and, with cut_brake_after_s at 0 or more, brakes at brake_mps2 to a stop that long
    # after the cut-in is complete (fifth review's matrix: a slow cut-in, then the actor
    # brakes). Negative: it never brakes.
    cut_brake_after_s: float = -1.0
    # Oncoming car at a left turn (M08, ``--oncoming``): oncoming_m past the junction in the
    # opposing lane, straight across; "stopped" pulls away at oncoming_mps go_after_s after
    # the ego first comes within oncoming_trigger_m of the junction entry; "parked" stands at
    # its lane's kerb side and never moves.
    # Braking lead with a second car ahead of it in the lane, second_gap_m bumper to bumper
    # (negative: none): two cars close along the radar's line of sight. The second holds
    # the lead's speed and brakes when the lead does (fifth review's matrix).
    second_gap_m: float = -1.0
    oncoming_m: float = 12.0
    oncoming_mps: float = 6.0
    go_after_s: float = 2.0
    oncoming_trigger_m: float = 12.0
    parked_offset_m: float = 0.9
    # Crossing vehicle (M07): waits side_m right of the lane centre (beyond the pavement:
    # a driveway; more: a side road), crosses at cross_mps, stops clear_beyond_m past the
    # lane's left edge.
    side_m: float = 6.0
    cross_mps: float = 2.0
    clear_beyond_m: float = 2.0
    # With radar_first at 1, the run is the radar-first variant (sixth review, F6-06): it is
    # exercised only if the radar tracked the car alone, before any LiDAR track on it, for
    # radar_first_min_s or more (``score_radar_timeline``); a far trigger_m and side_m set
    # it up. At 0 the plain crossing.
    radar_first: float = 0.0
    radar_first_min_s: float = 0.25

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

    def roles(self):
        """(role, CARLA actor) of every vehicle or walker the scenario drives."""
        return [("primary", self.actor)]

    def truths(self):
        """Evaluator truth of every actor the scenario drives, each with its ``role`` (eighth
        review, F8-05: a second car was driven but never recorded, so no metric could say
        which car a track was). Log only: the stack never sees it."""
        return [dict(_box_truth(actor, self.core_xy), role=role) for role, actor in self.roles() if actor is not None and getattr(actor, "is_alive", True)]

    def hold_ends_run(self):
        """Whether a sustained hold may end the run now (review: holding is not a result
        for every scenario; a car that waits out a crossing must be allowed to resume)."""
        return True

    def destroy(self):
        if self.actor is not None and self.actor.is_alive:
            self.actor.destroy()
        self.actor = None


def braking_profile_control(speed_mps, start_mps, elapsed_s, decel_mps2, gain=1.5):
    """The brake that holds a scripted car to a constant-deceleration stop: the speed the
    profile asks for by the next tick (``start_mps`` less ``decel_mps2`` times the time
    braking, that tick included), and
    the brake in proportion to the excess, full once stopped (a hold). A fixed pedal
    (brake_mps2 / 8, "CARLA's brake ~8 m/s^2 at 1") stopped the lead at 2.8 m/s within 0.15
    s, about 18 m/s^2 against the 3 m/s^2 asked for: CARLA's brake is far stronger at low
    speed (PC 2, r10_lead_pair, a73f5e0; ninth review, N9-01's braking matrix)."""
    target = max(0.0, start_mps - decel_mps2 * elapsed_s)
    if target <= 0.0 and speed_mps < 0.3:
        return 1.0
    return float(np.clip(gain * (speed_mps - target), 0.0, 1.0))


class BrakingLeadCar(ScriptedActor):
    """M05: a lead car that brakes to a stop at ``brake_at_m``; with ``second_gap_m`` at 0
    or more, a second car ahead of it in the lane that holds its speed and brakes with it."""

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
        self.second = None
        if p.second_gap_m >= 0.0:
            length = 2.0 * float(getattr(getattr(self.actor, "bounding_box", None), "extent", SimpleNamespace(x=2.2)).x)
            wp2 = waypoints[min(len(waypoints) - 1, int(round(p.lead_start_m + p.second_gap_m + length)))]
            t2 = wp2.transform
            t2.location.z += 0.3
            self.second = owner.try_spawn(library.find(LEAD_CAR), t2)
            if self.second is None:
                raise RuntimeError("Could not place the second car ahead of the lead: try another --actor-param second_gap_m")
        return self.actor

    def _drive(self, actor, braking, tick=0):
        """Hold lead_mps along the lane, or brake at brake_mps2 (``braking_profile_control``)."""
        carla, p = self.carla, self.params
        if braking:
            v = actor.get_velocity()
            speed = math.hypot(v.x, v.y)
            starts = self.__dict__.setdefault("_brake_starts", {})
            start_tick, start_speed = starts.setdefault(id(actor), (tick, speed))
            brake = braking_profile_control(speed, start_speed, (tick - start_tick + 1) * 0.05, p.brake_mps2)
            actor.apply_control(carla.VehicleControl(throttle=0.0, brake=brake))
            return
        t = actor.get_transform()
        x, y = self.core_xy(t.location)
        lane = np.array([self.core_xy(w.transform.location) for w in self.waypoints])
        s = float(np.argmin(np.hypot(lane[:, 0] - x, lane[:, 1] - y)))
        v = actor.get_velocity()
        speed = math.hypot(v.x, v.y)
        # A plain speed loop along the lane: throttle to hold lead_mps, steer to the lane.
        throttle = float(np.clip(0.35 + 0.3 * (p.lead_mps - speed), 0.0, 0.7))
        ahead = self.waypoints[min(len(self.waypoints) - 1, int(s) + 4)]
        ax, ay = self.core_xy(ahead.transform.location)
        yaw = -math.radians(t.rotation.yaw)
        bearing = math.atan2(ay - y, ax - x) - yaw
        bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
        actor.apply_control(carla.VehicleControl(throttle=throttle, steer=float(np.clip(-bearing / math.radians(35), -1, 1))))

    def step(self, tick, ego_truth):
        p = self.params
        x, y = self.core_xy(self.actor.get_transform().location)
        lane = np.array([self.core_xy(w.transform.location) for w in self.waypoints])
        s = float(np.argmin(np.hypot(lane[:, 0] - x, lane[:, 1] - y)))
        if s >= p.brake_at_m:
            self.braking = True
        self._drive(self.actor, self.braking, tick)
        if self.second is not None:
            self._drive(self.second, self.braking, tick)

    def roles(self):
        return [("lead", self.actor)] + ([("second", self.second)] if getattr(self, "second", None) is not None else [])

    def destroy(self):
        if getattr(self, "second", None) is not None and self.second.is_alive:
            self.second.destroy()
        self.second = None
        super().destroy()


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
        if self.cut_start is not None and p.cut_brake_after_s >= 0.0 and (tick - self.cut_start) * 0.05 >= p.cut_in_s + p.cut_brake_after_s:
            self.braking = True
            starts = self.__dict__.setdefault("_brake_starts", {})
            start_tick, start_speed = starts.setdefault(id(self.actor), (tick, speed))
            throttle, brake = 0.0, braking_profile_control(speed, start_speed, (tick - start_tick + 1) * 0.05, p.brake_mps2)
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


class CrossingVehicle(ScriptedActor):
    """M07: a car crossing the ego lane from the right, out of a driveway, a car park or a
    side road (fifth review: F5-02's driveway exit, F5-06's crossing target). It waits
    ``side_m`` right of the lane centre at ``brake_at_m`` along the lane, facing across it,
    and moves off at ``cross_mps`` once the ego is within ``trigger_m``. Its speed is held
    by CARLA's constant-velocity mode, so its velocity (and the radar's Doppler) is real and
    the kerb does not slow it. It stops once its whole body is ``clear_beyond_m`` past the
    lane's left edge."""

    def spawn(self, owner, library, waypoints):
        p = self.params
        wp = waypoints[min(len(waypoints) - 1, int(round(p.brake_at_m)))]
        self.cross_wp = wp
        self.lane_width_m = float(getattr(wp, "lane_width", 3.5))
        t = wp.transform
        yaw = math.radians(t.rotation.yaw)
        right = (-math.sin(yaw), math.cos(yaw))  # CARLA frame: y to the right
        self.cross_xy = self.core_xy(t.location)
        rx, ry = self.core_xy(SimpleNamespace(x=t.location.x + right[0], y=t.location.y + right[1], z=t.location.z))
        self.right_core = (rx - self.cross_xy[0], ry - self.cross_xy[1])
        t.location.x += p.side_m * right[0]
        t.location.y += p.side_m * right[1]
        t.location.z += 0.5
        t.rotation.yaw = t.rotation.yaw - 90.0  # facing left, across the lane
        self.actor = owner.try_spawn(library.find(LEAD_CAR), t)
        if self.actor is None:
            raise RuntimeError(f"Could not place the crossing car {p.side_m:g} m right of the lane centre: try another --site or --actor-param side_m")
        self.actor.apply_control(self.carla.VehicleControl(brake=1.0, hand_brake=True))
        self.stopped = False
        return self.actor

    def _offset(self):
        """The car's box centre right of the crossing point's lane centre (m; negative: left)."""
        truth = _box_truth(self.actor, self.core_xy)
        x, y = truth["centre_xy"]
        return (x - self.cross_xy[0]) * self.right_core[0] + (y - self.cross_xy[1]) * self.right_core[1], truth["extent_m"][0]

    def step(self, tick, ego_truth):
        carla, p = self.carla, self.params
        if self.stopped:
            return
        if not self.started and ego_truth is not None:
            cx, cy = self.cross_xy
            ex, ey = ego_truth["rear_axle_xy"]
            if math.dist((cx, cy), (ex, ey)) <= p.trigger_m:
                self.started = True
                self.actor.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=False))
                self.actor.enable_constant_velocity(carla.Vector3D(p.cross_mps, 0.0, 0.0))
        if self.started:
            offset, half_length = self._offset()
            if offset <= -(self.lane_width_m / 2 + half_length + p.clear_beyond_m):
                self.actor.disable_constant_velocity()
                self.actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
                self.stopped = True

    def cleared(self):
        """The car's whole body is a metre past the lane's left edge."""
        if self.actor is None or not self.started:
            return False
        offset, half_length = self._offset()
        return offset <= -(self.lane_width_m / 2 + half_length + 1.0)

    def hold_ends_run(self):
        return self.cleared()


class OncomingCar(ScriptedActor):
    """M08, for a left turn (fifth review, F5-07): a car in the opposing lane across the
    junction. ``mode`` "stopped": standing in the middle of its lane, it pulls away towards
    the junction and straight through it ``go_after_s`` after the ego first comes within
    ``oncoming_trigger_m`` of the junction entry, so the left-turn gate must not have
    excused it as static. "parked": standing at its lane's kerb side, it never moves, and
    the gate may excuse it once it has been seen standing long enough."""

    def __init__(self, carla, params, core_xy, mode):
        super().__init__(carla, params, core_xy)
        if mode not in ("stopped", "parked"):
            raise ValueError("oncoming mode is 'stopped' or 'parked'")
        self.mode = mode

    @staticmethod
    def _straight_on(carla, wp, yaw_deg, step_m):
        """The next waypoint ``step_m`` on with the heading closest to ``yaw_deg``, or None."""
        options = wp.next(step_m)
        if not options:
            return None
        best = min(options, key=lambda w: abs((w.transform.rotation.yaw - yaw_deg + 180.0) % 360.0 - 180.0))
        return best if abs((best.transform.rotation.yaw - yaw_deg + 180.0) % 360.0 - 180.0) < 25.0 else None

    def spawn_at_turn(self, owner, library, site):
        carla, p = self.carla, self.params
        entry = site["approach"][-1]
        self.entry_xy = self.core_xy(entry.transform.location)
        yaw = entry.transform.rotation.yaw
        # Straight across the junction, then oncoming_m beyond it.
        wp, inside, beyond = entry, False, None
        for _ in range(120):
            wp = self._straight_on(carla, wp, yaw, 1.0)
            if wp is None:
                break
            inside = inside or wp.is_junction
            if inside and not wp.is_junction:
                beyond = wp
                break
        if beyond is None:
            raise RuntimeError("No road straight across this junction for an oncoming car: try another --site")
        ahead = beyond.next(p.oncoming_m)
        ahead = [w for w in ahead if not w.is_junction] or [beyond]
        far = min(ahead, key=lambda w: abs((w.transform.rotation.yaw - yaw + 180.0) % 360.0 - 180.0))
        lane = far.get_left_lane()
        if lane is None or lane.lane_type != carla.LaneType.Driving or lane.lane_id * far.lane_id > 0:
            raise RuntimeError("No opposing Driving lane across this junction: try another --site")
        t = lane.transform
        self.lane_wp = lane
        if self.mode == "parked":
            heading = math.radians(t.rotation.yaw)
            right = (-math.sin(heading), math.cos(heading))  # its own right: the kerb side
            t.location.x += p.parked_offset_m * right[0]
            t.location.y += p.parked_offset_m * right[1]
        t.location.z += 0.3
        self.actor = owner.try_spawn(library.find(LEAD_CAR), t)
        if self.actor is None:
            raise RuntimeError("Could not place the oncoming car: try another --site or --actor-param oncoming_m")
        self.actor.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
        # Its way: along its lane, straight through the junction, 60 m.
        route, w = [lane], lane
        for _ in range(60):
            w = self._straight_on(carla, w, w.transform.rotation.yaw, 1.0)
            if w is None:
                break
            route.append(w)
        self.route = np.array([self.core_xy(w.transform.location) for w in route])
        self.near_since = None
        self.done = False
        return self.actor

    def spawn(self, owner, library, waypoints):
        raise RuntimeError("the oncoming car is placed with spawn_at_turn (left turns only)")

    def step(self, tick, ego_truth):
        carla, p = self.carla, self.params
        if self.mode == "parked" or self.done:
            return
        if self.near_since is None and ego_truth is not None and math.dist(ego_truth["rear_axle_xy"], self.entry_xy) <= p.oncoming_trigger_m:
            self.near_since = tick
        if not self.started and self.near_since is not None and (tick - self.near_since) * 0.05 >= p.go_after_s:
            self.started = True
        if not self.started:
            return
        t = self.actor.get_transform()
        x, y = self.core_xy(t.location)
        k = int(np.argmin(np.hypot(self.route[:, 0] - x, self.route[:, 1] - y)))
        if k >= len(self.route) - 3:
            self.actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            self.done = True
            return
        ax, ay = self.route[min(len(self.route) - 1, k + 4)]
        bearing = (math.atan2(ay - y, ax - x) - (-math.radians(t.rotation.yaw)) + math.pi) % (2 * math.pi) - math.pi
        v = self.actor.get_velocity()
        speed = math.hypot(v.x, v.y)
        throttle = float(np.clip(0.45 + 0.3 * (p.oncoming_mps - speed), 0.0, 0.8))
        self.actor.apply_control(carla.VehicleControl(throttle=throttle, hand_brake=False, steer=float(np.clip(-bearing / math.radians(35), -1, 1))))

    def hold_ends_run(self):
        return False  # waiting at the junction is the point; the turn gates judge the run


def make_oncoming(carla, mode, core_xy, **overrides):
    return OncomingCar(carla, ScenarioParams(f"oncoming_{mode}", **overrides), core_xy, mode)


SCENARIOS = {
    "crossing_pedestrian": (CrossingPedestrian, "M04"),
    "braking_lead": (BrakingLeadCar, "M05"),
    "cut_in": (CutInCar, "M06"),
    "crossing_vehicle": (CrossingVehicle, "M07"),
}


def make_scenario(carla, name, core_xy, **overrides):
    cls, _ = SCENARIOS[name]
    return cls(carla, ScenarioParams(name, **overrides), core_xy)


def ego_to_actor_distance(truth, actor, point_ego):
    """Distance from a point in the ego's rear-axle frame (x forward, y left) to the actor's
    true box; 0 inside it."""
    x, y = truth["rear_axle_xy"]
    c, s = math.cos(truth["yaw_rad"]), math.sin(truth["yaw_rad"])
    u, v = point_ego
    px, py = x + c * u - s * v, y + s * u + c * v
    ax, ay = actor["centre_xy"]
    ca, sa = math.cos(actor["yaw_rad"]), math.sin(actor["yaw_rad"])
    dx, dy = px - ax, py - ay
    hl, hw = actor["extent_m"]
    return math.hypot(max(abs(ca * dx + sa * dy) - hl, 0.0), max(abs(-sa * dx + ca * dy) - hw, 0.0))


def lead_is_actor(row, actor, match_m=1.0):
    """Whether the stack's lead this tick was the actor: its logged position within
    ``match_m`` of the actor's box. ``None`` when the log has no lead position (before the
    third review) or there was no lead."""
    lead = (row.get("driver") or {}).get("lead")
    if not lead or lead.get("x_m") is None or actor is None:
        return None
    return ego_to_actor_distance(row["truth"], actor, (lead["x_m"], lead["y_m"])) <= match_m


def score_actor_tracking(log, actor_truths, body, dt_s=0.05, match_m=1.0, view_ahead_m=50.0, view_side_m=10.0):
    """Which tracks were the scenario's actor, from the logged tracks around the car and the
    actor's true box (third review, F1: the lead's errors were scored against the actor
    without asking whether the lead was the actor). Truth stays on this side: the stack
    logs its tracks, the evaluator matches them. Reported, not gated.

    Over ticks with the actor in view (centre within ``view_ahead_m`` ahead, 5 m behind, and
    ``view_side_m`` to the side): ``matched_ticks`` (a track within ``match_m`` of its box),
    ``missed_ticks`` and the longest run of them, ``not_judged_ticks`` (no match, but the
    log left tracks out that tick), ``fragmented_ticks`` (two or more tracks on it;
    of those, ticks with a radar-only one, the tracker's reasons for not merging it, and
    ticks where the second track rather than the lead set the cap), ``track_ids`` (the nearest matched track's ids, in order) and ``id_switches``
    between consecutive matched ticks. Over ticks with a lead: ``lead_is_actor_ticks``,
    ``false_lead_ticks`` (a lead that was not the actor); and ``actor_in_lane_not_lead_ticks``
    (the actor ahead in the ego's lane, in view, and not the lead).
    """
    rows = [(row, actor) for row, actor in zip(log, actor_truths) if actor is not None]
    if not rows or "tracks_near" not in (rows[0][0].get("driver") or {}):
        return dict(available=False, note="the log has no tracks_near (recorded before the third review)")
    in_view = matched = missed = fragmented = unknown = 0
    radar_fragments = twin_caps = 0
    twins_kept = {}
    run = longest = 0
    ids = []
    lead_ticks = lead_actor = false_lead = not_lead = 0
    for row, actor in rows:
        d = row.get("driver") or {}
        cx, cy = _ego_frame(row["truth"], actor["centre_xy"])
        view = -5.0 <= cx <= view_ahead_m and abs(cy) <= view_side_m
        is_actor = lead_is_actor(row, actor, match_m)
        if d.get("lead"):
            lead_ticks += 1
            lead_actor += bool(is_actor)
            false_lead += is_actor is False
        if view and _ahead_in_band(row["truth"], actor, body, 0.5)[0] and not is_actor:
            not_lead += 1
        if not view:
            run = 0
            continue
        in_view += 1
        hits = sorted((ego_to_actor_distance(row["truth"], actor, (e[1], e[2])), e[0]) for e in d.get("tracks_near") or ())
        hits = [h for h in hits if h[0] <= match_m]
        if not hits:
            if d.get("tracks_near_omitted"):
                # The log left tracks out (a crowded tick): the actor's may be among them,
                # so this is not counted as a miss (fourth review).
                unknown += 1
                continue
            missed += 1
            run += 1
            longest = max(longest, run)
            continue
        run = 0
        matched += 1
        fragmented += len(hits) >= 2
        ids.append(hits[0][1])
        if len(hits) >= 2:
            # What the second track is, and why the tracker kept it (fourth review).
            entries = {e[0]: e for e in d.get("tracks_near") or ()}
            on_actor = {tid for _, tid in hits}
            radar_fragments += any(len(entries[tid]) > 8 and "L" not in entries[tid][8] for _, tid in hits[1:])
            for radar_id, lidar_id, _, why in (d.get("radar_stats") or {}).get("twins_kept") or ():
                if radar_id in on_actor and lidar_id in on_actor:
                    key = why.split(" by ")[0].split(" (")[0]
                    twins_kept[key] = twins_kept.get(key, 0) + 1
            cause, lead = d.get("cap_track") or {}, d.get("lead") or {}
            if cause.get("track_id") in on_actor and lead.get("track_id") in on_actor and cause["track_id"] != lead["track_id"]:
                twin_caps += 1  # a second track on the actor set the cap, not the lead
    order = [tid for k, tid in enumerate(ids) if k == 0 or tid != ids[k - 1]]
    return dict(
        available=True,
        match_m=match_m,
        in_view_ticks=in_view,
        matched_ticks=matched,
        matched_share=round(matched / in_view, 3) if in_view else None,
        missed_ticks=missed,
        not_judged_ticks=unknown,
        longest_missed_s=round(longest * dt_s, 2),
        fragmented_ticks=fragmented,
        fragmented_with_radar_only_ticks=radar_fragments,
        twins_kept_why=twins_kept,
        second_track_set_cap_ticks=twin_caps,
        track_ids=order[:20],
        id_switches=len(order) - 1 if order else 0,
        lead_ticks=lead_ticks,
        lead_is_actor_ticks=lead_actor,
        false_lead_ticks=false_lead,
        false_lead_s=round(false_lead * dt_s, 2),
        actor_in_lane_not_lead_ticks=not_lead,
    )


def score_lead_tracking(log, actor_truths, body, dt_s=0.05, match_m=1.0):
    """What the stack knew about the lead, against the evaluator's truth (G3: radar must
    earn its place). Reported, not gated:

    - ``first_lead_s`` / ``first_lead_truth_gap_m``: when the actor was first the lead and
      the true bumper gap then;
    - ``speed_error_rmse_mps`` / ``speed_error_p95_mps``: the lead's speed along the path,
      as the stack had it, against the actor's, over ticks with the actor as the lead;
    - ``gap_error_median_m``: the stack's gap minus the true one (negative: conservative);
    - ``brake_s``: when the actor began to brake (the first tick below its cruise, of a
      drop past 0.3 m/s), looked for from the tick it first led the car, against the
      fastest it went from then on (None if it braked only before it led);
      ``brake_seen_s``: the first tick after it with the actor as the lead and its speed
      0.5 m/s below the actor's cruise, and ``brake_delay_s`` between them;
    - ``stopped_before_lead_s``: time the ego stood still while the lead still moved
      (false stops), and ``support``: ticks of each sensor set behind the lead.

    Only ticks whose lead was the actor (``lead_is_actor``, the third review's F1) are
    scored; ``unmatched_lead_ticks`` counts the others. A log without lead positions
    (before the third review) is scored on every lead tick, as ``matching`` says.
    """
    gaps = separations(log, actor_truths, body)
    errors, gap_errors, support = [], [], {}
    nees, nis = {}, []
    by_phase, lead_since = {}, {}
    first = None
    speeds = [a["speed_mps"] if a else None for a in actor_truths]
    # The actor's braking is looked for from the tick it first led the car, against the
    # fastest it went from then on: a cut-in car slowing from its lane change to its
    # following speed read as a brake 2.5 s before it led (live r7, braking cut-in).
    led_from = next(
        (k for k, (row, a) in enumerate(zip(log, actor_truths)) if a is not None and (row.get("driver") or {}).get("lead") and not row["driver"]["lead"].get("crossing") and lead_is_actor(row, a, match_m) is not False),
        None,
    )
    cruise = None if led_from is None else max((v for v in speeds[led_from:] if v is not None), default=None)
    brake_tick = seen_tick = at_cruise = None
    stopped_ticks, moved = 0, False
    unmatched, positions = 0, False
    for k, (row, actor) in enumerate(zip(log, actor_truths)):
        if actor is None:
            continue
        lead = (row.get("driver") or {}).get("lead")
        if cruise is not None and brake_tick is None and k >= led_from:
            if actor["speed_mps"] >= cruise - 0.1:
                at_cruise = k
            elif at_cruise is not None and actor["speed_mps"] < cruise - 0.3:
                brake_tick = at_cruise + 1  # the first tick below cruise
        moved = moved or row["truth"]["speed_mps"] >= 0.5
        goal = row["truth"].get("goal_distance_m")
        if moved and row["truth"]["speed_mps"] < 0.05 and actor["speed_mps"] > 0.5 and (goal is None or goal > 5.0):
            stopped_ticks += 1  # after the car first moved, short of the goal: neither the start-up wait nor the goal stop is a false stop
        if lead and lead.get("crossing") and brake_tick is not None and seen_tick is None and k >= brake_tick:
            # A stopped lead whose last motion still reads "moving" is predicted as a crossing
            # conflict where it stands for a few ticks (r10 cut-in: 0.45 s): its speed still
            # shows the stop being seen.
            if lead_is_actor(row, actor, match_m) is not False and lead["speed_mps"] < cruise - 0.5:
                seen_tick = k
        if not lead or lead.get("crossing"):
            continue
        is_actor = lead_is_actor(row, actor, match_m)
        positions = positions or is_actor is not None
        if is_actor is False:
            unmatched += 1
            continue
        if first is None:
            first = (row["time_s"], gaps[k])
        errors.append(lead["speed_mps"] - actor["speed_mps"])
        if gaps[k] is not None and lead.get("gap_m") is not None:
            gap_errors.append(lead["gap_m"] - gaps[k])
        key = "+".join(lead.get("support") or ()) or "none"
        support[key] = support.get(key, 0) + 1
        sigma = lead.get("speed_sigma_mps")
        if sigma:
            value = ((lead["speed_mps"] - actor["speed_mps"]) / max(sigma, 1e-3)) ** 2
            nees.setdefault(key, []).append(value)
            # ... and by phase (ninth review, N9-03): the first second of a lead track (an
            # acquisition, a hand-over or a merge), the actor braking, stopped, or neither.
            since = lead_since.setdefault(lead.get("track_id"), k)
            if k - since < round(1.0 / dt_s):
                phase = "acquired"
            elif brake_tick is not None and k >= brake_tick:
                phase = "braking" if actor["speed_mps"] > 0.05 else "stopped"
            else:
                phase = "cruise"
            by_phase.setdefault(phase, []).append(value)
        if lead.get("lidar_nis") is not None:
            nis.append(lead["lidar_nis"])
        if brake_tick is not None and seen_tick is None and k >= brake_tick and lead["speed_mps"] < cruise - 0.5:
            seen_tick = k
    e = np.abs(np.array(errors)) if errors else None
    t = lambda k: None if k is None else round(log[k]["time_s"], 2)  # noqa: E731
    # The braking the actor achieved (ninth review: the scripted pedal gave about 18 m/s^2
    # where 3 was asked): from the tick it first fell below cruise to the first at rest.
    achieved = None
    if brake_tick is not None and cruise:
        stop_k = next((k for k in range(brake_tick, len(speeds)) if speeds[k] is not None and speeds[k] < 0.1), None)
        if stop_k is not None and stop_k > brake_tick:
            achieved = round(cruise / ((stop_k - brake_tick + 1) * dt_s), 2)
    return dict(
        brake_achieved_mps2=achieved,
        matching="lead matched to the actor within %.1f m" % match_m if positions else "unavailable: the log has no lead positions",
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
        # Is the lead's speed as certain as the filter says (fifth review, F5-04)? The
        # normalised estimation error squared against the truth, by the sensors behind it
        # (one degree of freedom: mean about 1, 95% under 3.84 when consistent; much more is
        # overconfidence), and the LiDAR innovations' (two: about 2).
        speed_nees={k: dict(ticks=len(v), mean=round(float(np.mean(v)), 2), within_95=round(float(np.mean(np.array(v) <= 3.84)), 2)) for k, v in nees.items()},
        speed_nees_by_phase={k: dict(ticks=len(v), mean=round(float(np.mean(v)), 2), within_95=round(float(np.mean(np.array(v) <= 3.84)), 2)) for k, v in sorted(by_phase.items())},
        lidar_nis_mean=round(float(np.mean(nis)), 2) if nis else None,
        # 2 when the LiDAR's position noise is right; far under it, the stated noise is
        # conservative (slower position updates), far over it, overconfident.
        lidar_nis_verdict=None if not nis else ("conservative" if np.mean(nis) < 0.5 else "overconfident" if np.mean(nis) > 6.0 else "consistent"),
        ticks=len(errors),
        unmatched_lead_ticks=unmatched,
    )


def _cause_stem(reason):
    """A speed-cap reason without its "(held: ...)" suffix."""
    return (reason or "speed cap").split(" (held")[0]


def restrictions(log, actor_truths, body, cruise_mps, match_m=1.0):
    """Per tick: what restricted the car's command (``None`` when nothing did), and whether
    that was the scenario's actor (third review, F4: brake-reason attribution).

    - A speed cap below cruise: it was computed on the tick before (the stack plans the
      next tick under it), so its reason and its track are that tick's. It is the actor's
      when the track behind it (``cap_track``), or the lead it follows, was the actor.
    - A stop the planner holds (obstacle, hold, junction) within braking range: the
      actor's when the actor is ahead in the ego's band (1 m margin) no further than the
      stop point plus 6 m.
    """
    out = []
    for k, (row, actor) in enumerate(zip(log, actor_truths)):
        d = row.get("driver") or {}
        prev = log[k - 1] if k else row  # the first row's cap was computed before the log began: its own stands in
        pd = prev.get("driver") or {}
        cap = d.get("speed_cap_mps")
        cause, mine = None, False
        v = row["truth"]["speed_mps"]
        stop = d.get("stop_distance_m")
        if d.get("stop_reason") in ("obstacle", "obstacle_hold", "junction") and stop is not None and stop <= v * v / 2.0 + 3.0:
            cause = "stop: " + d["stop_reason"]
            if actor is not None:
                inside, _, _ = _ahead_in_band(row["truth"], actor, body, 1.0)
                gap = polygon_distance(ego_vertices(row["truth"], body), box_vertices(actor["centre_xy"], actor["yaw_rad"], actor["extent_m"]))
                mine = inside and gap <= stop + 6.0
        elif cap is not None and cap < cruise_mps - 0.25:
            cause = "cap: " + _cause_stem(pd.get("speed_cap_reason"))
            prev_actor = actor_truths[k - 1] if k else actor
            track = pd.get("cap_track")
            if prev is not None and prev_actor is not None:
                if track and track.get("x_m") is not None:
                    mine = ego_to_actor_distance(prev["truth"], prev_actor, (track["x_m"], track["y_m"])) <= match_m
                elif "lead" in cause:
                    is_actor = lead_is_actor(prev, prev_actor, match_m)
                    if is_actor is None:
                        # A log without the lead's position (before the third review): the
                        # actor ahead in the ego's band stands in for the match.
                        is_actor = _ahead_in_band(prev["truth"], prev_actor, body, 1.0)[0]
                    mine = bool(is_actor)
        out.append((cause, mine))
    return out


def score_behaviour(name, log, actor_truths, body, cruise_mps, dt_s=0.05, gap0_m=4.0, headway_s=2.0, goal_stop_m=5.0):
    """Driving quality beside the safety gates (third review, F4): what restricted the car,
    whether it was the actor, and the time it cost. Reported, not gated: these make
    unnecessary braking visible; thresholds wait for a baseline.

    For every scenario, after the car first moved: ``restricted_s`` and ``time_lost_s``
    (the sum of dt x (1 - v / cruise) over restricted ticks), each split into the actor's
    and the rest, with the rest's causes; ``brake_ticks`` (brake commanded at 0.3 m/s or
    more) and the share of them the actor's. Then:

    - braking lead, before the actor began to brake: the restricted time not the actor's,
      ``pre_brake_unjustified_slow_s`` (slower than the lead by 0.5 m/s with the gap at
      least the following rule's desired gap plus 2 m; the lead's speed or cruise, whichever
      is lower) and the lowest speed after the car first reached it;
    - crossing pedestrian, once the walker had cleared and the car resumed: the near stops
      (below 0.5 m/s after 1 m/s, short of the goal by ``goal_stop_m`` or more), the
      restricted time not the walker's, and the progress
      (distance over time to the end of the run);
    - cut-in: the first restriction that was the cut-in car and the first brake command,
      in seconds from the cut-in, and the least time to collision from 3 s before it to 5 s
      after (bumper gap over closing speed, closing at 0.1 m/s or more).
    """
    causes = restrictions(log, actor_truths, body, cruise_mps)
    speeds = [r["truth"]["speed_mps"] for r in log]
    moved = next((k for k, v in enumerate(speeds) if v >= 0.5), None)
    if moved is None:
        return dict(note="the car never moved")

    def summary(ticks, prefix=""):
        restricted = [k for k in ticks if causes[k][0] is not None]
        mine = [k for k in restricted if causes[k][1]]
        other = [k for k in restricted if not causes[k][1]]
        lost = lambda ks: round(sum(max(0.0, 1.0 - speeds[k] / cruise_mps) for k in ks) * dt_s, 2)  # noqa: E731
        by = {}
        for k in other:
            by[causes[k][0]] = round(by.get(causes[k][0], 0.0) + dt_s, 2)
        return {
            f"{prefix}restricted_s": round(len(restricted) * dt_s, 2),
            f"{prefix}restricted_by_actor_s": round(len(mine) * dt_s, 2),
            f"{prefix}restricted_other_s": round(len(other) * dt_s, 2),
            f"{prefix}time_lost_s": lost(restricted),
            f"{prefix}time_lost_other_s": lost(other),
            f"{prefix}other_causes_s": dict(sorted(by.items(), key=lambda kv: -kv[1])[:6]),
        }

    after = range(moved, len(log))
    out = summary(after)
    brakes = [k for k in after if (log[k].get("command") or [0, 0, 0])[2] >= 0.05 and speeds[k] >= 0.3]
    out["brake_ticks"] = len(brakes)
    out["brake_ticks_actor_share"] = round(sum(causes[k][1] for k in brakes) / len(brakes), 3) if brakes else None
    rows = [(k, a) for k, a in enumerate(actor_truths) if a is not None]
    if not rows:
        return out | dict(note="no actor measurements")
    t0 = lambda k: round(log[k]["time_s"], 2)  # noqa: E731
    if name == "braking_lead":
        lead_cruise = max(a["speed_mps"] for _, a in rows)
        at_cruise = brake = None
        for k, a in rows:
            if a["speed_mps"] >= lead_cruise - 0.1:
                at_cruise = k
            elif at_cruise is not None and a["speed_mps"] < lead_cruise - 0.3:
                brake = at_cruise + 1
                break
        if brake is None:
            return out | dict(note="the lead never braked")
        before = range(moved, brake)
        gaps = separations(log, actor_truths, body)
        unjustified = [
            k for k in before
            if actor_truths[k] is not None and gaps[k] is not None
            and speeds[k] < min(cruise_mps, actor_truths[k]["speed_mps"]) - 0.5 and gaps[k] >= gap0_m + headway_s * speeds[k] + 2.0
        ]
        reached = next((k for k in before if actor_truths[k] is not None and speeds[k] >= min(cruise_mps, actor_truths[k]["speed_mps"]) - 0.3), None)
        out |= summary(before, "pre_brake_")
        out |= dict(
            lead_brake_s=t0(brake),
            pre_brake_unjustified_slow_s=round(len(unjustified) * dt_s, 2),
            pre_brake_min_speed_after_lead_speed_mps=None if reached is None else round(min(speeds[reached:brake]), 2),
        )
        return out
    if name in ("crossing_pedestrian", "crossing_vehicle"):
        vehicle = name == "crossing_vehicle"
        first = next((k for k, a in rows if _ahead_in_band(log[k]["truth"], a, body, 1.0)[0]), None)
        if first is None:
            return out | dict(note="the " + ("crossing car" if vehicle else "walker") + " never entered the path")
        cleared = next((k for k, a in rows if k >= first and _cleared_left(log[k]["truth"], a, body, vehicle)), None)
        if cleared is None:
            return out | dict(note="the walker had not cleared the path when the run ended")
        resumed = next((k for k in range(cleared, len(log)) if speeds[k] >= 0.5), None)
        if resumed is None:
            return out | dict(walker_cleared_s=t0(cleared), note="the car did not resume")
        near_stops, up = 0, False
        for k in range(resumed, len(log)):
            goal = log[k]["truth"].get("goal_distance_m")
            if speeds[k] >= 1.0:
                up = True
            elif up and speeds[k] < 0.5 and (goal is None or goal > goal_stop_m):
                near_stops, up = near_stops + 1, False  # the stop at the goal is not one
        span = (len(log) - cleared) * dt_s
        out |= summary(range(resumed, len(log)), "post_clear_")
        out |= dict(
            walker_cleared_s=t0(cleared),
            resumed_s=t0(resumed),
            post_clear_near_stops=near_stops,
            post_clear_progress_mps=round(sum(speeds[cleared:]) * dt_s / span, 2) if span > 0 else None,
        )
        return out
    if name == "cut_in":
        cut = next((k for k, a in rows if _ahead_in_band(log[k]["truth"], a, body, 0.5)[0]), None)
        if cut is None:
            return out | dict(note="the cut-in car never entered the ego's lane")
        window = range(max(moved, cut - int(3.0 / dt_s)), min(len(log), cut + int(5.0 / dt_s)))
        response = next((k for k in window if causes[k][1]), None)
        braked = next((k for k in window if (log[k].get("command") or [0, 0, 0])[2] >= 0.05), None)
        gaps = separations(log, actor_truths, body)
        ttc = []
        for k in window:
            a = actor_truths[k]
            if a is None or gaps[k] is None:
                continue
            truth = log[k]["truth"]
            heading = truth["yaw_rad"]
            v_actor = a["speed_mps"] * math.cos(a["yaw_rad"] - heading)
            closing = speeds[k] - v_actor
            if closing >= 0.1:
                ttc.append(gaps[k] / closing)
        out |= dict(
            cut_in_s=t0(cut),
            first_actor_restriction_from_cut_in_s=None if response is None else round((response - cut) * dt_s, 2),
            first_brake_from_cut_in_s=None if braked is None else round((braked - cut) * dt_s, 2),
            min_ttc_s=round(min(ttc), 2) if ttc else None,
        )
        return out
    return out


def score_radar_timeline(log, actor_truths, body, cruise_mps, dt_s=0.05, match_m=1.5):
    """When the stack knew what about the actor once it moved (sixth review, F6-02: whether
    the radar's warning came early enough to matter). Reported, not gated.

    From ``actor_moving_s`` (the actor at 0.3 m/s or faster) on, times (s, run clock) of the
    first track on its box (within ``match_m``) with radar support (``first_radar_s``),
    radar alone (no LiDAR-supported track on it that tick: ``first_radar_only_s``) and LiDAR
    support (``first_lidar_s``); the first tick the stack predicted a radar-only track on
    it as a provisional or a full threat (``first_provisional_threat_s``,
    ``first_full_threat_s``), the first restriction it set for the actor (a cap below
    cruise whose responsible track is on it: ``first_restriction_s``), the first brake
    command (``first_brake_s``) and the actor entering the ego's band within 30 m
    (``entered_band_s``). ``lidar_before_moving_s``: the first tick a LiDAR track was on it
    before it moved (a car the LiDAR had standing is no radar-first case).

    The radar-only interval, from the actor's moving off to the first LiDAR track on it,
    counted from the ticks a radar-only track was actually on it (seventh review, F7-01:
    the span from the first radar-only tick to the first LiDAR tick credited ticks the
    radar had lost it): ``radar_only_span_s`` (that span), ``radar_only_observed_s`` (its
    ticks with a radar-only track on the actor), ``radar_only_longest_s`` (the longest
    unbroken run of them, which is ``radar_only_s``), ``radar_only_gaps`` and
    ``radar_only_longest_gap_s`` (ticks in the span without one), and
    ``threat_before_lidar`` (a provisional or full radar threat on it before the LiDAR
    had it: the warning was there for the plan to use). ``radar_first``: a radar-only
    run at all, with the LiDAR not on it standing. At the first restriction, the room
    from the car's front to the crossing car (``restriction_room_m``), the speed then and
    the deceleration a stop in that room needs (``restriction_decel_needed_mps2``).

    A radar restriction whose responsible track is not on the actor is not the actor's
    (eighth review, F8-04: "radar" in the cap's reason also counted, so an unrelated radar
    track could set the first restriction and its room): such caps from the actor's moving
    off are reported apart as possible false interventions (``unmatched_radar_restrictions``:
    ticks, seconds, first time and the tracks)."""
    rows = [(k, a) for k, a in enumerate(actor_truths) if a is not None]
    if not rows or "tracks_near" not in (log[rows[0][0]].get("driver") or {}):
        return dict(available=False, note="no actor measurements or no tracks_near in the log")
    keys = ("first_radar", "first_radar_only", "first_lidar", "first_provisional_threat", "first_full_threat", "first_restriction", "first_brake", "entered_band")
    first = dict.fromkeys(keys)
    moving = lidar_before = None
    alone = []  # (tick, radar-only track on it) from moving off until the LiDAR has it
    unmatched = []  # (tick, responsible track) of radar restrictions not on the actor
    for k, a in rows:
        row = log[k]
        d = row.get("driver") or {}
        on = {}
        for e in d.get("tracks_near") or ():
            if ego_to_actor_distance(row["truth"], a, (e[1], e[2])) <= match_m:
                on[e[0]] = e[8] if len(e) > 8 else ""
        lidar = any("L" in sup for sup in on.values())
        if moving is None and a["speed_mps"] >= 0.3:
            moving = k
        if moving is None:
            if lidar and lidar_before is None:
                lidar_before = k
            continue
        def mark(key):
            if first[key] is None:
                first[key] = k
        if any("R" in sup for sup in on.values()):
            mark("first_radar")
        radar_only = not lidar and any(sup == "R" for sup in on.values())
        if radar_only:
            mark("first_radar_only")
        if lidar:
            mark("first_lidar")
        if first["first_lidar"] is None:
            alone.append((k, radar_only))
        for entry in d.get("radar_threats") or ():
            tid, kind = (entry[0], entry[1]) if isinstance(entry, (list, tuple)) else (entry, "full")
            if tid in on:
                mark("first_provisional_threat" if kind == "provisional" else "first_full_threat")
        cap = d.get("speed_cap_mps")
        cause = (d.get("cap_track") or {}).get("track_id")
        if cap is not None and cap < cruise_mps - 0.25:
            if cause is not None and cause in on:
                mark("first_restriction")
            elif "radar" in (d.get("speed_cap_reason") or ""):
                unmatched.append((k, cause))
        if (row.get("command") or [0, 0, 0])[2] >= 0.05 and row["truth"]["speed_mps"] >= 0.3:
            mark("first_brake")
        inside, dx, _ = _ahead_in_band(row["truth"], a, body, 0.5)
        if inside and dx <= 30.0:
            mark("entered_band")
    if moving is None:
        return dict(available=True, match_m=match_m, actor_moving_s=None, note="the actor never moved")
    at = lambda k: None if k is None else round(log[k]["time_s"], 2)  # noqa: E731
    out = dict(available=True, match_m=match_m, actor_moving_s=at(moving), lidar_before_moving_s=at(lidar_before)) | {f"{key}_s": at(k) for key, k in first.items()}
    radar, lidar = first["first_radar_only"], first["first_lidar"]
    radar_first = lidar_before is None and radar is not None and (lidar is None or radar < lidar)
    span = [hit for k, hit in alone if radar is not None and k >= radar]
    runs, gaps, run, gap = [], [], 0, 0
    for hit in span:
        if hit:
            run += 1
            if gap:
                gaps.append(gap)
            gap = 0
        else:
            gap += 1
            if run:
                runs.append(run)
            run = 0
    runs += [run] if run else []
    gaps += [gap] if gap else []  # lost before the LiDAR had it: a gap too
    threat = first["first_provisional_threat"], first["first_full_threat"]
    before = [k for k in threat if k is not None and (lidar is None or k < lidar)]
    out |= dict(
        radar_first=radar_first,
        radar_only_span_s=round(len(span) * dt_s, 2) if radar_first else 0.0,
        radar_only_observed_s=round(sum(runs) * dt_s, 2) if radar_first else 0.0,
        radar_only_longest_s=round(max(runs, default=0) * dt_s, 2) if radar_first else 0.0,
        radar_only_gaps=len(gaps) if radar_first else 0,
        radar_only_longest_gap_s=round(max(gaps, default=0) * dt_s, 2) if radar_first else 0.0,
        threat_before_lidar=bool(before),
    )
    out["radar_only_s"] = out["radar_only_longest_s"]
    k = first["first_restriction"]
    if k is not None and actor_truths[k] is not None:
        # Whether the restriction came with room to stop (seventh review, F7-03): the car's
        # front to the crossing car's near side, and the deceleration a stop there needs.
        row, a = log[k], actor_truths[k]
        _, dx, _ = _ahead_in_band(row["truth"], a, body, 1.0)
        room = dx - body.front_m - min(a["extent_m"])
        v = row["truth"]["speed_mps"]
        out |= dict(
            restriction_room_m=round(room, 2),
            restriction_speed_mps=round(v, 2),
            restriction_decel_needed_mps2=None if room <= 0.1 else round(v * v / (2.0 * room), 2),
        )
    out["from_actor_moving_s"] = {key: None if k is None else round((k - moving) * dt_s, 2) for key, k in first.items()}
    out["unmatched_radar_restrictions"] = dict(
        ticks=len(unmatched),
        seconds=round(len(unmatched) * dt_s, 2),
        first_s=at(unmatched[0][0]) if unmatched else None,
        tracks=sorted({tid for _, tid in unmatched if tid is not None}),
    )
    return out



def score_multi_actor_tracking(log, actor_sets, body, dt_s=0.05, match_m=1.0):
    """Every scripted actor's tracking, and whether the tracks kept them apart (eighth review,
    F8-05). ``actor_sets``: per tick, the actors' truths, each with its ``role``. Per role,
    ``score_actor_tracking``'s summary (matched share, track ids, id switches, lead ticks).
    Across them: ``shared_track_ticks`` (one track the nearest match of two actors at once:
    two cars as one), ``tracks_on_two_actors`` (track ids matched to more than one actor
    over the run, with the roles: an identity carried from one car to the other) and, over
    ticks with a lead, which actor it was (``lead_by_role``) and how often that changed
    (``lead_hand_overs``, with the times). Reported, not gated; the runtime never sees it."""
    roles = sorted({a["role"] for actors in actor_sets for a in actors or ()})
    if len(roles) < 2:
        return dict(available=False, note="fewer than two actors recorded")
    by_role = {role: [next((a for a in actors or () if a["role"] == role), None) for actors in actor_sets] for role in roles}
    keep = ("matched_share", "track_ids", "id_switches", "lead_ticks", "lead_is_actor_ticks", "missed_ticks", "in_view_ticks")
    per_role = {}
    for role, truths in by_role.items():
        out = score_actor_tracking(log, truths, body, dt_s, match_m)
        per_role[role] = {k: out.get(k) for k in keep} if out.get("available") else out
    shared, owners, lead_roles, hand_overs, last = 0, {}, {}, [], None
    for k, row in enumerate(log):
        d = row.get("driver") or {}
        nearest = {}
        for role in roles:
            a = by_role[role][k]
            if a is None:
                continue
            hits = sorted((ego_to_actor_distance(row["truth"], a, (e[1], e[2])), e[0]) for e in d.get("tracks_near") or ())
            if hits and hits[0][0] <= match_m:
                nearest[role] = hits[0][1]
                owners.setdefault(hits[0][1], set()).add(role)
        shared += len(set(nearest.values())) < len(nearest)
        if d.get("lead"):
            matched = [role for role in roles if by_role[role][k] is not None and lead_is_actor(row, by_role[role][k], match_m)]
            role = matched[0] if len(matched) == 1 else ("several" if matched else "none")
            lead_roles[role] = lead_roles.get(role, 0) + 1
            if last is not None and role != last:
                hand_overs.append(dict(time_s=round(row["time_s"], 2), from_role=last, to_role=role))
            last = role
    return dict(
        available=True,
        match_m=match_m,
        roles=roles,
        per_role=per_role,
        shared_track_ticks=shared,
        tracks_on_two_actors={str(tid): sorted(rs) for tid, rs in owners.items() if len(rs) > 1},
        lead_by_role=lead_roles,
        lead_hand_overs=len(hand_overs),
        lead_hand_over_times=hand_overs[:10],
    )

EXERCISE_CONDITION = {
    "braking_lead": "the lead slows by 0.5 m/s or more while within 50 m ahead",
    "crossing_pedestrian": "the walker enters the ego's band (0.5 m margin) within 30 m ahead",
    "cut_in": "the cut-in car becomes the lead while a conflict or yield cap is in force, and the car then closes on it",
    "crossing_vehicle": "the crossing car enters the ego's band (0.5 m margin) within 30 m ahead",
    "oncoming_stopped": "the oncoming car pulls away while the ego stands within 40 m of it",
    "oncoming_parked": "the ego comes within 25 m of the parked oncoming car",
}
# A variant's condition on top of its scenario's (sixth review, F6-06: a radar-first run in
# which the LiDAR saw the car first, or a cut-in that brakes before the hand-over, is not
# the variant it is filed as).
EXERCISE_VARIANT = {
    "radar_first": "and the radar alone tracked the moving crossing car without a break for radar_first_min_s or more before the LiDAR did, with a radar threat on it in that time",
    "cut_brake": "and the cut-in car braked (0.5 m/s or more below its speed then) after it became the lead",
}


def exercise_variant(name, params):
    """The variant of ``name`` that ``params`` (``ScenarioParams.record()``) set up, or None."""
    params = params or {}
    if name == "crossing_vehicle" and params.get("radar_first", 0.0) >= 0.5:
        return "radar_first"
    if name == "cut_in" and params.get("cut_brake_after_s", -1.0) >= 0.0:
        return "cut_brake"
    return None


def score_exercised(name, log, actor_truths, body, cruise_mps, dt_s=0.05, handover_window_s=1.0, params=None):
    """Whether the run exercised what its scenario is for (fourth review, 9.1): a pass on a
    run that never reached the targeted transition is no evidence for it. The live cut-in
    passed with the car pulling away, so the conflict-to-following hand-over (R1) never
    happened. Reported as ``exercised`` with the condition and, when it was not met, what
    was missing. Given ``params``, a variant's own condition too (``EXERCISE_VARIANT``)."""
    variant = exercise_variant(name, params)
    out = _score_exercised(name, log, actor_truths, body, cruise_mps, dt_s, handover_window_s)
    if variant is None:
        return out
    out = out | dict(variant=variant, condition=out["condition"] + " " + EXERCISE_VARIANT[variant])
    if not out.get("exercised"):
        return out
    if variant == "radar_first":
        timeline = score_radar_timeline(log, actor_truths, body, cruise_mps, dt_s)
        need = (params or {}).get("radar_first_min_s", 0.25)
        out |= {k: timeline.get(k) for k in ("radar_only_s", "radar_only_observed_s", "radar_only_span_s", "radar_only_gaps", "threat_before_lidar")}
        if not timeline.get("available") or (timeline.get("radar_only_s") or 0.0) < need:
            return out | dict(exercised=False, note=f"the radar alone had the car for at most {timeline.get('radar_only_s')} s unbroken before the LiDAR (needs {need:g} s)")
        if not timeline.get("threat_before_lidar"):
            return out | dict(exercised=False, note="no radar threat on the car before the LiDAR had it")
        return out
    # cut_brake: the actor's speed falls 0.5 m/s below its speed at the hand-over, after it.
    start = next((k for k, row in enumerate(log) if out.get("lead_from_s") is not None and round(row["time_s"], 2) >= out["lead_from_s"]), None)
    at = actor_truths[start]["speed_mps"] if start is not None and actor_truths[start] is not None else None
    braked = None if at is None else next((k for k in range(start, len(log)) if actor_truths[k] is not None and actor_truths[k]["speed_mps"] <= at - 0.5), None)
    if braked is None:
        return out | dict(exercised=False, note="the cut-in car did not brake after it became the lead")
    return out | dict(braked_after_hand_over_s=round(log[braked]["time_s"], 2))


def _score_exercised(name, log, actor_truths, body, cruise_mps, dt_s=0.05, handover_window_s=1.0):
    out = dict(condition=EXERCISE_CONDITION.get(name, "none defined"))
    rows = [(k, a) for k, a in enumerate(actor_truths) if a is not None]
    if name not in EXERCISE_CONDITION or not rows:
        return out | dict(exercised=None, note="no actor measurements" if name in EXERCISE_CONDITION else "no condition for this scenario")
    if name == "braking_lead":
        top = 0.0
        for k, a in rows:
            cx, _ = _ego_frame(log[k]["truth"], a["centre_xy"])
            top = max(top, a["speed_mps"])
            if 0.0 <= cx <= 50.0 and a["speed_mps"] <= top - 0.5:
                return out | dict(exercised=True, at_s=round(log[k]["time_s"], 2))
        return out | dict(exercised=False, note="the lead did not slow within 50 m ahead")
    if name == "oncoming_stopped":
        for k, a in rows:
            ego = log[k]["truth"]
            if a["speed_mps"] >= 1.0 and ego["speed_mps"] < 0.1 and math.dist(ego["rear_axle_xy"], a["centre_xy"]) <= 40.0:
                return out | dict(exercised=True, at_s=round(log[k]["time_s"], 2))
        return out | dict(exercised=False, note="the oncoming car did not pull away while the ego stood near it")
    if name == "oncoming_parked":
        for k, a in rows:
            if math.dist(log[k]["truth"]["rear_axle_xy"], a["centre_xy"]) <= 25.0:
                return out | dict(exercised=True, at_s=round(log[k]["time_s"], 2))
        return out | dict(exercised=False, note="the ego never came within 25 m of the parked car")
    if name in ("crossing_pedestrian", "crossing_vehicle"):
        for k, a in rows:
            inside, dx, _ = _ahead_in_band(log[k]["truth"], a, body, 0.5)
            if inside and dx <= 30.0:
                return out | dict(exercised=True, at_s=round(log[k]["time_s"], 2))
        who = "walker" if name == "crossing_pedestrian" else "crossing car"
        return out | dict(exercised=False, note=f"the {who} never entered the ego's band within 30 m")
    window = int(round(handover_window_s / dt_s))
    first = None
    slower = False
    for k, a in rows:
        row = log[k]
        if not (row.get("driver") or {}).get("lead"):
            continue
        is_actor = lead_is_actor(row, a)
        if is_actor is None:
            is_actor = _ahead_in_band(row["truth"], a, body, 0.5)[0]
        if not is_actor:
            continue
        first = k if first is None else first
        slower = slower or a["speed_mps"] < row["truth"]["speed_mps"] - 0.2
    if first is None:
        return out | dict(exercised=False, note="the cut-in car never became the lead")
    before = [log[j].get("driver") or {} for j in range(max(0, first - window), first + 1)]
    restricted = any(
        d.get("speed_cap_mps") is not None and d["speed_cap_mps"] < cruise_mps - 0.25 and any(w in (d.get("speed_cap_reason") or "") for w in ("conflict", "yield"))
        for d in before
    )
    out |= dict(lead_from_s=round(log[first]["time_s"], 2), restricted_at_hand_over=restricted, closing_on_it=slower)
    if restricted and slower:
        return out | dict(exercised=True)
    missing = [m for m, ok in (("no conflict or yield cap was in force when it became the lead", restricted), ("the car never closed on it", slower)) if not ok]
    return out | dict(exercised=False, note="; ".join(missing))


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
    "cut_in": "a gap at or above the separation threshold once the cut-in is complete, then reach the goal (or, if the car braked to a stop, hold behind it)",
    "crossing_vehicle": "yield while the crossing car is in the path if the car would otherwise reach it before it clears (a second's margin), then resume and reach the goal",
}


def _cleared_left(truth, actor, body, vehicle):
    """A crossing road user is past the ego's band to the left (its whole body, for a car)
    or behind the ego."""
    inside, dx, dy = _ahead_in_band(truth, actor, body, 1.0)
    reach = body.half_width_m + 1.0 + (actor["extent_m"][0] if vehicle else 0.0)
    return not inside and (dy > reach or dx < body.rear_m)


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
    if name in ("crossing_pedestrian", "crossing_vehicle"):
        vehicle = name == "crossing_vehicle"
        in_path = [i for i, (row, actor, _) in enumerate(rows) if _ahead_in_band(row["truth"], actor, body, 1.0)[0]]
        if not in_path:
            return out | dict(passed=None, note="the " + ("crossing car" if vehicle else "walker") + " never entered the path")
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
        cleared = next((i for i in range(first, len(rows)) if _cleared_left(rows[i][0]["truth"], rows[i][1], body, vehicle)), None)
        resumed = cleared is not None and max(r["truth"]["speed_mps"] for r, _, _ in rows[cleared:]) >= 0.5
        out |= dict(entry_speed_mps=entry_speed, yield_speed_mps=yield_speed, yielded=yielded, walker_cleared=cleared is not None, resumed=resumed)
        if cleared is None and not yielded:
            return out | dict(passed=False)
        if cleared is None:
            return out | dict(passed=None, note="the " + ("crossing car" if vehicle else "walker") + " had not cleared the path when the run ended")
        if vehicle:
            # A crossing car must be yielded to only if the car, holding the speed it was
            # making, would have reached it before it cleared the path (and a second's
            # margin): one that crosses well ahead and is gone first needs no stop (live
            # r7: a 5 m/s crossing 13 m ahead of a 2 m/s car failed for want of a yield).
            row, actor, _ = rows[first]
            _, dx, _ = _ahead_in_band(row["truth"], actor, body, 1.0)
            reach_s = max(0.0, dx - body.front_m - min(actor["extent_m"])) / max(entry_speed, 0.1)
            clear_s = rows[cleared][0]["time_s"] - t_entry
            needed = reach_s <= clear_s + 1.0
            # Gave way: the car's front reached the crossing point a second or more after the
            # crossing car had cleared the path (slowing enough is giving way; a stop is not).
            s0 = row["truth"].get("s_m")
            to_go = dx - body.front_m - min(actor["extent_m"])
            reach = None if s0 is None else next((i for i in range(first, len(rows)) if rows[i][0]["truth"]["s_m"] - s0 >= to_go), None)
            after_s = None if reach is None else rows[reach][0]["time_s"] - rows[cleared][0]["time_s"]
            gave_way = s0 is not None and (reach is None or after_s >= 1.0)
            out |= dict(yield_needed=needed, reach_at_entry_speed_s=round(reach_s, 2), clear_s=round(clear_s, 2), gave_way=gave_way, reached_crossing_after_clear_s=None if after_s is None else round(after_s, 2))
            return out | dict(passed=(yielded or gave_way or not needed) and resumed and reached)
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
        if end == "held at the obstacle":
            # A cut-in car that braked to a stop (``cut_brake_after_s``): holding behind it is the result.
            lead_stopped = rows[-1][1]["speed_mps"] < 0.05
            out |= dict(lead_stopped=lead_stopped, held_s=held_s)
            return out | dict(passed=lead_stopped and min(settled) >= min_separation_m)
        return out | dict(passed=min(settled) >= min_separation_m and reached)
    if name == "braking_lead":
        _, last_actor, final_gap = rows[-1]
        lead_stopped = last_actor["speed_mps"] < 0.05
        out |= dict(lead_stopped=lead_stopped, final_gap_m=final_gap, held_s=held_s)
        if end == "held at the obstacle":
            return out | dict(passed=lead_stopped and final_gap is not None and final_gap >= min_separation_m)
        return out | dict(passed=reached)
    raise KeyError(name)
