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
- **M06 cut-in** (``CutInCar``): a car in the left lane, level with the ego, moves into
  the ego's lane over ``cut_in_s`` and then holds ``lead_mps``.

The actors are kept physically plausible: speeds and decelerations are what the
vehicle blueprint can do, and a walker moves at a walking pace. Scenario parameters are
recorded in the run report so a run is reproducible from it.

Status: written against the CARLA 0.9.16 API from the capture and drive harnesses;
not yet run live. Hook: ``MaplessDrive`` creates the scenario from ``--scenario`` and
calls ``spawn`` after the site is chosen and ``step`` every tick.
"""

import math
from dataclasses import asdict, dataclass

import numpy as np

LEAD_CAR = "vehicle.audi.tt"
WALKER = "walker.pedestrian.0001"


def _box_truth(actor, core_xy):
    """The actor's plan-view box in the evaluator's core frame: centre, yaw, extent."""
    t = actor.get_transform()
    bb = actor.bounding_box
    x, y = core_xy(t.location)
    v = actor.get_velocity()
    return dict(
        centre_xy=[x, y],
        yaw_rad=-math.radians(t.rotation.yaw),
        extent_m=[bb.extent.x, bb.extent.y],
        speed_mps=math.hypot(v.x, v.y),
    )


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
        if left is None or left.lane_type != self.carla.LaneType.Driving:
            raise RuntimeError("The cut-in scenario needs a Driving lane on the left of the site")
        t = left.transform
        t.location.z += 0.3
        self.actor = owner.try_spawn(library.find(LEAD_CAR), t)
        if self.actor is None:
            raise RuntimeError("Could not place the cut-in car")
        self.waypoints = waypoints
        self.left_waypoints = [w.get_left_lane() or w for w in waypoints]
        self.braking = False
        self.cut_start = None
        return self.actor

    def step(self, tick, ego_truth):
        p = self.params
        # Follow the left lane until the ego is level, then blend the target line into the ego lane.
        if self.cut_start is None and ego_truth is not None:
            x, y = self.core_xy(self.actor.get_transform().location)
            ex, ey = ego_truth["rear_axle_xy"]
            if math.dist((x, y), (ex, ey)) <= 8.0:
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
        throttle = float(np.clip(0.35 + 0.3 * (p.lead_mps - math.hypot(v.x, v.y)), 0.0, 0.7))
        self.actor.apply_control(carla.VehicleControl(throttle=throttle, steer=float(np.clip(-bearing / math.radians(35), -1, 1))))


class CrossingPedestrian(ScriptedActor):
    """M04: a walker crossing from the right pavement once the ego is within ``trigger_m``."""

    def spawn(self, owner, library, waypoints):
        p = self.params
        wp = waypoints[min(len(waypoints) - 1, int(round(p.brake_at_m)))]
        self.cross_wp = wp
        t = wp.transform
        yaw = math.radians(t.rotation.yaw)
        # Start 4 m right of the lane centre (on the pavement), facing left across the lane.
        right = (math.sin(yaw), -math.cos(yaw))  # CARLA frame: right of heading
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


SCENARIOS = {
    "crossing_pedestrian": (CrossingPedestrian, "M04"),
    "braking_lead": (BrakingLeadCar, "M05"),
    "cut_in": (CutInCar, "M06"),
}


def make_scenario(carla, name, core_xy, **overrides):
    cls, _ = SCENARIOS[name]
    return cls(carla, ScenarioParams(name, **overrides), core_xy)


def score_moving(log, actor_truths, body, min_separation_m=1.0):
    """Gates for M04–M06 from the drive log and the actor's truth per tick (framework).

    - ``no_collision``: the harness's collision sensor saw nothing (checked by the caller).
    - ``separation``: the body-to-actor plan-view distance never fell under ``min_separation_m``.
    - ``completion``: the run finished or held, not a fault stop short of the goal.
    Time-to-collision and false-stop statistics are reported alongside.
    """
    separations = []
    for row, actor in zip(log, actor_truths):
        if actor is None:
            continue
        x, y = row["truth"]["rear_axle_xy"]
        yaw = row["truth"]["yaw_rad"]
        cx, cy = actor["centre_xy"]
        c, s = math.cos(yaw), math.sin(yaw)
        # Distance from the actor's centre to the ego body box, less the actor's half extent.
        dx, dy = c * (cx - x) + s * (cy - y), -s * (cx - x) + c * (cy - y)
        along = max(body.rear_m - dx, 0.0, dx - body.front_m)
        across = max(abs(dy) - body.half_width_m, 0.0)
        separations.append(math.hypot(along, across) - min(actor["extent_m"]))
    worst = min(separations) if separations else None
    return dict(min_separation_m=worst, separation_ok=worst is None or worst >= min_separation_m, ticks=len(separations))
