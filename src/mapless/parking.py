"""Parallel parking into a gap on the right (the showcase's parking clip, 2 Nov).

A deterministic manoeuvre beside the driver, from the car's own sensors only:

1. **Search.** The car creeps along a straight at ``search_mps`` holding its heading,
   and the LiDAR's obstacle points to its right (``band_m`` beyond its side, between
   ``height_m``) fill a profile along its track in the dead-reckoned frame. A gap is a
   run of free profile between two occupied runs, at least ``car length + extra_m`` long,
   with the parked cars' near faces giving the line to park on.
2. **Stop** past the gap, at the place the two-arc manoeuvre needs (below), or beyond it.
3. **Reverse** straight to the arc start, then on two arcs of radius
   ``wheelbase / tan(lock_gain x steer_rad)``: full right until the heading has turned by
   theta, full left until it is back to the start heading. Two equal arcs shift the rear
   axle ``2R(1 - cos theta)`` sideways over ``2R sin theta``: theta is chosen for the shift
   to the parking line. The wheels are turned only at a standstill. The radius the car
   actually turns on is measured on the first arc (distance over heading turned) and theta
   re-solved from where the car is, for the arcs to end on the parking line.
4. **Centre** between the two cars and stop. If the rear runs out of room before the second
   arc has straightened the car, it finishes forwards with the wheels the other way.

Odometry is dead reckoning from the wheel speed (its sign from the gear selected) and the
gyro's yaw rate, the heading pulled towards the compass's (against its mean at the
start) as the car moves and re-referenced to it at every standstill, where the gyro's
bias is refined too: a 1 s bias estimate leaves about 0.001 rad/s, 4 deg over the minute
the manoeuvre takes (park_t5: 2 deg), and a heading off by that bends the dead-reckoned
frame the gap is measured in. A point of an obstacle
within ``clear_m`` of the body in the direction of travel stops the car where it is
(``blocked``), whatever the phase.

``ParkCommand`` carries the gear (``reverse``); adapters that drive forward only refuse it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from mapless.geometry import wrap, yaw_from_compass


@dataclass(frozen=True)
class ParkingConfig:
    search_mps: float = 1.5
    reverse_mps: float = 0.7
    centre_mps: float = 0.4
    steer_rad: float = math.radians(35.0)  # the arcs' steering, half the Tesla's lock (the drive's controller cap is 20 deg)
    # The share of steer_rad the car's turning achieves: the arcs are planned on
    # wheelbase / tan(lock_gain x steer_rad). 1: the bicycle model (the synthetic road);
    # the CARLA drive passes the Tesla's (park_t5: 5.14 m at 35 deg, 0.85).
    lock_gain: float = 1.0
    radius_after_deg: float = 10.0  # the first arc's radius measured once it has turned this far ...
    radius_min_mps: float = 0.15  # ... over its ticks at least this fast (a crawl reads as a standstill)
    radius_bounds: tuple = (0.7, 1.5)  # ... and used if within these shares of the planned one
    steer_settle_s: float = 0.6  # wheels turned at a standstill, then this long before moving
    extra_m: float = 2.5  # a gap must be the car's length plus this
    bin_m: float = 0.25
    band_m: tuple = (0.2, 3.5)  # obstacle points this far beyond the car's right side
    height_m: tuple = (0.3, 1.8)  # ... between these heights above the ground
    min_run_m: float = 1.5  # an occupied run this long is a parked car ...
    max_run_m: float = 11.0  # ... up to two nose to tail; longer is a wall or a hedge
    ahead_m: float = 2.0  # a gap counts only from this far ahead of the search's start
    pass_m: float = 1.0  # stop this much beyond the arc start
    clear_m: float = 0.35  # an obstacle point this close to the body ahead of its travel stops it
    centre_tol_m: float = 0.15
    heading_tol_deg: float = 4.0  # arc 2 stopped short of the rear car with more than this: finish forwards
    heading_kp: float = 1.5
    speed_kp: float = 0.6
    calibrate_s: float = 2.0  # standing at the start: the gyro's bias is its mean yaw rate, the compass's mean the reference heading
    standstill_settle_s: float = 0.15  # at a standstill, compass and gyro readings count from this long after it began
    compass_gain: float = 0.02  # per tick, moving: the compass's pull on the gyro's heading (2.5 s; its 0.02 rad noise to ~0.1 deg)
    standstill_mps: float = 0.08  # wheel speeds below this are a standstill (the encoder's noise is 0.05 m/s)
    rear_margin_m: float = 0.8  # the arcs end with the rear this far from the rear car
    sweep_margin_m: float = 0.4  # the planned path's body keeps this clear of both cars
    arc_slow_gain: float = 1.5  # in an arc, the speed is at most 0.15 + this x the heading still to turn (rad): the stop lands on it


@dataclass(frozen=True)
class ParkCommand:
    steering_angle_rad: float
    throttle: float
    brake: float
    reverse: bool = False


@dataclass
class Slot:
    rear_end_m: float  # the rear car's front, along the track (dead-reckoned x)
    front_end_m: float  # the front car's rear
    face_y_m: float  # the parked cars' near faces (dead-reckoned y, negative: right)

    @property
    def length_m(self):
        return self.front_end_m - self.rear_end_m


@dataclass
class ParkingState:
    phase: str = "search"
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    speed: float = 0.0
    slot: Slot | None = None
    theta: float | None = None
    arc_start_x: float | None = None
    target_x: float | None = None
    target_y: float | None = None
    centre_x: float | None = None
    phase_t: float = 0.0
    log: list = field(default_factory=list)


class ParallelParker:
    """The manoeuvre, one ``step`` per control tick: a ``SensorFrame`` in, a ``ParkCommand``
    out. ``body`` is the car's ``VehicleBody`` (rear-axle frame), ``wheelbase_m`` its wheelbase."""

    def __init__(self, body, wheelbase_m, config=None, dt_s=0.05):
        self.body, self.wheelbase_m, self.dt = body, float(wheelbase_m), float(dt_s)
        self.config = config or ParkingConfig()
        self.state = ParkingState()
        self.radius_m = self.planned_radius_m = self.wheelbase_m / math.tan(self.config.lock_gain * self.config.steer_rad)
        self.measured_radius_m = None
        self._arc_turn, self._arc_dist = 0.0, 0.0  # the first arc's heading turned and distance, over ticks at a measurable speed
        self._reverse = False
        self._bins = {}  # bin index -> nearest face y seen there
        self._seen = (math.inf, -math.inf)  # the profile's covered range
        self._bias = 0.0
        self._last_gaps = None
        self._bias_sum, self._bias_n = 0.0, 0
        self._compass0 = None  # the compass's mean heading at the start (vehicle-frame yaw, rad)
        self._compass = []  # (sin, cos) of the compass headings, this standstill or the start's
        self.state.phase = "calibrate"

    # -- odometry ----------------------------------------------------------------------
    def _integrate(self, frame):
        st = self.state
        speed = abs(float(frame.speed_mps or 0.0))
        speed = 0.0 if speed < self.config.standstill_mps else speed * (-1.0 if self._reverse else 1.0)  # the encoder's noise at a standstill is not motion
        rate = 0.0 if frame.imu is None else float(frame.imu.yaw_rate_rps) - self._bias
        heading = st.yaw + 0.5 * rate * self.dt
        st.x += speed * self.dt * math.cos(heading)
        st.y += speed * self.dt * math.sin(heading)
        st.yaw += rate * self.dt
        if frame.imu is not None and self._compass0 is not None:
            # The compass pulls the gyro's heading back (a complementary filter): a residual
            # bias turned the car 2.3 deg over the 30 s pass (synthetic seed 5310), 0.9 m off
            # its dead-reckoned line, and the arcs, solved in that frame, ended 0.57 m out.
            error = wrap(yaw_from_compass(frame.imu.compass_rad) - self._compass0 - st.yaw)
            st.yaw += self.config.compass_gain * error
        st.speed = speed

    def _standstill(self, frame):
        """At a standstill (the wheels turned, nothing moving): the gyro's readings refine its
        bias, the compass's mean re-references the heading (park_t5: the body does not turn
        while the wheels do)."""
        if frame.imu is None or self.state.phase_t < self.config.standstill_settle_s:
            return
        self._bias_sum += float(frame.imu.yaw_rate_rps)
        self._bias_n += 1
        self._bias = self._bias_sum / self._bias_n
        heading = yaw_from_compass(frame.imu.compass_rad)
        self._compass.append((math.sin(heading), math.cos(heading)))
        if self._compass0 is not None:
            s, c = np.mean(self._compass, axis=0)
            self.state.yaw = wrap(math.atan2(s, c) - self._compass0)

    def _points(self, frame):
        """Obstacle points in the dead-reckoned frame, and in the body frame (N x 2 each)."""
        if not frame.lidars:
            return np.zeros((0, 2)), np.zeros((0, 2))
        ego = np.asarray(frame.lidars[0].points, dtype=float)[:, :3]  # already in the vehicle frame (ground at z 0)
        lo, hi = self.config.height_m
        ego = ego[(ego[:, 2] >= lo) & (ego[:, 2] <= hi)]
        st = self.state
        c, s = math.cos(st.yaw), math.sin(st.yaw)
        world = np.column_stack((st.x + c * ego[:, 0] - s * ego[:, 1], st.y + s * ego[:, 0] + c * ego[:, 1]))
        return world, ego[:, :2]

    # -- the gap -----------------------------------------------------------------------
    def _profile(self, world, body_xy):
        c, b = self.config, self.body
        side = -b.half_width_m
        right = (body_xy[:, 1] <= side - c.band_m[0]) & (body_xy[:, 1] >= side - c.band_m[1])
        for (x, y) in world[right]:
            k = int(math.floor(x / c.bin_m))
            self._bins[k] = max(self._bins.get(k, -math.inf), float(y))  # the nearest face (least negative y)
        # The profile covers what the band has passed beside: from the rear of the band's
        # view to its front, relative to the car (the LiDAR sees ahead and behind).
        reach = c.band_m[1] + 6.0
        lo, hi = self.state.x - reach, self.state.x + reach
        self._seen = (min(self._seen[0], lo), max(self._seen[1], hi))

    def find_slot(self):
        """The first gap ahead of where the search began (``ahead_m`` past the car's rear
        axle there), between two runs the size of parked cars, long enough. A gap behind the
        start is not one the car was sent to (park_t3, PC 1: Town10HD's own roadside objects
        behind the start made one, and the car reversed towards it at once)."""
        c, b = self.config, self.body
        if not self._bins:
            return None
        k0, k1 = min(self._bins), max(self._bins)
        occupied = [k for k in range(k0, k1 + 1) if k in self._bins]
        runs, start, prev = [], occupied[0], occupied[0]
        for k in occupied[1:]:
            if k > prev + 2:  # a free stretch of more than two bins ends the run
                runs.append((start, prev))
                start = k
            prev = k
        runs.append((start, prev))
        cars = [(a, z) for a, z in runs if c.min_run_m <= (z - a + 1) * c.bin_m <= c.max_run_m]
        need = (b.front_m - b.rear_m) + c.extra_m
        for (a0, z0), (a1, z1) in zip(cars, cars[1:]):
            rear_end, front_end = (z0 + 1) * c.bin_m, a1 * c.bin_m
            if rear_end >= c.ahead_m and front_end - rear_end >= need:
                faces = [self._bins[k] for k in list(range(a0, z0 + 1)) + list(range(a1, z1 + 1)) if k in self._bins]
                return Slot(rear_end, front_end, float(np.median(faces)))
        return None

    def _plan(self, slot):
        """theta, the arc start, the arcs' end and the centred pose for this slot from the
        car's lateral place now, or None when the swept body would come within
        ``sweep_margin_m`` of either car. The arcs end at the back of the gap: the rear
        ``rear_margin_m`` from the rear car at most, less when the swept body's clearance to
        the front car is less, the two balanced (starting the arcs further forward swings
        the rear corner into the front car's side, simulated 2 Nov; with the Tesla's 5.1 m
        arcs the 9 m gap, seen as 8.25 m, kept 0.36 m from the front car at a 0.8 m rear
        margin, 0.58 m from both at 0.58). ``centre`` then drives forward."""
        b, c, st = self.body, self.config, self.state
        target_y = slot.face_y_m - b.half_width_m - 0.05  # the car's near side just inside the parked cars' line
        shift = st.y - target_y
        cos_theta = 1.0 - shift / (2.0 * self.radius_m)
        if not -1.0 < cos_theta < 1.0:
            return None
        theta = math.acos(cos_theta)
        back_x = slot.rear_end_m - b.rear_m  # the arcs' end with the rear at the rear car
        sweep = 2.0 * self.radius_m * math.sin(theta)
        front, rear = self._clearances(theta, back_x + sweep, st.y, slot)
        # The path moves along x with the margin: front - m to the front car, rear + m behind.
        margin = min(c.rear_margin_m, 0.5 * (front - rear)) if math.isfinite(front) and math.isfinite(rear) else c.rear_margin_m
        if min(front - margin, rear + margin) < c.sweep_margin_m:
            return None
        target_x = back_x + margin
        arc_start_x = target_x + sweep
        centre_x = max(target_x, 0.5 * (slot.rear_end_m + slot.front_end_m) - 0.5 * (b.front_m + b.rear_m))
        return theta, arc_start_x, target_x, target_y, centre_x

    def _path_clear(self, theta, x0, y0, slot):
        """The planned arcs' body corners keep ``sweep_margin_m`` from both cars."""
        front, rear = self._clearances(theta, x0, y0, slot)
        return min(front, rear) >= self.config.sweep_margin_m

    def _clearances(self, theta, x0, y0, slot, samples=40):
        """The planned arcs' body corners against the two cars: the least distance along the
        lane to the front car's end and past the rear car's, of the corners within
        ``sweep_margin_m`` of the parked cars' line or inside it (inf: none there)."""
        b, c, r = self.body, self.config, self.radius_m
        u = np.array([b.front_m, b.front_m, b.rear_m, b.rear_m])
        v = np.array([b.half_width_m, -b.half_width_m, -b.half_width_m, b.half_width_m])
        # Arc 1 about the centre to the right of the start, arc 2 about one to the left of its end.
        a1 = theta * np.arange(samples + 1) / samples
        x1, y1 = x0 - r * np.sin(a1), y0 - r * (1.0 - np.cos(a1))
        a2 = theta * (1.0 - np.arange(samples + 1) / samples)
        x2, y2 = x1[-1] - r * (math.sin(theta) - np.sin(a2)), y1[-1] - r * (np.cos(a2) - math.cos(theta))
        x, y, h = np.concatenate((x1, x2))[:, None], np.concatenate((y1, y2))[:, None], np.concatenate((a1, a2))[:, None]
        px = x + np.cos(h) * u - np.sin(h) * v
        py = y + np.sin(h) * u + np.cos(h) * v
        inside = py < slot.face_y_m + c.sweep_margin_m
        if not inside.any():
            return math.inf, math.inf
        return float((slot.front_end_m - px[inside]).min()), float((px[inside] - slot.rear_end_m).min())

    def _resolve_theta(self):
        """theta for the arcs from where the car is on the first arc (heading yaw, radius
        R): the second arc then ends, at heading 0, with the rear axle at
        y - R cos(yaw) + 2R cos(theta) - R; theta puts that on target_y."""
        st, r = self.state, self.radius_m
        cos_theta = (st.target_y - st.y + r * math.cos(st.yaw) + r) / (2.0 * r)
        st.theta = math.acos(float(np.clip(cos_theta, -1.0, 1.0)))

    # -- control -----------------------------------------------------------------------
    def _blocked(self, body_xy, reverse):
        """An obstacle point within clear_m of the body, on the side it is moving towards."""
        b, c = self.body, self.config
        if len(body_xy) == 0:
            return False
        within = np.abs(body_xy[:, 1]) <= b.half_width_m + c.clear_m
        if reverse:
            near = within & (body_xy[:, 0] <= b.rear_m) & (body_xy[:, 0] >= b.rear_m - c.clear_m)
        else:
            near = within & (body_xy[:, 0] >= b.front_m) & (body_xy[:, 0] <= b.front_m + c.clear_m)
        return bool(near.any())

    def _end_gaps(self, body_xy, reach_m=4.0):
        """(front gap, rear gap): the nearest obstacle point ahead of the front and behind
        the rear within the body's width, each None beyond ``reach_m`` or unseen."""
        b = self.body
        if len(body_xy) == 0:
            return None, None
        within = np.abs(body_xy[:, 1]) <= b.half_width_m
        ahead = body_xy[within & (body_xy[:, 0] > b.front_m), 0] - b.front_m
        behind = b.rear_m - body_xy[within & (body_xy[:, 0] < b.rear_m), 0]
        front = float(ahead.min()) if len(ahead) and ahead.min() <= reach_m else None
        rear = float(behind.min()) if len(behind) and behind.min() <= reach_m else None
        return front, rear

    def _drive(self, target_mps, steer, reverse):
        """Speed control towards |target| in the gear asked for."""
        c, st = self.config, self.state
        if reverse != self._reverse:
            if abs(st.speed) > 0.05:
                return ParkCommand(steer, 0.0, 1.0, self._reverse)
            self._reverse = reverse
        # A gentle loop with a coasting band: a 0.25 throttle floor and braking on any
        # excess made the live car surge and stop in reverse (park_t3: 0.0-0.74 m/s).
        error = abs(target_mps) - abs(st.speed)
        if error >= 0:
            return ParkCommand(steer, float(min(0.45, 0.12 + c.speed_kp * error)), 0.0, reverse)
        if error > -0.15:
            return ParkCommand(steer, 0.0, 0.0, reverse)  # coast
        return ParkCommand(steer, 0.0, float(min(0.5, 0.5 * (-error - 0.15) + 0.05)), reverse)

    def _stop(self, steer=0.0):
        return ParkCommand(steer, 0.0, 1.0, self._reverse)

    def _enter(self, phase):
        self.state.phase, self.state.phase_t = phase, 0.0
        self._compass = []

    def step(self, frame):
        c, st = self.config, self.state
        yaw_before = st.yaw
        self._integrate(frame)
        st.phase_t += self.dt
        world, body_xy = self._points(frame)
        standing = abs(st.speed) < 0.03
        if st.phase == "calibrate":
            if frame.imu is not None:
                self._bias_sum += float(frame.imu.yaw_rate_rps)
                self._bias_n += 1
                heading = yaw_from_compass(frame.imu.compass_rad)
                self._compass.append((math.sin(heading), math.cos(heading)))
            if st.phase_t >= c.calibrate_s:
                self._bias = self._bias_sum / max(1, self._bias_n)
                if self._compass:
                    s, co = np.mean(self._compass, axis=0)
                    self._compass0 = math.atan2(s, co)
                st.x = st.y = st.yaw = 0.0
                self._enter("search")
            return self._stop()
        if st.phase in ("search", "pass"):
            # The profile and the plan are refined until the car stops: the far end of the
            # gap is seen best from beside it.
            self._profile(world, body_xy)
            slot = self.find_slot()
            plan = None if slot is None else self._plan(slot)
            # The same gap, seen better (its ends move as more of each car comes into view).
            same = st.slot is not None and slot is not None and abs(slot.rear_end_m - st.slot.rear_end_m) <= 3.0 and abs(slot.front_end_m - st.slot.front_end_m) <= 3.0
            if plan is not None and (st.phase == "search" or same):
                st.slot = slot
                st.theta, st.arc_start_x, st.target_x, st.target_y, st.centre_x = plan
                if st.phase == "search":
                    self._enter("pass")
            elif st.phase == "pass" and same and plan is None:
                # The gap, seen better, is too short for the planned sweep (its far end is
                # seen best from beside it): not this one; search on.
                st.slot = None
                self._enter("search")
        if st.phase == "search":
            steer = float(np.clip(-c.heading_kp * st.yaw, -0.2, 0.2))
            return self._drive(c.search_mps, steer, False)
        if st.phase == "pass":
            if st.x >= st.arc_start_x + c.pass_m:
                if standing:
                    self._enter("back")
                return self._stop()
            if self._blocked(body_xy, False):
                return self._stop()
            steer = float(np.clip(-c.heading_kp * st.yaw, -0.2, 0.2))
            return self._drive(c.search_mps, steer, False)
        if st.phase == "back":  # straight back to the arc start
            if st.x <= st.arc_start_x:
                if standing:
                    self._enter("steer_right")
                return self._stop()
            if self._blocked(body_xy, True):
                self._enter("blocked")
                return self._stop()
            return self._drive(c.reverse_mps, 0.0, True)
        if st.phase in ("steer_right", "steer_left"):
            steer = -c.steer_rad if st.phase == "steer_right" else c.steer_rad
            self._standstill(frame)
            if st.phase_t >= c.steer_settle_s:
                self._enter("arc1" if st.phase == "steer_right" else "arc2")
                if st.phase == "arc1":
                    self._arc_turn, self._arc_dist = 0.0, 0.0
                    self._resolve_theta()  # from the heading the compass gave at the standstill
            return self._stop(steer)
        if st.phase == "arc1":
            if abs(st.speed) >= c.radius_min_mps:
                self._arc_dist += abs(st.speed) * self.dt
                self._arc_turn += st.yaw - yaw_before
            if self._arc_turn >= math.radians(c.radius_after_deg):
                # The radius the car turns on (the Tesla's wheels reach less than they are
                # asked: park_t5, 5.14 m where 4.29 was planned, the arcs ended 0.55 m deep).
                radius = self._arc_dist / self._arc_turn
                lo, hi = c.radius_bounds
                if lo * self.planned_radius_m <= radius <= hi * self.planned_radius_m:
                    self.radius_m = self.measured_radius_m = radius
                    self._resolve_theta()
            if st.yaw >= st.theta:
                if standing:
                    self._enter("steer_left")
                return self._stop(-c.steer_rad)
            if self._blocked(body_xy, True):
                self._enter("blocked")
                return self._stop(-c.steer_rad)
            return self._drive(min(c.reverse_mps, 0.15 + c.arc_slow_gain * max(0.0, st.theta - st.yaw)), -c.steer_rad, True)
        if st.phase == "arc2":
            if st.yaw <= 0.0:
                if standing:
                    self._enter("straighten")
                return self._stop(c.steer_rad)
            if self._blocked(body_xy, True):
                # Out of room behind with the nose still out (park_t4: 23 deg, 0.6 m from the
                # rear car): finish forwards, wheels the other way, as a driver would.
                if standing:
                    self._enter("steer_fix" if st.yaw > math.radians(c.heading_tol_deg) else "straighten")
                return self._stop(c.steer_rad)
            return self._drive(min(c.reverse_mps, 0.15 + c.arc_slow_gain * max(0.0, st.yaw)), c.steer_rad, True)
        if st.phase == "steer_fix":
            self._standstill(frame)
            if st.phase_t >= c.steer_settle_s:
                self._enter("fix")
            return self._stop(-c.steer_rad)
        if st.phase == "fix":  # forwards, wheels right: the heading comes back as the nose swings in
            if st.yaw <= math.radians(0.5) or self._blocked(body_xy, False):
                if standing:
                    self._enter("straighten")
                return self._stop(-c.steer_rad)
            return self._drive(c.centre_mps, -c.steer_rad, False)
        if st.phase == "straighten":
            self._standstill(frame)
            if st.phase_t >= c.steer_settle_s:
                self._enter("centre")
            return self._stop(0.0)
        if st.phase == "centre":
            # Centred on the two cars as the LiDAR sees them now (front and rear gaps along the
            # body), the dead-reckoned target when either is out of view.
            gaps = self._last_gaps = self._end_gaps(body_xy)
            error = st.centre_x - st.x if None in gaps else 0.5 * (gaps[0] - gaps[1])
            if abs(error) <= c.centre_tol_m:
                if standing:
                    self._enter("parked")
                return self._stop()
            reverse = error < 0
            if self._blocked(body_xy, reverse):
                self._enter("parked")
                return self._stop()
            # Steered on the heading as it goes (4 Nov, park_kerb6: the arcs ended 3.6 deg
            # out by the compass at the standstill, and 3.2 m forwards with the wheels straight
            # carried the car 0.2 m back out over the shoulder's line). Reversing, the same
            # correction takes the wheels the other way.
            steer = float(np.clip(-c.heading_kp * st.yaw, -0.2, 0.2))
            return self._drive(min(c.centre_mps, 0.3 + abs(error)), -steer if reverse else steer, reverse)
        return self._stop()  # parked, blocked: hold the brake

    @property
    def done(self):
        return self.state.phase in ("parked", "blocked")

    def record(self):
        st = self.state
        return dict(
            phase=st.phase, radius_m=round(self.radius_m, 2), planned_radius_m=round(self.planned_radius_m, 2),
            measured_radius_m=None if self.measured_radius_m is None else round(self.measured_radius_m, 2), theta_deg=None if st.theta is None else round(math.degrees(st.theta), 1),
            slot=None if st.slot is None else dict(rear_end_m=round(st.slot.rear_end_m, 2), front_end_m=round(st.slot.front_end_m, 2), length_m=round(st.slot.length_m, 2), face_y_m=round(st.slot.face_y_m, 2)),
            target_xy_m=None if st.target_x is None else [round(st.target_x, 2), round(st.target_y, 2)],
            centre_x_m=None if st.centre_x is None else round(st.centre_x, 2), gyro_bias_rps=round(self._bias, 5),
            pose=[round(st.x, 2), round(st.y, 2), round(math.degrees(st.yaw), 1)],
            end_gaps_m=None if self._last_gaps is None else [None if g is None else round(g, 2) for g in self._last_gaps],
        )
