"""The mapless car for G1b: sensors and a sat-nav route in, one control command out per tick.

Per tick:

1. Sensor health, then ego motion (odometry frame, and the route pose once start-up
   averaging has finished).
2. The LiDAR sweep is classified and added to the terrain grid, from the first tick on.
3. Until the lane centre's distance from the kerb is known, it is measured at a
   standstill (``CorridorEstimator.calibrate``), alongside the GNSS averaging.
4. The corridor and lane centre are estimated; the path is observed in its tube: the
   nearest obstacle, and how far the road is seen clear.
5. The planner builds a trajectory (lane follow, obstacle stop or goal stop) and the
   ``LocalTrajectoryController`` tracks it in the odometry frame.

**Start-up assumption.** The car starts stopped, centred and aligned in a lane with a
kerb on its right; the calibration measures that lane. It is checked as far as it can be
(a clean, stable kerb fit within 10 degrees of the heading) and fails otherwise. It is
assumed again, at a standstill, to re-acquire a lost corridor.

**Modes:**
- STARTUP: holding the brake while GNSS and compass are averaged.
- CALIBRATING: holding the brake while the kerb distance is measured.
- DRIVING / STOPPING: following the plan (STOPPING when its target speed is zero).
- FINISHED: the route goal was reached (latched).
- FAULT: stale LiDAR, IMU, speedometer or GNSS, an uncertain route position, a failed
  calibration (latched) or no corridor. The car brakes along its last trajectory while
  odometry is trusted, otherwise with the steering held, and recovers by itself once the
  cause is gone. It does not move off a standstill on kerb evidence older than
  ``start_evidence_max_age_s``. After an IMU or speedometer outage the odometry frame is
  broken (nothing was integrated), so the car stops and rebuilds grid, corridor, obstacle
  hold and trajectory together from a standstill before driving again.

Nothing here reads the simulator; the only inputs are ``SensorFrame``s, the sensor suite,
the route and the car's own body and controller geometry.
"""

import math
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from mapless.corridor import CorridorConfig, CorridorEstimator
from mapless.crossing import CrossingConfig, StraightCrossing
from mapless.ego_motion import EgoMotion, EgoMotionConfig, Pose2D
from mapless.frame import SensorFrame, SensorState
from mapless.geometry import wrap as wrap_angle
from mapless.geometry import yaw_from_compass
from mapless.health import HealthTracker
from mapless.junction import JunctionConfig, JunctionTurn, kerb_opening, oncoming_gate
from mapless.obstacles import ObstacleConfig, observe_path
from mapless.planner import LanePlanner, PlannerConfig
from mapless.satnav import Command, RouteProgress, SatNavRoute
from mapless.sensor_suite import SensorSuite
from mapless.terrain import TerrainConfig, TerrainGrid, VehicleBody, classify_sweep
from navigation_control.local import LocalControlConfig, LocalStatus, LocalTrajectoryController
from navigation_control.types import ControlCommand, VehicleState


class DriveMode(str, Enum):
    STARTUP = "startup"
    CALIBRATING = "calibrating"
    DRIVING = "driving"
    TURNING = "turning"
    STOPPING = "stopping"
    FINISHED = "finished"
    FAULT = "fault"


@dataclass(frozen=True)
class DriverConfig:
    body: VehicleBody
    control: LocalControlConfig
    ego: EgoMotionConfig = field(default_factory=EgoMotionConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    corridor: CorridorConfig = field(default_factory=CorridorConfig)
    obstacles: ObstacleConfig = field(default_factory=ObstacleConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    junction: JunctionConfig = field(default_factory=JunctionConfig)
    crossing: CrossingConfig = field(default_factory=CrossingConfig)  # STRAIGHT through a junction (``crossing``)
    turn_start_m: float = 25.0  # a commanded turn is looked for when the route's entry point is this close
    turn_route_tolerance_m: float = 8.0  # the entry point must be within this of the car's path
    lidar_max_age_s: float = 0.15
    gnss_max_age_s: float = 3.0
    route_max_sigma_m: float = 4.0  # route position this uncertain: the goal cannot be judged
    start_evidence_max_age_s: float = 0.5  # move off a standstill only on kerb evidence this fresh
    hold_within_m: float = 1.5  # stopped this close to an obstacle stop point: hold the brake
    # A junction hold point only moves when the view does, and at a standstill it does not:
    # the brake is held only at the hold point itself, so a car that stopped a metre short
    # of a verified frontier creeps up to it instead of waiting there for ever (review).
    junction_hold_within_m: float = 0.4
    # Cells between the LiDAR's rings are not re-hit while the car stands, so a standstill's
    # duration is added to the junction's freshness windows. It is kept after the car moves
    # off until it has travelled this far (the rings then sweep those cells again): dropped
    # at the first metre, the windows collapsed and a car that had waited 12 s at the mouth
    # found its road ahead "unseen" and never moved (full review of 73ae759, S7).
    still_credit_m: float = 3.0
    mouth_release_m: float = 0.5  # a standing car's mouth moves on to an opening read this much further ...
    mouth_release_s: float = 1.0  # ... steadily (within mouth_release_m) for this long
    # The path the far points' ground is read along (``classify_sweep``'s path_xy) is the last
    # one planned, while it is no older than this (a fault tick plans none).
    profile_path_max_age_s: float = 0.5

    def __post_init__(self):
        if not isinstance(self.body, VehicleBody) or not isinstance(self.control, LocalControlConfig):
            raise TypeError("DriverConfig needs a VehicleBody and a LocalControlConfig")
        if self.planner.decel_mps2 > self.control.controller.stopping_deceleration_mps2 + 1e-9:
            raise ValueError("The planner may not brake harder than the controller's stopping deceleration")


@dataclass(frozen=True)
class DriverStep:
    command: ControlCommand
    mode: DriveMode
    reason: str
    diagnostics: dict


HOLD = 0.6  # brake while holding at standstill or faulted


def refused_speed_limit(trajectory, state, lookahead_m=2.0):
    """The speed a refused plan allows where the car is: its lowest sample speed from the
    sample nearest the car to ``lookahead_m`` beyond it (a stop the plan holds near the
    car reads 0). The controller drives a kept trajectory no faster (third review, R7)."""
    samples = trajectory.samples
    if not samples:
        return 0.0
    xy = np.array([[smp.x_m, smp.y_m] for smp in samples])
    speeds = np.array([smp.speed_mps for smp in samples])
    arcs = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T)))) if len(xy) > 1 else np.zeros(1)
    nearest = int(np.argmin(np.hypot(xy[:, 0] - state.x_m, xy[:, 1] - state.y_m)))
    window = (arcs >= arcs[nearest]) & (arcs <= arcs[nearest] + lookahead_m)
    return float(speeds[window].min())


def resolve_oncoming_override(override, now_s, max_age_s):
    """The stack's oncoming-traffic verdict as the left-turn gate may use it now (eighth
    review, F8-02). The stack computes it after each step from that tick's tracks, so the
    gate normally reads it one tick old. Older (a tick or more without perception, which
    leaves it as it was), it is no basis for anything: its excused footprints are dropped,
    so the LiDAR gate counts every occupied cell again, and its "clear" lapses too (review
    of 519b8fe, F-02: a clear read with a radar-only car 50 m out is not clear with the
    tracks gone), so the car waits at the mouth until the stack speaks again. A "not clear"
    stands. Returns the override to use (None: none), with ``stale`` and ``age_s`` when it
    carries a time."""
    if override is None:
        return None
    when = override.get("time_s")
    if when is None:
        return override
    age = float(now_s) - float(when)
    if age <= max_age_s + 1e-9:
        return dict(override, stale=False, age_s=age)
    note = override.get("note") if not override.get("clear", True) else f"the stack's verdict is {age:.2f} s old: waiting for a fresh one"
    return dict(override, excused=(), stale=True, age_s=age, clear=False, note=note)


def _wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _polyline_distance(point, polyline):
    """Distance from ``point`` to the polyline through ``polyline`` (N x 2; a single point
    when N is 1)."""
    p = np.asarray(point, dtype=float)
    pts = np.asarray(polyline, dtype=float).reshape(-1, 2)
    if len(pts) == 1:
        return float(np.hypot(*(p - pts[0])))
    best = math.inf
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        n = float(np.dot(ab, ab))
        t = 0.0 if n == 0 else float(np.clip(np.dot(p - a, ab) / n, 0.0, 1.0))
        best = min(best, float(np.hypot(*(p - (a + t * ab)))))
    return best


def _traced_seed(evidence, y_at_car, step_m=1.0):
    """A seed for the kerb corridor through pooled lane-centre evidence (vehicle frame,
    N x 2) from the car: a quadratic through the car's point and the evidence, sampled
    every step_m from 2 m behind the car to the evidence's far end plus 10 m."""
    x = np.concatenate([[0.0], evidence[:, 0]])
    y = np.concatenate([[y_at_car], evidence[:, 1]])
    w = np.concatenate([[3.0], np.ones(len(evidence))])
    degree = 2 if float(np.ptp(x)) >= 6.0 else 1
    coefficients = np.polyfit(x, y, degree, w=w)
    xs = np.arange(-2.0, float(evidence[:, 0].max()) + 10.0 + 1e-9, step_m)
    return np.column_stack((xs, np.polyval(coefficients, xs)))


class MaplessDriver:
    """One car, one episode. Feed ``step()`` every tick in order."""

    def __init__(self, suite, route, config, dt_s):
        if not isinstance(suite, SensorSuite) or not isinstance(route, SatNavRoute):
            raise TypeError("MaplessDriver(SensorSuite, SatNavRoute, DriverConfig, dt_s)")
        if not isinstance(config, DriverConfig):
            raise TypeError("config must be a DriverConfig")
        lidars = [s.name for s in suite.sensors if s.kind == "lidar"]
        if not lidars:
            raise ValueError("G1b needs a LiDAR in the suite")
        self.config = config
        self.dt_s = dt_s
        self.lidar = lidars[0]
        self.health = HealthTracker(suite, dt_s)
        self.progress = RouteProgress(route, north_sign=suite.gnss.north_sign if suite.gnss is not None else 1)
        self.ego = EgoMotion(self.progress.plane, config.ego)
        self.grid = TerrainGrid(config.terrain)
        self.corridor = CorridorEstimator(config.corridor)
        self.planner = LanePlanner(config.planner, config.body, tube_half_m=config.body.half_width_m + config.obstacles.margin_m)
        self.turn = JunctionTurn(config.junction, config.body, self.corridor)
        self._turning = False
        self.crossing = StraightCrossing(config.crossing)  # a junction crossed straight on
        self.speed_cap_mps = None  # the stack's following/prediction cap for this tick (None: none)
        self.last_motion = self.last_classified = self.last_plan = self.last_corridor = None
        self._profile_path = None  # (time, odometry polyline) of the last path planned along, for the far points' ground (terrain)
        self._turn_done = None  # route index of the junction entry whose turn is complete
        self._mouth_odo = None  # the junction mouth, odometry frame, nearest seen since the approach began
        self._lane_heading_odo = None  # the entry lane's direction, odometry frame, latched as the approach begins
        self._turn_corridor = None  # the last turn path steered along (the stop line if the latched path fails mid-arc)
        self._moving_t = None  # when the car last moved (freshness at a standstill is measured from here)
        self._opening_odo = None  # the kerb opening last seen, odometry frame: kept while the car waits at the mouth
        self._mouth_candidate = None  # (odometry point, first time) of a further opening read at a standstill
        self._turn_command = None  # LEFT or RIGHT while a turn is under way
        self._latest_odo = None  # the furthest a held car may stand and still begin the nominal arc (odometry)
        self._gate_frame = None  # a left turn's entry lane (odometry pose on its centre), latched as the approach begins
        self._zone_entered = False  # the body has entered the opposing lane: clear it, never stop in it for the gate
        self.last_gate = None  # this tick's OncomingGate (for the stack)
        # How far ahead (vehicle x) the route turns off at a junction, while the path planned
        # along is still the kerb corridor (not the turn's): past it the car leaves this road
        # (fifth review, F5-03: a lead on the line continued straight may be on the branch the
        # car does not take). None without a turn ahead, or on the turn's own path.
        self.turn_entry_m = None
        # The stack's verdict on oncoming traffic for the next tick (None: the LiDAR grid
        # alone): dict(clear=bool, excused=[footprints], note=str); see ``oncoming_gate``.
        self.oncoming_override = None
        self.controller = LocalTrajectoryController(config.control)
        self._last_time = None
        self._finished = False
        self._last_steer = 0.0
        self._calibrated = False
        self._rebuild = False  # odometry was lost: grid, corridor, hold and trajectory are stale
        self._rebuild_since = None  # when the odometry was first lost
        self._still_since = None  # when the current standstill began (once the car has moved)
        self._still_base = 0.0  # credit carried into it from a standstill just before
        self._still_credit = 0.0  # time stood still, kept until the car has moved still_credit_m
        self._still_odo = None  # where the last standstill was (odometry)

    def _brake(self, mode, reason, diagnostics):
        """Standstill hold: wheels straight, brake on."""
        return self._emit(ControlCommand(0.0, 0.0, HOLD), mode, reason, diagnostics)

    def _emit(self, command, mode, reason, diagnostics):
        self._last_steer = command.steering_angle_rad
        return DriverStep(command, mode, reason, diagnostics)

    def _fault(self, reason, diagnostics, motion, t, dt):
        """Brake for a fault without leaving the lane.

        Along the last trajectory while odometry is trusted (the controller steers it with a
        zero speed limit). Otherwise the last steering is held: with a dropped IMU the
        odometry heading is stale, but a car in a bend still needs to keep turning. Never
        wheels straight in motion (review of 94b5caa: a fault in a bend ran wide).
        """
        # Stationary is judged from the speedometer's own readings, never from the odometry
        # speed, which is zeroed while odometry is untrusted (review of 697d53c: a fault
        # with a dropped speedometer straightened the wheels while the car was moving).
        moving = not motion.stationary
        steer, how = (self._last_steer if moving else 0.0), "held"
        brake = HOLD
        if motion.odometry_trusted and not self._rebuild and self.controller.trajectory is not None:
            state = VehicleState(motion.odometry.x_m, motion.odometry.y_m, motion.odometry.yaw_rad, motion.speed_mps)
            result = self.controller.update(state, t, dt, speed_limit_mps=0.0)
            steer, how = result.command.steering_angle_rad, "trajectory"
            brake = max(HOLD, result.command.brake)
        diagnostics.update(fault_steering=how)
        return self._emit(ControlCommand(steer, 0.0, brake), DriveMode.FAULT, reason, diagnostics)

    def step(self, frame):
        if not isinstance(frame, SensorFrame):
            raise TypeError("frame must be a SensorFrame")
        c = self.config
        t = frame.sim_time_s
        dt = self.dt_s if self._last_time is None else max(1e-3, t - self._last_time)
        self._last_time = t
        # Wall-clock per stage, for the run's timing report (third review: profile before
        # optimising). Laps are kept in the diagnostics, early returns included.
        timing, clock = {}, [time.perf_counter()]

        def lap(name):
            now = time.perf_counter()
            timing[name] = round((now - clock[0]) * 1000, 2)
            clock[0] = now

        self.health.update(frame)
        motion = self.ego.update(frame)
        lap("ego")
        if not motion.stationary:
            self._moving_t = t  # before any early return: a fault while moving must not stretch a later standstill's freshness
        classified = None
        if frame.status_of(self.lidar).state is SensorState.OK:
            # The far points' ground is read along the path planned last tick (terrain's
            # module docstring), while it is recent and the odometry frame still the same:
            # the lane's, or a committed turn's. Not a turn still being chosen at the
            # approach, whose path moves with each reading (the synthetic left turns,
            # tuned on the bands' ground there, were held at the mouth).
            path_body = None
            choosing = self._turning and not self.turn.committed
            if self._profile_path is not None and not choosing and motion.odometry_trusted and not self._rebuild and t - self._profile_path[0] <= c.profile_path_max_age_s:
                path_body = motion.odometry.to_body(self._profile_path[1])
            classified = classify_sweep(frame.lidar(self.lidar).xyz, c.body, c.terrain, path_xy=path_body)
        lap("classify")
        # For the stack around the driver (G2/G3): this tick's inputs, read after step().
        self.last_motion, self.last_classified, self.last_plan, self.last_corridor = motion, classified, None, None
        self.turn_entry_m = None
        if classified is not None and motion.odometry_trusted and not self._rebuild:
            self.grid.add(classified, motion.odometry, t, stationary=motion.stationary)  # never placed with a broken frame
        lap("grid")
        if not motion.odometry_trusted:
            if not self._rebuild:
                self._rebuild_since = t
            self._rebuild = True
        d = dict(
            time_s=t,
            odometry=(motion.odometry.x_m, motion.odometry.y_m, motion.odometry.yaw_rad),
            route_pose=None if motion.route is None else (motion.route.x_m, motion.route.y_m, motion.route.yaw_rad),
            route_sigma_m=motion.route_sigma_m,
            speed_mps=motion.speed_mps,
            gyro_bias_rps=motion.gyro_bias_rps,
            imu_age_s=motion.imu_age_s,
            speed_age_s=motion.speed_age_s,
            gnss_age_s=motion.gnss_age_s,
            lidar_points=None if classified is None else int(len(classified.xyz)),
            odometry_trusted=motion.odometry_trusted,
            rebuilding=self._rebuild,
            keep_right_m=self.corridor.keep_right_m,
            keep_right_source=self.corridor.keep_right_source,
            driver_timing_ms=timing,
        )
        if self._finished:
            return self._brake(DriveMode.FINISHED, "goal reached", d)
        calibration = "configured" if self.corridor.keep_right_source == "configured" else "accepted"
        if self.corridor.keep_right_source != "configured" and not self._calibrated:
            calibration = self.corridor.calibrate(classified, motion.odometry, t, motion.stationary)
            self._calibrated = calibration == "accepted"
            d.update(
                calibration=self.corridor.calibration_summary,
                keep_right_m=self.corridor.keep_right_m,
                keep_right_source=self.corridor.keep_right_source,
            )
        if calibration.startswith("failed"):
            return self._fault(f"kerb calibration {calibration}", d, motion, t, dt)
        if not motion.localised:
            return self._brake(DriveMode.STARTUP, "averaging GNSS and compass at standstill", d)
        if calibration == "calibrating":
            return self._brake(DriveMode.CALIBRATING, "measuring the kerb at standstill", d)
        stale = self.health.stale([self.lidar], c.lidar_max_age_s)
        standing = 0.0 if self._moving_t is None or not motion.stationary else max(0.0, t - self._moving_t)
        self.corridor.standing_s = standing
        here = np.array([motion.odometry.x_m, motion.odometry.y_m])
        if standing > 0.0:
            if self._still_since is None:
                self._still_since, self._still_base = t - standing, self._still_credit
            self._still_credit, self._still_odo = self._still_base + (t - self._still_since), here
        else:
            self._still_since = None
            if self._still_odo is not None and float(np.hypot(*(here - self._still_odo))) > c.still_credit_m:
                self._still_credit, self._still_odo = 0.0, None
        corridor = self.corridor.update(self.grid, motion.odometry, t)
        lap("corridor")
        d.update(
            corridor_ok=corridor.ok,
            corridor_reason=corridor.reason,
            corridor_valid_m=corridor.valid_length_m,
            corridor_support_m=corridor.support_m,
            evidence_age_s=corridor.evidence_age_s if math.isfinite(corridor.evidence_age_s) else None,
            lane_offset_m=corridor.lateral_offset_at(0.0) if corridor.ok else None,
            corridor_evidence=int(len(corridor.evidence_body)),
            right_edge_5m=corridor.right_edge_from_centre_m(5.0),
            right_kerb_body=corridor.right_edge_body(5.0),
        )
        guidance = self.progress.update(motion.route.x_m, motion.route.y_m)
        d.update(route_index=guidance.index, goal_distance_m=guidance.goal_distance_m, route_finished=guidance.finished)
        if stale:
            return self._fault(f"LiDAR older than {c.lidar_max_age_s:.2f} s", d, motion, t, dt)
        if not motion.odometry_trusted:
            which = "IMU" if motion.imu_age_s > c.ego.imu_max_age_s else "speedometer"
            age = max(motion.imu_age_s, motion.speed_age_s)
            return self._fault(f"{which} {age:.2f} s old: odometry not trusted", d, motion, t, dt)
        if self._rebuild:
            # The odometry frame skipped the motion during the outage, so everything placed
            # in it (grid, corridor, obstacle hold, trajectory) is off by that much. Brake
            # to a standstill, then rebuild from the start-up assumption: the car stands in
            # its lane (review of 2b7e14d: resetting the corridor alone left the rest).
            if not motion.stationary:
                return self._fault("odometry recovered: stopping to rebuild the local frame", d, motion, t, dt)
            self.grid.reset()
            self.corridor.reset()
            # An obstacle hold is an obligation to stay stopped, not a map: when the car stood
            # still from before the outage to now (brake held), its odometry position is still
            # where the car is, and the hold is kept (full review of 73ae759: a 0.3 s IMU dropout
            # at a standstill dropped the hold and the car drove 1.24 m towards the obstacle).
            still = self._moving_t is None or (self._rebuild_since is not None and self._moving_t <= self._rebuild_since)
            self.planner.reset(keep_holds=still)
            d.update(holds_kept=bool(still and self.planner.holding))
            self.controller.reset()
            self.turn.reset()  # a latched turn path lived in the old frame too
            self.crossing.reset()
            self._turning, self._mouth_odo, self._lane_heading_odo, self._turn_corridor, self._opening_odo, self._mouth_candidate = False, None, None, None, None, None
            self._gate_frame = self._turn_command = self._latest_odo = None
            self._zone_entered = False
            self._still_since, self._still_base, self._still_credit, self._still_odo = None, 0.0, 0.0, None
            self._rebuild = False
            self._profile_path = None
            d.update(frame_rebuilt=True)
            if classified is not None:
                self.grid.add(classified, motion.odometry, t, stationary=motion.stationary)
            return self._fault("local frame rebuilt from a standstill: waiting for a corridor", d, motion, t, dt)
        if motion.gnss_age_s > c.gnss_max_age_s:
            return self._fault(f"GNSS {motion.gnss_age_s:.1f} s old", d, motion, t, dt)
        if motion.route_sigma_m > c.route_max_sigma_m:
            return self._fault(f"route position uncertain ({motion.route_sigma_m:.1f} m)", d, motion, t, dt)
        # -- straight through a junction (crossing.py) ------------------------------------
        self.crossing.observe(motion.odometry, not motion.stationary, None if frame.imu is None else yaw_from_compass(frame.imu.compass_rad))
        crossing = self._crossing_context(guidance, motion)
        if crossing is None and self.crossing.active:
            # The route has moved past the exit point while the car still crosses on the
            # line: the junction is this one until the kerb is found again.
            crossing = self._crossing_context(guidance, motion, entry_i=self.crossing.entry_i)
        if crossing is not None and crossing[-1] == self._turn_done:
            crossing = None  # crossed: kerb following to the exit point and on
        if crossing is None:
            if self.crossing.active or self.crossing.line is not None:
                self.crossing.reset()  # the route moved on
        elif not self.turn.committed:
            entry_b, exit_b, after_b, entry_i = crossing
            d.update(crossing_entry_m=round(float(entry_b[0]), 1), crossing_exit_m=round(float(exit_b[0]), 1))
            self.crossing.latch(corridor, motion.odometry, entry_b)
            if self.crossing.should_enter(entry_b):
                self.crossing.enter(entry_i)
            if self.crossing.active:
                handed = False
                d.update(crossing_heading_hold_deg=round(math.degrees(self.crossing.hold_heading(motion.odometry)), 2))
                path = self.crossing.path_body(motion.odometry, exit_b, after_b)
                self._crossing_tick = getattr(self, "_crossing_tick", 0) + 1
                if self.crossing.may_try_kerb(entry_b) and self._crossing_tick % c.crossing.seed_every_ticks == 0:
                    # Into the junction: seed kerb following from the path, at the path and
                    # beside it (the line drifts: r12 on Town10HD, 0.7-1.5 m over a 43 m
                    # junction with nothing to correct it), and take the first fit with kerb
                    # evidence of its own. Near the path, it is handed the car; further off,
                    # it re-anchors the line, which then converges on it.
                    found, gathered = None, []
                    for offset in c.crossing.seed_offsets_m:
                        self.corridor.seed(path + np.array([0.0, offset]), motion.odometry, t)
                        resumed = self.corridor.update(self.grid, motion.odometry, t)
                        if resumed.ok and len(resumed.evidence_body):
                            gathered.append(resumed.evidence_body)
                        if resumed.ok and len(resumed.evidence_body) >= c.crossing.handback_min_evidence:
                            found = resumed
                            break
                    # The kerb as every seed saw it: with the exit line a few degrees out, each
                    # seed holds a few metres of it (its gate is 0.75 m) and none holds enough;
                    # together they do, and their line has the kerb's direction too.
                    pooled = np.vstack(gathered) if gathered else np.zeros((0, 2))
                    if found is None and len(pooled) >= c.crossing.handback_min_evidence and float(np.ptp(pooled[:, 0])) >= c.crossing.handback_span_m:
                        # No seed held the kerb on its own (a curving exit: each parallel seed
                        # holds a few metres of it), but together they trace it. Seed the kerb
                        # corridor along what they traced, from the car, and let it fit.
                        traced = _traced_seed(pooled, float(np.interp(0.0, path[:, 0], path[:, 1])))
                        self.corridor.seed(traced, motion.odometry, t)
                        resumed = self.corridor.update(self.grid, motion.odometry, t)
                        d.update(crossing_traced=f"{len(resumed.evidence_body)} {'ok' if resumed.ok else resumed.reason}")
                        if resumed.ok and len(resumed.evidence_body) >= c.crossing.handback_min_evidence:
                            found = resumed
                    d.update(crossing_pooled=len(pooled), crossing_found=None if found is None else len(found.evidence_body))
                    if found is not None:
                        # Judged where the evidence is (the seed is the fit's prior where there
                        # is none: at the car, in the junction, it says nothing). Only a fit
                        # with evidence of its own counts: a line through the pooled points
                        # took in the exit corner's fillet and moved the car 1.8 m off.
                        ex = found.evidence_body[:, 0]
                        span = float(np.ptp(ex))
                        # Agreement is judged where the evidence begins, not at the car: a fit
                        # through kerb 10-30 m ahead on a curve extrapolates to the car a metre
                        # or two off (seed 5310 of the curve sweep: -2.4 m, and the hand-back
                        # never came). Handed back, the kerb corridor is seeded with the path to
                        # there and the fit beyond, so the car sees no step.
                        ahead = path[path[:, 0] >= -0.5]
                        k = min(5, len(ahead) - 1)
                        heading_here = math.atan2(ahead[k, 1] - ahead[0, 1], ahead[k, 0] - ahead[0, 0]) if k > 0 else 0.0
                        x_near = max(0.0, float(ex.min()) - 2.0)
                        y_path_near = float(np.interp(x_near, path[:, 0], path[:, 1]))
                        y_path_on = float(np.interp(x_near + 1.0, path[:, 0], path[:, 1]))
                        lateral_near = found.lateral_offset_at(x_near) - y_path_near
                        turn_near = _wrap(found.heading_at(x_near) - math.atan2(y_path_on - y_path_near, 1.0))
                        agrees_near = abs(lateral_near) <= c.crossing.handback_offset_m and abs(turn_near) <= c.crossing.handback_heading_rad
                        d.update(crossing_fit_near=(round(x_near, 1), round(lateral_near, 2), round(math.degrees(turn_near), 1), round(span, 1)))
                        # Hand back: a kerb fit with real evidence over handback_span_m that
                        # agrees with the path where its evidence begins and runs to within
                        # handback_route_m (plus the route's own uncertainty) of the route past
                        # the exit. The exit may curve (Town10HD 468: 50 degrees over 30 m, r13):
                        # the kerb corridor follows a curve, no line or sat-nav arc does; a side
                        # road's fillet runs away from the route and is refused by the route test.
                        if span >= c.crossing.handback_span_m and agrees_near:
                            x_far = min(float(ex.max()), found.valid_length_m)
                            far = (x_far, found.lateral_offset_at(x_far))
                            off_route = _polyline_distance(far, np.vstack([np.asarray(exit_b)[None, :], np.asarray(after_b).reshape(-1, 2)]) if after_b is not None else np.asarray(exit_b)[None, :])
                            d.update(crossing_handback_route_m=round(off_route, 1))
                            if off_route <= c.crossing.handback_route_m + motion.route_sigma_m:
                                fitted = found.centre_body[found.centre_body[:, 0] >= x_near]
                                composite = np.vstack([path[path[:, 0] < x_near], fitted]) if len(fitted) >= 2 else path
                                self.corridor.seed(composite, motion.odometry, t)
                                taken = self.corridor.update(self.grid, motion.odometry, t)
                                # The refit through the composite and the kerb must still start
                                # at the car (a cubic over 30 m can leave it 2 m off there: the
                                # controller refuses such a trajectory and the car stops).
                                step_here = abs(taken.lateral_offset_at(0.0) - float(np.interp(0.0, path[:, 0], path[:, 1])))
                                d.update(crossing_handback_step_m=round(step_here, 2))
                                if taken.ok and step_here <= c.crossing.handback_offset_m and abs(_wrap(taken.heading_at(0.0) - heading_here)) <= c.crossing.handback_heading_rad:
                                    corridor, handed = taken, True
                                    self._turn_done = entry_i
                                    self.crossing.reset()
                                    d.update(crossing="handed back to the kerb corridor")
                        past_exit_near = self.crossing.past_exit_point_m(motion.odometry, (x_near, y_path_near))
                        if not handed and past_exit_near is not None and past_exit_near <= c.crossing.reanchor_past_exit_m and span >= 6.0 and abs(turn_near) <= c.crossing.handback_heading_rad:
                            # Kerb along the latched line, short of the exit point, moves the
                            # path sideways (the odometry drifts a metre over a kerbless box).
                            # Past the exit point nothing is moved: a difference there is the
                            # sat-nav arc's error, or the exit corner's fillet (moved on it, the
                            # car went 0.9-2 m off), and the hand-back resolves it. An edge at
                            # an angle is some other edge (a side road's hedge, the box's far
                            # kerb) and is left alone.
                            if self.crossing.reanchor(motion.odometry, lateral_near, found.heading_at(x_near)):
                                d.update(crossing_reanchor_m=round(lateral_near, 2), crossing_reanchor_deg=round(math.degrees(turn_near), 1), crossing_reanchor_x_m=round(x_near, 1))
                if not handed:
                    if self.crossing.given_up(exit_b):
                        self._turn_done = entry_i
                        self.crossing.reset()
                        d.update(crossing="no kerb past the exit: kerb following on its own")
                    else:
                        corridor = self.crossing.corridor(self.grid, motion.odometry, t, self.corridor.support, exit_b, after_b)
                        d.update(crossing=self.crossing.note, crossing_bend=round(math.degrees(self.crossing.bend_rad), 1), corridor_ok=corridor.ok, corridor_reason=corridor.reason, corridor_valid_m=corridor.valid_length_m, corridor_support_m=corridor.support_m, lane_offset_m=corridor.lateral_offset_at(0.0))
            elif self.crossing.line is not None:
                d.update(crossing=self.crossing.note)
        if not corridor.ok and not self.turn.committed:
            if motion.stationary and corridor.evidence_age_s > c.corridor.evidence_max_age_s:
                # Stopped and the corridor is gone for good: assume again that the car stands
                # in its lane, so the next sweeps search for the kerb from the car's own line.
                self.corridor.reset()
                d.update(corridor_reacquire=True)
            return self._fault(f"no corridor: {corridor.reason}", d, motion, t, dt)
        # -- junction turns (G1c) -------------------------------------------------------
        hold_at = None
        junction = self._junction_context(guidance, motion, corridor)
        if junction is not None and junction[-1] == self._turn_done:
            junction = None  # this junction's turn is done: kerb following to the exit point
        self.turn.standing_s = standing
        self.turn.still_credit_s = self._still_credit
        d.update(still_credit_s=round(self._still_credit, 2) if self._still_credit else None)
        self.turn.observe(classified, motion.odometry, t)
        if self.turn.committed:
            # In the turn: follow the latched path to its end, whatever the route says.
            exit_heading_body = self.turn.exit_heading_body(motion.odometry)
            plan_turn = self.turn.follow(self.grid, motion.odometry, t)
            block = self.turn.block
            if block is not None:
                d.update(turn_block=block)
                if plan_turn is not None:
                    d.update(turn=f"an object on the latched path {block['at_m']:.1f} m on: held short of it ({block['waited_s']:.1f} s)")
            if plan_turn is None:
                # A newly seen kerb or obstacle cell under the latched path (review of 1931070:
                # the first version sat in the junction for good). Near the entry heading the
                # car is still on the straight: drop the latch and plan afresh this tick. Near
                # the exit heading it is on the exit lane: hand over to kerb following now. In
                # between: one fresh arc from here; failing that, stop along the turn path (not
                # the pre-turn kerb line, which would be a steering jump).
                turned = abs(wrap_angle(motion.odometry.yaw_rad - self._lane_heading_odo)) if self._lane_heading_odo is not None else math.inf
                if turned <= c.junction.replan_max_heading_rad and junction is not None:
                    self.turn.uncommit()
                    d.update(turn="latched path blocked: re-planning")
                else:
                    resumed = None
                    if abs(exit_heading_body) <= c.junction.exit_tolerance_rad and self._turn_corridor is not None:
                        self.corridor.seed(motion.odometry.to_body(self._turn_corridor.centre_odometry), motion.odometry, t)
                        resumed = self.corridor.update(self.grid, motion.odometry, t)
                    if resumed is not None and resumed.ok:
                        corridor, hold_at = resumed, None
                        self._turning = False
                        self._turn_done = None if junction is None else junction[-1]
                        self._mouth_odo = self._lane_heading_odo = self._turn_corridor = self._opening_odo = self._mouth_candidate = self._gate_frame = self._turn_command = self._latest_odo = None
                        self._zone_entered = False
                        self.turn.reset()
                        junction = None
                        d.update(turn="latched path blocked at the exit: handed back to the kerb corridor")
                    else:
                        plan_turn = self.turn.replan_same_arc(self.grid, motion.odometry, t)
                        if plan_turn is not None:
                            d.update(turn="latched path blocked: re-planned on the re-measured exit line" + (" (same arc)" if plan_turn.arc_start_m > 0.0 else " (fresh arc from here)"))
                        else:
                            plan_turn = self.turn.replan_mid_arc(self.grid, motion.odometry, t)
                            if plan_turn is None:
                                if self._turn_corridor is not None:
                                    corridor = self._turn_corridor
                                hold_at = 0.0
                                d.update(turn="latched path no longer clears the terrain: stopping on it")
                            else:
                                d.update(turn="latched path blocked: re-planned mid-arc")
            if plan_turn is not None and plan_turn.arc_end_m > 0.0:
                # Still on the arc: the exit kerb is read better from here than from where
                # the turn was committed; move the exit straight (same arc, same tangent
                # point) when the re-measured line has shifted by more than the latch width.
                refined = self.turn.replan_same_arc(self.grid, motion.odometry, t, min_shift_m=c.junction.refine_shift_m)
                d.update(turn_refine=self.turn.refine_note)
                if refined is not None:
                    plan_turn = refined
                    d.update(turn="exit line refined from closer by")
            d.update(self.turn.exit_diagnostics())
            if plan_turn is not None:
                corridor = self._turn_corridor = plan_turn.corridor
                hold_at = max(0.0, plan_turn.verified_m - c.body.front_m)
                d.update(turn_radius_m=plan_turn.radius_m, turn_arc_end_m=plan_turn.arc_end_m, turn_verified_m=plan_turn.verified_m, turn_clearance_m=round(plan_turn.clearance_m, 2) if math.isfinite(plan_turn.clearance_m) else None)
                if abs(plan_turn.exit_heading_rad) <= c.junction.exit_tolerance_rad and plan_turn.arc_end_m < 0.0:
                    # Hand back to kerb following only once it can take over: seed it with
                    # the latched exit line and see that it finds its own corridor.
                    self.corridor.seed(plan_turn.corridor.centre_body, motion.odometry, t)
                    resumed = self.corridor.update(self.grid, motion.odometry, t)
                    if resumed.ok:
                        corridor, hold_at = resumed, None
                        self._turning = False
                        self._turn_done = None if junction is None else junction[-1]
                        self._mouth_odo = self._lane_heading_odo = self._turn_corridor = self._opening_odo = self._mouth_candidate = self._gate_frame = self._turn_command = self._latest_odo = None
                        self._zone_entered = False
                        self.turn.reset()
                        junction = None  # this junction is done this tick too: no hold at its entry
                        d.update(turn="handed back to the kerb corridor")
                    else:
                        d.update(turn=f"kerb corridor not ready: {resumed.reason}")
        if not self.turn.committed and junction is not None:
            entry, exit_point, exit_heading, command, _ = junction
            opening = kerb_opening(corridor, self.grid, c.junction.opening_min_m, max_age_s=3.0 + max(standing, self._still_credit)) if corridor.ok else None
            # An opening seen once is kept in the odometry frame: at a standstill the cells
            # that confirmed it are not refreshed and it vanished, and the hold point fell
            # back to the route's entry, 9 m short of the real mouth (live run g1c_right).
            if opening is not None:
                self._opening_odo = motion.odometry.to_world(np.array([[opening, 0.0]]))[0]
                d.update(kerb_opening_fresh_m=opening)
            elif self._opening_odo is not None:
                remembered = float(motion.odometry.to_body(self._opening_odo[None])[0, 0])
                opening = remembered if remembered > -2.0 else None
            d.update(junction_command=command.value, junction_entry_m=float(entry[0]), kerb_opening_m=opening)
            if not self._turning:
                # Within turn_start_m of the entry, or already past it (a route point behind
                # the car still names a junction the car has not turned at).
                near = entry[0] <= c.turn_start_m and abs(entry[1]) <= c.turn_route_tolerance_m
                confirmed = entry[0] >= 0.0 and (
                    command is Command.LEFT or (opening is not None and abs(opening - entry[0]) <= c.junction.max_entry_offset_m)
                )
                if near and confirmed:
                    self._turning = True
                    self._turn_command = command
                    # The lane's direction from the kerb evidence beside and behind the car: the
                    # whole-corridor fit is already tilted by the kerb running out into the mouth.
                    self._lane_heading_odo = motion.odometry.yaw_rad + (corridor.heading_near() if corridor.ok else 0.0)
                    if command is Command.LEFT:
                        centre = motion.odometry.to_world(np.array([[0.0, corridor.lateral_offset_at(0.0) if corridor.ok else 0.0]]))[0]
                        self._gate_frame = Pose2D(float(centre[0]), float(centre[1]), self._lane_heading_odo)
                elif near:
                    # The route says a junction is here and the kerb does not: do not enter.
                    hold_at = max(0.0, float(entry[0]) - c.body.front_m - c.planner.stop_gap_m)
                    d.update(turn="junction not confirmed by the kerb")
            if self._turning:
                plan_turn = self.turn.plan(self.grid, motion.odometry, corridor, exit_point, exit_heading, t)
                # The junction mouth: the observed kerb opening for a right turn, else the
                # route's entry point less a margin. The car does not pass it until the
                # exit lane is measured and a turn with room to spare is planned.
                if command is Command.RIGHT and opening is not None:
                    mouth = opening
                else:
                    mouth = float(entry[0]) - c.planner.stop_gap_m
                # The mouth only comes nearer: an opening that flickers at the edge of the
                # observed range must not let the car creep forward in steps. The exception is
                # a car standing at the mouth that reads the opening further on, steadily,
                # for mouth_release_s: read from 25 m the kerb's last few metres are between
                # rings and the opening came 3 m early, the car waited 7 m short of the real
                # mouth, and from there the exit kerb was read too poorly to commit (full
                # review of 73ae759, S7: a walker on the exit for 3 s left the car there).
                if self._mouth_odo is not None:
                    held = float(motion.odometry.to_body(self._mouth_odo[None])[0, 0])
                    fresh = d.get("kerb_opening_fresh_m")
                    if command is Command.RIGHT and motion.stationary and fresh is not None and fresh > held + c.mouth_release_m:
                        seen = self._mouth_candidate
                        here = motion.odometry.to_world(np.array([[fresh, 0.0]]))[0]
                        if seen is None or float(np.hypot(*(seen[0] - here))) > c.mouth_release_m:
                            self._mouth_candidate = (here, t)
                        elif t - seen[1] >= c.mouth_release_s:
                            held = fresh
                            d.update(turn_mouth="moved to the opening read from here")
                    else:
                        self._mouth_candidate = None
                    mouth = min(mouth, held)
                self._mouth_odo = motion.odometry.to_world(np.array([[mouth, 0.0]]))[0]
                latest = self.turn.latest_start_m()
                # Latched like the mouth (it only comes nearer): the exit reading drifts while
                # something stands on the exit (a walker split the exit road, and the car crept
                # on after a tangent point that moved 4 m in 8 s; Stage 3).
                if latest is not None:
                    if self._latest_odo is not None:
                        latest = min(latest, float(motion.odometry.to_body(self._latest_odo[None])[0, 0]))
                    self._latest_odo = motion.odometry.to_world(np.array([[latest, 0.0]]))[0]
                if plan_turn is None:
                    hold_at = max(0.0, mouth - c.body.front_m)
                    if latest is not None:
                        hold_at = min(hold_at, max(0.0, latest))
                    d.update(turn="no radius clears the terrain")
                else:
                    d.update(
                        turn_radius_m=plan_turn.radius_m,
                        turn_exit=plan_turn.exit_source,
                        turn_arc_end_m=plan_turn.arc_end_m,
                        turn_verified_m=plan_turn.verified_m,
                        turn_exit_fit=plan_turn.exit_fit,
                        turn_clearance_m=round(plan_turn.clearance_m, 2) if math.isfinite(plan_turn.clearance_m) else None,
                        **self.turn.exit_diagnostics(),
                    )
                    measured = plan_turn.exit_source == "kerb" and plan_turn.clearance_m >= c.junction.commit_clearance_m
                    if measured:
                        # Steer along the planned turn, as far as its footprint is on observed road.
                        corridor = self._turn_corridor = plan_turn.corridor
                        hold_at = max(0.0, plan_turn.verified_m - c.body.front_m)
                    else:
                        # Keep to the lane (the route's point never places the exit lane) and
                        # wait at the mouth, where the exit road is in view.
                        hold_at = max(0.0, min(plan_turn.verified_m, mouth) - c.body.front_m)
                        if latest is not None:
                            hold_at = min(hold_at, max(0.0, latest))
        elif not self.turn.committed and self._turning:
            self._turning = False  # the route moved on before the turn began: kerb following again
            self._mouth_odo = self._lane_heading_odo = self._turn_corridor = self._opening_odo = self._mouth_candidate = self._gate_frame = self._turn_command = self._latest_odo = None
            self._zone_entered = False
            self.turn.reset()
        self.last_gate = None
        if self._gate_frame is not None and c.junction.oncoming_check and corridor is self._turn_corridor and corridor is not None and not self._zone_entered:
            # A left turn on its way across the opposing lane: enter it only with the approach
            # seen clear (S5); once in it, clear it (the gate never stops the car there).
            override = resolve_oncoming_override(self.oncoming_override, t, c.junction.oncoming_override_max_age_s)
            if override is not None and override.get("stale"):
                d.update(turn_gate_override=f"stale ({override['age_s']:.2f} s): excuses and clear dropped")
            gate = oncoming_gate(
                self.grid, motion.odometry, corridor.centre_body, self._gate_frame, c.body, c.junction,
                motion.speed_mps, c.planner.cruise_mps, excused=() if override is None else override.get("excused", ()),
            )
            self.last_gate = gate
            if gate is not None:
                clear = gate.clear and (override is None or bool(override.get("clear", True)))
                note = gate.note if override is None or override.get("clear", True) else override.get("note", "the stack sees oncoming traffic")
                if gate.entry_m <= 0.0:
                    self._zone_entered = True
                    d.update(turn_gate="in the opposing lane: clearing it")
                elif not clear:
                    stop = max(0.0, gate.entry_m - c.junction.gate_gap_m)
                    hold_at = stop if hold_at is None else min(hold_at, stop)
                    d.update(turn_gate="waiting: " + note)
                else:
                    d.update(turn_gate=note)
                d.update(
                    turn_gate_entry_m=round(gate.entry_m, 2),
                    turn_gate_required_s=round(gate.required_s, 1),
                    turn_gate_view_s=round(gate.checked_m / c.junction.oncoming_mps, 1),  # how long the view covers at oncoming_mps
                    turn_gate_cells=gate.cells,
                )
        lap("junction")
        if not corridor.ok:
            return self._fault(f"no corridor: {corridor.reason}", d, motion, t, dt)
        observation = None if classified is None else observe_path(classified, corridor, c.body, c.obstacles)
        lap("obstacles")
        obstacle = None if observation is None else observation.obstacle
        goal_body = None
        if guidance.index == len(self.progress.xy) - 1 and guidance.goal_distance_m < corridor.valid_length_m + 5:
            goal_body = motion.odometry.to_body(motion.route_to_odometry(np.array([self.progress.xy[-1]])))[0]
        plan = self.planner.plan(corridor, observation, goal_body, t, hold_at_m=hold_at, speed_cap_mps=self.speed_cap_mps)
        lap("plan")
        self.last_plan, self.last_corridor = plan, corridor  # the path planned along this tick (a turn's, in a turn)
        self._profile_path = (t, corridor.centre_odometry) if corridor.ok else None
        self.turn_entry_m = float(junction[0][0]) if junction is not None and corridor is not self._turn_corridor else None
        d.update(
            obstacle_m=None if obstacle is None else obstacle.distance_m,
            obstacle_points=None if obstacle is None else obstacle.points,
            obstacle_lateral_m=None if obstacle is None else obstacle.lateral_m,
            clear_to_m=None if observation is None else observation.clear_to_m,
            sight_speed_mps=None if math.isinf(plan.sight_speed_mps) else round(plan.sight_speed_mps, 2),
            hold_state=plan.hold_state,
            hold_note=plan.hold_note or None,
            holds=self.planner.hold_count,
            hold_reports=list(plan.holds) or None,  # each hold's id and release prerequisites (F2)
            stop_reason=plan.stop_reason,
            stop_distance_m=plan.stop_distance_m if math.isfinite(plan.stop_distance_m) else None,
        )
        if (
            plan.stop_reason in ("obstacle", "obstacle_hold", "junction")
            and plan.stop_distance_m < (c.junction_hold_within_m if plan.stop_reason == "junction" else c.hold_within_m)
            and motion.stationary
        ):
            # Stopped at the stop point: hold the brake rather than let the speed loop
            # creep (or the car roll back on a slope, which the live harness refuses as
            # reverse motion). Driving resumes once the path is seen clear for
            # resume_clear_s, or the junction's hold point moves on.
            return self._brake(DriveMode.STOPPING, f"holding for {plan.stop_reason} ({plan.hold_state})", d)
        if motion.stationary and not self.turn.committed and corridor.evidence_age_s > c.start_evidence_max_age_s:
            # Moving off needs a kerb seen now, not a line carried from before the stop.
            # The line is kept: it expires on its own after evidence_max_age_s.
            return self._fault(
                f"kerb evidence {corridor.evidence_age_s:.1f} s old: not moving off on a carried corridor", d, motion, t, dt
            )
        if plan.trajectory is None:
            return self._fault("no plan", d, motion, t, dt)
        state = VehicleState(motion.odometry.x_m, motion.odometry.y_m, motion.odometry.yaw_rad, motion.speed_mps)
        acceptance = self.controller.set_local_trajectory(plan.trajectory, state, t)
        limit = math.inf
        if not acceptance.accepted:
            # This tick's plan was refused (it starts off the car, or has expired), and the
            # controller keeps the last one it accepted: planned before this tick's
            # obstacles, holds and caps. That one is driven no faster than the refused plan
            # allows where the car is (third review, R7).
            limit = refused_speed_limit(plan.trajectory, state, self.controller.config.speed_lookahead_m)
        result = self.controller.update(state, t, dt, speed_limit_mps=limit)
        lap("control")
        d.update(
            trajectory_accepted=acceptance.accepted,
            trajectory_reason=acceptance.reason,
            refused_plan_limit_mps=None if math.isinf(limit) else round(limit, 2),
            control_status=result.status.value,
            target_speed_mps=None if result.diagnostics is None else result.diagnostics.target_speed_mps,
            cross_track_m=None if result.diagnostics is None else result.diagnostics.cross_track_error_m,
        )
        if not acceptance.accepted and result.status in (LocalStatus.NO_TRAJECTORY, LocalStatus.EXPIRED):
            # Braking because plans are refused: say why rather than sit in DRIVING (first live run).
            return self._fault(f"trajectory refused: {acceptance.reason}", d, motion, t, dt)
        if result.status is LocalStatus.FINISHED or guidance.finished and motion.stationary:
            self._finished = True
            return self._brake(DriveMode.FINISHED, "goal reached", d)
        mode = DriveMode.STOPPING if result.status in (LocalStatus.STOPPING, LocalStatus.EXPIRED) else DriveMode.DRIVING
        if mode is DriveMode.DRIVING and self._turning:
            mode = DriveMode.TURNING
        return self._emit(result.command, mode, plan.stop_reason, d)

    @property
    def gate_frame(self):
        """A left turn's entry lane (odometry pose on its centre, along it), latched as the
        approach began; None otherwise."""
        return self._gate_frame

    @property
    def clearing_opposing_lane(self):
        """A left turn's body is in the opposing lane it crosses (it entered with the gate
        clear): the stack lets it clear the lane rather than stop in it for a vehicle
        predicted where it stands (S5)."""
        if not self._zone_entered or self._gate_frame is None or self.last_motion is None:
            return False
        from mapless.junction import body_corners  # noqa: PLC0415

        lo = self.config.junction.lane_width_m / 2
        hi = lo + self.config.junction.lane_width_m
        pose = self.last_motion.odometry
        here = self._gate_frame.to_body(pose.to_world(np.array([[0.0, 0.0], [1.0, 0.0]])))
        y = body_corners(here, self.config.body)[0, :, 1]
        return bool(y.max() > lo and y.min() < hi)

    def _crossing_context(self, guidance, motion, entry_i=None):
        """(entry point, exit point, the point after the exit or None, entry route index),
        vehicle frame, for a junction the route crosses STRAIGHT that the car is heading
        for or in (or the given entry index's), else None."""
        points = self.progress.xy
        i = guidance.index
        if entry_i is not None:
            pass
        elif guidance.command is Command.STRAIGHT:
            entry_i = i
        elif guidance.previous is Command.STRAIGHT and i >= 1:
            entry_i = i - 1
        else:
            return None
        if entry_i + 1 >= len(points):
            return None
        # The points after the exit within 60 m of it along the route, to the first that
        # is itself a junction command: the exit lane's line is fitted through them (one
        # 25 m pair's direction is 6 degrees out with the sat-nav's 2 m noise; three
        # points, half that).
        exit_i = entry_i + 1
        after = []
        route_points = self.progress.route.points
        for j in range(exit_i + 1, len(points)):
            if math.dist(points[j], points[exit_i]) > 60.0:
                break
            after.append(j)
            if route_points[j].command is not Command.LANEFOLLOW:
                break
        odo = motion.route_to_odometry(np.array([points[entry_i], points[exit_i]] + [points[j] for j in after]))
        body = motion.odometry.to_body(odo)
        return body[0], body[1], (body[2:] if after else None), entry_i

    def _junction_context(self, guidance, motion, corridor):
        """(entry point, exit point, exit heading, command, entry route index) in the
        vehicle frame for the junction the route is heading for or in, else None."""
        points = self.progress.xy
        i = guidance.index
        if guidance.command.is_junction:
            entry_i = i
        elif guidance.previous.is_junction and i >= 1:
            entry_i = i - 1  # the entry point was just passed: the exit is the target now
        else:
            return None
        if entry_i + 1 >= len(points):
            return None
        command = self.progress.route.points[entry_i].command
        if command not in (Command.LEFT, Command.RIGHT):
            return None  # STRAIGHT keeps kerb following for now
        exit_i = entry_i + 1
        after_i = min(exit_i + 1, len(points) - 1)
        odo = motion.route_to_odometry(np.array([points[entry_i], points[exit_i], points[after_i]]))
        body = motion.odometry.to_body(odo)
        if after_i != exit_i:
            route_heading = math.atan2(body[2, 1] - body[1, 1], body[2, 0] - body[1, 0])
        else:
            route_heading = math.atan2(body[1, 1] - body[0, 1], body[1, 0] - body[0, 0])
        # Two GNSS points 20 m apart give the exit direction to about 8 degrees; a commanded
        # turn is a right angle to the entry lane, which the car's own corridor gives. The
        # lane's direction, not the car's: a 1.5 degree heading error at the commit drifted
        # the latched exit line 0.6 m by the end of the exit straight.
        # Latched in the odometry frame when the approach begins: nearer the junction the
        # kerb runs out into the mouth and the corridor's fit tilts with it (0.8 m on the
        # exit lane for the right turn).
        if self._lane_heading_odo is not None:
            lane = wrap_angle(self._lane_heading_odo - motion.odometry.yaw_rad)
        else:
            lane = corridor.heading_near() if corridor.ok else 0.0
        prior = lane + (-math.pi / 2 if command is Command.RIGHT else math.pi / 2)
        heading = prior if abs(wrap_angle(route_heading - prior)) <= math.radians(35) else route_heading
        return body[0], body[1], heading, command, entry_i
