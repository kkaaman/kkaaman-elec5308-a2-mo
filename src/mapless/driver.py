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
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from mapless.corridor import CorridorConfig, CorridorEstimator
from mapless.ego_motion import EgoMotion, EgoMotionConfig
from mapless.frame import SensorFrame, SensorState
from mapless.geometry import wrap as wrap_angle
from mapless.health import HealthTracker
from mapless.junction import JunctionConfig, JunctionTurn, kerb_opening
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
    turn_start_m: float = 25.0  # a commanded turn is looked for when the route's entry point is this close
    turn_route_tolerance_m: float = 8.0  # the entry point must be within this of the car's path
    lidar_max_age_s: float = 0.15
    gnss_max_age_s: float = 3.0
    route_max_sigma_m: float = 4.0  # route position this uncertain: the goal cannot be judged
    start_evidence_max_age_s: float = 0.5  # move off a standstill only on kerb evidence this fresh
    hold_within_m: float = 1.5  # stopped this close to an obstacle stop point: hold the brake

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
        self.planner = LanePlanner(config.planner, config.body)
        self.turn = JunctionTurn(config.junction, config.body, self.corridor)
        self._turning = False
        self.speed_cap_mps = None  # the stack's following/prediction cap for this tick (None: none)
        self.last_motion = self.last_classified = self.last_plan = None
        self._turn_done = None  # route index of the junction entry whose turn is complete
        self._mouth_odo = None  # the junction mouth, odometry frame, nearest seen since the approach began
        self._lane_heading_odo = None  # the entry lane's direction, odometry frame, latched as the approach begins
        self._turn_corridor = None  # the last turn path steered along (the stop line if the latched path fails mid-arc)
        self.controller = LocalTrajectoryController(config.control)
        self._last_time = None
        self._finished = False
        self._last_steer = 0.0
        self._calibrated = False
        self._rebuild = False  # odometry was lost: grid, corridor, hold and trajectory are stale

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
        self.health.update(frame)
        motion = self.ego.update(frame)
        classified = None
        if frame.status_of(self.lidar).state is SensorState.OK:
            classified = classify_sweep(frame.lidar(self.lidar).xyz, c.body, c.terrain)
        # For the stack around the driver (G2/G3): this tick's inputs, read after step().
        self.last_motion, self.last_classified, self.last_plan = motion, classified, None
        if classified is not None and motion.odometry_trusted and not self._rebuild:
            self.grid.add(classified, motion.odometry, t)  # never placed with a broken frame
        if not motion.odometry_trusted:
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
        corridor = self.corridor.update(self.grid, motion.odometry, t)
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
            self.planner.reset()
            self.controller.reset()
            self.turn.reset()  # a latched turn path lived in the old frame too
            self._turning, self._mouth_odo, self._lane_heading_odo, self._turn_corridor = False, None, None, None
            self._rebuild = False
            d.update(frame_rebuilt=True)
            if classified is not None:
                self.grid.add(classified, motion.odometry, t)
            return self._fault("local frame rebuilt from a standstill: waiting for a corridor", d, motion, t, dt)
        if motion.gnss_age_s > c.gnss_max_age_s:
            return self._fault(f"GNSS {motion.gnss_age_s:.1f} s old", d, motion, t, dt)
        if motion.route_sigma_m > c.route_max_sigma_m:
            return self._fault(f"route position uncertain ({motion.route_sigma_m:.1f} m)", d, motion, t, dt)
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
        self.turn.observe(classified, motion.odometry, t)
        if self.turn.committed:
            # In the turn: follow the latched path to its end, whatever the route says.
            exit_heading_body = self.turn.exit_heading_body(motion.odometry)
            plan_turn = self.turn.follow(self.grid, motion.odometry, t)
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
                        self._mouth_odo = self._lane_heading_odo = self._turn_corridor = None
                        self.turn.reset()
                        junction = None
                        d.update(turn="latched path blocked at the exit: handed back to the kerb corridor")
                    else:
                        plan_turn = self.turn.replan_same_arc(self.grid, motion.odometry, t)
                        if plan_turn is not None:
                            d.update(turn="latched path blocked: exit line re-anchored, same arc")
                        else:
                            plan_turn = self.turn.replan_mid_arc(self.grid, motion.odometry, t)
                            if plan_turn is None:
                                if self._turn_corridor is not None:
                                    corridor = self._turn_corridor
                                hold_at = 0.0
                                d.update(turn="latched path no longer clears the terrain: stopping on it")
                            else:
                                d.update(turn="latched path blocked: re-planned mid-arc")
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
                        self._mouth_odo = self._lane_heading_odo = self._turn_corridor = None
                        self.turn.reset()
                        junction = None  # this junction is done this tick too: no hold at its entry
                        d.update(turn="handed back to the kerb corridor")
                    else:
                        d.update(turn=f"kerb corridor not ready: {resumed.reason}")
        if not self.turn.committed and junction is not None:
            entry, exit_point, exit_heading, command, _ = junction
            opening = kerb_opening(corridor, self.grid, c.junction.opening_min_m) if corridor.ok else None
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
                    # The lane's direction from the kerb evidence beside and behind the car: the
                    # whole-corridor fit is already tilted by the kerb running out into the mouth.
                    self._lane_heading_odo = motion.odometry.yaw_rad + (corridor.heading_near() if corridor.ok else 0.0)
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
                # observed range must not let the car creep forward in steps.
                if self._mouth_odo is not None:
                    mouth = min(mouth, float(motion.odometry.to_body(self._mouth_odo[None])[0, 0]))
                self._mouth_odo = motion.odometry.to_world(np.array([[mouth, 0.0]]))[0]
                if plan_turn is None:
                    hold_at = max(0.0, mouth - c.body.front_m)
                    d.update(turn="no radius clears the terrain")
                else:
                    d.update(
                        turn_radius_m=plan_turn.radius_m,
                        turn_exit=plan_turn.exit_source,
                        turn_arc_end_m=plan_turn.arc_end_m,
                        turn_verified_m=plan_turn.verified_m,
                        turn_exit_fit=plan_turn.exit_fit,
                        turn_clearance_m=round(plan_turn.clearance_m, 2) if math.isfinite(plan_turn.clearance_m) else None,
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
        elif not self.turn.committed and self._turning:
            self._turning = False  # the route moved on before the turn began: kerb following again
            self._mouth_odo = self._lane_heading_odo = self._turn_corridor = None
            self.turn.reset()
        if not corridor.ok:
            return self._fault(f"no corridor: {corridor.reason}", d, motion, t, dt)
        observation = None if classified is None else observe_path(classified, corridor, c.body, c.obstacles)
        obstacle = None if observation is None else observation.obstacle
        goal_body = None
        if guidance.index == len(self.progress.xy) - 1 and guidance.goal_distance_m < corridor.valid_length_m + 5:
            goal_body = motion.odometry.to_body(motion.route_to_odometry(np.array([self.progress.xy[-1]])))[0]
        plan = self.planner.plan(corridor, observation, goal_body, t, hold_at_m=hold_at, speed_cap_mps=self.speed_cap_mps)
        self.last_plan = plan
        d.update(
            obstacle_m=None if obstacle is None else obstacle.distance_m,
            obstacle_points=None if obstacle is None else obstacle.points,
            obstacle_lateral_m=None if obstacle is None else obstacle.lateral_m,
            clear_to_m=None if observation is None else observation.clear_to_m,
            hold_state=plan.hold_state,
            stop_reason=plan.stop_reason,
            stop_distance_m=plan.stop_distance_m if math.isfinite(plan.stop_distance_m) else None,
        )
        if (
            plan.stop_reason in ("obstacle", "obstacle_hold")
            and plan.stop_distance_m < c.hold_within_m
            and motion.stationary
        ):
            # Stopped at the stop point: hold the brake rather than let the speed loop
            # creep. Driving resumes once the path is seen clear for resume_clear_s.
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
        result = self.controller.update(state, t, dt)
        d.update(
            trajectory_accepted=acceptance.accepted,
            trajectory_reason=acceptance.reason,
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
