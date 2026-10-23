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
from mapless.health import HealthTracker
from mapless.obstacles import ObstacleConfig, observe_path
from mapless.planner import LanePlanner, PlannerConfig
from mapless.satnav import RouteProgress, SatNavRoute
from mapless.sensor_suite import SensorSuite
from mapless.terrain import TerrainConfig, TerrainGrid, VehicleBody, classify_sweep
from navigation_control.local import LocalControlConfig, LocalStatus, LocalTrajectoryController
from navigation_control.types import ControlCommand, VehicleState


class DriveMode(str, Enum):
    STARTUP = "startup"
    CALIBRATING = "calibrating"
    DRIVING = "driving"
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
        c = self.config
        moving = motion.speed_mps > c.control.controller.stopped_speed_mps
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
            if motion.odometry_trusted and not self._rebuild:
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
            self._rebuild = False
            d.update(frame_rebuilt=True)
            if classified is not None:
                self.grid.add(classified, motion.odometry, t)
            return self._fault("local frame rebuilt from a standstill: waiting for a corridor", d, motion, t, dt)
        if motion.gnss_age_s > c.gnss_max_age_s:
            return self._fault(f"GNSS {motion.gnss_age_s:.1f} s old", d, motion, t, dt)
        if motion.route_sigma_m > c.route_max_sigma_m:
            return self._fault(f"route position uncertain ({motion.route_sigma_m:.1f} m)", d, motion, t, dt)
        if not corridor.ok:
            if motion.stationary and corridor.evidence_age_s > c.corridor.evidence_max_age_s:
                # Stopped and the corridor is gone for good: assume again that the car stands
                # in its lane, so the next sweeps search for the kerb from the car's own line.
                self.corridor.reset()
                d.update(corridor_reacquire=True)
            return self._fault(f"no corridor: {corridor.reason}", d, motion, t, dt)
        observation = None if classified is None else observe_path(classified, corridor, c.body, c.obstacles)
        obstacle = None if observation is None else observation.obstacle
        goal_body = None
        if guidance.index == len(self.progress.xy) - 1 and guidance.goal_distance_m < corridor.valid_length_m + 5:
            goal_body = motion.odometry.to_body(motion.route_to_odometry(np.array([self.progress.xy[-1]])))[0]
        plan = self.planner.plan(corridor, observation, goal_body, t)
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
        if motion.stationary and corridor.evidence_age_s > c.start_evidence_max_age_s:
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
        return self._emit(result.command, mode, plan.stop_reason, d)
