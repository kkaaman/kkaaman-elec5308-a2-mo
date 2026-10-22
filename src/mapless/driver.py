"""The mapless car for G1b: sensors and a sat-nav route in, one control command out per tick.

Per tick:

1. Sensor health, then ego motion (odometry frame, and the route pose once start-up
   averaging has finished).
2. The LiDAR sweep is classified and added to the terrain grid, from the first tick on,
   so a corridor is ready by the time the car may move.
3. The corridor and lane centre are estimated; obstacles are looked for in its tube.
4. The planner builds a trajectory (lane follow, obstacle stop or goal stop) and the
   ``LocalTrajectoryController`` tracks it in the odometry frame.

**Modes:**
- STARTUP: holding the brake while GNSS and compass are averaged.
- DRIVING / STOPPING: following the plan (STOPPING when its target speed is zero).
- FINISHED: the route goal was reached (latched).
- FAULT: LiDAR older than ``lidar_max_age_s`` or no corridor. The car brakes and
  recovers by itself once both are back.

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
from mapless.obstacles import ObstacleConfig, path_obstacle
from mapless.planner import LanePlanner, PlannerConfig
from mapless.satnav import RouteProgress, SatNavRoute
from mapless.sensor_suite import SensorSuite
from mapless.terrain import PointClass, TerrainConfig, TerrainGrid, VehicleBody, classify_sweep
from navigation_control.local import LocalControlConfig, LocalStatus, LocalTrajectoryController
from navigation_control.types import ControlCommand, VehicleState


class DriveMode(str, Enum):
    STARTUP = "startup"
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
        self.progress = RouteProgress(route)
        self.ego = EgoMotion(self.progress.plane, config.ego)
        self.grid = TerrainGrid(config.terrain)
        self.corridor = CorridorEstimator(config.corridor)
        self.planner = LanePlanner(config.planner, config.body)
        self.controller = LocalTrajectoryController(config.control)
        self._last_time = None
        self._finished = False

    def _brake(self, mode, reason, diagnostics):
        return DriverStep(ControlCommand(0.0, 0.0, HOLD), mode, reason, diagnostics)

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
            self.grid.add(classified, motion.odometry, t)
        d = dict(
            time_s=t,
            odometry=(motion.odometry.x_m, motion.odometry.y_m, motion.odometry.yaw_rad),
            route_pose=None if motion.route is None else (motion.route.x_m, motion.route.y_m, motion.route.yaw_rad),
            route_sigma_m=motion.route_sigma_m,
            speed_mps=motion.speed_mps,
            gyro_bias_rps=motion.gyro_bias_rps,
        )
        if self._finished:
            return self._brake(DriveMode.FINISHED, "goal reached", d)
        if not motion.localised:
            if classified is not None and motion.stationary:
                self.corridor.observe_start(classified)
            return self._brake(DriveMode.STARTUP, "averaging GNSS and compass at standstill", d)
        stale = self.health.stale([self.lidar], c.lidar_max_age_s)
        corridor = self.corridor.update(self.grid, motion.odometry, t)
        d.update(
            corridor_ok=corridor.ok,
            corridor_reason=corridor.reason,
            corridor_valid_m=corridor.valid_length_m,
            lane_offset_m=corridor.lateral_offset_at(0.0) if corridor.ok else None,
            corridor_evidence=int(len(corridor.evidence_body)),
            right_edge_5m=corridor.right_edge_from_centre_m(5.0),
            right_kerb_body=corridor.right_edge_body(5.0),
            keep_right_m=self.corridor.keep_right_m,
            keep_right_source=self.corridor.keep_right_source,
        )
        guidance = self.progress.update(motion.route.x_m, motion.route.y_m)
        d.update(route_index=guidance.index, goal_distance_m=guidance.goal_distance_m, route_finished=guidance.finished)
        if stale:
            return self._brake(DriveMode.FAULT, f"LiDAR older than {c.lidar_max_age_s:.2f} s", d)
        if not corridor.ok:
            return self._brake(DriveMode.FAULT, f"no corridor: {corridor.reason}", d)
        obstacle = None
        if classified is not None:
            obstacle = path_obstacle(classified.of(PointClass.OBSTACLE), corridor, c.body, c.obstacles)
        goal_body = None
        if guidance.index == len(self.progress.xy) - 1 and guidance.goal_distance_m < corridor.valid_length_m + 5:
            goal_body = motion.odometry.to_body(motion.route_to_odometry(np.array([self.progress.xy[-1]])))[0]
        plan = self.planner.plan(corridor, obstacle, goal_body, t)
        d.update(
            obstacle_m=None if obstacle is None else obstacle.distance_m,
            obstacle_points=None if obstacle is None else obstacle.points,
            obstacle_lateral_m=None if obstacle is None else obstacle.lateral_m,
            stop_reason=plan.stop_reason,
            stop_distance_m=plan.stop_distance_m if math.isfinite(plan.stop_distance_m) else None,
        )
        if (
            plan.stop_reason in ("obstacle", "obstacle_hold")
            and plan.stop_distance_m < c.hold_within_m
            and motion.stationary
        ):
            # Stopped at the stop point: hold the brake rather than let the speed loop
            # creep. Driving resumes when the obstacle has been clear for resume_clear_s.
            return self._brake(DriveMode.STOPPING, f"holding for {plan.stop_reason}", d)
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
            return self._brake(DriveMode.FAULT, f"trajectory refused: {acceptance.reason}", d)
        if result.status is LocalStatus.FINISHED or guidance.finished and motion.stationary:
            self._finished = True
            return self._brake(DriveMode.FINISHED, "goal reached", d)
        mode = DriveMode.STOPPING if result.status in (LocalStatus.STOPPING, LocalStatus.EXPIRED) else DriveMode.DRIVING
        return DriverStep(result.command, mode, plan.stop_reason, d)

