"""Coordinate path progress, lateral control, longitudinal control and stopping."""

import math
from typing import Iterable, Optional

from ._validation import clamp, finite, nonnegative, positive
from .path import ReferencePath
from .speed import SpeedPID
from .steering import PurePursuit
from .types import (
    ControlCommand,
    ControlDiagnostics,
    ControllerConfig,
    ControlResult,
    ControlStatus,
    VehicleState,
    Waypoint,
)


class PathFollower:
    """Stateful forward-only controller independent of any simulator API.

    One instance controls one vehicle/episode; it is not thread-safe. Set a route
    before update. Invalid input raises ValueError/TypeError; no-route use raises
    RuntimeError. The caller, not this library, must stop actuators on exceptions.
    """

    def __init__(self, config: ControllerConfig) -> None:
        if not isinstance(config, ControllerConfig):
            raise TypeError("config must be a ControllerConfig")
        self.config = config
        self._path: Optional[ReferencePath] = None
        self._steering = PurePursuit(config)
        self._speed = SpeedPID(config)
        self.reset()

    def set_path(self, waypoints: Iterable[Waypoint]) -> None:
        """Copy a valid open path and reset history; rejected paths leave state intact."""
        path = ReferencePath(waypoints)
        self._path = path
        self.reset()

    def reset(self) -> None:
        """Clear progress, completion and PID history while retaining path/config."""
        self._progress_m = 0.0
        self._finished = False
        self._speed.reset()

    def update(
        self,
        state: VehicleState,
        target_speed_mps: float,
        dt_s: float,
        *,
        completion_allowed: bool = True,
    ) -> ControlResult:
        """Calculate one command from a rear-axle state and caller's speed ceiling.

        The endpoint speed ceiling uses sqrt(2*a*remaining_braking_distance).
        Completion is latched until reset/set_path. The caller can suppress a
        new completion latch during an obstacle hold with ``completion_allowed``.
        This does not release an already latched completion or change endpoint
        braking. A zero requested speed pauses motion away from the goal.
        """
        if not isinstance(state, VehicleState):
            raise TypeError("state must be a VehicleState")
        nonnegative("target_speed_mps", target_speed_mps)
        positive("dt_s", dt_s)
        if type(completion_allowed) is not bool:
            raise TypeError("completion_allowed must be bool")
        if self._path is None:
            raise RuntimeError("set_path() must be called before update()")

        path, config = self._path, self.config
        projection = path.project(state.x_m, state.y_m)
        progress = max(self._progress_m, projection.progress_m)
        remaining = max(0.0, path.length_m - progress)
        goal = path.points[-1]
        goal_distance = math.hypot(goal.x_m - state.x_m, goal.y_m - state.y_m)
        # A short route must not be declared finished at its starting point.
        near_end = progress >= max(path.length_m * 0.5, path.length_m - config.goal_tolerance_m)
        finished = self._finished or (
            completion_allowed
            and near_end
            and goal_distance <= config.goal_tolerance_m
            and state.speed_mps <= config.stopped_speed_mps
        )
        margin = min(config.goal_tolerance_m * 0.5, path.length_m * 0.1)
        braking_distance = max(0.0, remaining - margin)
        speed_cap = math.sqrt(2.0 * config.stopping_deceleration_mps2 * braking_distance)
        finite("endpoint speed limit", speed_cap)
        internal_speed = 0.0 if finished else min(target_speed_mps, speed_cap)
        target = path.point_at(min(path.length_m, progress + config.lookahead_m))

        if finished:
            status = ControlStatus.FINISHED
        elif internal_speed == 0 or speed_cap < target_speed_mps:
            status = ControlStatus.STOPPING
        else:
            status = ControlStatus.TRACKING

        # A zero speed request must not keep accelerating due to an old integral.
        if internal_speed == 0:
            angle = 0.0
            # Keep following a curve while braking for a temporary stop. At the
            # route end, do not chase a target that may already be behind us.
            if not finished and state.speed_mps > config.stopped_speed_mps and remaining > margin:
                angle = self._steering.steering_angle(state, target)
            command = ControlCommand(
                steering_angle_rad=angle,
                throttle=0.0,
                brake=max(config.hold_brake, clamp(config.kp * state.speed_mps, 0.0, 1.0)),
            )
            self._speed.reset()
        else:
            angle = self._steering.steering_angle(state, target)
            effort = self._speed.update(state.speed_mps, internal_speed, dt_s)
            command = ControlCommand(angle, max(0.0, effort), max(0.0, -effort))

        self._progress_m = progress
        self._finished = finished
        diagnostics = ControlDiagnostics(
            cross_track_error_m=projection.distance_m,
            speed_error_mps=internal_speed - state.speed_mps,
            progress_m=progress,
            progress_fraction=progress / path.length_m,
            remaining_distance_m=remaining,
            internal_target_speed_mps=internal_speed,
            distance_to_goal_m=goal_distance,
            target_waypoint=target,
        )
        return ControlResult(command, diagnostics, status)
