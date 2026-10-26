"""Follow a moving-horizon local trajectory, for a planner that replans several times a second.

The baseline ``PathFollower`` is built for one fixed route. Its ``set_path()`` clears
the speed PID and progress, and its stopping logic treats the path end as the
destination. Feeding it a new short path every planning cycle would discard the PID
integral on every update and could latch "finished" at the end of a horizon.
``LocalTrajectoryController`` fixes both problems.

**Updates keep controller state.** ``set_local_trajectory()`` keeps the speed PID. A
change of ``manoeuvre_id`` only restarts progress along the new trajectory, because that
is the only state the change invalidates.

**The horizon end is not the goal.** Only a trajectory with ``final_goal=True`` can end
the drive. For any other trajectory the speed is capped so the car can still stop
``stop_margin_m`` before the trajectory's end. That is a physical limit: never outrun
the path you were given. The planner avoids horizon braking by sending horizons longer
than ``minimum_horizon_m(speed)``.

**Trajectories expire.** After ``valid_until_s`` the controller brakes along the last
valid trajectory and reports EXPIRED. It never keeps driving on a stale plan.

**Implausible trajectories are refused.** A trajectory that starts far from the car, or
has already expired, is rejected, and the previous one is kept.

Frames: the planner's local odometry frame. ``VehicleState`` must be in that same frame
and refer to the rear axle.
"""

import math
from dataclasses import dataclass, field
from enum import Enum

from ._validation import clamp, finite, nonnegative, positive
from .path import ReferencePath
from .speed import SpeedPID
from .steering import PurePursuit
from .types import ControlCommand, ControllerConfig, VehicleState, Waypoint


@dataclass(frozen=True)
class TrajectorySample:
    x_m: float
    y_m: float
    speed_mps: float

    def __post_init__(self):
        finite("x_m", self.x_m)
        finite("y_m", self.y_m)
        nonnegative("speed_mps", self.speed_mps)


@dataclass(frozen=True)
class LocalTrajectory:
    """Timed path with a speed profile, valid until ``valid_until_s`` (simulation time)."""

    samples: tuple
    created_s: float
    valid_until_s: float
    manoeuvre_id: str = "lane_follow"
    final_goal: bool = False

    def __post_init__(self):
        samples = tuple(self.samples)
        if len(samples) < 2 or any(not isinstance(s, TrajectorySample) for s in samples):
            raise ValueError("A trajectory needs at least two TrajectorySample points")
        nonnegative("created_s", self.created_s)
        finite("valid_until_s", self.valid_until_s)
        if self.valid_until_s <= self.created_s:
            raise ValueError("valid_until_s must be after created_s")
        if not isinstance(self.manoeuvre_id, str) or not self.manoeuvre_id.strip():
            raise ValueError("manoeuvre_id must be a non-empty string")
        if self.final_goal and samples[-1].speed_mps != 0:
            raise ValueError("A final-goal trajectory must end at zero speed")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "final_goal", bool(self.final_goal))


class LocalStatus(str, Enum):
    TRACKING = "tracking"
    LIMITED = "limited"  # slowed by the horizon end or an external speed limit
    STOPPING = "stopping"  # zero speed requested by the trajectory
    EXPIRED = "expired"  # braking along a trajectory past its validity
    FINISHED = "finished"  # final goal reached (latched)
    NO_TRAJECTORY = "no_trajectory"


@dataclass(frozen=True)
class LocalControlConfig:
    controller: ControllerConfig
    max_start_offset_m: float = 2.0
    stop_margin_m: float = 1.0
    speed_lookahead_m: float = 1.0
    # Below ``coast_below_mps`` and short of a stop, brake efforts under ``brake_deadband``
    # are coasting, not braking (see tracking).
    brake_deadband: float = 0.3
    coast_below_mps: float = 1.5

    def __post_init__(self):
        if not isinstance(self.controller, ControllerConfig):
            raise TypeError("controller must be a ControllerConfig")
        positive("max_start_offset_m", self.max_start_offset_m)
        nonnegative("stop_margin_m", self.stop_margin_m)
        nonnegative("speed_lookahead_m", self.speed_lookahead_m)
        nonnegative("brake_deadband", self.brake_deadband)
        nonnegative("coast_below_mps", self.coast_below_mps)
        if self.brake_deadband >= 1:
            raise ValueError("brake_deadband must be below 1")

    def minimum_horizon_m(self, speed_mps):
        """Shortest non-final horizon that does not cap ``speed_mps`` (plus the lookahead)."""
        nonnegative("speed_mps", speed_mps)
        a = self.controller.stopping_deceleration_mps2
        return speed_mps**2 / (2 * a) + self.stop_margin_m + self.speed_lookahead_m


@dataclass(frozen=True)
class Acceptance:
    accepted: bool
    reason: str


@dataclass(frozen=True)
class LocalDiagnostics:
    cross_track_error_m: float
    progress_m: float
    remaining_m: float
    profile_speed_mps: float
    horizon_cap_mps: float
    target_speed_mps: float
    trajectory_age_s: float
    manoeuvre_id: str
    target_waypoint: Waypoint | None


@dataclass(frozen=True)
class LocalControlResult:
    command: ControlCommand
    status: LocalStatus
    diagnostics: LocalDiagnostics | None = None
    rejections: tuple = field(default_factory=tuple)


class LocalTrajectoryController:
    """Stateful controller for one vehicle and one episode. Not thread-safe."""

    def __init__(self, config):
        if not isinstance(config, LocalControlConfig):
            raise TypeError("config must be a LocalControlConfig")
        self.config = config
        self._steering = PurePursuit(config.controller)
        self._speed = SpeedPID(config.controller)
        self._trajectory = None
        self._path = None
        self._progress_m = 0.0
        self._finished = False

    def reset(self):
        """Episode start: forget the trajectory, progress, completion and PID history."""
        self._trajectory = self._path = None
        self._progress_m = 0.0
        self._finished = False
        self._speed.reset()

    @property
    def trajectory(self):
        return self._trajectory

    def set_local_trajectory(self, trajectory, state, now_s):
        """Adopt a new trajectory if it is plausible; otherwise keep the current one."""
        if not isinstance(trajectory, LocalTrajectory):
            raise TypeError("trajectory must be a LocalTrajectory")
        if not isinstance(state, VehicleState):
            raise TypeError("state must be a VehicleState")
        finite("now_s", now_s)
        if now_s > trajectory.valid_until_s:
            return Acceptance(False, "already expired")
        path = ReferencePath(Waypoint(s.x_m, s.y_m) for s in trajectory.samples)
        projection = path.project(state.x_m, state.y_m)
        if projection.distance_m > self.config.max_start_offset_m:
            return Acceptance(False, f"starts {projection.distance_m:.2f} m from the car")
        previous = self._trajectory
        self._trajectory, self._path = trajectory, path
        self._progress_m = projection.progress_m
        if previous is None or previous.manoeuvre_id != trajectory.manoeuvre_id or not trajectory.final_goal:
            self._finished = False
        return Acceptance(True, "accepted")

    def _profile_speed(self, progress_m):
        path, samples = self._path, self._trajectory.samples
        s = clamp(progress_m, 0.0, path.length_m)
        arcs = path.arc_lengths
        # ReferencePath drops consecutive duplicate points; samples line up otherwise.
        if len(arcs) != len(samples):
            return min(sample.speed_mps for sample in samples)
        for i in range(len(arcs) - 1):
            if s <= arcs[i + 1]:
                ratio = (s - arcs[i]) / (arcs[i + 1] - arcs[i])
                return samples[i].speed_mps + ratio * (samples[i + 1].speed_mps - samples[i].speed_mps)
        return samples[-1].speed_mps

    def _brake(self, state):
        c = self.config.controller
        self._speed.reset()  # the cruise integral is invalid once a stop is requested
        return ControlCommand(0.0, 0.0, max(c.hold_brake, clamp(c.kp * state.speed_mps, 0.0, 1.0)))

    def update(self, state, now_s, dt_s, speed_limit_mps=math.inf):
        if not isinstance(state, VehicleState):
            raise TypeError("state must be a VehicleState")
        finite("now_s", now_s)
        positive("dt_s", dt_s)
        if not (speed_limit_mps == math.inf or (math.isfinite(speed_limit_mps) and speed_limit_mps >= 0)):
            raise ValueError("speed_limit_mps must be non-negative")
        if self._trajectory is None:
            return LocalControlResult(self._brake(state), LocalStatus.NO_TRAJECTORY)
        c, path, trajectory = self.config.controller, self._path, self._trajectory
        projection = path.project(state.x_m, state.y_m)
        progress = max(self._progress_m, projection.progress_m)
        remaining = max(0.0, path.length_m - progress)
        profile = self._profile_speed(progress + self.config.speed_lookahead_m)
        margin = c.goal_tolerance_m * 0.5 if trajectory.final_goal else self.config.stop_margin_m
        cap = math.sqrt(2.0 * c.stopping_deceleration_mps2 * max(0.0, remaining - margin))
        expired = now_s > trajectory.valid_until_s
        if trajectory.final_goal and not self._finished:
            self._finished = remaining <= c.goal_tolerance_m and state.speed_mps <= c.stopped_speed_mps
        target_speed = 0.0 if (expired or self._finished) else min(profile, cap, speed_limit_mps)
        lookahead = path.point_at(min(path.length_m, progress + c.lookahead_m))
        angle = 0.0
        if state.speed_mps > c.stopped_speed_mps or target_speed > 0:
            try:
                angle = self._steering.steering_angle(state, lookahead)
            except ValueError:
                angle = 0.0  # lookahead behind the car at the very end: hold the wheel straight
        if target_speed == 0:
            command = self._brake(state)
            command = ControlCommand(angle if state.speed_mps > c.stopped_speed_mps else 0.0, 0.0, command.brake)
        else:
            effort = self._speed.update(state.speed_mps, target_speed, dt_s)
            brake = max(0.0, -effort)
            if (c.stopped_speed_mps < state.speed_mps < self.config.coast_below_mps
                    and target_speed > c.stopped_speed_mps and brake < self.config.brake_deadband):
                # A trace of brake on a slowly moving car stops it dead in the simulator
                # (the tyres stick below walking pace): a speed excess at walking pace is
                # shed by coasting, not by the pads, unless the target is a stop or the
                # excess is large. The first full live stack run sawtoothed between 0 and
                # 1 m/s on brake commands of 0.006.
                brake = 0.0
            command = ControlCommand(angle, max(0.0, effort), brake)
        self._progress_m = progress
        if self._finished:
            status = LocalStatus.FINISHED
        elif expired:
            status = LocalStatus.EXPIRED
        elif target_speed == 0:
            status = LocalStatus.STOPPING
        elif target_speed < profile - 1e-9:
            status = LocalStatus.LIMITED
        else:
            status = LocalStatus.TRACKING
        diagnostics = LocalDiagnostics(
            cross_track_error_m=projection.distance_m,
            progress_m=progress,
            remaining_m=remaining,
            profile_speed_mps=profile,
            horizon_cap_mps=cap,
            target_speed_mps=target_speed,
            trajectory_age_s=now_s - trajectory.created_s,
            manoeuvre_id=trajectory.manoeuvre_id,
            target_waypoint=lookahead,
        )
        return LocalControlResult(command, status, diagnostics)
