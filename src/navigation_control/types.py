"""Immutable public input, output and configuration types."""

import math
from dataclasses import dataclass
from enum import Enum

from ._validation import finite, nonnegative, positive


@dataclass(frozen=True)
class VehicleState:
    """Rear-axle pose and forward speed in a shared planar world frame.

    ``x_m`` and ``y_m`` are metres; ``yaw_rad`` is counterclockwise from +X.
    ``speed_mps`` is nonnegative. Angles need not be wrapped to [-pi, pi].
    """

    x_m: float
    y_m: float
    yaw_rad: float
    speed_mps: float

    def __post_init__(self) -> None:
        for name in ("x_m", "y_m", "yaw_rad"):
            finite(name, getattr(self, name))
        nonnegative("speed_mps", self.speed_mps)


@dataclass(frozen=True)
class Waypoint:
    """A path position in metres, using the same world frame as VehicleState."""

    x_m: float
    y_m: float

    def __post_init__(self) -> None:
        finite("x_m", self.x_m)
        finite("y_m", self.y_m)


@dataclass(frozen=True)
class ControllerConfig:
    """Vehicle geometry and tuning, fixed for the lifetime of a controller.

    Geometry must be supplied for the chosen vehicle. Remaining defaults are
    illustrative baseline tuning, not universal or safety-certified settings.
    PID gains map speed error to a dimensionless signed pedal effort.
    """

    wheelbase_m: float
    max_steering_angle_rad: float
    lookahead_m: float = 5.0
    kp: float = 0.4
    ki: float = 0.05
    kd: float = 0.0
    stopping_deceleration_mps2: float = 1.0
    goal_tolerance_m: float = 1.0
    stopped_speed_mps: float = 0.1
    hold_brake: float = 0.2

    def __post_init__(self) -> None:
        for name in (
            "wheelbase_m",
            "max_steering_angle_rad",
            "lookahead_m",
            "stopping_deceleration_mps2",
            "goal_tolerance_m",
            "stopped_speed_mps",
            "hold_brake",
        ):
            positive(name, getattr(self, name))
        if self.max_steering_angle_rad >= math.pi / 2:
            raise ValueError("max_steering_angle_rad must be less than pi/2")
        if self.hold_brake > 1:
            raise ValueError("hold_brake must be in (0, 1]")
        for name in ("kp", "ki", "kd"):
            nonnegative(name, getattr(self, name))
        if self.kp == self.ki == self.kd == 0:
            raise ValueError("at least one PID gain must be positive")


class ControlStatus(str, Enum):
    """TRACKING, STOPPING (including temporary stops), or latched FINISHED."""

    TRACKING = "tracking"
    STOPPING = "stopping"
    FINISHED = "finished"


@dataclass(frozen=True)
class ControlCommand:
    """Front-wheel-equivalent steering angle and mutually exclusive pedals.

    Positive steering means left, in radians. Throttle and brake are each in
    [0, 1]. An adapter must translate these to its vehicle's actuator conventions.
    """

    steering_angle_rad: float
    throttle: float
    brake: float

    def __post_init__(self) -> None:
        finite("steering_angle_rad", self.steering_angle_rad)
        for name in ("throttle", "brake"):
            nonnegative(name, getattr(self, name))
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.throttle > 0 and self.brake > 0:
            raise ValueError("throttle and brake cannot both be positive")


@dataclass(frozen=True)
class ControlDiagnostics:
    """Measurements before applying the returned command.

    Cross-track error is the nonnegative distance to the nearest point on the
    path polyline (including endpoints). Speed error is internal target minus
    measured speed. Progress is monotonic arc length, not proof of successful
    completion. ``target_waypoint`` is the interpolated Pure Pursuit target.
    """

    cross_track_error_m: float
    speed_error_mps: float
    progress_m: float
    progress_fraction: float
    remaining_distance_m: float
    internal_target_speed_mps: float
    distance_to_goal_m: float
    target_waypoint: Waypoint


@dataclass(frozen=True)
class ControlResult:
    """One control-cycle command, diagnostics and execution status."""

    command: ControlCommand
    diagnostics: ControlDiagnostics
    status: ControlStatus
