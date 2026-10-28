"""Opt-in P1 contracts; A1 VehicleState and ControlCommand retain their APIs."""

import math
from dataclasses import dataclass
from enum import Enum

from navigation_control import VehicleState


def finite(name, value, *, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def angle_error(a, b):
    """Signed shortest difference in radians, including the +/-pi boundary."""
    return math.atan2(math.sin(a - b), math.cos(a - b))


class DriveMode(str, Enum):
    DRIVE = "drive"
    PARK = "park"


class ParkingPhase(str, Enum):
    READY = "ready"
    APPROACH = "approach"
    STOPPING = "stopping"
    HOLD = "hold"
    PARKED = "parked"
    FAULT = "fault"


@dataclass(frozen=True)
class MotionState:
    """New signed longitudinal-speed API; P1 execution accepts forward motion only."""

    x_m: float
    y_m: float
    yaw_rad: float
    signed_speed_mps: float
    mode: DriveMode = DriveMode.DRIVE

    def __post_init__(self):
        for name in ("x_m", "y_m", "yaw_rad", "signed_speed_mps"):
            finite(name, getattr(self, name))
        if not isinstance(self.mode, DriveMode):
            raise TypeError("mode must be DriveMode")

    @classmethod
    def from_forward(cls, state, mode=DriveMode.DRIVE):
        if not isinstance(state, VehicleState):
            raise TypeError("Expected VehicleState")
        return cls(state.x_m, state.y_m, state.yaw_rad, state.speed_mps, mode)

    def as_forward(self):
        if self.signed_speed_mps < 0:
            raise ValueError("P1 does not execute reverse motion")
        return VehicleState(self.x_m, self.y_m, self.yaw_rad, self.signed_speed_mps)


@dataclass(frozen=True)
class ParkingVehicle:
    """Ego geometry only; bumper offsets are measured from the rear axle."""

    wheelbase_m: float
    steering_limit_rad: float
    front_m: float
    rear_m: float
    half_width_m: float
    height_m: float

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            finite(name, getattr(self, name), minimum=1e-6)
        if self.steering_limit_rad >= math.pi / 2:
            raise ValueError("steering limit must be below pi/2")
        if self.front_m < self.wheelbase_m:
            raise ValueError("front bumper must cover the front axle")

    @property
    def curvature_limit(self):
        return math.tan(self.steering_limit_rad) / self.wheelbase_m

    @property
    def corner_radius_m(self):
        return math.hypot(max(self.front_m, self.rear_m), self.half_width_m)


@dataclass(frozen=True)
class ParkingBay:
    """Oriented rectangular bay, centred independently of the rear-axle goal."""

    x_m: float
    y_m: float
    yaw_rad: float
    length_m: float
    width_m: float

    def __post_init__(self):
        for name in ("x_m", "y_m", "yaw_rad"):
            finite(name, getattr(self, name))
        for name in ("length_m", "width_m"):
            finite(name, getattr(self, name), minimum=1e-6)


@dataclass(frozen=True)
class ParkingGoal:
    x_m: float
    y_m: float
    yaw_rad: float
    bay: ParkingBay
    clearance_m: float = 0.15
    position_tolerance_m: float = 0.3
    heading_tolerance_rad: float = math.radians(5)

    def __post_init__(self):
        for name in ("x_m", "y_m", "yaw_rad"):
            finite(name, getattr(self, name))
        if not isinstance(self.bay, ParkingBay):
            raise TypeError("bay must be ParkingBay")
        for name in ("clearance_m", "position_tolerance_m", "heading_tolerance_rad"):
            finite(name, getattr(self, name), minimum=1e-6)
        if self.position_tolerance_m > 0.3 or self.heading_tolerance_rad > math.radians(5):
            raise ValueError("P1 tolerances cannot exceed 0.3 m and 5 degrees")


@dataclass(frozen=True)
class ParkingConfig:
    """Explicit harness policy; values are not a CARLA calibration or G1 admission."""

    max_speed_mps: float = 0.8
    lookahead_m: float = 0.8
    dt_s: float = 0.05
    deceleration_mps2: float = 1.0
    response_budget_s: float = 0.4
    stop_margin_m: float = 0.35
    emergency_margin_m: float = 0.1
    acceleration_mps2: float = 0.5
    stopped_speed_mps: float = 0.05
    hold_brake: float = 0.5
    preview_m: float = 4.0
    sample_step_m: float = 0.15
    straight_finish_m: float = 2.0
    tracking_margin_m: float = 0.15
    tracking_heading_rad: float = math.radians(5)
    max_observation_age_s: float = 0.25
    command_ttl_s: float = 0.1
    depth_margin_m: float = 0.02
    ground_exclusion_m: float = 0.08
    max_depth_m: float = 50.0
    max_duration_s: float = 120.0

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            finite(name, getattr(self, name), minimum=1e-6)
        if self.max_speed_mps > 1 or self.hold_brake > 1:
            raise ValueError("P1 speed <=1 m/s and brake <=1 required")
        if self.stopped_speed_mps >= self.max_speed_mps:
            raise ValueError("stopped threshold must be below approach speed")
        if self.emergency_margin_m >= self.stop_margin_m:
            raise ValueError("emergency margin must be below stop margin")
        if self.tracking_heading_rad > math.radians(15) or self.sample_step_m > 0.25:
            raise ValueError("tracking/sampling envelope exceeds P1 bounds")
        if self.command_ttl_s < self.dt_s or self.command_ttl_s > self.max_observation_age_s:
            raise ValueError("require dt <= command TTL <= observation age")
        if self.response_budget_s < self.max_observation_age_s + self.command_ttl_s:
            raise ValueError("response budget must cover observation age and delivery")
        if self.depth_margin_m >= self.ground_exclusion_m:
            raise ValueError("depth margin must be below ground exclusion height")

    def tracking_padding(self, vehicle):
        return self.tracking_margin_m + 2 * vehicle.corner_radius_m * math.sin(
            self.tracking_heading_rad / 2
        )
