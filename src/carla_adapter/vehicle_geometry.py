"""Validated fixed-vehicle geometry and simulator-free coordinate conversions."""

import math
from dataclasses import dataclass

import numpy as np

from navigation_control.types import ControlCommand, VehicleState


def _finite(*values):
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Values must be finite")


@dataclass(frozen=True)
class VehicleGeometry:
    wheelbase_m: float
    rear_axle_offset_m: float
    physical_steering_limit_rad: float
    controller_steering_limit_rad: float

    def __post_init__(self):
        _finite(
            self.wheelbase_m,
            self.rear_axle_offset_m,
            self.physical_steering_limit_rad,
            self.controller_steering_limit_rad,
        )
        if not 1.0 < self.wheelbase_m < 5.0:
            raise ValueError("Implausible passenger-car wheelbase")
        if not 0.0 < self.rear_axle_offset_m < self.wheelbase_m:
            raise ValueError("Rear axle offset must lie inside the wheelbase")
        if (
            not 0
            < self.controller_steering_limit_rad
            <= self.physical_steering_limit_rad
            < math.pi / 2
        ):
            raise ValueError("Invalid steering limits")


def geometry_from_wheels(
    world_positions_cm, world_to_actor, max_steer_degrees, controller_limit_rad
):
    """Interpret installed-build wheel positions as world centimetres.

    Requires FL, FR, RL, RR order, a proper rigid inverse actor transform and
    front-steering/rear-fixed wheels. Ambiguous or implausible data raises.
    """
    positions = np.asarray(world_positions_cm, dtype=float)
    matrix = np.asarray(world_to_actor, dtype=float)
    angles = np.asarray(max_steer_degrees, dtype=float)
    if positions.shape != (4, 3) or matrix.shape != (4, 4) or angles.shape != (4,):
        raise ValueError("Expected four wheels and a 4x4 transform")
    if not all(np.isfinite(a).all() for a in (positions, matrix, angles)):
        raise ValueError("Geometry must be finite")
    rotation = matrix[:3, :3]
    if (
        not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
        or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)
    ):
        raise ValueError("Transform must be a proper rigid transform")
    local = (matrix @ np.column_stack((positions / 100, np.ones(4))).T).T[:, :3]
    if abs(local[0, 0] - local[1, 0]) > 0.03 or abs(local[2, 0] - local[3, 0]) > 0.03:
        raise ValueError("Wheel pairs do not define aligned axles")
    front = local[:2].mean(axis=0)
    rear = local[2:].mean(axis=0)
    for pair in (local[:2], local[2:]):
        if not (pair[0, 1] < 0 < pair[1, 1] and 0.8 < pair[1, 1] - pair[0, 1] < 2.5):
            raise ValueError("Invalid left/right wheel order or track width")
    if abs(front[1]) > 0.03 or abs(rear[1]) > 0.03 or front[0] <= 0 or rear[0] >= 0:
        raise ValueError("Actor origin must lie between centred axles")
    if not (
        0 < angles[0] < 90
        and abs(angles[0] - angles[1]) < 0.1
        and np.all(np.abs(angles[2:]) < 0.1)
    ):
        raise ValueError("Expected symmetric front steering and fixed rear wheels")
    return VehicleGeometry(
        float(front[0] - rear[0]),
        float(-rear[0]),
        math.radians(float(angles[:2].mean())),
        controller_limit_rad,
    )


def forward_state(x, y, yaw_deg, vx, vy, geometry, reverse_tolerance_mps=0.05):
    """CARLA planar actor pose -> right-handed rear-axle state.

    Speed is signed longitudinal planar velocity; reverse motion beyond a small
    numerical tolerance is rejected rather than hidden by a velocity magnitude.
    """
    _finite(x, y, yaw_deg, vx, vy, reverse_tolerance_mps)
    if reverse_tolerance_mps < 0:
        raise ValueError("Reverse tolerance must be nonnegative")
    yaw = math.radians(yaw_deg)
    speed = vx * math.cos(yaw) + vy * math.sin(yaw)
    if speed < -reverse_tolerance_mps:
        raise ValueError("Reverse motion is outside the forward-only mission")
    return VehicleState(
        x - geometry.rear_axle_offset_m * math.cos(yaw),
        -y + geometry.rear_axle_offset_m * math.sin(yaw),
        -yaw,
        max(0.0, speed),
    )


def normalized_steering(command: ControlCommand, geometry: VehicleGeometry):
    """Nominal low-speed mapping; speed-dependent response needs BL-E-02 calibration.

    Controller cap and physical full-scale angle are deliberately distinct.
    Commands exceeding the controller cap are rejected, never silently clipped.
    """
    if abs(command.steering_angle_rad) > geometry.controller_steering_limit_rad + 1e-12:
        raise ValueError("Command exceeds controller steering cap")
    return -command.steering_angle_rad / geometry.physical_steering_limit_rad
