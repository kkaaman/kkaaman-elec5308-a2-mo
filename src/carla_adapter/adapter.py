"""CARLA-specific translation between CARLA and navigation_control types."""
import math

import carla

from navigation_control.types import ControlCommand, VehicleState


def carla_vehicle_to_state(
    vehicle: carla.Vehicle,
    rear_axle_offset_m: float = 0.0,
) -> VehicleState:
    """Convert CARLA vehicle data into the controller's shared world frame."""
    if rear_axle_offset_m < 0:
        raise ValueError("rear_axle_offset_m must be nonnegative")

    transform = vehicle.get_transform()
    location = transform.location
    velocity = vehicle.get_velocity()

    # CARLA: +Y right, left-handed.
    # Controller: +Y left, right-handed.
    yaw_rad = -math.radians(transform.rotation.yaw)
    x_m = location.x
    y_m = -location.y

    # Controller expects rear-axle position.
    x_m -= rear_axle_offset_m * math.cos(yaw_rad)
    y_m -= rear_axle_offset_m * math.sin(yaw_rad)

    speed_mps = math.sqrt(
        velocity.x**2 + velocity.y**2 + velocity.z**2
    )

    return VehicleState(
        x_m=x_m,
        y_m=y_m,
        yaw_rad=yaw_rad,
        speed_mps=speed_mps,
    )


def command_to_carla_control(
    command: ControlCommand,
    max_steering_angle_rad: float,
) -> carla.VehicleControl:
    """Convert one controller command into CARLA actuator controls."""
    if max_steering_angle_rad <= 0:
        raise ValueError("max_steering_angle_rad must be positive")

    # Controller positive steering = left; CARLA positive steering = right.
    steer = -command.steering_angle_rad / max_steering_angle_rad
    steer = max(-1.0, min(1.0, steer))

    return carla.VehicleControl(
        throttle=command.throttle,
        steer=steer,
        brake=command.brake,
        hand_brake=False,
        reverse=False,
        manual_gear_shift=False,
    )


def apply_command(
    vehicle: carla.Vehicle,
    command: ControlCommand,
    max_steering_angle_rad: float,
) -> carla.VehicleControl:
    """Convert and apply a controller command to a CARLA vehicle."""
    control = command_to_carla_control(command, max_steering_angle_rad)
    vehicle.apply_control(control)
    return control
