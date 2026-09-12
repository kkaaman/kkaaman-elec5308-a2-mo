"""Unit tests for the CARLA adapter.

These tests use fake CARLA-like objects, so no running simulator is required.
"""

import math
from types import SimpleNamespace

import pytest

from carla_adapter.adapter import (
    carla_vehicle_to_state,
    command_to_carla_control,
)
from navigation_control.types import ControlCommand


class FakeVehicle:
    """Small stand-in for the CARLA vehicle methods used by the adapter."""

    def __init__(self, x, y, yaw_deg, vx, vy, vz=0.0):
        self._transform = SimpleNamespace(
            location=SimpleNamespace(x=x, y=y),
            rotation=SimpleNamespace(yaw=yaw_deg),
        )
        self._velocity = SimpleNamespace(x=vx, y=vy, z=vz)

    def get_transform(self):
        return self._transform

    def get_velocity(self):
        return self._velocity


def test_vehicle_state_converts_carla_coordinates_and_speed():
    vehicle = FakeVehicle(x=10.0, y=4.0, yaw_deg=90.0, vx=3.0, vy=4.0)

    state = carla_vehicle_to_state(vehicle)

    assert state.x_m == pytest.approx(10.0)
    assert state.y_m == pytest.approx(-4.0)
    assert state.yaw_rad == pytest.approx(-math.pi / 2)
    assert state.speed_mps == pytest.approx(5.0)


def test_vehicle_state_applies_rear_axle_offset():
    vehicle = FakeVehicle(x=10.0, y=4.0, yaw_deg=0.0, vx=0.0, vy=0.0)

    state = carla_vehicle_to_state(vehicle, rear_axle_offset_m=2.0)

    assert state.x_m == pytest.approx(8.0)
    assert state.y_m == pytest.approx(-4.0)


def test_control_command_maps_left_steering_to_carla():
    command = ControlCommand(
        steering_angle_rad=0.1,
        throttle=0.2,
        brake=0.0,
    )

    control = command_to_carla_control(command, max_steering_angle_rad=0.5)

    assert control.throttle == pytest.approx(0.2)
    assert control.brake == pytest.approx(0.0)
    assert control.steer == pytest.approx(-0.2)


def test_control_command_clamps_steering_to_carla_range():
    command = ControlCommand(
        steering_angle_rad=-2.0,
        throttle=0.0,
        brake=0.3,
    )

    control = command_to_carla_control(command, max_steering_angle_rad=0.5)

    assert control.steer == pytest.approx(1.0)
    assert control.brake == pytest.approx(0.3)


def test_invalid_max_steering_angle_is_rejected():
    command = ControlCommand(steering_angle_rad=0.0, throttle=0.0, brake=0.0)

    with pytest.raises(ValueError, match="max_steering_angle_rad"):
        command_to_carla_control(command, max_steering_angle_rad=0.0)