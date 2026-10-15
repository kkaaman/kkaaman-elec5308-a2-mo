"""Geometry, units, frame and actuator regression tests without a CARLA server."""

import math

import numpy as np
import pytest

from carla_adapter.vehicle_geometry import (
    VehicleGeometry,
    forward_state,
    geometry_from_wheels,
    normalized_steering,
)
from navigation_control.types import ControlCommand


@pytest.fixture
def geometry():
    return VehicleGeometry(3.00465, 1.38641, math.radians(70), math.radians(20))


@pytest.mark.parametrize("yaw", [0, 90, -90, 180])
def test_geometry_at_multiple_world_headings(yaw):
    a = math.radians(yaw)
    rotation = np.array(
        [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
    )
    origin = np.array([-64, 24, 0.6])
    local = np.array(
        [
            [1.61824, -0.8334, 0.3],
            [1.61824, 0.8334, 0.3],
            [-1.38641, -0.8334, 0.3],
            [-1.38641, 0.8334, 0.3],
        ]
    )
    world_cm = (local @ rotation.T + origin) * 100
    inverse = np.eye(4)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ origin
    measured = geometry_from_wheels(world_cm, inverse, [70, 70, 0, 0], math.radians(20))
    assert measured.wheelbase_m == pytest.approx(3.00465)
    assert measured.rear_axle_offset_m == pytest.approx(1.38641)


@pytest.mark.parametrize("yaw", [0, 90, -90, 180])
def test_rear_pose_and_signed_speed(yaw, geometry):
    a = math.radians(yaw)
    state = forward_state(10, 20, yaw, 2 * math.cos(a), 2 * math.sin(a), geometry)
    assert state.x_m == pytest.approx(10 - 1.38641 * math.cos(a))
    assert state.y_m == pytest.approx(-20 + 1.38641 * math.sin(a))
    assert state.yaw_rad == pytest.approx(-a)
    assert state.speed_mps == pytest.approx(2)


def test_reverse_rejected_and_lateral_velocity_not_speed(geometry):
    with pytest.raises(ValueError, match="Reverse"):
        forward_state(0, 0, 0, -1, 0, geometry)
    assert forward_state(0, 0, 0, 0, 5, geometry).speed_mps == 0
    assert forward_state(0, 0, 0, -0.01, 0, geometry).speed_mps == 0


def test_controller_cap_is_not_actuator_full_scale(geometry):
    left = ControlCommand(math.radians(20), 0.2, 0)
    assert normalized_steering(left, geometry) == pytest.approx(-20 / 70)
    right = ControlCommand(-math.radians(20), 0, 1)
    assert normalized_steering(right, geometry) == pytest.approx(20 / 70)
    with pytest.raises(ValueError, match="cap"):
        normalized_steering(ControlCommand(math.radians(21), 0, 1), geometry)


@pytest.mark.parametrize(
    "fault", ["metres", "swapped", "nan", "reflection", "rear_steer"]
)
def test_ambiguous_geometry_rejected(fault):
    points = np.array(
        [
            [161.824, -83.34, 30],
            [161.824, 83.34, 30],
            [-138.641, -83.34, 30],
            [-138.641, 83.34, 30],
        ]
    )
    matrix = np.eye(4)
    angles = [70, 70, 0, 0]
    if fault == "metres":
        points /= 100
    elif fault == "swapped":
        points = points[[1, 0, 2, 3]]
    elif fault == "nan":
        points[0, 0] = np.nan
    elif fault == "reflection":
        matrix[1, 1] = -1
    else:
        angles[2] = 10
    with pytest.raises(ValueError):
        geometry_from_wheels(points, matrix, angles, math.radians(20))


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_pose_rejected(value, geometry):
    with pytest.raises(ValueError):
        forward_state(value, 0, 0, 0, 0, geometry)


def test_adapter_brakes_on_reverse_or_oversized_command(monkeypatch):
    import sys
    from types import SimpleNamespace

    from carla_adapter.vehicle import FixedVehicleAdapter

    controls = []

    def position(x, y, z):
        return SimpleNamespace(x=x, y=y, z=z)

    wheels = [
        SimpleNamespace(position=position(x, y, 30), max_steer_angle=angle)
        for x, y, angle in [
            (161.824, -83.34, 70),
            (161.824, 83.34, 70),
            (-138.641, -83.34, 0),
            (-138.641, 83.34, 0),
        ]
    ]
    velocity = position(1, 0, 0)
    pose = SimpleNamespace(
        location=position(0, 0, 0),
        rotation=SimpleNamespace(yaw=0, pitch=0, roll=0),
        get_inverse_matrix=lambda: np.eye(4),
    )
    vehicle = SimpleNamespace(
        type_id="vehicle.tesla.model3",
        is_alive=True,
        get_physics_control=lambda: SimpleNamespace(wheels=wheels),
        get_transform=lambda: pose,
        get_velocity=lambda: velocity,
        apply_control=controls.append,
    )
    monkeypatch.setitem(
        sys.modules, "carla", SimpleNamespace(VehicleControl=SimpleNamespace)
    )
    adapter = FixedVehicleAdapter(vehicle)
    control = adapter.apply(ControlCommand(math.radians(10), 0.2, 0))
    assert control.steer == pytest.approx(-10 / 70)
    assert control.reverse is False
    with pytest.raises(ValueError):
        adapter.apply(ControlCommand(math.radians(21), 0.2, 0))
    assert controls[-1].brake == 1
    velocity.x = -1
    with pytest.raises(ValueError, match="Reverse"):
        adapter.apply(ControlCommand(0, 0.2, 0))
    assert controls[-1].brake == 1
