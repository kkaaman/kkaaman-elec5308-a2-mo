"""Public API tests using deterministic navigation-control states."""

import math
from dataclasses import replace

import pytest

from navigation_control import (
    ControlCommand,
    ControllerConfig,
    ControlStatus,
    PathFollower,
    VehicleState,
    Waypoint,
)
from navigation_control.steering import PurePursuit


def test_straight_route_returns_zero_steering_and_interpolated_target(
    follower,
):
    result = follower.update(VehicleState(0, 0, 0, 0), 5, 0.05)

    assert result.command.steering_angle_rad == 0
    assert result.command.throttle > 0
    assert result.command.brake == 0
    assert result.diagnostics.target_waypoint == Waypoint(5, 0)
    assert result.status == ControlStatus.TRACKING


@pytest.mark.parametrize("y_offset,expected_sign", [(-1, 1), (1, -1)])
def test_steering_corrects_lateral_offset(follower, y_offset, expected_sign):
    result = follower.update(VehicleState(0, y_offset, 0, 2), 5, 0.05)

    assert result.command.steering_angle_rad * expected_sign > 0
    assert result.diagnostics.cross_track_error_m == pytest.approx(1)


def test_pure_pursuit_uses_actual_target_chord_length(config):
    angle = PurePursuit(config).steering_angle(
        VehicleState(0, 0, 0, 1), Waypoint(4, 1)
    )

    assert angle == pytest.approx(math.atan(2 * 2.8 / 17))


def test_steering_angle_is_limited_to_vehicle_maximum(config):
    angle = PurePursuit(config).steering_angle(
        VehicleState(0, 0, 0, 1), Waypoint(0.1, 1)
    )

    assert angle == config.max_steering_angle_rad


def test_heading_representations_across_pi_boundary_are_equivalent(config):
    steering = PurePursuit(config)

    first = steering.steering_angle(
        VehicleState(0, 0, math.pi - 0.01, 1), Waypoint(-5, 0)
    )
    second = steering.steering_angle(
        VehicleState(0, 0, -math.pi - 0.01, 1), Waypoint(-5, 0)
    )

    assert first == pytest.approx(second)
    assert abs(first) < 0.02


def test_rotated_world_frame_supports_straight_driving(config):
    controller = PathFollower(config)
    controller.set_path([Waypoint(10, 10), Waypoint(10, 100)])

    result = controller.update(VehicleState(10, 10, math.pi / 2, 2), 3, 0.05)

    assert result.command.steering_angle_rad == pytest.approx(0, abs=1e-12)


def test_target_behind_vehicle_is_rejected(follower):
    with pytest.raises(ValueError, match="behind"):
        follower.update(VehicleState(0, 0, math.pi, 0), 5, 0.05)


def test_path_progress_does_not_rewind_from_small_position_noise(follower):
    forward = follower.update(VehicleState(30, 0, 0, 1), 2, 0.05)
    backward_noise = follower.update(VehicleState(29.9, 0, 0, 1), 2, 0.05)

    assert forward.diagnostics.progress_m == 30
    assert backward_noise.diagnostics.progress_m == 30
    assert backward_noise.diagnostics.remaining_distance_m == 70


def test_endpoint_speed_cap_never_exceeds_requested_speed(follower):
    result = follower.update(VehicleState(90, 0, 0, 5), 5, 0.05)

    assert result.status == ControlStatus.STOPPING
    assert result.diagnostics.internal_target_speed_mps == pytest.approx(
        math.sqrt(19)
    )
    assert result.command.brake > 0


def test_finished_state_is_latched_with_holding_brake(follower):
    result = follower.update(VehicleState(99.5, 0, 0, 0.05), 5, 0.05)

    assert result.status == ControlStatus.FINISHED
    assert result.command.throttle == 0
    assert result.command.brake >= follower.config.hold_brake

    later = follower.update(VehicleState(99.5, 0, 0, 0), 10, 0.05)
    assert later.status == ControlStatus.FINISHED
    assert later.diagnostics.internal_target_speed_mps == 0


def test_full_path_projection_is_not_success_when_far_from_goal(follower):
    result = follower.update(VehicleState(100, 5, 0, 0), 5, 0.05)

    assert result.diagnostics.progress_fraction == 1
    assert result.status != ControlStatus.FINISHED
    assert result.command.brake > 0


def test_temporary_stop_clears_integral_and_positive_request_resumes(follower):
    state = VehicleState(10, 0, 0, 1)
    for _ in range(100):
        follower.update(state, 2, 0.05)

    stopped = follower.update(state, 0, 0.05)
    assert stopped.status == ControlStatus.STOPPING
    assert stopped.command.throttle == 0
    assert stopped.command.brake > 0

    held = follower.update(VehicleState(10, 0, 0, 0), 0, 0.05)
    assert held.command.brake >= follower.config.hold_brake

    resumed = follower.update(VehicleState(10, 0, 0, 0), 2, 0.05)
    assert resumed.status == ControlStatus.TRACKING
    assert resumed.command.throttle > 0
    assert resumed.command.brake == 0


def test_temporary_stop_preserves_lateral_control_while_braking(follower):
    result = follower.update(VehicleState(10, -1, 0, 5), 0, 0.05)

    assert result.command.steering_angle_rad > 0
    assert result.command.brake > 0
    assert result.command.throttle == 0


def test_stop_after_overshoot_does_not_chase_a_target_behind_vehicle(follower):
    result = follower.update(VehicleState(101, 0, 0, 2), 0, 0.05)

    assert result.status == ControlStatus.STOPPING
    assert result.command.steering_angle_rad == 0
    assert result.command.brake > 0


def test_reset_retains_path_and_restarts_control_episode(follower):
    initial = follower.update(VehicleState(0, 0, 0, 0), 5, 0.05)
    follower.update(VehicleState(99.5, 0, 0, 0), 5, 0.05)

    follower.reset()

    assert follower.update(VehicleState(0, 0, 0, 0), 5, 0.05) == initial


def test_set_path_replaces_route_and_resets_state(follower):
    follower.update(VehicleState(99.5, 0, 0, 0), 5, 0.05)
    follower.set_path([Waypoint(0, 0), Waypoint(0, 20)])

    result = follower.update(VehicleState(0, 0, math.pi / 2, 0), 2, 0.05)

    assert result.status == ControlStatus.TRACKING
    assert result.diagnostics.remaining_distance_m == 20


def test_rejected_set_path_leaves_previous_route_intact(follower):
    follower.update(VehicleState(10, 0, 0, 1), 2, 0.05)

    with pytest.raises(ValueError):
        follower.set_path([])

    result = follower.update(VehicleState(10, 0, 0, 1), 2, 0.05)
    assert result.diagnostics.progress_m == 10


def test_short_route_is_not_finished_at_start(config):
    controller = PathFollower(config)
    controller.set_path([Waypoint(0, 0), Waypoint(0.2, 0)])

    initial = controller.update(VehicleState(0, 0, 0, 0), 1, 0.05)
    near_goal = controller.update(VehicleState(0.18, 0, 0, 0), 1, 0.05)
    assert initial.status != ControlStatus.FINISHED
    assert near_goal.status == ControlStatus.FINISHED


def test_update_requires_a_path(config):
    with pytest.raises(RuntimeError, match="set_path"):
        PathFollower(config).update(VehicleState(0, 0, 0, 0), 1, 0.05)


@pytest.mark.parametrize("dt_s", [0, -1, math.inf, math.nan, True, "0.05"])
def test_invalid_timestep_does_not_change_controller_history(
    follower, config, dt_s
):
    with pytest.raises(ValueError):
        follower.update(VehicleState(50, 0, 0, 2), 3, dt_s)

    fresh = PathFollower(config)
    fresh.set_path([Waypoint(0, 0), Waypoint(100, 0)])
    assert follower.update(VehicleState(0, 0, 0, 0), 2, 0.05) == fresh.update(
        VehicleState(0, 0, 0, 0), 2, 0.05
    )


@pytest.mark.parametrize("speed_mps", [-1, math.inf, math.nan, True, "5"])
def test_invalid_requested_speed_is_rejected(follower, speed_mps):
    with pytest.raises(ValueError):
        follower.update(VehicleState(0, 0, 0, 0), speed_mps, 0.05)


@pytest.mark.parametrize(
    "field,value",
    [
        ("wheelbase_m", 0),
        ("wheelbase_m", math.inf),
        ("max_steering_angle_rad", math.pi / 2),
        ("max_steering_angle_rad", -0.1),
        ("lookahead_m", 0),
        ("kp", -1),
        ("ki", math.nan),
        ("stopping_deceleration_mps2", 0),
        ("goal_tolerance_m", 0),
        ("stopped_speed_mps", 0),
        ("hold_brake", 0),
        ("hold_brake", 1.1),
    ],
)
def test_invalid_configuration_is_rejected(config, field, value):
    with pytest.raises(ValueError):
        replace(config, **{field: value})


def test_all_zero_pid_gains_are_rejected():
    with pytest.raises(ValueError, match="PID gain"):
        ControllerConfig(2.8, 0.6, kp=0, ki=0, kd=0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("x_m", math.nan),
        ("y_m", math.inf),
        ("yaw_rad", -math.inf),
        ("speed_mps", -1),
    ],
)
def test_invalid_vehicle_state_is_rejected(field, value):
    with pytest.raises(ValueError):
        replace(VehicleState(0, 0, 0, 0), **{field: value})


@pytest.mark.parametrize(
    "values", [(0, 1, 1), (0, -1, 0), (0, 0, 2), (math.nan, 0, 0)]
)
def test_invalid_control_command_is_rejected(values):
    with pytest.raises(ValueError):
        ControlCommand(*values)


@pytest.mark.parametrize("value", [math.nan, math.inf, "1", True])
def test_invalid_waypoint_is_rejected(value):
    with pytest.raises(ValueError):
        Waypoint(value, 0)
