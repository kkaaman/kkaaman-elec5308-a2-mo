"""Evidence for the mission rules in docs/mission.md (BL-A-01).

These tests pin A1 PathFollower behaviour that the A2 mission design relies
on or must avoid. They describe the controller as it is; they are not
proposals to change it.
"""

import math

import pytest

from navigation_control import ControllerConfig, PathFollower, VehicleState, Waypoint

CONFIG = ControllerConfig(wheelbase_m=2.9, max_steering_angle_rad=0.6, lookahead_m=5.0)


def test_progress_jumps_to_a_nearby_later_leg_and_the_controller_fails():
    """Projection uses the nearest segment of the whole route, so a route that
    doubles back within a few metres can make progress jump ahead. The
    look-ahead target then lies behind the car and update() raises. Hence the
    rule: non-adjacent parts of the mission must stay >= 10 m apart, and the
    stack turns any controller exception into an emergency stop."""
    u_turn = [Waypoint(0, 0), Waypoint(30, 0), Waypoint(30, 6), Waypoint(0, 6)]
    follower = PathFollower(CONFIG)
    follower.set_path(u_turn)
    on_first_leg = follower.update(VehicleState(10.0, 2.5, 0.0, 3.0), 3.0, 0.1)
    assert on_first_leg.diagnostics.progress_m < 15.0
    # 1 m of lateral drift puts the return leg nearer than the current leg.
    with pytest.raises(ValueError, match="behind vehicle"):
        follower.update(VehicleState(11.0, 3.5, 0.0, 3.0), 3.0, 0.1)


def test_setting_the_path_every_cycle_discards_speed_integral():
    """set_path() resets PID history. Calling it each cycle (as the A1 demo
    did) leaves only proportional action, so the car under-tracks speed.
    Hence the rule: set_path once per mission."""
    route = [Waypoint(0, 0), Waypoint(200, 0)]
    state = VehicleState(5.0, 0.0, 0.0, 3.0)

    once = PathFollower(CONFIG)
    once.set_path(route)
    for _ in range(20):
        steady = once.update(state, 5.0, 0.1)

    every_cycle = PathFollower(CONFIG)
    for _ in range(20):
        every_cycle.set_path(route)
        replanned = every_cycle.update(state, 5.0, 0.1)

    assert steady.command.throttle > replanned.command.throttle
    assert replanned.command.throttle == CONFIG.kp * 2.0 + CONFIG.ki * 2.0 * 0.1


def test_endpoint_cap_brakes_only_near_the_real_goal():
    """The controller's own endpoint cap, sqrt(2*a*(remaining - margin)),
    brings the car to rest at the destination; the stack never needs to send
    zero speed to finish the mission."""
    follower = PathFollower(CONFIG)
    follower.set_path([Waypoint(0, 0), Waypoint(100, 0)])
    far = follower.update(VehicleState(10.0, 0.0, 0.0, 5.0), 5.0, 0.1)
    near = follower.update(VehicleState(95.0, 0.0, 0.0, 5.0), 5.0, 0.1)
    margin = min(CONFIG.goal_tolerance_m * 0.5, 100 * 0.1)
    assert far.diagnostics.internal_target_speed_mps == 5.0
    expected = math.sqrt(2 * CONFIG.stopping_deceleration_mps2 * (5.0 - margin))
    assert near.diagnostics.internal_target_speed_mps == expected
