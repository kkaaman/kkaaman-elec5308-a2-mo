"""Speed-governor tests for BL-D-07."""

import math

import pytest

from planning.speed_governor import (
    GovernorConfig,
    SpeedGovernor,
    speed_for_distance,
    stopping_distance,
)


@pytest.fixture
def config():
    return GovernorConfig(
        cruise_mps=5.0,
        deceleration_mps2=2.0,
        lateral_acceleration_mps2=1.5,
        response_budget_s=0.5,
        stop_margin_m=4.0,
        acceleration_mps2=1.0,
        emergency_margin_m=2.0,
    )


def test_stopping_distance_includes_response_and_braking():
    assert stopping_distance(4.0, 2.0, 0.5) == pytest.approx(6.0)


def test_speed_for_distance_inverts_stopping_budget():
    speed = speed_for_distance(6.0, 2.0, 0.5)

    assert stopping_distance(speed, 2.0, 0.5) == pytest.approx(6.0)


def test_straight_path_does_not_reduce_cruise_ceiling(config):
    governor = SpeedGovernor(config)

    target, emergency = governor.limit(
        speed_mps=0.0,
        dt_s=1.0,
        curvature=((10.0, 0.0),),
    )

    assert target == pytest.approx(1.0)
    assert not emergency


def test_left_and_right_curvature_have_same_speed_limit(config):
    left = SpeedGovernor(config)
    right = SpeedGovernor(config)

    left_target, _ = left.limit(
        speed_mps=4.0,
        dt_s=10.0,
        curvature=((0.0, 0.25),),
    )

    right_target, _ = right.limit(
        speed_mps=4.0,
        dt_s=10.0,
        curvature=((0.0, -0.25),),
    )

    assert left_target == pytest.approx(right_target)


def test_curvature_cap_matches_lateral_acceleration_limit(config):
    governor = SpeedGovernor(config)

    target, emergency = governor.limit(
        speed_mps=4.0,
        dt_s=10.0,
        curvature=((0.0, 0.375),),
    )

    expected = math.sqrt(
        config.lateral_acceleration_mps2 / 0.375
    )

    assert target == pytest.approx(expected)
    assert not emergency


def test_obstacle_cap_reduces_target_immediately(config):
    governor = SpeedGovernor(config)

    governor.previous_target_mps = config.cruise_mps

    target, emergency = governor.limit(
        speed_mps=5.0,
        dt_s=0.1,
        obstacle_gap_m=8.0,
    )

    expected = speed_for_distance(
        8.0 - config.stop_margin_m,
        config.deceleration_mps2,
        config.response_budget_s,
    )

    assert target == pytest.approx(expected)
    assert target < config.cruise_mps
    assert not emergency


def test_visible_distance_can_limit_target(config):
    governor = SpeedGovernor(config)

    governor.previous_target_mps = config.cruise_mps

    target, emergency = governor.limit(
        speed_mps=4.0,
        dt_s=0.1,
        visible_distance_m=7.0,
    )

    expected = speed_for_distance(
        7.0 - config.stop_margin_m,
        config.deceleration_mps2,
        config.response_budget_s,
    )

    assert target == pytest.approx(expected)
    assert not emergency


def test_nearest_distance_limit_wins(config):
    governor = SpeedGovernor(config)

    governor.previous_target_mps = config.cruise_mps

    target, emergency = governor.limit(
        speed_mps=4.0,
        dt_s=0.1,
        obstacle_gap_m=12.0,
        visible_distance_m=7.0,
    )

    expected = speed_for_distance(
        7.0 - config.stop_margin_m,
        config.deceleration_mps2,
        config.response_budget_s,
    )

    assert target == pytest.approx(expected)
    assert not emergency


def test_speed_increase_is_rate_limited(config):
    governor = SpeedGovernor(config)

    target, emergency = governor.limit(
        speed_mps=0.0,
        dt_s=0.2,
    )

    assert target == pytest.approx(0.2)
    assert not emergency


def test_emergency_override_returns_zero_same_cycle(config):
    governor = SpeedGovernor(config)

    governor.previous_target_mps = config.cruise_mps

    target, emergency = governor.limit(
        speed_mps=5.0,
        dt_s=0.1,
        obstacle_gap_m=2.4,
    )

    assert emergency
    assert target == 0.0


def test_hold_override_returns_zero_same_cycle(config):
    governor = SpeedGovernor(config)

    governor.previous_target_mps = config.cruise_mps

    target, emergency = governor.limit(
        speed_mps=4.0,
        dt_s=0.1,
        hold=True,
    )

    assert target == 0.0
    assert not emergency


def test_negative_available_distance_clamps_to_zero(config):
    governor = SpeedGovernor(config)

    governor.previous_target_mps = config.cruise_mps

    target, emergency = governor.limit(
        speed_mps=1.0,
        dt_s=0.1,
        obstacle_gap_m=config.stop_margin_m - 0.5,
    )

    assert target == 0.0
    assert not emergency


def test_reset_clears_previous_target(config):
    governor = SpeedGovernor(config)

    governor.previous_target_mps = 4.0
    governor.reset()

    assert governor.previous_target_mps == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cruise_mps": 0.0},
        {"deceleration_mps2": 0.0},
        {"lateral_acceleration_mps2": 0.0},
        {"stop_margin_m": 0.0},
        {"acceleration_mps2": 0.0},
        {"emergency_margin_m": 0.0},
        {"response_budget_s": -0.1},
    ],
)
def test_invalid_config_values_are_rejected(kwargs):
    values = {
        "cruise_mps": 5.0,
        "deceleration_mps2": 2.0,
        "lateral_acceleration_mps2": 1.5,
        "response_budget_s": 0.5,
        "stop_margin_m": 4.0,
        "acceleration_mps2": 1.0,
        "emergency_margin_m": 2.0,
    }
    values.update(kwargs)

    with pytest.raises(ValueError):
        GovernorConfig(**values)


def test_emergency_margin_must_be_below_stop_margin():
    with pytest.raises(
        ValueError,
        match="smaller than stop_margin_m",
    ):
        GovernorConfig(
            cruise_mps=5.0,
            deceleration_mps2=2.0,
            lateral_acceleration_mps2=1.5,
            response_budget_s=0.5,
            stop_margin_m=2.0,
            acceleration_mps2=1.0,
            emergency_margin_m=2.0,
        )