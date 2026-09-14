"""Unit tests for PID speed response, anti-windup, derivative and reset."""

from dataclasses import replace

import pytest

from navigation_control.speed import SpeedPID


def test_speed_error_sign_and_output_limits(config):
    pid = SpeedPID(config)

    assert 0 < pid.update(0, 1, 0.05) <= 1
    assert -1 <= pid.update(2, 1, 0.05) < 0
    assert pid.update(0, 100, 0.05) == 1
    assert pid.update(100, 0, 0.05) == -1


def test_proportional_and_integral_output_values(config):
    pid = SpeedPID(config)

    assert pid.update(0, 1, 1) == pytest.approx(0.45)
    assert pid.update(0, 1, 1) == pytest.approx(0.50)


@pytest.mark.parametrize("speed,target", [(0, 100), (100, 0)])
def test_sustained_saturation_does_not_accumulate_integral(
    config, speed, target
):
    pid = SpeedPID(config)

    for _ in range(1000):
        pid.update(speed, target, 0.05)

    assert pid.update(5, 5, 0.05) == 0
    assert pid.update(6, 5, 0.05) < 0


def test_derivative_uses_measurement_not_setpoint(config):
    pid = SpeedPID(replace(config, kp=0, ki=0, kd=0.1))

    assert pid.update(1, 1, 0.1) == 0
    assert pid.update(1, 100, 0.1) == 0
    assert pid.update(2, 100, 0.1) == -1

    pid.reset()
    assert pid.update(2, 100, 0.1) == 0


def test_reset_clears_integrator(config):
    pid = SpeedPID(config)
    first = pid.update(0, 1, 0.05)

    for _ in range(20):
        pid.update(0, 1, 0.05)

    assert pid.update(0, 1, 0.05) > first
    pid.reset()
    assert pid.update(0, 1, 0.05) == first
