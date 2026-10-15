"""Check speed settling under load, including the observed throttle shortfall."""

import pytest

from evaluation.steering_calibration import SpeedServo


@pytest.mark.parametrize("target", [2.0, 5.0, 8.0])
def test_servo_settles_under_constant_resistance(target):
    servo = SpeedServo()
    speed = 0.0
    stable = 0
    for _ in range(240):
        throttle, brake = servo.update(target, speed, 0.05)
        # Synthetic load needs 0.14 throttle to hold speed. This is a
        # regression plant, not a fitted CARLA dynamics model.
        speed = max(0.0, speed + (4 * throttle - 8 * brake - 0.56) * 0.05)
        stable = stable + 1 if abs(speed - target) < 0.2 else 0
        if stable >= 10:
            break
    assert stable >= 10
    assert servo.integral > 0


def test_integral_unwinds_when_error_reverses_at_saturation():
    servo = SpeedServo(integral=5.0)
    for _ in range(200):
        servo.update(2.0, 8.0, 0.05)
    assert servo.integral < 0
