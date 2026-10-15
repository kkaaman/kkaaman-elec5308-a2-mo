"""Speed settling distance under a synthetic resistive load."""

import pytest

from evaluation.steering_calibration import SpeedServo


@pytest.mark.parametrize("target", [2.0, 5.0, 8.0])
def test_settling_leaves_distance_for_sweep(target):
    servo = SpeedServo()
    speed = distance = 0.0
    stable = 0
    for _ in range(240):
        throttle, brake = servo.update(target, speed, 0.05)
        # Deliberately synthetic, not a CARLA-calibrated dynamics model.
        speed = max(0.0, speed + (8 * throttle - 12 * brake - 3.1) * 0.05)
        distance += speed * 0.05
        stable = stable + 1 if abs(speed - target) < 0.2 else 0
        if stable >= 10:
            break
    assert stable >= 10
    assert distance + target * 1.0 < 40.0
