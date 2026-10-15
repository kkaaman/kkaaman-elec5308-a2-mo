"""Known bicycle response, speed servo and synthetic CSV analysis checks."""

import csv
import json
import math

import pytest

from evaluation.steering_calibration import SpeedServo, analyze, inferred_angle


@pytest.mark.parametrize("speed", [2, 5, 8])
@pytest.mark.parametrize("angle", [-0.1, 0.1])
def test_bicycle_angle_roundtrip_at_each_speed(speed, angle):
    rate = speed * math.tan(angle) / 3.00465
    assert inferred_angle(3.00465, speed, rate) == pytest.approx(angle)


@pytest.mark.parametrize(
    "values", [(0, 2, 0), (3, 0, 1), (3, 0.5, 1), (3, 2, float("nan"))]
)
def test_invalid_bicycle_inputs_rejected(values):
    with pytest.raises(ValueError):
        inferred_angle(*values)


def test_speed_servo_pedals_exclusive_and_bounded():
    servo = SpeedServo()
    for measured in [0, 1, 7, 8, 9, 15]:
        throttle, brake = servo.update(8, measured, 0.05)
        assert 0 <= throttle <= 1
        assert 0 <= brake <= 1
        assert throttle * brake == 0
    assert servo.update(8, 0, 0.05)[0] > 0
    assert servo.update(8, 12, 0.05)[1] > 0


def test_new_trial_servo_has_no_integral_history():
    first = SpeedServo()
    for _ in range(10):
        first.update(2, 1, 0.05)
    assert first.integral > 0
    assert SpeedServo().integral == 0
    with pytest.raises(ValueError):
        first.update(2, 1, 0)


def test_analysis_keeps_speeds_signs_gain_and_plot(tmp_path):
    fields = [
        "trial",
        "phase",
        "phase_time_s",
        "requested_angle_rad",
        "target_speed_mps",
        "speed_mps",
        "inferred_angle_rad",
        "wheel_fl_deg_raw",
        "wheel_fr_deg_raw",
        "applied_steer",
    ]
    with (tmp_path / "samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        trial = 0
        for speed in (2, 5, 8):
            for sign in (-1, 1):
                trial += 1
                for index in range(20):
                    writer.writerow(
                        dict(
                            zip(
                                fields,
                                [
                                    trial,
                                    "sweep",
                                    (index + 1) * 0.05,
                                    sign * 0.1,
                                    speed,
                                    speed,
                                    sign * 0.08,
                                    -sign * 4.6,
                                    -sign * 4.5,
                                    -sign * 0.1 / math.radians(70),
                                ],
                            )
                        )
                    )
    report = analyze(tmp_path)
    assert len(report["trials"]) == 6
    assert all(t["response_gain"] == pytest.approx(0.8) for t in report["trials"])
    assert all(t["quality"] == "USABLE_SHORT_WINDOW" for t in report["trials"])
    assert report["automatic_adapter_update"] is False
    assert (tmp_path / "steering_response.png").stat().st_size > 1000
    assert (
        json.loads((tmp_path / "calibration.json").read_text())["status"]
        == "MEASURED_REVIEW_REQUIRED"
    )
