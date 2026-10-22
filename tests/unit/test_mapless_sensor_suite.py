"""Mapless sensor suite: Leaderboard SENSORS-track limits and banned ground-truth sensors."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from mapless.sensor_suite import (
    BANNED,
    BANNED_BLUEPRINTS,
    BLUEPRINTS,
    LIMITS,
    CameraSpec,
    Mount,
    SensorSuite,
    SpeedometerSpec,
    load_sensor_suite,
    suite_from_dict,
)

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "configs/sensor_suites/mapless_v1.json"


def document():
    return json.loads(SUITE.read_text(encoding="utf-8"))


def test_v1_suite_loads_within_limits():
    suite = load_sensor_suite(SUITE)
    assert [c.name for c in suite.cameras] == ["front", "front_left", "front_right", "rear"]
    assert len(suite.lidars) == 1 and len(suite.radars) == 3
    for kind, limit in LIMITS.items():
        assert len(suite.of_kind(kind)) <= limit
    assert suite.validate_for_timestep(0.05) == []
    assert suite.lidars[0].full_sweep_per_tick(0.05)
    assert len(suite.sha256) == 64 and suite.sha256 == load_sensor_suite(SUITE).sha256


def test_only_real_sensor_blueprints_are_mountable():
    assert set(BLUEPRINTS.values()).isdisjoint(BANNED_BLUEPRINTS)
    assert "sensor.camera.depth" in BANNED_BLUEPRINTS
    assert "sensor.lidar.ray_cast_semantic" in BANNED_BLUEPRINTS


@pytest.mark.parametrize("kind", sorted(BANNED))
def test_ground_truth_sensor_types_are_refused_with_the_reason(kind):
    doc = document()
    doc["sensors"].append({"type": kind, "name": "x", "mount": {"x_m": 0, "y_m": 0, "z_m": 1}})
    with pytest.raises(ValueError, match="not allowed on the car"):
        suite_from_dict(doc)


def test_limits_unknown_fields_and_duplicates_are_refused(tmp_path):
    doc = document()
    camera = doc["sensors"][0]
    doc["sensors"] += [dict(camera, name=f"extra{i}") for i in range(5)]  # 9 cameras
    with pytest.raises(ValueError, match="At most 8 camera"):
        suite_from_dict(doc)
    doc = document()
    doc["sensors"][0]["exposure"] = 1
    with pytest.raises(ValueError, match="Unknown camera fields"):
        suite_from_dict(doc)
    doc = document()
    doc["sensors"][1]["name"] = "front"
    with pytest.raises(ValueError, match="unique"):
        suite_from_dict(doc)
    path = tmp_path / "dup.json"
    path.write_text('{"name": "a", "name": "b", "sensors": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate key"):
        load_sensor_suite(path)


def test_lidar_must_sweep_once_per_tick_and_noise_free_copy_zeroes_noise():
    suite = load_sensor_suite(SUITE)
    slow = replace(suite, sensors=tuple(replace(s, rotation_frequency_hz=10.0) if s.kind == "lidar" else s for s in suite.sensors))
    assert "full sweep" in slow.validate_for_timestep(0.05)[0]
    quiet = suite.without_noise()
    assert quiet.gnss.white_m == quiet.gnss.bias_sigma_m == 0
    assert quiet.imu.compass_white_rad == 0 and quiet.speedometer.scale_sigma == 0
    assert quiet.lidars[0].noise_stddev_m == 0 and quiet.lidars[0].dropoff_general_rate == 0
    assert quiet.sha256 != suite.sha256


def test_a_suite_needs_camera_imu_and_speedometer():
    mount = Mount(0, 0, 1)
    with pytest.raises(ValueError, match="at least one imu"):
        SensorSuite("x", (CameraSpec("c", mount, 64, 64, 90.0), SpeedometerSpec("s")))
