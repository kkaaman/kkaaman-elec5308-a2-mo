"""E2-01: LiDAR sweep contract, range/density configuration and the synthetic fixture."""

import json

import numpy as np
import pytest

from autonomy.paths import FIXTURES_DIR
from obstacle_perception.lidar import LidarConfig, LidarSweep, validate_rigid_transform
from obstacle_perception.lidar_synthetic import (
    Scene,
    SceneBox,
    default_sensor_to_ego,
    load_fixture_sweep,
    synthetic_sweep,
)

IDENTITY = tuple(map(tuple, np.eye(4)))


def sweep(points, config=None, **overrides):
    values = dict(
        episode_id="e",
        frame_id=1,
        sim_time_s=0.05,
        received_monotonic_s=0.0,
        points_xyzi=points,
        sensor_to_ego=IDENTITY,
        config=config or LidarConfig(),
    )
    values.update(overrides)
    return LidarSweep(**values)


def test_contract_accepts_vehicle_frame_float32_and_snapshots_points():
    points = np.array([[10, 1, 0.5, 0.9], [20, -2, 1, 0.1]], np.float32)
    s = sweep(points)
    points[0, 0] = 99  # mutate the caller's buffer after construction
    assert s.points_xyzi[0, 0] == 10
    assert not s.points_xyzi.flags.writeable
    assert s.point_count == 2
    assert s.density_fraction == pytest.approx(2 / LidarConfig().expected_points_per_sweep)
    assert s.sensor_origin_ego_m == (0.0, 0.0, 0.0)
    assert set(s.summary()) >= {"frame_id", "sim_time_s", "point_count", "density_fraction"}


def test_contract_allows_empty_sweep_but_never_marks_it_clear():
    s = sweep(np.zeros((0, 4), np.float32))
    assert s.point_count == 0 and s.density_fraction == 0.0
    assert s.azimuth_coverage_deg() == 0.0


@pytest.mark.parametrize(
    "points",
    [
        np.zeros((3, 4), np.float64),
        np.zeros((3, 3), np.float32),
        np.zeros((12,), np.float32),
        np.array([[1, 0, 0, np.nan]], np.float32),
        np.array([[1, 0, 0, 1.5]], np.float32),
        np.array([[1, 0, 0, -0.1]], np.float32),
        np.array([[80, 0, 0, 0.5]], np.float32),
    ],
)
def test_contract_rejects_wrong_dtype_shape_nonfinite_intensity_or_range(points):
    with pytest.raises(ValueError):
        sweep(points)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(episode_id=""),
        dict(frame_id=-1),
        dict(frame_id=1.0),
        dict(sim_time_s=-0.1),
        dict(sim_time_s=float("nan")),
        dict(config="not-a-config"),
    ],
)
def test_contract_rejects_bad_frame_metadata(overrides):
    with pytest.raises((ValueError, TypeError)):
        sweep(np.zeros((1, 4), np.float32), **overrides)


def test_extrinsics_must_be_rigid():
    bad = np.eye(4)
    bad[0, 0] = 2
    with pytest.raises(ValueError):
        sweep(np.zeros((1, 4), np.float32), sensor_to_ego=tuple(map(tuple, bad)))
    reflected = np.diag([1.0, -1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="handedness"):
        validate_rigid_transform(reflected)
    assert not validate_rigid_transform(np.eye(4)).flags.writeable


def test_range_is_measured_from_the_sensor_not_the_rear_axle():
    config = LidarConfig(range_m=10.0)
    mount = default_sensor_to_ego(x_m=1.5, z_m=2.0)
    inside = np.array([[11.4, 0, 2.0, 0.5]], np.float32)  # 9.9 m from the sensor
    s = sweep(inside, config, sensor_to_ego=tuple(map(tuple, mount)))
    assert s.sensor_origin_ego_m == pytest.approx((1.5, 0.0, 2.0))
    with pytest.raises(ValueError, match="range"):
        sweep(
            np.array([[11.6, 0, 2.0, 0.5]], np.float32),
            config,
            sensor_to_ego=tuple(map(tuple, mount)),
        )


def test_config_rotation_matches_timestep_and_density_is_configurable():
    config = LidarConfig.for_timestep(0.05, channels=64, points_per_second=1_000_000, range_m=80.0)
    assert config.rotation_frequency_hz == pytest.approx(20.0)
    assert config.full_sweep_per_tick(0.05)
    assert not config.full_sweep_per_tick(0.1)
    assert config.expected_points_per_sweep == pytest.approx(50_000)
    assert config.to_dict()["range_m"] == 80.0
    with pytest.raises(ValueError):
        LidarConfig(channels=0)
    with pytest.raises(ValueError):
        LidarConfig(lower_fov_deg=5.0, upper_fov_deg=5.0)
    with pytest.raises(ValueError):
        LidarConfig(horizontal_fov_deg=400)
    with pytest.raises(ValueError):
        LidarConfig.for_timestep(0.0)


def test_azimuth_coverage_uses_sensor_frame():
    mount = default_sensor_to_ego(x_m=1.5, z_m=2.0)
    azimuth = np.radians(np.arange(0.5, 180, 1.0))
    xyz = np.column_stack(
        (5 * np.cos(azimuth) + 1.5, 5 * np.sin(azimuth), np.full_like(azimuth, 1.0))
    )
    s = sweep(
        np.column_stack((xyz, np.full(len(xyz), 0.5))).astype(np.float32),
        sensor_to_ego=tuple(map(tuple, mount)),
    )
    assert s.azimuth_coverage_deg() == pytest.approx(180.0)


def test_synthetic_sweep_is_deterministic_and_known_object_lands_at_its_box():
    scene = Scene(boxes=(SceneBox(20.0, 0.0, 4.0, 2.0, 1.5, label="car"),))
    a = synthetic_sweep(scene, episode_id="e", frame_id=3, sim_time_s=0.15)
    b = synthetic_sweep(scene, episode_id="e", frame_id=3, sim_time_s=0.15)
    assert np.array_equal(a.points_xyzi, b.points_xyzi)
    assert a.points_xyzi.dtype == np.float32 and a.points_xyzi.shape[1] == 4
    face = a.points_xyzi[(np.abs(a.points_xyzi[:, 1]) < 0.9) & (a.points_xyzi[:, 2] > 0.4)]
    assert face[:, 0].min() == pytest.approx(18.0, abs=0.05)
    assert 0 <= a.points_xyzi[:, 3].min() and a.points_xyzi[:, 3].max() <= 1


def test_checked_in_fixture_matches_manifest_and_contract():
    directory = FIXTURES_DIR / "lidar"
    manifest = json.loads((directory / "sweeps.json").read_text())
    assert manifest["origin"] == "synthetic" and manifest["schema_version"] == 1
    config = LidarConfig(**manifest["config"])
    assert config.full_sweep_per_tick(manifest["fixed_delta_seconds"])
    notes = [row["note"] for row in manifest["sweeps"]]
    assert "sparse_dropout" in notes and "empty_sweep" in notes and "depth_dropped" in notes
    for row in manifest["sweeps"]:
        s = load_fixture_sweep(manifest, row, directory)
        assert s.point_count == row["point_count"]
        assert (directory / row["npy"]).stat().st_size < 1_000_000
    corrupt = dict(manifest["sweeps"][0], sha256="0" * 64)
    with pytest.raises(ValueError, match="hash"):
        load_fixture_sweep(manifest, corrupt, directory)
