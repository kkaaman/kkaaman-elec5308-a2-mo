"""Shadow localisation: GNSS + gyro + wheel-speed estimate, logged and never driven on."""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from autonomy.baseline import BaselineConfig
from autonomy.localisation import (
    FilterTuning,
    LocalisationConfig,
    LocalisationInputs,
    PoseEstimator,
    SensorNoise,
    pose_errors,
)
from autonomy.paths import REPO_ROOT
from carla_adapter.localisation_sensors import LiveLocalisationSensors, geo_calibration
from evaluation.runner import run_episode
from navigation_control.types import VehicleState
from planning.corridor import RouteGeometry
from sim_harness.baseline import BaselineWorld, RasterDetector

STRAIGHT = RouteGeometry(tuple((float(x), 0.0) for x in range(0, 200, 5)))


def drive(estimator, gnss, *, seconds=20.0, speed=5.0, rate_bias=0.0, dt=0.05):
    estimate = None
    for k in range(int(seconds / dt) + 1):
        t = k * dt
        fix = gnss(speed * t) if k % 4 == 0 else None
        estimate = estimator.step(LocalisationInputs(t, speed, rate_bias, fix), STRAIGHT)
    return estimate


def test_noise_free_straight_drive_tracks_truth():
    estimate = drive(PoseEstimator(), lambda x: (x, 0.0))
    assert estimate.x_m == pytest.approx(100.0, abs=0.05)
    assert abs(estimate.y_m) < 0.01 and abs(estimate.yaw_rad) < 1e-3


def test_constant_gnss_offset_is_not_observable():
    # The filter follows a constant 1 m sideways offset: it is position error,
    # which only a lane-relative measurement could remove.
    estimate = drive(PoseEstimator(), lambda x: (x, 1.0))
    assert estimate.y_m == pytest.approx(1.0, abs=0.1)


def test_gyro_bias_is_learned_from_gnss():
    estimate = drive(PoseEstimator(), lambda x: (x, 0.0), seconds=40.0, rate_bias=0.01)
    assert estimate.gyro_bias_rps == pytest.approx(0.01, abs=0.003)
    assert abs(estimate.yaw_rad) < math.radians(1)


def test_first_fix_snaps_to_the_route_and_takes_its_heading():
    route = RouteGeometry(((0.0, 0.0), (0.0, 50.0)))
    estimate = PoseEstimator().step(LocalisationInputs(0.0, 0.0, 0.0, (0.8, 10.0)), route)
    assert (estimate.x_m, estimate.y_m) == pytest.approx((0.0, 10.0))
    assert estimate.yaw_rad == pytest.approx(math.pi / 2)
    assert PoseEstimator().step(LocalisationInputs(0.0, 0.0, 0.0, None), route) is None


def test_errors_are_split_along_and_across_the_route():
    estimate = SimpleNamespace(x_m=12.0, y_m=-0.5, yaw_rad=0.1)
    errors = pose_errors(estimate, VehicleState(10.0, 0.0, 0.0, 5.0), STRAIGHT)
    assert errors["along_error_m"] == pytest.approx(2.0)
    assert errors["lateral_error_m"] == pytest.approx(-0.5)
    assert errors["yaw_error_rad"] == pytest.approx(0.1)


def test_config_files_load_and_reject_negative_values(tmp_path):
    for name in ("consumer", "rtk"):
        LocalisationConfig.load(REPO_ROOT / f"configs/localisation_{name}_gnss.json")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(dict(schema_version=1, sensors=dict(gnss_white_m=-1))))
    with pytest.raises(ValueError):
        LocalisationConfig.load(bad)


# Live sensor wrapper, with a linear stand-in for map.transform_to_geolocation.
ORIGIN = (49.0, 8.0)
DEG_PER_M = (-9.0e-6, 1.37e-5)  # latitude falls along CARLA +y; longitude rises along +x


def geolocate(x, y):
    return ORIGIN[0] + DEG_PER_M[0] * y, ORIGIN[1] + DEG_PER_M[1] * x


def test_geo_calibration_round_trips_carla_xy():
    origin, to_geo, from_geo = geo_calibration(geolocate)
    lat, lon = geolocate(37.0, -12.0)
    assert from_geo @ (np.array([lat, lon]) - origin) == pytest.approx([37.0, -12.0])


def live_sensors(noise=None):
    spawned = {}

    def spawn(blueprint, attributes, mount, callback):
        spawned[blueprint] = (attributes, mount, callback)

    config = LocalisationConfig(noise or SensorNoise(), FilterTuning())
    sensors = LiveLocalisationSensors(config, 5308, geolocate, spawn, 1.38641, 0.05)
    return sensors, spawned


def test_live_inputs_convert_fixes_and_flip_the_gyro():
    sensors, spawned = live_sensors()
    _, mount, on_imu = spawned["sensor.other.imu"]
    on_gnss = spawned["sensor.other.gnss"][2]
    assert mount[0] == pytest.approx(-1.38641)  # antenna over the rear axle
    on_imu(SimpleNamespace(frame=7, timestamp=0.35, gyroscope=SimpleNamespace(z=0.2)))
    lat, lon = geolocate(20.0, 4.0)  # CARLA y = 4 -> core y = -4
    on_gnss(SimpleNamespace(frame=7, latitude=lat, longitude=lon))
    sensors.record_speed(7, 5.0)
    inputs = sensors.inputs(7)
    assert inputs.gnss_xy_m == pytest.approx((20.0, -4.0))
    assert inputs.yaw_rate_rps == pytest.approx(-0.2)
    assert inputs.sim_time_s == 0.35
    assert sensors.record()["counts"]["fixes"] == 1


def test_live_noise_attributes_are_in_degrees_and_seeded():
    sensors, spawned = live_sensors(SensorNoise(gnss_white_m=1.0, gnss_bias_m=0.0))
    gnss = spawned["sensor.other.gnss"][0]
    assert gnss["noise_lat_stddev"] == pytest.approx(abs(DEG_PER_M[0]))
    assert gnss["noise_lon_stddev"] == pytest.approx(DEG_PER_M[1])
    assert gnss["noise_lat_bias"] == 0 and gnss["noise_seed"] == 5308
    imu = spawned["sensor.other.imu"][0]
    assert imu["noise_gyro_bias_z"] == pytest.approx(-sensors.model.gyro_bias)


def harness_run(tmp_path, name, localisation=None):
    world = BaselineWorld("S0", route="mission")
    if localisation is not None:
        world.enable_localisation(localisation, 5308)
    result = run_episode(
        world,
        RasterDetector(),
        world.points,
        BaselineConfig(),
        tmp_path / name,
        dict(
            engine="harness",
            scenario="S0",
            seed=5308,
            step_wall_budget_s=10.0,
            event_sensors_verified=True,
            truth_channel_complete=True,
        ),
        max_sim_s=150,
        navigation_only=True,
    )
    rows = [json.loads(x) for x in (tmp_path / name / "samples.jsonl").read_text().splitlines()]
    return result, rows


def test_shadow_estimate_never_changes_the_drive(tmp_path):
    rtk = LocalisationConfig.load(REPO_ROOT / "configs/localisation_rtk_gnss.json")
    plain, plain_rows = harness_run(tmp_path, "plain")
    shadow, shadow_rows = harness_run(tmp_path, "shadow", rtk)
    assert plain["localisation"] is None
    assert [r["command"] for r in plain_rows] == [r["command"] for r in shadow_rows]
    assert [r["ego"] for r in plain_rows] == [r["ego"] for r in shadow_rows]
    report = shadow["localisation"]
    assert report["lateral_error_m"]["p95"] < 0.05
    assert report["gyro_truth_correlation"] > 0.9
    manifest = json.loads((tmp_path / "shadow" / "manifest.json").read_text())
    assert manifest["localisation"]["mode"].startswith("shadow")


def estimate_run(tmp_path, name, profile, seed):
    world = BaselineWorld("S0", route="mission")
    world.enable_localisation(
        LocalisationConfig.load(REPO_ROOT / f"configs/localisation_{profile}_gnss.json"), seed
    )
    return run_episode(
        world,
        RasterDetector(),
        world.points,
        BaselineConfig(),
        tmp_path / name,
        dict(
            engine="harness",
            scenario="S0",
            seed=seed,
            pose_source="estimate",
            step_wall_budget_s=10.0,
            event_sensors_verified=True,
            truth_channel_complete=True,
        ),
        max_sim_s=150,
        navigation_only=True,
    )


def test_driving_on_an_rtk_estimate_completes_in_lane(tmp_path):
    result = estimate_run(tmp_path, "rtk", "rtk", 5308)
    assert result["collection_status"] == "COLLECTED", result["error"]
    assert result["metrics"]["route_completion"] == 1
    assert result["metrics"]["cte_mean_m"] < 0.05  # scored on the true pose
    manifest = json.loads((tmp_path / "rtk" / "manifest.json").read_text())
    assert manifest["localisation"]["mode"].startswith("driving on the estimate")


def test_driving_on_a_biased_estimate_really_follows_it(tmp_path):
    # Seed 5308 draws a ~1.9 m consumer-GNSS offset; the true CTE shows the car
    # drove where it believed the lane was, so the estimate really is in control.
    result = estimate_run(tmp_path, "consumer", "consumer", 5308)
    assert result["metrics"]["cte_mean_m"] > 0.3


def test_estimate_mode_needs_localisation_sensors(tmp_path):
    world = BaselineWorld("S0", route="mission")
    with pytest.raises(ValueError, match="needs world localisation sensors"):
        run_episode(
            world,
            RasterDetector(),
            world.points,
            BaselineConfig(),
            tmp_path / "none",
            dict(engine="harness", scenario="S0", seed=1, pose_source="estimate"),
            max_sim_s=5,
            navigation_only=True,
        )
