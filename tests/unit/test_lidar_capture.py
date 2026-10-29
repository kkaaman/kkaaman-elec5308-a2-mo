"""E2-02: ray-cast blueprint configuration, single sensor-to-vehicle transform, known object."""

import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from carla_adapter.lidar import (
    LidarCapture,
    LidarRig,
    SweepQueue,
    known_object_check,
    lidar_points_to_ego,
)
from carla_adapter.world import WorldTiming
from obstacle_perception.lidar import LidarSweep
from obstacle_perception.lidar_synthetic import Scene, SceneBox, synthetic_sweep
from scripts.capture_lidar_sweeps import expected_target_box, serialise_sweep, validate_environment

REAR_AXLE = 1.38641


def raw(points):
    return np.asarray(points, np.float32).tobytes()


def test_decode_reflects_y_and_applies_mount_once():
    rig = LidarRig(x_actor_m=0.5, z_actor_m=2.0, yaw_deg=90.0)
    matrix = rig.sensor_to_ego(REAR_AXLE)
    # CARLA sensor frame: 3 m ahead, 1 m to the RIGHT. Sensor yawed 90 deg left on the car.
    points, dropped = lidar_points_to_ego(raw([[3, 1, 0.5, 0.7], [np.nan, 0, 0, 0]]), matrix)
    assert dropped == 1 and points.dtype == np.float32 and points.shape == (1, 4)
    # Right-handed sensor frame: (3, -1, 0.5); yaw +90 -> (1, 3); plus mount (0.5+1.386, 0, 2).
    assert points[0, :3] == pytest.approx([1 + 0.5 + REAR_AXLE, 3, 2.5], abs=1e-5)
    assert points[0, 3] == pytest.approx(0.7)


def test_decode_rejects_partial_points_and_clips_intensity_and_range():
    with pytest.raises(ValueError):
        lidar_points_to_ego(b"\x00" * 12, np.eye(4))
    points, dropped = lidar_points_to_ego(
        raw([[1, 0, 0, 1.4], [90, 0, 0, 0.2]]), np.eye(4), range_m=50
    )
    assert dropped == 1 and points[0, 3] == 1.0


def test_rig_blueprint_sets_one_rotation_per_tick_and_mount_signs():
    rig = LidarRig(
        y_actor_m=0.2, yaw_deg=10.0, channels=64, range_m=80.0, points_per_second=1_200_000
    )
    attributes = rig.blueprint_attributes(0.05)
    assert attributes["rotation_frequency"] == "20.0"
    assert attributes["channels"] == "64" and attributes["range"] == "80.0"
    assert attributes["points_per_second"] == "1200000" and attributes["sensor_tick"] == "0.0"
    assert rig.config(0.05).full_sweep_per_tick(0.05)
    location, rotation = rig.mount_arguments()
    assert location["y"] == -0.2 and rotation["yaw"] == -10.0
    with pytest.raises(ValueError):
        LidarRig(dropoff_general_rate=1.5)
    with pytest.raises(ValueError):
        LidarRig(channels=0)


def test_sweep_queue_bounded_ordered_and_stops_on_timeout():
    q = SweepQueue("e", capacity=2)
    stops = []
    for frame in range(5):
        q.put(frame, frame * 0.05, np.zeros((1, 4), np.float32))
    assert q.events["overflow"] == 3
    timestamp, points, dropped, received = q.wait(4, 0.1, lambda: stops.append(1))
    assert timestamp == pytest.approx(0.2) and not points.flags.writeable and stops == []
    q.put(4, 0.2, np.zeros((1, 4), np.float32))
    assert q.events["late"] == 1
    with pytest.raises(TimeoutError):
        q.wait(7, 0.001, lambda: stops.append(2))
    assert stops == [2]
    q.close()
    with pytest.raises(RuntimeError):
        q.wait(8, 0.01, lambda: stops.append(3))
    assert stops == [2, 3]


class FakeSensor:
    is_alive = True
    id = 7

    def __init__(self):
        self.stopped = False
        self.callback = None

    def listen(self, callback):
        self.callback = callback

    def stop(self):
        self.stopped = True

    def destroy(self):
        self.is_alive = False
        return True


class FakeBlueprint:
    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


def fake_carla(monkeypatch):
    module = SimpleNamespace(
        Transform=lambda location, rotation: ("transform", location, rotation),
        Location=lambda **kwargs: ("location", kwargs),
        Rotation=lambda **kwargs: ("rotation", kwargs),
        AttachmentType=SimpleNamespace(Rigid="rigid"),
    )
    monkeypatch.setitem(sys.modules, "carla", module)
    return module


def fake_owner(sensor, blueprint, fail=False):
    spawns = []

    class World:
        def get_blueprint_library(self):
            return self

        def find(self, key):
            assert key == "sensor.lidar.ray_cast"
            return blueprint

        def spawn_actor(self, bp, transform, attach_to, attachment_type):
            spawns.append((transform, attach_to, attachment_type))
            if fail:
                raise RuntimeError("injected spawn failure")
            return sensor

    stops = []
    owner = SimpleNamespace(
        world=World(),
        _actors=[],
        _require_owner=lambda: None,
        timing=WorldTiming(fixed_delta_seconds=0.05),
        emergency_stop=lambda: stops.append(True),
        spawns=spawns,
        stops=stops,
    )
    return owner


def test_capture_configures_blueprint_transforms_once_and_records_density(monkeypatch):
    fake_carla(monkeypatch)
    sensor, blueprint = FakeSensor(), FakeBlueprint()
    owner = fake_owner(sensor, blueprint)
    rig = LidarRig(z_actor_m=2.0, channels=32, points_per_second=640_000)
    scene = Scene(boxes=(SceneBox(20.0, 0.0, 4.0, 2.0, 1.5, label="car"),))
    with LidarCapture(owner, "vehicle", "e", rear_axle_offset_m=REAR_AXLE, rig=rig) as capture:
        assert blueprint.attributes["rotation_frequency"] == "20.0"
        assert blueprint.attributes["channels"] == "32"
        assert owner._actors == [sensor]
        transform, attach_to, attachment = owner.spawns[0]
        assert attach_to == "vehicle" and attachment == "rigid"
        assert transform[1] == ("location", dict(x=0.0, y=-0.0, z=2.0))
        # Build a sensor-frame CARLA buffer from a synthetic ego-frame sweep so the
        # adapter must undo the mount and the Y reflection to recover it.
        reference = synthetic_sweep(
            scene, episode_id="e", frame_id=12, sim_time_s=0.6, sensor_to_ego=capture.sensor_to_ego
        )
        ego = reference.points_xyzi[:, :3].astype(float)
        local = (ego - capture.sensor_to_ego[:3, 3]) @ capture.sensor_to_ego[:3, :3]
        carla_buffer = np.column_stack(
            (local[:, 0], -local[:, 1], local[:, 2], reference.points_xyzi[:, 3])
        )
        sensor.callback(SimpleNamespace(frame=12, timestamp=0.6, raw_data=raw(carla_buffer)))
        sweep = capture.read(12, timeout_s=0.5)
        assert isinstance(sweep, LidarSweep)
        assert sweep.frame_id == 12 and sweep.sim_time_s == 0.6
        assert np.allclose(sweep.points_xyzi, reference.points_xyzi, atol=1e-4)
        assert sweep.sensor_to_ego[0][3] == pytest.approx(REAR_AXLE)
        check = known_object_check(sweep, center_ego_m=(20.0, 0.0, 0.75), extent_m=(4.0, 2.0, 1.5))
        assert check["passed"] and check["points_in_box"] >= 6
        assert abs(check["near_face_error_m"]) <= 0.05 and check["intensity_in_unit_interval"]
        record = capture.records[0]
        assert record["frame_id"] == 12 and record["point_count"] == sweep.point_count
        assert record["density_fraction"] == pytest.approx(sweep.point_count / 32_000)
        assert record["fixed_delta_seconds"] == 0.05 and record["dropped_nonfinite"] == 0
        assert owner.stops == []
    assert sensor.stopped and not sensor.is_alive


def test_capture_cleanup_on_spawn_failure_and_missing_frame_stops(monkeypatch):
    fake_carla(monkeypatch)
    sensor, blueprint = FakeSensor(), FakeBlueprint()
    with pytest.raises(RuntimeError, match="injected"):
        with LidarCapture(
            fake_owner(sensor, blueprint, fail=True), "v", "e", rear_axle_offset_m=1.0
        ):
            pass
    owner = fake_owner(sensor, blueprint)
    with LidarCapture(owner, "v", "e", rear_axle_offset_m=1.0) as capture:
        sensor.callback(SimpleNamespace(frame=3, timestamp=0.15, raw_data=b"\x00" * 12))
        assert capture.callback_errors and "whole number" in capture.callback_errors[0]
        with pytest.raises(TimeoutError):
            capture.read(3, timeout_s=0.01)
        assert owner.stops == [True]


def test_known_object_check_fails_on_displaced_or_missing_object():
    scene = Scene(boxes=(SceneBox(20.0, 0.0, 4.0, 2.0, 1.5),))
    sweep = synthetic_sweep(scene, episode_id="e", frame_id=1, sim_time_s=0.05)
    assert known_object_check(sweep, center_ego_m=(20.0, 0.0, 0.75), extent_m=(4.0, 2.0, 1.5))[
        "passed"
    ]
    displaced = known_object_check(sweep, center_ego_m=(21.0, 0.0, 0.75), extent_m=(4.0, 2.0, 1.5))
    assert not displaced["passed"] and displaced["near_face_error_m"] < -0.5
    missing = known_object_check(sweep, center_ego_m=(20.0, 6.0, 5.0), extent_m=(1.0, 1.0, 1.5))
    assert not missing["passed"] and missing["points_in_box"] == 0
    with pytest.raises(ValueError):
        known_object_check(sweep, center_ego_m=(0, 0, 0), extent_m=(0, 1, 1))


def test_expected_target_box_converts_left_handed_actor_frame_to_rear_axle():
    # Ego actor at world (10, 5) yawed 90 deg (CARLA: +yaw clockwise seen from above).
    yaw = math.radians(90)
    c, s = math.cos(yaw), math.sin(yaw)
    actor_to_world = np.array([[c, -s, 0, 10], [s, c, 0, 5], [0, 0, 1, 0], [0, 0, 0, 1]])
    world_to_actor = np.linalg.inv(actor_to_world)
    # A 4 x 2 x 1.5 box centred 20 m ahead of the actor origin and 1 m to CARLA-right.
    corners = [[20 + dx, 1 + dy, dz] for dx in (-2, 2) for dy in (-1, 1) for dz in (0, 1.5)]
    world = [(actor_to_world @ np.array([*p, 1]))[:3] for p in corners]
    center, extent = expected_target_box(world_to_actor, world, REAR_AXLE)
    assert center == pytest.approx([20 + REAR_AXLE, -1.0, 0.75], abs=1e-6)
    assert extent == pytest.approx([4.0, 2.0, 1.5], abs=1e-6)


def test_serialise_sweep_and_environment_validation(tmp_path):
    scene = Scene(boxes=(SceneBox(20.0, 0.0, 4.0, 2.0, 1.5),))
    sweep = synthetic_sweep(scene, episode_id="e", frame_id=1, sim_time_s=0.05)
    check = known_object_check(sweep, center_ego_m=(20.0, 0.0, 0.75), extent_m=(4.0, 2.0, 1.5))
    row = serialise_sweep(
        sweep,
        "sweep_000001.npy",
        tmp_path,
        check=check,
        target=dict(label="target"),
        distance_m=20.0,
    )
    assert row["frame_id"] == 1 and row["point_count"] == sweep.point_count
    assert np.array_equal(np.load(tmp_path / "sweep_000001.npy"), sweep.points_xyzi)
    assert row["known_object_check"]["passed"] and row["depth_valid"] is False
    validate_environment("0.9.16", "0.9.16", "Carla/Maps/Town10HD_Opt")
    with pytest.raises(ValueError):
        validate_environment("0.9.15", "0.9.16", "Town10HD_Opt")
    with pytest.raises(ValueError):
        validate_environment("0.9.16", "0.9.16", "Town01")


def test_capture_completion_requires_verified_cleanup_and_all_measurements():
    from scripts.capture_lidar_sweeps import final_capture_status

    clean = {"settings_restored": True, "leaked_owned_ids": []}
    assert final_capture_status(None, [True], clean) == "COMPLETE"
    for error, checks, cleanup in (
        (None, [True], dict(clean, settings_restored=False)),
        (None, [True], dict(clean, leaked_owned_ids=[42])),
        (None, [True, False], clean),
        (None, [], clean),
        ("capture failure", [True], clean),
    ):
        assert final_capture_status(error, checks, cleanup) == "INCOMPLETE"


def test_capture_batch_pairs_all_distances_with_centre_and_both_roadsides():
    from scripts.capture_lidar_sweeps import capture_cases

    cases = capture_cases([10, 20, 30], [0, 4.5, -4.5])
    assert len(cases) == 9 and len(set(cases)) == 9
    for distance in (10.0, 20.0, 30.0):
        assert {lateral for d, lateral in cases if d == distance} == {0.0, 4.5, -4.5}
    with pytest.raises(ValueError):
        capture_cases([float("nan")], [0])


def test_known_car_check_excludes_road_search_padding_without_relaxing_face_tolerance():
    config = LidarRig().config(0.05)
    transform = tuple(map(tuple, np.eye(4)))
    vehicle = np.array([[18.04, y, 0.6, 0.9] for y in np.linspace(-0.5, 0.5, 10)], np.float32)
    road = np.array([[17.01, y, 0, 0.9] for y in np.linspace(-0.5, 0.5, 20)], np.float32)
    sweep = LidarSweep("e", 1, 0.05, 1, np.vstack((vehicle, road)), transform, config)
    result = known_object_check(sweep, center_ego_m=(20, 0, 0.75), extent_m=(4, 2, 1.5))
    assert result["passed"] and result["excluded_search_points_below_surface"] == 20
    displaced = LidarSweep(
        "e", 2, 0.1, 2, vehicle + np.array([0.5, 0, 0, 0], np.float32), transform, config
    )
    assert not known_object_check(displaced, center_ego_m=(20, 0, 0.75), extent_m=(4, 2, 1.5))[
        "passed"
    ]
    road_only = LidarSweep("e", 3, 0.15, 3, road, transform, config)
    assert not known_object_check(road_only, center_ego_m=(20, 0, 0.75), extent_m=(4, 2, 1.5))[
        "passed"
    ]


def test_cleanup_fence_observes_async_destruction_and_never_accepts_leak():
    from scripts.capture_lidar_sweeps import verify_cleanup

    class World:
        present = True

        def get_actors(self):
            return [SimpleNamespace(id=7)] if self.present else []

        def get_settings(self):
            return SimpleNamespace(synchronous_mode=False)

        def wait_for_tick(self, timeout):
            self.present = False

    world = World()
    result = verify_cleanup(world, [7], {"synchronous_mode": False})
    assert result["leaked_owned_ids"] == [] and result["observations"][0]["leaked_owned_ids"] == [7]
    result = verify_cleanup(World(), [7], {"synchronous_mode": False}, timeout_s=0)
    assert result["leaked_owned_ids"] == [7]


def test_optional_raw_sensor_capture_is_exact_and_bounded(tmp_path):
    owner = SimpleNamespace(timing=WorldTiming(), emergency_stop=lambda: None)
    capture = LidarCapture(
        owner,
        object(),
        "e",
        rear_axle_offset_m=REAR_AXLE,
        capacity=2,
        raw_directory=tmp_path / "raw",
    )
    for i in range(3):
        payload = raw([[i + 5, 0, -1, 0.8]])
        capture._receive(SimpleNamespace(frame=i, timestamp=i * 0.05, raw_data=payload))
    assert len(capture._raw_frames) == 2 and 0 not in capture._raw_frames
    capture.read(2)
    assert (tmp_path / "raw/sensor_000002.bin").read_bytes() == payload
    assert (
        capture.records[-1]["received_monotonic_s"]
        <= capture.records[-1]["read_completed_monotonic_s"]
    )


def test_live_audit_empty_sweep_stays_unknown_and_never_clear(tmp_path):
    import hashlib
    import json
    from dataclasses import asdict

    from scripts.audit_lidar_live import audit

    payload = b""
    (tmp_path / "raw-sensor").mkdir()
    (tmp_path / "raw-sensor/sensor.bin").write_bytes(payload)
    array = np.empty((0, 4), np.float32)
    np.save(tmp_path / "sweep.npy", array)
    row = dict(
        frame_id=1,
        sim_time_s=0.05,
        npy="sweep.npy",
        sha256=hashlib.sha256((tmp_path / "sweep.npy").read_bytes()).hexdigest(),
        target_distance_m=10,
        target_lateral_m=0,
        depth_valid=False,
        depth_gap_m=None,
    )
    doc = dict(
        schema_version=1,
        origin="recorded",
        episode_id="empty",
        route_xy_m=[[0, 0], [60, 0]],
        front_bumper_from_rear_axle_m=3.8,
        config=asdict(LidarRig().config(0.05)),
        sensor_to_ego=np.eye(4).tolist(),
        sweeps=[row],
        capture_records=[
            dict(
                frame_id=1,
                raw_sensor_file="sensor.bin",
                raw_sensor_sha256=hashlib.sha256(payload).hexdigest(),
                received_monotonic_s=1,
                read_completed_monotonic_s=1.01,
            )
        ],
    )
    (tmp_path / "sweeps.json").write_text(json.dumps(doc))
    result = audit(tmp_path / "sweeps.json", tmp_path / "audit")
    assert result["first_road_return_gap_m_min"] is None
    assert not result["any_positive_clearance"] and result["near_0_1m_road_return_total"] == 0
    data = np.load(tmp_path / "audit/processed_000001.npz")
    assert not data["bin_state"].any() and not data["road_supported_bins"].any()
