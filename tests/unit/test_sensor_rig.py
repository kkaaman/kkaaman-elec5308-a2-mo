"""Mapless sensor rig without a CARLA server: conversions, frame join and noise determinism.

A minimal fake ``carla`` module stands in for the simulator. Sensor callbacks are fed by
hand, in CARLA's left-handed conventions, and the resulting ``SensorFrame`` is checked
in the vehicle frame.
"""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from carla_adapter.sensor_rig import (
    SensorRig,
    WheelEncoder,
    carla_mount,
    imu_vectors,
    lidar_azimuth_coverage_deg,
    lidar_points_to_ego,
    radar_to_returns,
)
from evaluation.satnav_route import carla_geolocation
from mapless.frame import SensorState
from mapless.geometry import sensor_to_ego
from mapless.sensor_suite import Mount, load_sensor_suite

ROOT = Path(__file__).resolve().parents[2]
OFFSET = 1.38641
GEO = carla_geolocation(49.0, 8.0)


class Vec:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


@pytest.fixture
def carla(monkeypatch):
    fake = SimpleNamespace(
        Location=lambda x=0.0, y=0.0, z=0.0: Vec(x, y, z),
        Rotation=lambda pitch=0.0, yaw=0.0, roll=0.0: SimpleNamespace(pitch=pitch, yaw=yaw, roll=roll),
        Transform=lambda location=None, rotation=None: SimpleNamespace(location=location, rotation=rotation),
        AttachmentType=SimpleNamespace(Rigid="rigid"),
    )
    monkeypatch.setitem(sys.modules, "carla", fake)
    return fake


class Blueprint:
    def __init__(self, blueprint_id):
        self.id, self.values = blueprint_id, {}

    def has_attribute(self, name):
        return name != "noise_seed" or self.id.startswith("sensor.other")

    def set_attribute(self, name, value):
        self.values[name] = value


class Sensor:
    _next = 500

    def __init__(self, blueprint, transform):
        Sensor._next += 1
        self.id, self.blueprint, self.transform, self.is_alive = Sensor._next, blueprint, transform, True

    def listen(self, callback):
        self.callback = callback

    def stop(self):
        pass

    def destroy(self):
        self.is_alive = False
        return True


class World:
    def __init__(self):
        self.sensors = []

    def get_blueprint_library(self):
        return SimpleNamespace(find=Blueprint)

    def get_map(self):
        return SimpleNamespace(
            transform_to_geolocation=lambda loc: SimpleNamespace(
                latitude=GEO(loc.x, loc.y)[0], longitude=GEO(loc.x, loc.y)[1]
            )
        )

    def spawn_actor(self, blueprint, transform, attach_to=None, attachment_type=None):
        sensor = Sensor(blueprint, transform)
        self.sensors.append(sensor)
        return sensor


class Owner:
    def __init__(self):
        self.world, self._actors, self.stops = World(), [], 0

    def _require_owner(self):
        pass

    def emergency_stop(self):
        self.stops += 1


def suite(noise=False):
    loaded = load_sensor_suite(ROOT / "configs/sensor_suites/mapless_v1.json")
    return loaded if noise else loaded.without_noise()


def feed(rig, frame, t, *, gnss=True, skip=()):
    """Deliver one tick of CARLA-convention data to every sensor."""
    by_name = {s.blueprint.id + ":" + str(i): s for i, s in enumerate(rig.owner.world.sensors)}
    specs = [s for s in rig.suite.sensors if s.kind != "speedometer"]
    for spec, sensor in zip(specs, by_name.values()):
        if spec.name in skip:
            continue
        if spec.kind == "camera":
            raw = np.zeros((spec.height, spec.width, 4), np.uint8)
            raw[..., 2] = 200  # red in BGRA
            data = SimpleNamespace(frame=frame, timestamp=t, width=spec.width, height=spec.height, raw_data=raw.tobytes())
        elif spec.kind == "lidar":
            pts = np.array([[10.0, 2.0, -1.0, 0.5], [0.0, -5.0, 0.0, 0.2]], np.float32)  # 2 m right; 5 m left
            data = SimpleNamespace(frame=frame, timestamp=t, raw_data=pts.tobytes())
        elif spec.kind == "radar":
            data = RadarMeasurement(frame, t, [SimpleNamespace(depth=20.0, azimuth=0.1, altitude=0.05, velocity=-3.0)])
        elif spec.kind == "imu":
            data = SimpleNamespace(
                frame=frame,
                timestamp=t,
                accelerometer=Vec(1.0, 0.5, 9.81),
                gyroscope=Vec(0.01, 0.02, -0.3),  # CARLA z negative = turning left
                compass=math.radians(90),
            )
        elif spec.kind == "gnss":
            if not gnss:
                continue
            lat, lon = GEO(100.0, -20.0)
            data = SimpleNamespace(frame=frame, timestamp=t, latitude=lat, longitude=lon, altitude=2.0)
        sensor.callback(data)


class RadarMeasurement(list):
    def __init__(self, frame, timestamp, detections):
        super().__init__(detections)
        self.frame, self.timestamp = frame, timestamp


def test_lidar_conversion_flips_y_and_applies_the_mount():
    raw = np.array([[10.0, 2.0, -1.0, 0.5]], np.float32).tobytes()
    roof = sensor_to_ego(Mount(0.0, 0.0, 2.4), OFFSET)
    assert lidar_points_to_ego(raw, roof)[0] == pytest.approx([10 + OFFSET, -2.0, 1.4, 0.5], abs=1e-5)
    left = sensor_to_ego(Mount(0.0, 0.0, 2.0, yaw_deg=90), OFFSET)
    forward = np.array([[5.0, 0.0, 0.0, 1.0]], np.float32).tobytes()
    assert lidar_points_to_ego(forward, left)[0][:3] == pytest.approx([OFFSET, 5.0, 2.0], abs=1e-5)
    with pytest.raises(ValueError):
        lidar_points_to_ego(np.zeros(5, np.float32).tobytes(), roof)


def test_radar_imu_and_mount_signs():
    returns = radar_to_returns([SimpleNamespace(depth=20.0, azimuth=0.1, altitude=0.05, velocity=-3.0)])
    assert returns[0] == pytest.approx([20.0, -0.1, 0.05, -3.0])  # right in CARLA -> negative; velocity is range rate
    assert radar_to_returns([]).shape == (0, 4)
    accel, gyro = imu_vectors(Vec(1.0, 0.5, 9.81), Vec(0.01, 0.02, -0.3))
    assert accel == (1.0, -0.5, 9.81) and gyro == (-0.01, 0.02, 0.3)
    m = carla_mount(Mount(1.2, 0.5, 1.9, yaw_deg=60.0, pitch_deg=-5.0))
    assert (m["y"], m["yaw"], m["pitch"]) == (-0.5, -60.0, -5.0)


def test_azimuth_coverage():
    roof = sensor_to_ego(Mount(0.0, 0.0, 2.4), OFFSET)
    angles = np.radians(np.arange(0, 360, 0.5))
    ring = np.column_stack((10 * np.cos(angles) + OFFSET, 10 * np.sin(angles), np.full_like(angles, 2.4), np.ones_like(angles)))
    assert lidar_azimuth_coverage_deg(ring.astype(np.float32), roof) == pytest.approx(360.0)
    assert lidar_azimuth_coverage_deg(ring[: len(ring) // 2].astype(np.float32), roof) == pytest.approx(180.0)


def test_rig_builds_a_vehicle_frame_sensor_frame(carla):
    owner = Owner()
    rig = SensorRig(owner, object(), suite(), episode_id="e1", rear_axle_offset_m=OFFSET, dt_s=0.05, seed=7)
    with rig:
        assert len(owner.world.sensors) == 10 and owner._actors == owner.world.sensors
        ids = {s.blueprint.id for s in owner.world.sensors}
        assert ids == {"sensor.camera.rgb", "sensor.lidar.ray_cast", "sensor.other.radar", "sensor.other.gnss", "sensor.other.imu"}
        lidar_bp = next(s.blueprint for s in owner.world.sensors if s.blueprint.id == "sensor.lidar.ray_cast")
        assert lidar_bp.values["rotation_frequency"] == "20.0" and lidar_bp.values["noise_stddev"] == "0.0"
        imu_bp = next(s.blueprint for s in owner.world.sensors if s.blueprint.id == "sensor.other.imu")
        assert "noise_seed" not in lidar_bp.values and "noise_seed" in imu_bp.values
        assert rig.unsupported_attributes == {}
        feed(rig, 11, 0.55)
        frame = rig.read(11, 4.0, sim_time_s=0.55)
    assert [c.name for c in frame.cameras] == ["front", "front_left", "front_right", "rear"]
    assert frame.camera("front").rgb[0, 0].tolist() == [200, 0, 0]
    roof = frame.lidar("roof").points
    assert roof[0] == pytest.approx([10 + OFFSET, -2.0, 1.4, 0.5], abs=1e-5)
    assert roof[1][1] == pytest.approx(5.0)
    # CARLA's radar velocity is the range rate (measured live): -3 means closing at 3 m/s.
    assert frame.radar("radar_front").returns[0] == pytest.approx([20.0, -0.1, 0.05, -3.0])
    assert frame.imu.gyro_rps == pytest.approx((-0.01, 0.02, 0.3)) and frame.imu.yaw_rate_rps > 0
    assert frame.imu.compass_rad == pytest.approx(math.radians(90))  # noise-free
    assert frame.gnss is not None and frame.speed_mps == pytest.approx(4.0)
    assert frame.sim_time_s == pytest.approx(0.55) and not rig.sensors


def test_gnss_scheduling_and_degraded_frames(carla):
    owner = Owner()
    rig = SensorRig(owner, object(), suite(), episode_id="e", rear_axle_offset_m=OFFSET, dt_s=0.05, seed=1)
    with rig:
        feed(rig, 1, 0.05)
        first = rig.read(1, 0.0, sim_time_s=0.05)
        assert first.gnss is not None and first.degraded() == ()
        feed(rig, 2, 0.10, gnss=False)
        second = rig.read(2, 0.0, sim_time_s=0.10)
        assert second.gnss is None and second.status_of("gnss").state is SensorState.NOT_DUE
        # A missing LiDAR sweep gives a degraded frame, not a stop.
        feed(rig, 3, 0.15, gnss=False, skip=("roof",))
        third = rig.read(3, 0.0, sim_time_s=0.15, grace_s=0.05)
        assert third.status_of("roof").state is SensorState.MISSING and not third.lidars
        assert [s.name for s in third.degraded()] == ["roof"] and owner.stops == 0
        # The last fix was at 0.05 s and the period is 0.2 s. CARLA's schedule can slip one
        # tick, so GNSS is MISSING only once a fix is more than a tick late (after 0.30 s).
        for k, t in ((4, 0.20), (5, 0.25), (6, 0.30)):
            feed(rig, k, t, gnss=False)
            assert rig.read(k, 0.0, sim_time_s=t, grace_s=0.05).status_of("gnss").state is SensorState.NOT_DUE
        feed(rig, 7, 0.35, gnss=False)
        assert rig.read(7, 0.0, sim_time_s=0.35, grace_s=0.05).status_of("gnss").state is SensorState.MISSING
        # A sample stamped with another time is INVALID.
        feed(rig, 8, 0.41)
        stale = rig.read(8, 0.0, sim_time_s=0.40)
        assert all(s.state is SensorState.INVALID for s in stale.status if s.kind in ("camera", "lidar", "radar", "imu"))
        assert stale.imu is None and not stale.cameras
        # Strict mode (capture, calibration) raises and brakes instead.
        feed(rig, 9, 0.45, skip=("front",))
        with pytest.raises(TimeoutError, match="front missing"):
            rig.read(9, 0.0, sim_time_s=0.45, grace_s=0.05, strict=True)
        assert owner.stops == 1
        with pytest.raises(RuntimeError, match="already read"):
            rig.read(1, 0.0)


def test_a_bad_sample_is_invalid_with_its_reason(carla):
    owner = Owner()
    rig = SensorRig(owner, object(), suite(), episode_id="e", rear_axle_offset_m=OFFSET, dt_s=0.05, seed=1)
    with rig:
        feed(rig, 1, 0.05, skip=("rear",))
        rear = owner.world.sensors[3]  # spawn order follows the suite: front, front_left, front_right, rear
        rear.callback(SimpleNamespace(frame=1, timestamp=0.05, width=10, height=10, raw_data=b""))
        frame = rig.read(1, 0.0, sim_time_s=0.05, grace_s=0.05)
    status = frame.status_of("rear")
    assert status.state is SensorState.INVALID and "image size" in status.detail
    assert len(frame.cameras) == 3


def test_health_tracker_ages_against_expected_rates(carla):
    from mapless.health import HealthTracker

    owner = Owner()
    s = suite()
    rig = SensorRig(owner, object(), s, episode_id="e", rear_axle_offset_m=OFFSET, dt_s=0.05, seed=1)
    tracker = HealthTracker(s, 0.05)
    with rig:
        for k in range(1, 8):
            t = 0.05 * k
            feed(rig, k, t, gnss=(k == 1), skip=("radar_rear_left",) if k >= 4 else ())
            report = tracker.update(rig.read(k, 0.0, sim_time_s=t, grace_s=0.02))
    assert report["front"].age_s == 0 and not report["front"].overdue
    assert report["radar_rear_left"].consecutive_dropouts == 4
    assert report["radar_rear_left"].age_s == pytest.approx(0.20) and report["radar_rear_left"].overdue
    # GNSS fixed at 0.05 s; at 0.35 s its 0.30 s age is past the 0.2 s period, so it is overdue.
    assert report["gnss"].age_s == pytest.approx(0.30) and report["gnss"].overdue
    assert tracker.stale(["front", "roof", "radar_rear_left"], 0.15) == ["radar_rear_left"]


def test_noise_is_seeded_and_applied_on_the_main_thread(carla):
    def run(seed):
        owner = Owner()
        rig = SensorRig(owner, object(), suite(noise=True), episode_id="e", rear_axle_offset_m=OFFSET, dt_s=0.05, seed=seed)
        values = []
        with rig:
            for k in range(1, 4):
                feed(rig, k, 0.05 * k)
                f = rig.read(k, 5.0)
                values.append((f.speed_mps, f.imu.compass_rad))
            gnss_bp = next(s.blueprint for s in owner.world.sensors if s.blueprint.id == "sensor.other.gnss")
        return values, gnss_bp.values, rig.encoder.scale

    a, b, c = run(3), run(3), run(4)
    assert a == b and a[0] != c[0]
    values, gnss, scale = a
    assert float(gnss["noise_lat_stddev"]) > 0 and float(gnss["noise_lat_bias"]) != 0
    assert scale != 1.0 and values[0][0] != pytest.approx(5.0)


def test_rig_refuses_a_lidar_that_does_not_sweep_once_per_tick(carla):
    with pytest.raises(ValueError, match="full sweep"):
        SensorRig(Owner(), object(), suite(), episode_id="e", rear_axle_offset_m=OFFSET, dt_s=0.1, seed=1)


def test_wheel_encoder_scale_and_noise():
    spec = SimpleNamespace(white_mps=0.0, scale_sigma=0.0)
    assert WheelEncoder(spec, np.random.default_rng(0)).measure(5.0) == 5.0
