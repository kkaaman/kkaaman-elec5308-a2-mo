"""Mount a mapless ``SensorSuite`` on the ego vehicle and deliver one ``SensorFrame`` per tick.

Conversions from CARLA (left-handed: +x forward, +y right, +z up) to the vehicle frame
(right-handed: +x forward, +y left, +z up, origin at the rear axle):
- **LiDAR** points ``(x, y, z, i)`` in the sensor frame become ``(x, -y, z, i)``, then the
  mount transform is applied.
- **Radar** azimuth is positive to the right in CARLA, so the sign is flipped. Altitude is
  already positive upwards. CARLA's ``velocity`` is the relative velocity projected on the
  line of sight. ``RADAR_VELOCITY_IS_RANGE_RATE`` records its sign, which
  ``scripts/check_sensor_rig.py`` measures live.
- **IMU** acceleration ``(ax, ay, az)`` becomes ``(ax, -ay, az)``. Angular velocity is a
  pseudovector, so under the y-reflection it becomes ``(-gx, gy, -gz)``. The yaw-rate
  sign (``-gz``) was confirmed live in TEST-03; the rig check confirms all three axes.
- **Compass:** passed through, clockwise from north, with seeded white noise added here,
  because CARLA's IMU has no compass-noise setting.
- **Accelerometer:** CARLA's IMU reports huge accelerations for its first few ticks after
  spawning (measured live). Consumers must not trust acceleration from the first ticks
  of an episode; the calibration gate skips 5.

The **speedometer** is the one value taken from the simulator's state. It stands for a
wheel encoder: true forward speed with a constant seeded scale error and white noise.
The Leaderboard SENSORS track provides the same pseudo-sensor.

Noise draws use one seeded generator, applied on the main thread in tick order, so a seed
reproduces the same noise. Callbacks only convert and store; they never tick the
simulator.
"""

import hashlib
import math
import threading
import time
from collections import defaultdict

import numpy as np

from carla_adapter.conversions import decode_rgb
from carla_adapter.localisation_sensors import geo_calibration
from mapless.frame import (
    CameraImage,
    GnssFix,
    ImuSample,
    LidarSweep,
    RadarReturns,
    SensorFrame,
    SensorState,
    SensorStatus,
)
from mapless.geometry import camera_calibration, sensor_to_ego
from mapless.sensor_suite import BANNED_BLUEPRINTS, BLUEPRINTS

# CARLA 0.9.16 radar ``velocity`` sign: True if positive means the target is receding.
# Measured live by scripts/check_sensor_rig.py on 22 Oct 2026 (PC 2, Town10HD_Opt).
# With False, the moving-target and moving-ego cases both gave a median
# measured/expected ratio of -0.99 (834 and 6,838 returns), so CARLA's value is the range
# rate (positive when receding), despite the documentation's "towards the sensor".
RADAR_VELOCITY_IS_RANGE_RATE = True


def carla_mount(mount):
    """CARLA transform arguments for a vehicle-frame mount."""
    return dict(x=mount.x_m, y=-mount.y_m, z=mount.z_m, pitch=mount.pitch_deg, yaw=-mount.yaw_deg, roll=0.0)


def lidar_points_to_ego(raw, sensor_to_ego_matrix):
    """CARLA LiDAR buffer (float32 x, y, z, intensity) to vehicle-frame N x 4."""
    points = np.frombuffer(raw, dtype=np.float32)
    if points.size % 4:
        raise ValueError("LiDAR buffer is not a whole number of points")
    points = points.reshape(-1, 4).astype(np.float64)
    local = np.column_stack((points[:, 0], -points[:, 1], points[:, 2]))
    m = np.asarray(sensor_to_ego_matrix)
    ego = local @ m[:3, :3].T + m[:3, 3]
    return np.column_stack((ego, points[:, 3])).astype(np.float32)


def radar_to_returns(detections):
    """CARLA radar detections to N x 4: range, azimuth (+left), elevation (+up), range rate."""
    sign = 1.0 if RADAR_VELOCITY_IS_RANGE_RATE else -1.0
    rows = [(d.depth, -d.azimuth, d.altitude, sign * d.velocity) for d in detections]
    return np.asarray(rows, dtype=np.float32).reshape(-1, 4)


def imu_vectors(accelerometer, gyroscope):
    """CARLA IMU vectors to vehicle-frame acceleration and angular velocity."""
    accel = (accelerometer.x, -accelerometer.y, accelerometer.z)
    gyro = (-gyroscope.x, gyroscope.y, -gyroscope.z)
    return accel, gyro


def lidar_azimuth_coverage_deg(points, sensor_to_ego_matrix, bins=360):
    """Share of 1-degree azimuth bins, around the sensor, that hold at least one point."""
    if len(points) == 0:
        return 0.0
    m = np.asarray(sensor_to_ego_matrix)
    local = (points[:, :3].astype(float) - m[:3, 3]) @ m[:3, :3]
    azimuth = np.degrees(np.arctan2(local[:, 1], local[:, 0])) % 360.0
    filled = np.unique(np.floor(azimuth * bins / 360.0).astype(int))
    return float(len(filled) * 360.0 / bins)


class WheelEncoder:
    """True speed with a constant seeded scale error and white noise."""

    def __init__(self, spec, rng):
        self.spec = spec
        self.scale = 1.0 + float(rng.normal(0.0, spec.scale_sigma)) if spec.scale_sigma else 1.0
        self.rng = rng

    def measure(self, true_speed_mps):
        noise = float(self.rng.normal(0.0, self.spec.white_mps)) if self.spec.white_mps else 0.0
        return self.scale * true_speed_mps + noise


def _attributes(spec, *, dt_s, gnss_scale=None, rng=None):
    if spec.kind == "camera":
        return dict(
            image_size_x=spec.width,
            image_size_y=spec.height,
            fov=spec.fov_deg,
            sensor_tick=0.0,
            lens_k=0.0,
            lens_kcube=0.0,
        )
    if spec.kind == "lidar":
        return dict(
            channels=spec.channels,
            range=spec.range_m,
            rotation_frequency=spec.rotation_frequency_hz,
            points_per_second=spec.points_per_second,
            upper_fov=spec.upper_fov_deg,
            lower_fov=spec.lower_fov_deg,
            horizontal_fov=spec.horizontal_fov_deg,
            noise_stddev=spec.noise_stddev_m,
            dropoff_general_rate=spec.dropoff_general_rate,
            dropoff_intensity_limit=spec.dropoff_intensity_limit,
            dropoff_zero_intensity=spec.dropoff_zero_intensity,
            atmosphere_attenuation_rate=spec.atmosphere_attenuation_rate,
            sensor_tick=0.0,
        )
    if spec.kind == "radar":
        return dict(
            horizontal_fov=spec.horizontal_fov_deg,
            vertical_fov=spec.vertical_fov_deg,
            range=spec.range_m,
            points_per_second=spec.points_per_second,
            sensor_tick=0.0,
        )
    if spec.kind == "gnss":
        lat_per_m, lon_per_m = gnss_scale
        bias = rng.normal(0.0, spec.bias_sigma_m, 2) if spec.bias_sigma_m else np.zeros(2)
        # bias[0] is east (longitude), bias[1] is north (latitude), in metres.
        return dict(
            sensor_tick=spec.period_s,
            noise_lat_stddev=spec.white_m * lat_per_m,
            noise_lon_stddev=spec.white_m * lon_per_m,
            noise_lat_bias=float(bias[1]) * lat_per_m,
            noise_lon_bias=float(bias[0]) * lon_per_m,
            noise_alt_stddev=0.0,
            noise_alt_bias=0.0,
        )
    if spec.kind == "imu":
        bias = float(rng.normal(0.0, spec.gyro_bias_sigma_rps)) if spec.gyro_bias_sigma_rps else 0.0
        return dict(
            sensor_tick=0.0,
            noise_accel_stddev_x=spec.accel_white_mps2,
            noise_accel_stddev_y=spec.accel_white_mps2,
            noise_accel_stddev_z=spec.accel_white_mps2,
            noise_gyro_stddev_x=spec.gyro_white_rps,
            noise_gyro_stddev_y=spec.gyro_white_rps,
            noise_gyro_stddev_z=spec.gyro_white_rps,
            # Vehicle yaw rate is -gyro z, so a vehicle-frame bias of +b is a CARLA bias of -b.
            noise_gyro_bias_z=-bias,
        )
    raise ValueError(f"No simulator blueprint for {spec.kind}")


class SensorRig:
    """Own the suite's sensors on one ego vehicle inside a ``SynchronousWorldSession``."""

    def __init__(self, owner, vehicle, suite, *, episode_id, rear_axle_offset_m, dt_s, seed, capacity=8):
        problems = suite.validate_for_timestep(dt_s)
        if problems:
            raise ValueError("; ".join(problems))
        if type(seed) is not int or seed < 0:
            raise ValueError("Non-negative integer seed required")
        self.owner, self.vehicle, self.suite = owner, vehicle, suite
        self.episode_id, self.dt_s, self.seed, self.capacity = episode_id, dt_s, seed, capacity
        self.rear_axle_offset_m = float(rear_axle_offset_m)
        self.rng = np.random.default_rng(seed)
        self.encoder = WheelEncoder(suite.speedometer, self.rng)
        self.extrinsics = {
            s.name: sensor_to_ego(s.mount, self.rear_axle_offset_m)
            for s in suite.sensors
            if hasattr(s, "mount")
        }
        self.calibrations = {
            s.name: camera_calibration(s, self.rear_axle_offset_m) for s in suite.cameras
        }
        self.every_tick = tuple(s.name for s in suite.sensors if s.kind in ("camera", "lidar", "radar", "imu"))
        self.sensors = []
        self.attributes = {}
        self.unsupported_attributes = {}
        self.callback_errors = []
        self.counts = defaultdict(int)
        self.receipts = {}
        self._data = defaultdict(dict)
        self._first_receipt = {}
        self._gnss = {}
        self._errors = {}
        self._last_fix_s = None
        self._last_sim_s = None
        self._first_sim_s = None
        self.effective_attributes = {}
        identity = f"{suite.sha256}|{self.rear_axle_offset_m:.5f}"
        self.calibration_id = hashlib.sha256(identity.encode()).hexdigest()[:16]
        self._condition = threading.Condition()
        self._retired = -1
        self._closed = False

    # -- lifecycle -------------------------------------------------------------
    def __enter__(self):
        import carla

        self.owner._require_owner()
        try:
            world = self.owner.world
            library = world.get_blueprint_library()
            gnss_scale = None
            if self.suite.gnss is not None:
                carla_map = world.get_map()

                def geolocate(x, y):
                    fix = carla_map.transform_to_geolocation(carla.Location(x=x, y=y, z=0.0))
                    return fix.latitude, fix.longitude

                _, to_geo, _ = geo_calibration(geolocate)
                gnss_scale = (float(np.hypot(*to_geo[0])), float(np.hypot(*to_geo[1])))
            for offset, spec in enumerate(s for s in self.suite.sensors if s.kind != "speedometer"):
                blueprint_id = BLUEPRINTS[spec.kind]
                if blueprint_id in BANNED_BLUEPRINTS:  # defensive; BLUEPRINTS never lists one
                    raise ValueError(f"{blueprint_id} is not allowed on the car")
                bp = library.find(blueprint_id)
                values = _attributes(spec, dt_s=self.dt_s, gnss_scale=gnss_scale, rng=self.rng)
                if bp.has_attribute("noise_seed"):
                    values["noise_seed"] = self.seed * 101 + offset
                applied, missing = {}, []
                for key, value in values.items():
                    if bp.has_attribute(key):
                        bp.set_attribute(key, str(value))
                        applied[key] = value
                    else:
                        missing.append(key)
                required = {"image_size_x", "image_size_y", "fov", "channels", "range", "rotation_frequency"}
                if required & set(missing):
                    raise RuntimeError(f"{blueprint_id} lacks {sorted(required & set(missing))}")
                self.attributes[spec.name] = dict(blueprint=blueprint_id, **applied)
                if missing:
                    self.unsupported_attributes[spec.name] = missing
                mount = carla_mount(spec.mount)
                transform = carla.Transform(
                    carla.Location(x=mount["x"], y=mount["y"], z=mount["z"]),
                    carla.Rotation(pitch=mount["pitch"], yaw=mount["yaw"], roll=mount["roll"]),
                )
                sensor = world.spawn_actor(
                    bp, transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid
                )
                self.sensors.append(sensor)
                self.owner._actors.append(sensor)
                # What CARLA actually applied (read back from the actor), for the record.
                self.effective_attributes[spec.name] = {
                    k: str(v) for k, v in dict(getattr(sensor, "attributes", None) or {}).items()
                }
                sensor.listen(lambda data, s=spec: self._receive(s, data))
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        with self._condition:
            self._closed = True
            self._data.clear()
            self._gnss.clear()
            self._condition.notify_all()
        errors = []
        for sensor in reversed(self.sensors):
            try:
                if sensor.is_alive:
                    sensor.stop()
                    if not sensor.destroy():
                        errors.append(f"Sensor destroy failed: {sensor.id}")
            except Exception as error:
                errors.append(str(error))
        self.sensors.clear()
        if errors:
            raise RuntimeError("; ".join(errors))

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    # -- callbacks (CARLA thread) ------------------------------------------------
    def _receive(self, spec, data):
        received = time.perf_counter()
        try:
            if spec.kind == "camera":
                if (data.width, data.height) != (spec.width, spec.height):
                    raise ValueError(f"image size changed to {data.width}x{data.height}")
                payload = decode_rgb(data.raw_data, data.width, data.height)
            elif spec.kind == "lidar":
                payload = lidar_points_to_ego(data.raw_data, self.extrinsics[spec.name])
            elif spec.kind == "radar":
                payload = radar_to_returns(data)
            elif spec.kind == "imu":
                payload = (*imu_vectors(data.accelerometer, data.gyroscope), float(data.compass))
            elif spec.kind == "gnss":
                payload = (float(data.latitude), float(data.longitude), float(data.altitude))
            else:
                raise ValueError(f"Unexpected sensor kind {spec.kind}")
            with self._condition:
                if self._closed or data.frame <= self._retired:
                    self.counts["late"] += 1
                    return
                if spec.kind == "gnss":
                    self._gnss[data.frame] = (float(data.timestamp), payload)
                else:
                    self._data[data.frame][spec.name] = (float(data.timestamp), payload)
                    self._first_receipt.setdefault(data.frame, received)
                self.receipts.setdefault(data.frame, {})[spec.name] = received
                while len(self._data) > self.capacity:
                    del self._data[min(self._data)]
                    self.counts["overflow"] += 1
                self._condition.notify_all()
        except Exception as error:
            self.callback_errors.append(f"{spec.name}: {error}")
            del self.callback_errors[:-16]
            frame = getattr(data, "frame", None)
            if type(frame) is int:
                with self._condition:
                    self._errors.setdefault(frame, {})[spec.name] = str(error)
                    self._condition.notify_all()

    # -- main thread ---------------------------------------------------------------
    def _gnss_since(self, sim_time_s):
        """Time since the last fix (or since the rig's first frame), or None."""
        if self.suite.gnss is None or sim_time_s is None:
            return None
        base = self._last_fix_s if self._last_fix_s is not None else self._first_sim_s
        return None if base is None else sim_time_s - base

    def _gnss_expected(self, sim_time_s):
        """A fix is scheduled about now: worth waiting briefly for its callback."""
        since = self._gnss_since(sim_time_s)
        return since is not None and since >= self.suite.gnss.period_s - self.dt_s / 2

    def _gnss_overdue(self, sim_time_s):
        """MISSING only when a fix is more than one tick late. CARLA's sensor_tick
        schedule can land one tick either side of the nominal period."""
        since = self._gnss_since(sim_time_s)
        return since is not None and since > self.suite.gnss.period_s + self.dt_s + 1e-6

    def read(self, frame, true_speed_mps, *, sim_time_s=None, grace_s=0.5, strict=False):
        """The frame's ``SensorFrame``, degraded rather than refused when sensors are missing.

        Waits up to ``grace_s`` (wall clock) for every per-tick sensor. Anything still
        absent is reported MISSING, and a sample rejected on arrival is INVALID. With
        ``sim_time_s`` (the world snapshot's time), a sample stamped with any other time
        is INVALID. ``strict=True`` (capture and calibration) raises instead and brakes.
        ``true_speed_mps`` feeds only the wheel-encoder model.
        """
        if type(frame) is not int or frame < 0 or not math.isfinite(true_speed_mps):
            raise ValueError("Invalid frame or speed")
        if sim_time_s is not None and not math.isfinite(sim_time_s):
            raise ValueError("sim_time_s must be finite")
        deadline = time.perf_counter() + grace_s
        try:
            with self._condition:
                if self._closed or frame <= self._retired:
                    raise RuntimeError("Rig closed or frame already read")
                while True:
                    have = set(self._data.get(frame, {})) | set(self._errors.get(frame, {}))
                    remaining = deadline - time.perf_counter()
                    if all(name in have for name in self.every_tick) or remaining <= 0:
                        break
                    self._condition.wait(remaining)
                entries = self._data.pop(frame, {})
                errors = self._errors.pop(frame, {})
                reference = sim_time_s
                if reference is None and entries:
                    stamps = sorted(t for t, _ in entries.values())
                    reference = stamps[len(stamps) // 2]
                if self._gnss_expected(reference) and frame not in self._gnss:
                    wait_until = time.perf_counter() + min(0.3, grace_s)
                    while frame not in self._gnss and time.perf_counter() < wait_until:
                        self._condition.wait(0.01)
                gnss_entry = self._gnss.pop(frame, None)
                received = self._first_receipt.pop(frame, time.perf_counter())
                self._retire(frame)
        except Exception:
            self.owner.emergency_stop()
            raise
        if reference is None:
            reference = (self._last_sim_s + self.dt_s) if self._last_sim_s is not None else 0.0
        self._last_sim_s = reference
        if self._first_sim_s is None:
            self._first_sim_s = reference
        statuses, valid = [], {}
        due = self._gnss_overdue(reference)
        for spec in self.suite.sensors:
            name = spec.name
            if spec.kind == "speedometer":
                statuses.append(SensorStatus(name, spec.kind, SensorState.OK, reference))
                continue
            if name in errors:
                statuses.append(SensorStatus(name, spec.kind, SensorState.INVALID, detail=errors[name]))
                continue
            entry = gnss_entry if spec.kind == "gnss" else entries.get(name)
            if entry is None:
                state = SensorState.NOT_DUE if spec.kind == "gnss" and not due else SensorState.MISSING
                statuses.append(SensorStatus(name, spec.kind, state))
                continue
            stamp, payload = entry
            if abs(stamp - reference) > 1e-6:
                statuses.append(
                    SensorStatus(name, spec.kind, SensorState.INVALID, detail=f"timestamp {stamp:.6f} != {reference:.6f}")
                )
                continue
            statuses.append(SensorStatus(name, spec.kind, SensorState.OK, stamp))
            valid[name] = payload
        bad = [s for s in statuses if s.state in (SensorState.MISSING, SensorState.INVALID)]
        for status in statuses:
            self.counts[f"state_{status.state.value}"] += 1
        if bad:
            self.counts["degraded_frames"] += 1
            if strict:
                self.owner.emergency_stop()
                raise TimeoutError(
                    f"Frame {frame}: " + ", ".join(f"{s.name} {s.state.value} {s.detail}".strip() for s in bad)
                )
        gnss = None
        if self.suite.gnss is not None and self.suite.gnss.name in valid:
            gnss = GnssFix(*valid[self.suite.gnss.name])
            self._last_fix_s = reference
            self.counts["gnss_fixes"] += 1
        imu = None
        imu_spec = self.suite.imu
        if imu_spec.name in valid:
            accel, gyro, compass = valid[imu_spec.name]
            if imu_spec.compass_white_rad:
                compass = (compass + float(self.rng.normal(0.0, imu_spec.compass_white_rad))) % (2 * math.pi)
            imu = ImuSample(accel, gyro, compass)
        self.counts["frames"] += 1
        return SensorFrame(
            episode_id=self.episode_id,
            frame_id=frame,
            sim_time_s=reference,
            received_monotonic_s=received,
            cameras=tuple(
                CameraImage(s.name, valid[s.name], self.calibrations[s.name])
                for s in self.suite.cameras
                if s.name in valid
            ),
            lidars=tuple(
                LidarSweep(s.name, valid[s.name], self.extrinsics[s.name]) for s in self.suite.lidars if s.name in valid
            ),
            radars=tuple(
                RadarReturns(s.name, valid[s.name], self.extrinsics[s.name]) for s in self.suite.radars if s.name in valid
            ),
            gnss=gnss,
            imu=imu,
            speed_mps=self.encoder.measure(true_speed_mps),
            status=tuple(statuses),
            calibration_id=self.calibration_id,
        )

    def _retire(self, frame):
        self._retired = max(self._retired, frame)
        for store in (self._data, self._gnss, self._first_receipt, self._errors):
            for key in [k for k in store if k <= frame]:
                del store[key]
        for key in [k for k in self.receipts if k < frame - 64]:
            del self.receipts[key]

    def record(self):
        return dict(
            suite=self.suite.as_dict(),
            suite_sha256=self.suite.sha256,
            seed=self.seed,
            rear_axle_offset_m=self.rear_axle_offset_m,
            calibration_id=self.calibration_id,
            attributes=self.attributes,
            effective_attributes=self.effective_attributes,
            unsupported_attributes=self.unsupported_attributes,
            wheel_encoder_scale=self.encoder.scale,
            radar_velocity_is_range_rate=RADAR_VELOCITY_IS_RANGE_RATE,
            counts=dict(self.counts),
            callback_errors=list(self.callback_errors),
        )


def true_forward_speed(snapshot, vehicle_id):
    """Forward speed of an actor from a world snapshot (wheel-encoder input only)."""
    actor = snapshot.find(vehicle_id)
    if actor is None:
        raise RuntimeError("Ego vehicle missing from the world snapshot")
    v = actor.get_velocity()
    f = actor.get_transform().get_forward_vector()
    return float(v.x * f.x + v.y * f.y + v.z * f.z)
