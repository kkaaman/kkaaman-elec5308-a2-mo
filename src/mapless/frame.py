"""One tick of sensor data: everything the mapless car is allowed to know.

A ``SensorFrame`` holds no pose, route, map, actor list or label. Its fields are fixed;
``tests/unit/test_mapless_firewall.py`` fails if a field is added without review.

Frames and units:
- Vehicle frame: right-handed, +x forward, +y left, +z up, origin at the rear axle.
- LiDAR points: in the vehicle frame (x, y, z, intensity).
- Radar returns: in the sensor's own polar frame, with azimuth positive to the left and
  elevation positive upwards. ``sensor_to_ego`` maps them into the vehicle frame.
- IMU: vehicle frame. Acceleration includes gravity, as an accelerometer measures it.
  The compass is a heading in radians, clockwise from north, as the sensor reports it.
- GNSS: WGS84 degrees. ``None`` on ticks without a fix.
"""

import math
from dataclasses import dataclass

import numpy as np

from autonomy.interfaces import CameraCalibration


def _finite(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _readonly(array, dtype=None):
    copy = np.array(array, dtype=dtype, copy=True)
    copy.setflags(write=False)
    return copy


def _transform(matrix):
    m = np.asarray(matrix, dtype=float)
    if m.shape != (4, 4) or not np.isfinite(m).all() or not np.allclose(m[3], (0, 0, 0, 1)):
        raise ValueError("sensor_to_ego must be a finite homogeneous 4x4 transform")
    r = m[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(r), 1.0):
        raise ValueError("sensor_to_ego rotation must be proper and orthonormal")
    return tuple(tuple(float(v) for v in row) for row in m)


@dataclass(frozen=True)
class CameraImage:
    name: str
    rgb: np.ndarray
    calibration: CameraCalibration

    def __post_init__(self):
        if not isinstance(self.calibration, CameraCalibration):
            raise TypeError("calibration must be a CameraCalibration")
        rgb = self.rgb
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
            raise ValueError("rgb must be a uint8 array")
        if rgb.ndim != 3 or rgb.shape[2] != 3 or 0 in rgb.shape:
            raise ValueError("rgb must be a non-empty HxWx3 array")
        object.__setattr__(self, "rgb", _readonly(rgb))


@dataclass(frozen=True)
class LidarSweep:
    name: str
    points: np.ndarray  # N x 4 float32: x, y, z (vehicle frame, metres), intensity
    sensor_to_ego: tuple

    def __post_init__(self):
        points = np.asarray(self.points)
        if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all():
            raise ValueError("LiDAR points must be a finite N x 4 array")
        object.__setattr__(self, "points", _readonly(points, np.float32))
        object.__setattr__(self, "sensor_to_ego", _transform(self.sensor_to_ego))

    @property
    def xyz(self):
        return self.points[:, :3]


@dataclass(frozen=True)
class RadarReturns:
    name: str
    # N x 4 float32: range_m, azimuth_rad (+left), elevation_rad (+up),
    # range_rate_mps (+ = moving away from the sensor).
    returns: np.ndarray
    sensor_to_ego: tuple

    def __post_init__(self):
        returns = np.asarray(self.returns)
        if returns.ndim != 2 or returns.shape[1] != 4 or not np.isfinite(returns).all():
            raise ValueError("Radar returns must be a finite N x 4 array")
        if (returns[:, 0] < 0).any():
            raise ValueError("Radar range must be non-negative")
        object.__setattr__(self, "returns", _readonly(returns, np.float32))
        object.__setattr__(self, "sensor_to_ego", _transform(self.sensor_to_ego))

    def points_ego(self):
        """Returns as vehicle-frame points (N x 3)."""
        r, az, el = (self.returns[:, i].astype(float) for i in range(3))
        local = np.column_stack((r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el)))
        m = np.asarray(self.sensor_to_ego)
        return local @ m[:3, :3].T + m[:3, 3]


@dataclass(frozen=True)
class GnssFix:
    latitude_deg: float
    longitude_deg: float
    altitude_m: float

    def __post_init__(self):
        for name in ("latitude_deg", "longitude_deg", "altitude_m"):
            _finite(name, getattr(self, name))
        if not -90 <= self.latitude_deg <= 90 or not -180 <= self.longitude_deg <= 180:
            raise ValueError("GNSS fix out of range")


@dataclass(frozen=True)
class ImuSample:
    accel_mps2: tuple  # x, y, z in the vehicle frame
    gyro_rps: tuple  # x, y, z in the vehicle frame (z = yaw rate, counter-clockwise)
    compass_rad: float  # clockwise from north

    def __post_init__(self):
        for name in ("accel_mps2", "gyro_rps"):
            value = tuple(float(v) for v in getattr(self, name))
            if len(value) != 3 or not all(math.isfinite(v) for v in value):
                raise ValueError(f"{name} must be three finite values")
            object.__setattr__(self, name, value)
        _finite("compass_rad", self.compass_rad)

    @property
    def yaw_rate_rps(self):
        return self.gyro_rps[2]


@dataclass(frozen=True)
class SensorFrame:
    episode_id: str
    frame_id: int
    sim_time_s: float
    received_monotonic_s: float
    cameras: tuple
    lidars: tuple
    radars: tuple
    gnss: GnssFix | None
    imu: ImuSample
    speed_mps: float

    def __post_init__(self):
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise ValueError("episode_id must be a non-empty string")
        if type(self.frame_id) is not int or self.frame_id < 0:
            raise ValueError("frame_id must be a non-negative integer")
        for name in ("sim_time_s", "received_monotonic_s", "speed_mps"):
            _finite(name, getattr(self, name))
        if self.sim_time_s < 0:
            raise ValueError("sim_time_s must be non-negative")
        for name, kind in (
            ("cameras", CameraImage),
            ("lidars", LidarSweep),
            ("radars", RadarReturns),
        ):
            items = tuple(getattr(self, name))
            if any(not isinstance(item, kind) for item in items):
                raise TypeError(f"{name} must hold {kind.__name__} values")
            object.__setattr__(self, name, items)
        if not self.cameras:
            raise ValueError("A frame needs at least one camera image")
        names = [item.name for group in (self.cameras, self.lidars, self.radars) for item in group]
        if len(set(names)) != len(names):
            raise ValueError("Sensor names must be unique within a frame")
        if self.gnss is not None and not isinstance(self.gnss, GnssFix):
            raise TypeError("gnss must be a GnssFix or None")
        if not isinstance(self.imu, ImuSample):
            raise TypeError("imu must be an ImuSample")

    def camera(self, name):
        return next(c for c in self.cameras if c.name == name)

    def lidar(self, name):
        return next(item for item in self.lidars if item.name == name)

    def radar(self, name):
        return next(item for item in self.radars if item.name == name)
