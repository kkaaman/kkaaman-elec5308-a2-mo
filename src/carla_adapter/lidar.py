"""CARLA ray-cast LiDAR capture delivering vehicle-frame :class:`LidarSweep` packets (E2-02).

CARLA's LiDAR buffer is float32 ``(x, y, z, intensity)`` in the **sensor** frame of a
left-handed world (+X forward, +Y right, +Z up). The adapter reflects Y once and
applies the rigid sensor-to-ego transform once, so the core receives right-handed
rear-axle ego coordinates (+X forward, +Y left, +Z up, metres) and never sees a raw
buffer. The blueprint's ``rotation_frequency`` is set to ``1 / fixed_delta_seconds``
so one callback holds exactly one full rotation per simulation tick.

Only this module and the capture script touch ``carla``; decoding and the known
object check are plain NumPy so they are unit-tested without a server.
"""

from __future__ import annotations

import math
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np

from obstacle_perception.lidar import LidarConfig, LidarSweep, validate_rigid_transform


@dataclass(frozen=True)
class LidarRig:
    """Actor-relative mount (metres, degrees) and ray-cast blueprint settings.

    ``x_actor_m``/``y_actor_m``/``z_actor_m`` are in the ego actor frame (+Y left);
    ``yaw_deg`` is counterclockwise about +Z. Blueprint settings mirror CARLA's
    ``sensor.lidar.ray_cast`` attributes; ``rotation_frequency`` is not stored
    because it is always derived from the world timestep.
    """

    x_actor_m: float = 0.0
    y_actor_m: float = 0.0
    z_actor_m: float = 2.4
    yaw_deg: float = 0.0
    channels: int = 32
    range_m: float = 50.0
    points_per_second: int = 640000
    upper_fov_deg: float = 10.0
    lower_fov_deg: float = -30.0
    horizontal_fov_deg: float = 360.0
    noise_stddev_m: float = 0.0
    dropoff_general_rate: float = 0.0
    dropoff_intensity_limit: float = 1.0
    dropoff_zero_intensity: float = 0.0
    atmosphere_attenuation_rate: float = 0.004

    def __post_init__(self):
        for name in ("x_actor_m", "y_actor_m", "z_actor_m", "yaw_deg"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        for name in (
            "noise_stddev_m",
            "dropoff_general_rate",
            "dropoff_intensity_limit",
            "dropoff_zero_intensity",
            "atmosphere_attenuation_rate",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("dropoff_general_rate", "dropoff_intensity_limit", "dropoff_zero_intensity"):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must be at most 1")
        self.config(0.05)  # validate the shared range/density fields once

    def config(self, dt_s):
        """Core :class:`LidarConfig` with ``rotation_frequency_hz = 1 / dt_s``."""
        return LidarConfig.for_timestep(
            dt_s,
            channels=self.channels,
            range_m=self.range_m,
            points_per_second=self.points_per_second,
            upper_fov_deg=self.upper_fov_deg,
            lower_fov_deg=self.lower_fov_deg,
            horizontal_fov_deg=self.horizontal_fov_deg,
        )

    def sensor_to_ego(self, rear_axle_offset_m):
        """Rigid sensor-to-rear-axle transform; the actor origin sits ``rear_axle_offset_m`` ahead of the axle."""
        if not math.isfinite(rear_axle_offset_m):
            raise ValueError("rear_axle_offset_m must be finite")
        yaw = math.radians(self.yaw_deg)
        c, s = math.cos(yaw), math.sin(yaw)
        return validate_rigid_transform(
            [
                [c, -s, 0.0, self.x_actor_m + rear_axle_offset_m],
                [s, c, 0.0, self.y_actor_m],
                [0.0, 0.0, 1.0, self.z_actor_m],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    def blueprint_attributes(self, dt_s):
        """``sensor.lidar.ray_cast`` attribute values as strings, one rotation per tick."""
        config = self.config(dt_s)
        return {
            key: str(value)
            for key, value in dict(
                channels=config.channels,
                range=config.range_m,
                rotation_frequency=config.rotation_frequency_hz,
                points_per_second=config.points_per_second,
                upper_fov=config.upper_fov_deg,
                lower_fov=config.lower_fov_deg,
                horizontal_fov=config.horizontal_fov_deg,
                noise_stddev=self.noise_stddev_m,
                dropoff_general_rate=self.dropoff_general_rate,
                dropoff_intensity_limit=self.dropoff_intensity_limit,
                dropoff_zero_intensity=self.dropoff_zero_intensity,
                atmosphere_attenuation_rate=self.atmosphere_attenuation_rate,
                sensor_tick=0.0,
            ).items()
        }

    def mount_arguments(self):
        """CARLA ``Location``/``Rotation`` keyword values (left-handed: Y and yaw negated)."""
        return dict(x=self.x_actor_m, y=-self.y_actor_m, z=self.z_actor_m), dict(
            pitch=0.0, yaw=-self.yaw_deg, roll=0.0
        )

    def to_dict(self):
        return asdict(self)


def lidar_points_to_ego(raw, sensor_to_ego, *, range_m=None):
    """Decode a CARLA LiDAR buffer into ego-frame N-by-4 float32 points.

    Reflects CARLA's +Y-right to +Y-left, applies ``sensor_to_ego`` exactly once and
    drops nonfinite rows and rows beyond ``range_m`` (sensor range, metres) if given.
    Intensity (column 3) is clipped to ``[0, 1]``. Returns ``(points, dropped)``
    where ``dropped`` counts removed rows.
    """
    matrix = validate_rigid_transform(sensor_to_ego)
    data = np.frombuffer(raw, dtype=np.float32)
    if data.size % 4:
        raise ValueError("LiDAR buffer is not a whole number of (x, y, z, i) points")
    data = data.reshape(-1, 4).astype(np.float64)
    finite = np.isfinite(data).all(axis=1)
    if range_m is not None:
        finite &= np.linalg.norm(data[:, :3], axis=1) <= range_m + 1e-3
    data = data[finite]
    local = np.column_stack((data[:, 0], -data[:, 1], data[:, 2]))
    ego = local @ matrix[:3, :3].T + matrix[:3, 3]
    intensity = np.clip(data[:, 3], 0.0, 1.0)
    return np.column_stack((ego, intensity)).astype(np.float32), int((~finite).sum())


def known_object_check(
    sweep, *, center_ego_m, extent_m, tolerance_m=0.3, min_points=6, search_margin_m=1.0
):
    """Verify a known box shows up where expected in a vehicle-frame sweep.

    ``center_ego_m`` and ``extent_m`` describe the object's axis-aligned box in the
    rear-axle frame (metres, full extents). Points inside the box grown by
    ``search_margin_m`` must number at least ``min_points``, the nearest observed X
    face among them must lie within ``tolerance_m`` of the expected near face, and
    all intensities must lie in ``[0, 1]``. Returns a JSON-friendly dict with
    ``passed``.
    """
    if not math.isfinite(search_margin_m) or search_margin_m < tolerance_m:
        raise ValueError("search_margin_m must be finite and at least tolerance_m")
    if not isinstance(sweep, LidarSweep):
        raise TypeError("known_object_check expects a LidarSweep")
    centre = np.asarray(center_ego_m, dtype=float)
    extent = np.asarray(extent_m, dtype=float)
    if centre.shape != (3,) or extent.shape != (3,) or (extent <= 0).any():
        raise ValueError("Expected XYZ centre and positive XYZ extent")
    low, high = centre - extent / 2 - search_margin_m, centre + extent / 2 + search_margin_m
    xyz = sweep.points_xyzi[:, :3]
    inside = ((xyz >= low) & (xyz <= high)).all(axis=1)
    points = xyz[inside]
    expected_face = float(centre[0] - extent[0] / 2)
    observed_face = float(points[:, 0].min()) if len(points) else None
    error = None if observed_face is None else observed_face - expected_face
    intensity = sweep.points_xyzi[:, 3]
    intensity_ok = bool(sweep.point_count == 0 or (intensity.min() >= 0 and intensity.max() <= 1))
    passed = bool(
        len(points) >= min_points
        and error is not None
        and abs(error) <= tolerance_m
        and intensity_ok
    )
    return dict(
        passed=passed,
        points_in_box=int(len(points)),
        min_points=min_points,
        expected_near_face_x_m=expected_face,
        observed_near_face_x_m=observed_face,
        near_face_error_m=error,
        tolerance_m=tolerance_m,
        search_margin_m=search_margin_m,
        observed_center_ego_m=None if not len(points) else [float(v) for v in points.mean(axis=0)],
        expected_center_ego_m=[float(v) for v in centre],
        intensity_min=None if sweep.point_count == 0 else float(intensity.min()),
        intensity_max=None if sweep.point_count == 0 else float(intensity.max()),
        intensity_in_unit_interval=intensity_ok,
    )


class SweepQueue:
    """Bounded per-frame store filled by the sensor callback; never reuses old sweeps."""

    def __init__(self, episode_id, capacity=8):
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("Nonempty episode required")
        if type(capacity) is not int or capacity < 1:
            raise ValueError("Positive queue capacity required")
        self.episode_id, self.capacity = episode_id, capacity
        self._frames = {}
        self._condition = threading.Condition()
        self._retired = -1
        self._closed = False
        self.events = Counter()

    def put(self, frame, timestamp, points, dropped=0):
        if type(frame) is not int or frame < 0 or not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Invalid sweep metadata")
        with self._condition:
            if self._closed or frame <= self._retired:
                self.events["late"] += 1
                return
            if frame in self._frames:
                self.events["duplicate"] += 1
                return
            points = np.array(points, dtype=np.float32, copy=True)
            points.setflags(write=False)
            self._frames[frame] = (timestamp, points, dropped, time.monotonic())
            while len(self._frames) > self.capacity:
                del self._frames[min(self._frames)]
                self.events["overflow"] += 1
            self._condition.notify_all()

    def _retire(self, frame):
        self._retired = max(self._retired, frame)
        for key in list(self._frames):
            if key <= frame:
                del self._frames[key]
                self.events["discarded"] += 1

    def wait(self, frame, timeout_s, stop):
        """Return ``(timestamp, points, dropped, received_monotonic_s)``; call ``stop`` on failure."""
        if type(frame) is not int or frame < 0 or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Invalid sweep request")
        deadline = time.monotonic() + timeout_s
        try:
            with self._condition:
                while True:
                    if self._closed or frame <= self._retired:
                        raise RuntimeError("Sweep queue closed or frame already retired")
                    if frame in self._frames:
                        result = self._frames[frame]
                        self._retire(frame)
                        self.events["delivered"] += 1
                        return result
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self.events["timeout"] += 1
                        self._retire(frame)
                        raise TimeoutError(f"Missing LiDAR sweep for frame {frame}")
                    self._condition.wait(remaining)
        except Exception:
            stop()
            raise

    def stats(self):
        with self._condition:
            return dict(self.events)

    def close(self):
        with self._condition:
            self._closed = True
            self._frames.clear()
            self._condition.notify_all()


class LidarCapture:
    """Own one ray-cast LiDAR within an active ``SynchronousWorldSession``.

    ``rear_axle_offset_m`` (metres, actor origin ahead of the rear axle) fixes the
    ego frame; ``owner.timing.fixed_delta_seconds`` fixes the rotation frequency.
    """

    def __init__(self, owner, vehicle, episode_id, *, rear_axle_offset_m, rig=None, capacity=8):
        self.owner, self.vehicle = owner, vehicle
        self.rig = rig or LidarRig()
        self.dt_s = owner.timing.fixed_delta_seconds
        self.config = self.rig.config(self.dt_s)
        self.sensor_to_ego = self.rig.sensor_to_ego(rear_axle_offset_m)
        self.queue = SweepQueue(episode_id, capacity)
        self.sensor = None
        self.callback_errors = []
        self.records = []

    def __enter__(self):
        import carla

        self.owner._require_owner()
        try:
            bp = self.owner.world.get_blueprint_library().find("sensor.lidar.ray_cast")
            for key, value in self.rig.blueprint_attributes(self.dt_s).items():
                bp.set_attribute(key, value)
            location, rotation = self.rig.mount_arguments()
            self.sensor = self.owner.world.spawn_actor(
                bp,
                carla.Transform(carla.Location(**location), carla.Rotation(**rotation)),
                attach_to=self.vehicle,
                attachment_type=carla.AttachmentType.Rigid,
            )
            self.owner._actors.append(self.sensor)
            self.sensor.listen(self._receive)
            return self
        except BaseException:
            self.close()
            raise

    def _receive(self, measurement):
        try:
            points, dropped = lidar_points_to_ego(
                measurement.raw_data, self.sensor_to_ego, range_m=self.config.range_m
            )
            self.queue.put(measurement.frame, measurement.timestamp, points, dropped)
        except Exception as error:
            self.callback_errors.append(str(error))
            del self.callback_errors[:-8]

    def read(self, frame, timeout_s=5.0):
        """Return the :class:`LidarSweep` for ``frame`` and append its capture record."""
        timestamp, points, dropped, received = self.queue.wait(
            frame, timeout_s, self.owner.emergency_stop
        )
        sweep = LidarSweep(
            self.queue.episode_id,
            frame,
            timestamp,
            received,
            points,
            tuple(map(tuple, self.sensor_to_ego)),
            self.config,
        )
        record = dict(sweep.summary(), dropped_nonfinite=dropped, fixed_delta_seconds=self.dt_s)
        self.records.append(record)
        return sweep

    def close(self):
        self.queue.close()
        sensor, self.sensor = self.sensor, None
        if sensor is None:
            return
        try:
            if sensor.is_alive:
                sensor.stop()
                if not sensor.destroy():
                    raise RuntimeError(f"Sensor destroy failed: {sensor.id}")
        except Exception as error:
            raise RuntimeError(str(error)) from error

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


__all__ = ["LidarCapture", "LidarRig", "SweepQueue", "known_object_check", "lidar_points_to_ego"]
