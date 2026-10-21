"""Live GNSS + IMU inputs for shadow localisation (CARLA 0.9.16).

The GNSS antenna is mounted over the rear axle, so a fix is directly comparable with
the rear-axle state. Fixes (latitude/longitude) are mapped back to CARLA x/y with a
linear fit of ``map.transform_to_geolocation`` around the origin; over a few hundred
metres the projection is linear to well under a millimetre. Noise is set on CARLA's
own sensors: white noise plus a constant offset drawn once per run from the seed
(CARLA's noise model takes degrees, converted from metres here). CARLA has no wheel
encoder, so wheel speed is the true longitudinal speed through the same seeded
scale-and-noise model the harness uses. The core frame flips CARLA's y, so the core
yaw rate is minus the gyro's z rate.
"""

import threading
import time

import numpy as np

from autonomy.localisation import LocalisationInputs, NoisySensors


def geo_calibration(geolocate):
    """Linear map CARLA (x, y) -> (lat, lon) from three probes; returns origin, A, A^-1."""
    origin = np.asarray(geolocate(0.0, 0.0), dtype=float)
    along_x = (np.asarray(geolocate(100.0, 0.0), dtype=float) - origin) / 100.0
    along_y = (np.asarray(geolocate(0.0, 100.0), dtype=float) - origin) / 100.0
    matrix = np.column_stack((along_x, along_y))
    if abs(np.linalg.det(matrix)) < 1e-18:
        raise ValueError("Map geolocation is degenerate; cannot convert GNSS fixes")
    return origin, matrix, np.linalg.inv(matrix)


class LiveLocalisationSensors:
    def __init__(self, config, seed, geolocate, spawn_sensor, rear_axle_offset_m, dt_s):
        self.config, self.dt_s = config, dt_s
        self.model = NoisySensors(config.sensors, seed)
        self.origin, self.to_geo, self.from_geo = geo_calibration(geolocate)
        lat_per_m = float(np.hypot(*self.to_geo[0]))
        lon_per_m = float(np.hypot(*self.to_geo[1]))
        # Core offset (x, y) -> CARLA (x, -y) -> degrees.
        bias_lat, bias_lon = self.to_geo @ np.array(
            [self.model.gnss_bias[0], -self.model.gnss_bias[1]]
        )
        noise = config.sensors
        self.gnss_attributes = dict(
            sensor_tick=noise.gnss_period_s,
            noise_lat_stddev=noise.gnss_white_m * lat_per_m,
            noise_lon_stddev=noise.gnss_white_m * lon_per_m,
            noise_lat_bias=float(bias_lat),
            noise_lon_bias=float(bias_lon),
            noise_alt_stddev=0.0,
            noise_alt_bias=0.0,
            noise_seed=int(seed),
        )
        self.imu_attributes = dict(
            sensor_tick=0.0,
            noise_gyro_stddev_z=noise.gyro_white_rps,
            # Core rate = -gyro z, so a CARLA bias of -b is a core bias of +b.
            noise_gyro_bias_z=-self.model.gyro_bias,
            noise_seed=int(seed) + 1,
        )
        self._lock = threading.Lock()
        self._imu, self._gnss, self._speed = {}, {}, {}
        self._last_fix_s = None
        self.counts = dict(frames=0, fixes=0, late_or_missing_fixes=0)
        mount = (-float(rear_axle_offset_m), 0.0, 1.6)
        spawn_sensor("sensor.other.imu", self.imu_attributes, mount, self._on_imu)
        spawn_sensor("sensor.other.gnss", self.gnss_attributes, mount, self._on_gnss)

    def _on_imu(self, measurement):
        with self._lock:
            self._imu[measurement.frame] = (measurement.timestamp, -measurement.gyroscope.z)

    def _on_gnss(self, measurement):
        with self._lock:
            self._gnss[measurement.frame] = (measurement.latitude, measurement.longitude)

    def record_speed(self, frame_id, speed_mps):
        self._speed[frame_id] = speed_mps

    def fix_to_core(self, latitude, longitude):
        x, y = self.from_geo @ (np.array([latitude, longitude]) - self.origin)
        return float(x), float(-y)

    def _wait(self, store, frame_id, timeout_s):
        deadline = time.monotonic() + timeout_s
        while True:
            with self._lock:
                if frame_id in store:
                    return store.pop(frame_id)
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.0005)

    def inputs(self, frame_id):
        imu = self._wait(self._imu, frame_id, 2.0)
        if imu is None:
            raise RuntimeError(f"No IMU measurement for frame {frame_id}")
        sim_time, yaw_rate = imu
        due = (
            self._last_fix_s is None
            or sim_time - self._last_fix_s >= self.config.sensors.gnss_period_s - self.dt_s / 2
        )
        fix = self._wait(self._gnss, frame_id, 0.3 if due else 0.0)
        gnss = None
        if fix is not None:
            gnss = self.fix_to_core(*fix)
            self._last_fix_s = sim_time
            self.counts["fixes"] += 1
        elif due:
            self.counts["late_or_missing_fixes"] += 1
        self.counts["frames"] += 1
        with self._lock:
            for store in (self._imu, self._gnss):
                for stale in [f for f in store if f < frame_id]:
                    del store[stale]
        speed = self._speed.pop(frame_id, 0.0)
        return LocalisationInputs(sim_time, self.model.wheel_speed(speed), yaw_rate, gnss)

    def record(self):
        return dict(
            source="CARLA GNSS + IMU; wheel speed from true speed with the seeded noise model",
            gnss_attributes=self.gnss_attributes,
            imu_attributes=self.imu_attributes,
            geo_origin=[float(v) for v in self.origin],
            geo_per_metre=[[float(v) for v in row] for row in self.to_geo],
            counts=dict(self.counts),
            **self.model.record(),
        )
