"""Planar pose estimate from wheel speed, gyro yaw rate and GNSS, without the simulator pose.

Shadow use: the estimate is computed and logged beside the true pose while the car
still drives on the true pose, so the localisation error can be measured before any
driving depends on it.

State ``[x, y, yaw, gyro_bias]`` in the core frame (x forward/east, y left, yaw
counter-clockwise). Prediction integrates wheel speed along the heading and the
bias-corrected gyro rate; GNSS fixes update x/y. A constant GNSS offset is not
observable from these sensors alone: it shows up directly as position error, which is
what a lane-relative correction (camera lane estimate) would have to remove.
"""

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SensorNoise:
    """How the simulated sensors are corrupted (per run, seeded)."""

    gnss_period_s: float = 0.2
    gnss_white_m: float = 0.5
    gnss_bias_m: float = 1.0  # 1-sigma of a constant per-axis offset, drawn once per run
    gyro_white_rps: float = 0.005
    gyro_bias_rps: float = 0.002  # 1-sigma of a constant offset, drawn once per run
    wheel_white_mps: float = 0.05
    wheel_scale_sigma: float = 0.01  # 1-sigma of a constant scale error, drawn once per run


@dataclass(frozen=True)
class FilterTuning:
    """What the estimator assumes; independent of the true noise above."""

    gnss_sigma_m: float = 1.2
    wheel_sigma_mps: float = 0.08
    wheel_scale_sigma: float = 0.015
    gyro_sigma_rps: float = 0.01
    bias_walk_rps_per_sqrt_s: float = 0.0002
    initial_yaw_sigma_rad: float = math.radians(5)
    initial_bias_sigma_rps: float = 0.005


@dataclass(frozen=True)
class LocalisationConfig:
    sensors: SensorNoise = SensorNoise()
    filter: FilterTuning = FilterTuning()

    def __post_init__(self):
        for section in (self.sensors, self.filter):
            for item in fields(section):
                value = getattr(section, item.name)
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"{item.name} must be finite and non-negative")
        if self.sensors.gnss_period_s <= 0:
            raise ValueError("gnss_period_s must be positive")

    @classmethod
    def load(cls, path):
        document = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if document.get("schema_version") != 1:
            raise ValueError("Unsupported localisation config")
        return cls(
            SensorNoise(**document.get("sensors", {})),
            FilterTuning(**document.get("filter", {})),
        )

    def as_dict(self):
        return dict(schema_version=1, sensors=asdict(self.sensors), filter=asdict(self.filter))


@dataclass(frozen=True)
class LocalisationInputs:
    sim_time_s: float
    wheel_speed_mps: float
    yaw_rate_rps: float
    gnss_xy_m: tuple | None = None


@dataclass(frozen=True)
class PoseEstimate:
    x_m: float
    y_m: float
    yaw_rad: float
    gyro_bias_rps: float
    position_sigma_m: float


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class PoseEstimator:
    """EKF over [x, y, yaw, gyro bias]. Initialised from the first GNSS fix.

    The first fix is snapped to the nearest route centreline point and takes the
    route heading there (map matching at a standstill, where GNSS gives no heading).
    """

    def __init__(self, tuning=None):
        self.tuning = tuning or FilterTuning()
        self.x = None
        self.P = None
        self.time_s = None

    @property
    def initialised(self):
        return self.x is not None

    def initialise(self, gnss_xy, route, time_s):
        projection = route.project(*gnss_xy)
        x, y, yaw = route.at(projection.s_m)
        t = self.tuning
        self.x = np.array([x, y, yaw, 0.0])
        self.P = np.diag(
            [
                t.gnss_sigma_m**2,
                t.gnss_sigma_m**2,
                t.initial_yaw_sigma_rad**2,
                t.initial_bias_sigma_rps**2,
            ]
        )
        self.time_s = time_s

    def initialise_at(self, x, y, yaw, time_s):
        """Start from a known pose (the mission start, where the car is parked)."""
        t = self.tuning
        self.x = np.array([x, y, yaw, 0.0])
        self.P = np.diag(
            [
                t.gnss_sigma_m**2,
                t.gnss_sigma_m**2,
                t.initial_yaw_sigma_rad**2,
                t.initial_bias_sigma_rps**2,
            ]
        )
        self.time_s = time_s

    def predict(self, dt, speed, yaw_rate):
        if dt <= 0:
            return
        t = self.tuning
        x, y, yaw, bias = self.x
        omega = yaw_rate - bias
        heading = yaw + 0.5 * omega * dt
        c, s = math.cos(heading), math.sin(heading)
        self.x = np.array([x + speed * dt * c, y + speed * dt * s, wrap(yaw + omega * dt), bias])
        F = np.eye(4)
        F[0, 2], F[1, 2] = -speed * dt * s, speed * dt * c
        F[0, 3], F[1, 3] = 0.5 * speed * dt * dt * s, -0.5 * speed * dt * dt * c
        F[2, 3] = -dt
        speed_sigma = t.wheel_sigma_mps + abs(speed) * t.wheel_scale_sigma
        G = np.array([[dt * c, 0.0], [dt * s, 0.0], [0.0, dt], [0.0, 0.0]])
        Q = G @ np.diag([speed_sigma**2, t.gyro_sigma_rps**2]) @ G.T
        Q[3, 3] += t.bias_walk_rps_per_sqrt_s**2 * dt
        self.P = F @ self.P @ F.T + Q

    def update_position(self, xy, sigma_m):
        H = np.zeros((2, 4))
        H[0, 0] = H[1, 1] = 1.0
        innovation = np.asarray(xy, dtype=float) - self.x[:2]
        S = H @ self.P @ H.T + np.eye(2) * sigma_m**2
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.x[2] = wrap(self.x[2])
        self.P = (np.eye(4) - K @ H) @ self.P

    def step(self, inputs, route):
        """Advance to ``inputs.sim_time_s``; returns None until the first GNSS fix."""
        if not self.initialised:
            if inputs.gnss_xy_m is None:
                return None
            self.initialise(inputs.gnss_xy_m, route, inputs.sim_time_s)
        else:
            self.predict(
                inputs.sim_time_s - self.time_s, inputs.wheel_speed_mps, inputs.yaw_rate_rps
            )
            self.time_s = inputs.sim_time_s
            if inputs.gnss_xy_m is not None:
                self.update_position(inputs.gnss_xy_m, self.tuning.gnss_sigma_m)
        return PoseEstimate(
            float(self.x[0]),
            float(self.x[1]),
            float(self.x[2]),
            float(self.x[3]),
            float(math.sqrt(max(0.0, self.P[0, 0] + self.P[1, 1]))),
        )


class NoisySensors:
    """Corrupt true signals the same way in the harness and in CARLA (seeded per run).

    CARLA's own GNSS/IMU sensors are used live; this model supplies the wheel-speed
    odometry (CARLA has no wheel-encoder sensor) and the harness's GNSS and gyro.
    """

    def __init__(self, noise, seed):
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.gnss_bias = self.rng.normal(0.0, noise.gnss_bias_m, size=2)
        self.gyro_bias = float(self.rng.normal(0.0, noise.gyro_bias_rps))
        self.wheel_scale = 1.0 + float(self.rng.normal(0.0, noise.wheel_scale_sigma))
        self._next_fix_s = None

    def wheel_speed(self, true_speed):
        noisy = true_speed * self.wheel_scale + self.rng.normal(0.0, self.noise.wheel_white_mps)
        return max(0.0, float(noisy))

    def yaw_rate(self, true_rate):
        return float(true_rate + self.gyro_bias + self.rng.normal(0.0, self.noise.gyro_white_rps))

    def gnss_due(self, time_s):
        if self._next_fix_s is None or time_s + 1e-9 >= self._next_fix_s:
            self._next_fix_s = (self._next_fix_s or time_s) + self.noise.gnss_period_s
            return True
        return False

    def gnss(self, true_xy):
        white = self.rng.normal(0.0, self.noise.gnss_white_m, size=2)
        return tuple(float(v) for v in np.asarray(true_xy) + self.gnss_bias + white)

    def record(self):
        return dict(
            gnss_bias_m=[float(v) for v in self.gnss_bias],
            gyro_bias_rps=self.gyro_bias,
            wheel_scale=self.wheel_scale,
        )


def pose_errors(estimate, true_state, route):
    """Estimate minus truth: total, lateral and along-route at the true position, yaw."""
    projection = route.project(true_state.x_m, true_state.y_m)
    _, _, heading = route.at(projection.s_m)
    dx, dy = estimate.x_m - true_state.x_m, estimate.y_m - true_state.y_m
    return dict(
        error_m=math.hypot(dx, dy),
        along_error_m=dx * math.cos(heading) + dy * math.sin(heading),
        lateral_error_m=-dx * math.sin(heading) + dy * math.cos(heading),
        yaw_error_rad=wrap(estimate.yaw_rad - true_state.yaw_rad),
    )
