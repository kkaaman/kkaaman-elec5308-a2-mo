"""Where the car is, from its own sensors: a smooth odometry frame and a route-level pose.

Two estimates, for two jobs.

**Odometry frame (for driving).** Wheel speed and gyro yaw rate are integrated from the
episode start, so the frame starts at the car's rear axle, facing +x. It drifts slowly
but never jumps. Perception accumulates LiDAR evidence in this frame, and the planner
and controller work in it. Lane keeping needs centimetre-level smoothness over the
next 30 m, not metre-level accuracy on the globe. The gyro bias is measured while the
car stands still and is removed from then on.

**Route pose (for knowing where the route is).** An EKF over [x, y, yaw, gyro bias] in
the sat-nav route's east/north tangent plane. It is initialised from GNSS fixes and
compass readings averaged while the car stands still at the start, never from the
simulator's spawn pose. GNSS fixes then correct position and the compass corrects
heading. GNSS has a per-run bias (about 1 m), so this pose decides which route point is
next and how far the goal is, never where the lane is.

``odometry_to_route`` maps odometry points into the route plane through the two current
poses, which is how route geometry and perception meet.
"""

import math
from dataclasses import dataclass, replace

import numpy as np

from autonomy.localisation import FilterTuning, PoseEstimator
from mapless.frame import SensorFrame, SensorState
from mapless.geometry import wrap, yaw_from_compass
from mapless.satnav import LocalTangentPlane


@dataclass(frozen=True)
class EgoMotionConfig:
    startup_s: float = 1.0  # stand still this long, averaging GNSS and compass, before driving
    stationary_speed_mps: float = 0.08  # on the mean of the last ``speed_window`` readings
    stationary_max_reading_mps: float = 0.3  # and no single reading above this
    speed_window: int = 10
    min_startup_fixes: int = 4
    compass_sigma_rad: float = 0.03
    gnss_sigma_m: float = 1.2  # white noise plus the per-run bias the filter cannot see
    gyro_sigma_rps: float = 0.01
    max_bias_samples: int = 400
    imu_max_age_s: float = 0.2  # yaw rate older than this: odometry is not trusted ...
    speed_max_age_s: float = 0.2  # ... nor with a speed older than this

    def __post_init__(self):
        for name in ("startup_s", "compass_sigma_rad", "gnss_sigma_m", "gyro_sigma_rps"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.min_startup_fixes < 1 or self.max_bias_samples < 1 or self.speed_window < 1:
            raise ValueError("sample counts must be positive")


@dataclass(frozen=True)
class Pose2D:
    x_m: float
    y_m: float
    yaw_rad: float

    def to_world(self, xy):
        """Points in this pose's body frame (N x 2) to the frame the pose is expressed in."""
        p = np.asarray(xy, dtype=float).reshape(-1, 2)
        c, s = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        return np.column_stack((self.x_m + c * p[:, 0] - s * p[:, 1], self.y_m + s * p[:, 0] + c * p[:, 1]))

    def to_body(self, xy):
        p = np.asarray(xy, dtype=float).reshape(-1, 2) - (self.x_m, self.y_m)
        c, s = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        return np.column_stack((c * p[:, 0] + s * p[:, 1], -s * p[:, 0] + c * p[:, 1]))


class Odometry:
    """Dead reckoning in the odometry frame, with a stationary gyro-bias estimate."""

    def __init__(self, config):
        self.config = config
        self.pose = Pose2D(0.0, 0.0, 0.0)
        self.time_s = None
        self.distance_m = 0.0
        self._bias_samples = []
        self.gyro_bias_rps = 0.0

    def update(self, time_s, speed_mps, yaw_rate_rps, stationary, fresh_gyro=True):
        if self.time_s is None:
            self.time_s = time_s
        dt = time_s - self.time_s
        self.time_s = time_s
        if stationary and not fresh_gyro:
            return self.pose  # never learn the bias from a reused reading
        if stationary:
            samples = self._bias_samples
            samples.append(yaw_rate_rps)
            if len(samples) > self.config.max_bias_samples:
                del samples[0]
            self.gyro_bias_rps = float(np.mean(samples))
            return self.pose
        if dt <= 0:
            return self.pose
        omega = yaw_rate_rps - self.gyro_bias_rps
        heading = self.pose.yaw_rad + 0.5 * omega * dt
        step = speed_mps * dt
        self.distance_m += step
        self.pose = Pose2D(
            self.pose.x_m + step * math.cos(heading),
            self.pose.y_m + step * math.sin(heading),
            wrap(self.pose.yaw_rad + omega * dt),
        )
        return self.pose


class RoutePose(PoseEstimator):
    """The baseline EKF plus a compass update and a stationary GNSS/compass start-up."""

    def update_yaw(self, yaw_rad, sigma_rad):
        H = np.array([[0.0, 0.0, 1.0, 0.0]])
        innovation = wrap(yaw_rad - self.x[2])
        S = (H @ self.P @ H.T).item() + sigma_rad**2
        K = (self.P @ H.T / S).ravel()
        self.x = self.x + K * innovation
        self.x[2] = wrap(self.x[2])
        self.P = (np.eye(4) - np.outer(K, H)) @ self.P


@dataclass(frozen=True)
class EgoMotionState:
    time_s: float
    odometry: Pose2D
    route: Pose2D | None  # None until the start-up average is complete
    speed_mps: float
    yaw_rate_rps: float  # bias-corrected
    stationary: bool
    route_sigma_m: float
    gyro_bias_rps: float
    imu_age_s: float = 0.0  # since the last IMU sample
    speed_age_s: float = 0.0  # since the last speedometer reading
    gnss_age_s: float = math.inf  # since the last GNSS fix
    odometry_trusted: bool = True  # IMU and speedometer fresh: the odometry frame is sound
    odometry_broken_s: float = 0.0  # time the odometry has spent untrusted since it was last sound

    @property
    def localised(self):
        return self.route is not None

    def odometry_to_route(self, xy):
        """Odometry-frame points (N x 2) in the route plane, through the current poses."""
        if self.route is None:
            raise ValueError("Route pose not initialised yet")
        return self.route.to_world(self.odometry.to_body(xy))

    def route_to_odometry(self, xy):
        if self.route is None:
            raise ValueError("Route pose not initialised yet")
        return self.odometry.to_world(self.route.to_body(xy))


class EgoMotion:
    """Feed one ``SensorFrame`` per tick; read ``EgoMotionState``."""

    def __init__(self, plane, config=None):
        if not isinstance(plane, LocalTangentPlane):
            raise TypeError("plane must be a LocalTangentPlane (the route's)")
        self.plane = plane
        self.config = config or EgoMotionConfig()
        self.odometry = Odometry(self.config)
        tuning = replace(FilterTuning(), gnss_sigma_m=self.config.gnss_sigma_m, gyro_sigma_rps=self.config.gyro_sigma_rps)
        self.route = RoutePose(tuning)
        self._startup_fixes = []
        self._startup_compass = []
        self._startup_began_s = None
        self._yaw_rate = 0.0
        self._speeds = []
        self._imu_time = -math.inf
        self._speed_time = -math.inf
        self._gnss_time = -math.inf
        self._speed = 0.0
        self._untrusted_since = None

    def _start(self, time_s):
        xy = np.mean(self._startup_fixes, axis=0)
        compass = np.asarray(self._startup_compass)
        yaw = math.atan2(np.mean(np.sin(compass)), np.mean(np.cos(compass)))
        self.route.initialise_at(float(xy[0]), float(xy[1]), yaw, time_s)
        # The fixes were averaged, so their white noise is smaller than one fix's.
        self.route.P[0, 0] = self.route.P[1, 1] = self.config.gnss_sigma_m**2
        self.route.P[2, 2] = self.config.compass_sigma_rad**2 / max(1, len(compass))
        self.route.x[3] = self.odometry.gyro_bias_rps

    def update(self, frame):
        if not isinstance(frame, SensorFrame):
            raise TypeError("frame must be a SensorFrame")
        c = self.config
        t = frame.sim_time_s
        # The speedometer reads noise around zero at a standstill. Judge standstill on a short
        # mean, and integrate zero speed while stationary (clamping noise at zero would
        # otherwise creep forwards).
        if _ok(frame, "speedometer"):
            reading = float(frame.speed_mps)
            self._speed_time = t
            self._speeds.append(reading)
            if len(self._speeds) > c.speed_window:
                del self._speeds[0]
        recent = np.asarray(self._speeds) if self._speeds else np.zeros(1)
        stationary = abs(recent.mean()) < c.stationary_speed_mps and np.abs(recent).max() < c.stationary_max_reading_mps
        if _ok(frame, "speedometer"):
            self._speed = 0.0 if stationary else max(0.0, float(frame.speed_mps))
        fresh_gyro = frame.imu is not None
        if fresh_gyro:
            self._yaw_rate = frame.imu.yaw_rate_rps
            self._imu_time = t
        if frame.gnss is not None:
            self._gnss_time = t
        trusted = t - self._imu_time <= c.imu_max_age_s and t - self._speed_time <= c.speed_max_age_s
        if trusted:
            self._untrusted_since = None
        elif self._untrusted_since is None:
            self._untrusted_since = t
        # With a stale speedometer or IMU, integrating held readings invents motion (a 3 s
        # speedometer outage integrated 6 m for 0.8 m driven, review of 2b7e14d). Integrate
        # nothing: the frame then lags the car by what it drove, which the driver treats as a
        # broken frame and rebuilds from a standstill.
        speed = self._speed if trusted else 0.0
        odometry = self.odometry.update(t, speed, self._yaw_rate if trusted else 0.0, stationary, fresh_gyro and trusted)
        fix = None if frame.gnss is None else self.plane.to_local(frame.gnss.latitude_deg, frame.gnss.longitude_deg)
        compass = None if frame.imu is None else yaw_from_compass(frame.imu.compass_rad)
        if not self.route.initialised:
            if stationary:
                if self._startup_began_s is None:
                    self._startup_began_s = t
                if fix is not None:
                    self._startup_fixes.append(fix)
                if compass is not None:
                    self._startup_compass.append(compass)
                long_enough = t - self._startup_began_s >= c.startup_s
                if long_enough and len(self._startup_fixes) >= c.min_startup_fixes and self._startup_compass:
                    self._start(t)
            else:
                # Moving before the start-up average finished: begin again once stopped.
                self._startup_fixes.clear()
                self._startup_compass.clear()
                self._startup_began_s = None
        else:
            self.route.predict(t - self.route.time_s, speed, self._yaw_rate)
            self.route.time_s = t
            if fix is not None:
                self.route.update_position(fix, c.gnss_sigma_m)
            if compass is not None:
                self.route.update_yaw(compass, c.compass_sigma_rad)
        route = None
        sigma = math.inf
        if self.route.initialised:
            route = Pose2D(float(self.route.x[0]), float(self.route.x[1]), float(self.route.x[2]))
            sigma = float(math.sqrt(max(0.0, self.route.P[0, 0] + self.route.P[1, 1])))
        return EgoMotionState(
            time_s=t,
            odometry=odometry,
            route=route,
            speed_mps=speed,
            yaw_rate_rps=self._yaw_rate - self.odometry.gyro_bias_rps,
            stationary=stationary,
            route_sigma_m=sigma,
            gyro_bias_rps=self.odometry.gyro_bias_rps,
            imu_age_s=t - self._imu_time,
            speed_age_s=t - self._speed_time,
            gnss_age_s=t - self._gnss_time,
            odometry_trusted=trusted,
            odometry_broken_s=0.0 if trusted else t - self._untrusted_since,
        )


def _ok(frame, kind):
    return any(status.kind == kind and status.state is SensorState.OK for status in frame.status)
