"""Vehicle-frame LiDAR sweep contract (E2-01).

A :class:`LidarSweep` is the only LiDAR object that crosses into the core. It
carries one full rotation of returns already expressed in the rear-axle ego
frame (+X forward, +Y left, +Z up, metres), the frame identity needed to reject
stale observations, the rigid sensor-to-ego extrinsics that were applied
exactly once in the adapter, and the configured range/density so sparse sweeps
can be judged against what the sensor should have delivered.

No simulator object, raw byte buffer or actor identity belongs here. Nothing in
this module authorises clearance: an empty or sparse sweep is an invalid
observation, never an obstacle-free scene.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


def _finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class LidarConfig:
    """Configured range and density of one spinning LiDAR.

    Units: ``range_m`` metres; ``rotation_frequency_hz`` rotations per second;
    ``points_per_second`` returns per second over all channels; field-of-view
    angles in degrees (``upper_fov_deg`` above the sensor horizon is positive,
    ``lower_fov_deg`` below is negative; ``horizontal_fov_deg`` is the swept
    azimuth, 360 for a full rotation).
    """

    channels: int = 32
    range_m: float = 50.0
    rotation_frequency_hz: float = 20.0
    points_per_second: int = 640000
    upper_fov_deg: float = 10.0
    lower_fov_deg: float = -30.0
    horizontal_fov_deg: float = 360.0

    def __post_init__(self) -> None:
        if type(self.channels) is not int or not 1 <= self.channels <= 256:
            raise ValueError("channels must be an integer in 1..256")
        if type(self.points_per_second) is not int or self.points_per_second <= 0:
            raise ValueError("points_per_second must be a positive integer")
        for name in ("range_m", "rotation_frequency_hz", "horizontal_fov_deg"):
            _finite(name, getattr(self, name))
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.horizontal_fov_deg > 360:
            raise ValueError("horizontal_fov_deg must be at most 360")
        _finite("upper_fov_deg", self.upper_fov_deg)
        _finite("lower_fov_deg", self.lower_fov_deg)
        if not -90 <= self.lower_fov_deg < self.upper_fov_deg <= 90:
            raise ValueError("Vertical field of view must satisfy -90 <= lower < upper <= 90")

    @property
    def expected_points_per_sweep(self) -> float:
        """Nominal returns in one rotation (dimensionless count)."""
        return self.points_per_second / self.rotation_frequency_hz

    def full_sweep_per_tick(self, dt_s: float) -> bool:
        """True when one simulation step of ``dt_s`` seconds covers exactly one rotation."""
        _finite("dt_s", dt_s)
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        return math.isclose(self.rotation_frequency_hz * dt_s, 1.0, rel_tol=1e-9)

    @classmethod
    def for_timestep(cls, dt_s: float, **overrides) -> LidarConfig:
        """Configuration whose rotation frequency is ``1 / dt_s`` (one sweep per tick)."""
        _finite("dt_s", dt_s)
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        overrides.pop("rotation_frequency_hz", None)
        return cls(rotation_frequency_hz=1.0 / dt_s, **overrides)

    def to_dict(self) -> dict:
        return asdict(self)


def validate_rigid_transform(matrix, name: str = "sensor_to_ego") -> np.ndarray:
    """Return a copied, read-only 4x4 rigid transform (rotation + metres translation)."""
    m = np.asarray(matrix, dtype=float)
    if m.shape != (4, 4) or not np.isfinite(m).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(m[3], (0, 0, 0, 1), atol=1e-7):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    rotation = m[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{name} rotation must preserve handedness")
    m = m.copy()
    m.setflags(write=False)
    return m


@dataclass(frozen=True)
class LidarSweep:
    """One rotation of LiDAR returns in rear-axle ego coordinates.

    ``points_xyzi`` is an N-by-4 ``float32`` array: columns are X forward, Y
    left, Z up in metres and a dimensionless intensity in ``[0, 1]``. Every
    value must be finite; N may be zero (an empty sweep is retained so the
    consumer can report it as invalid rather than clear). ``sensor_to_ego`` is
    the rigid transform the adapter applied once to move the returns from the
    sensor frame to the ego frame; it is retained for provenance and azimuth
    coverage checks, never applied again. ``sim_time_s`` is simulation seconds
    within the episode; ``received_monotonic_s`` is host ``perf_counter`` /
    ``monotonic`` seconds and must never be compared with simulation time.
    """

    episode_id: str
    frame_id: int
    sim_time_s: float
    received_monotonic_s: float
    points_xyzi: np.ndarray
    sensor_to_ego: tuple[tuple[float, ...], ...]
    config: LidarConfig

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise ValueError("episode_id must be a nonempty string")
        if type(self.frame_id) is not int or self.frame_id < 0:
            raise ValueError("frame_id must be a nonnegative integer")
        for name in ("sim_time_s", "received_monotonic_s"):
            _finite(name, getattr(self, name))
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not isinstance(self.config, LidarConfig):
            raise TypeError("config must be LidarConfig")
        points = self.points_xyzi
        if not isinstance(points, np.ndarray) or points.dtype != np.float32:
            raise ValueError("points_xyzi must be a float32 ndarray")
        if points.ndim != 2 or points.shape[1] != 4:
            raise ValueError("points_xyzi must be N-by-4 (x, y, z, intensity)")
        if not np.isfinite(points).all():
            raise ValueError("points_xyzi must be finite; drop invalid returns in the adapter")
        if points.size and ((points[:, 3] < 0).any() or (points[:, 3] > 1).any()):
            raise ValueError("intensity must lie in [0, 1]")
        matrix = validate_rigid_transform(self.sensor_to_ego)
        if points.size:
            # Range is measured from the sensor origin, not the rear axle.
            local = points[:, :3].astype(float) - matrix[:3, 3]
            if (np.linalg.norm(local, axis=1) > self.config.range_m + 1e-3).any():
                raise ValueError("points exceed the configured range")
        snapshot = np.array(points, copy=True)
        snapshot.setflags(write=False)
        object.__setattr__(self, "points_xyzi", snapshot)
        object.__setattr__(self, "sensor_to_ego", tuple(tuple(float(v) for v in r) for r in matrix))

    @property
    def point_count(self) -> int:
        """Number of returns in this sweep (dimensionless)."""
        return int(self.points_xyzi.shape[0])

    @property
    def density_fraction(self) -> float:
        """Returns delivered divided by the configured nominal returns per rotation."""
        return self.point_count / self.config.expected_points_per_sweep

    @property
    def sensor_origin_ego_m(self) -> tuple[float, float, float]:
        """Sensor position in the ego frame, metres."""
        return tuple(float(v) for v in np.asarray(self.sensor_to_ego)[:3, 3])

    def azimuth_coverage_deg(self, bins: int = 360) -> float:
        """Degrees of azimuth (about the sensor axis) holding at least one return."""
        if type(bins) is not int or bins < 1:
            raise ValueError("bins must be a positive integer")
        if self.point_count == 0:
            return 0.0
        m = np.asarray(self.sensor_to_ego, dtype=float)
        local = (self.points_xyzi[:, :3].astype(float) - m[:3, 3]) @ m[:3, :3]
        azimuth = np.degrees(np.arctan2(local[:, 1], local[:, 0])) % 360.0
        filled = np.unique(np.floor(azimuth * bins / 360.0).astype(int))
        return float(len(filled) * 360.0 / bins)

    def summary(self) -> dict:
        """JSON-friendly frame identity and density record (no point data)."""
        return dict(
            episode_id=self.episode_id,
            frame_id=self.frame_id,
            sim_time_s=self.sim_time_s,
            point_count=self.point_count,
            expected_points_per_sweep=self.config.expected_points_per_sweep,
            density_fraction=self.density_fraction,
            azimuth_coverage_deg=self.azimuth_coverage_deg(),
            sensor_origin_ego_m=list(self.sensor_origin_ego_m),
        )
