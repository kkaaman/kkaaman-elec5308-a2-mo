"""Deterministic synthetic LiDAR sweeps for fixtures and offline tests (E2-01/E2-03).

A spinning LiDAR is simulated by casting rays from the sensor origin through a
scene of a flat road, raised pavements beyond a kerb and axis-aligned boxes
(vehicles, poles, walls). Each ray keeps the nearest hit within range; rays that
hit nothing (sky) are dropped, so density is below one like a real sensor. The
output is a vehicle-frame :class:`~obstacle_perception.lidar.LidarSweep`, with the
same sensor-to-ego extrinsics the adapter would record.

Scenes are plain dataclasses in the rear-axle ego frame (metres). They are test
inputs, never evaluation truth fed to control: the fixture records each object's
expected front-bumper gap only so the replay can measure the pipeline's error.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from obstacle_perception.lidar import LidarConfig, LidarSweep, validate_rigid_transform

FIXTURE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SceneBox:
    """Axis-aligned box in the ego frame: centre XY, full XY size, base Z and height (metres)."""

    x_m: float
    y_m: float
    size_x_m: float
    size_y_m: float
    height_m: float
    base_z_m: float = 0.0
    label: str = "object"

    def __post_init__(self):
        for name in ("size_x_m", "size_y_m", "height_m"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def bounds(self):
        low = np.array([self.x_m - self.size_x_m / 2, self.y_m - self.size_y_m / 2, self.base_z_m])
        high = np.array(
            [
                self.x_m + self.size_x_m / 2,
                self.y_m + self.size_y_m / 2,
                self.base_z_m + self.height_m,
            ]
        )
        return low, high

    def front_gap_m(self, front_bumper_m):
        """Nearest X face minus the front bumper offset (metres) on a straight route."""
        return self.x_m - self.size_x_m / 2 - front_bumper_m


@dataclass(frozen=True)
class Scene:
    """Flat road of ``2 * road_half_width_m`` with pavements ``kerb_height_m`` higher."""

    road_half_width_m: float = 3.5
    kerb_height_m: float = 0.15
    pavement_extent_m: float = 12.0
    boxes: tuple[SceneBox, ...] = ()
    road_slope: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "boxes", tuple(self.boxes))


def fixture_config(dt_s):
    """Forward 120-degree sector, 32 channels, 6000 nominal returns per sweep, one sweep per tick.

    The angular density (about 0.64 degrees of azimuth per return) matches a full
    rotation at 18000 returns per sweep; the sector keeps checked-in fixtures small.
    A live capture uses the full rotation (see :class:`LidarConfig` defaults).
    """
    return LidarConfig(
        channels=32,
        range_m=50.0,
        rotation_frequency_hz=1.0 / dt_s,
        points_per_second=int(round(6000 / dt_s)),
        upper_fov_deg=10.0,
        lower_fov_deg=-30.0,
        horizontal_fov_deg=120.0,
    )


def default_sensor_to_ego(x_m=1.5, y_m=0.0, z_m=2.4, yaw_rad=0.0):
    """Roof-mounted sensor pose in the rear-axle frame as a 4x4 rigid transform."""
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return validate_rigid_transform([[c, -s, 0, x_m], [s, c, 0, y_m], [0, 0, 1, z_m], [0, 0, 0, 1]])


def ray_directions(config, azimuth_offset_rad=0.0):
    """Unit ray directions (sensor frame) for one rotation of ``config``."""
    per_channel = max(1, int(round(config.expected_points_per_sweep / config.channels)))
    elevation = np.radians(np.linspace(config.lower_fov_deg, config.upper_fov_deg, config.channels))
    span = math.radians(config.horizontal_fov_deg)
    azimuth = azimuth_offset_rad + np.arange(per_channel) * span / per_channel - span / 2
    el, az = np.meshgrid(elevation, azimuth, indexing="ij")
    return np.column_stack(
        (
            np.cos(el.ravel()) * np.cos(az.ravel()),
            np.cos(el.ravel()) * np.sin(az.ravel()),
            np.sin(el.ravel()),
        )
    )


def _plane_hits(origin, directions, z_plane, inside):
    """Ray parameter for z = z_plane where the hit satisfies ``inside(xy)``, else inf."""
    dz = directions[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (z_plane - origin[2]) / dz
    t = np.where((dz < 0) & (t > 0), t, np.inf)
    ok = np.isfinite(t)
    keep = np.zeros_like(ok)
    xy = origin[None, :2] + t[ok, None] * directions[ok, :2]
    keep[ok] = inside(xy)
    return np.where(keep, t, np.inf)


def _box_hits(origin, directions, low, high):
    with np.errstate(divide="ignore", invalid="ignore"):
        t_low = (low[None, :] - origin[None, :]) / directions
        t_high = (high[None, :] - origin[None, :]) / directions
    t_near = np.nanmax(np.minimum(t_low, t_high), axis=1)
    t_far = np.nanmin(np.maximum(t_low, t_high), axis=1)
    hit = (t_far >= t_near) & (t_far > 0)
    t = np.where(t_near > 0, t_near, t_far)
    return np.where(hit, t, np.inf)


def cast_sweep(
    scene, config, sensor_to_ego, *, rng, noise_m=0.01, dropout=0.0, azimuth_offset_rad=0.0
):
    """Ego-frame N-by-4 float32 returns for one rotation through ``scene``.

    ``noise_m`` is Gaussian range noise (metres); ``dropout`` is the fraction of
    returns removed at random (dimensionless, simulates a degraded sensor).
    """
    m = np.asarray(sensor_to_ego, dtype=float)
    origin = m[:3, 3]
    directions = ray_directions(config, azimuth_offset_rad) @ m[:3, :3].T
    half = scene.road_half_width_m
    t_road = _plane_hits(origin, directions, 0.0, lambda xy: np.abs(xy[:, 1]) <= half)
    if scene.road_slope:
        # Sloped road: solve z = slope * x along the ray.
        dz = directions[:, 2] - scene.road_slope * directions[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (scene.road_slope * origin[0] - origin[2]) / dz
            xy = origin[None, :2] + t[:, None] * directions[:, :2]
        t_road = np.where((t > 0) & np.isfinite(t) & (np.abs(xy[:, 1]) <= half), t, np.inf)
    t_pavement = _plane_hits(
        origin,
        directions,
        scene.kerb_height_m,
        lambda xy: (np.abs(xy[:, 1]) > half) & (np.abs(xy[:, 1]) <= half + scene.pavement_extent_m),
    )
    best = np.minimum(t_road, t_pavement)
    for box in scene.boxes:
        low, high = box.bounds
        best = np.minimum(best, _box_hits(origin, directions, low, high))
    keep = np.isfinite(best) & (best <= config.range_m)
    t = best[keep]
    if noise_m:
        t = np.maximum(0.0, t + rng.normal(0.0, noise_m, t.shape))
    t = np.minimum(t, config.range_m)
    points = origin[None, :] + t[:, None] * directions[keep]
    intensity = np.exp(-0.004 * t)
    xyzi = np.column_stack((points, intensity)).astype(np.float32)
    if dropout:
        xyzi = xyzi[rng.random(len(xyzi)) >= dropout]
    return xyzi


def synthetic_sweep(
    scene,
    *,
    episode_id,
    frame_id,
    sim_time_s,
    config=None,
    sensor_to_ego=None,
    seed=0,
    noise_m=0.01,
    dropout=0.0,
):
    """Validated :class:`LidarSweep` for ``scene`` with a deterministic seed."""
    config = config or fixture_config(0.05)
    sensor_to_ego = (
        default_sensor_to_ego()
        if sensor_to_ego is None
        else validate_rigid_transform(sensor_to_ego)
    )
    rng = np.random.default_rng([seed, frame_id])
    points = cast_sweep(scene, config, sensor_to_ego, rng=rng, noise_m=noise_m, dropout=dropout)
    return LidarSweep(
        episode_id, frame_id, sim_time_s, 0.0, points, tuple(map(tuple, sensor_to_ego)), config
    )


def approach_sequence(
    *, frames=8, start_gap_m=24.0, closing_mps=4.0, dt_s=0.05, front_bumper_m=3.8, every=5
):
    """Scenes of a car approaching along a straight road, plus roadside clutter.

    Returns ``[(frame_id, sim_time_s, scene, expected_gap_m)]`` sampled every
    ``every`` ticks. Roadside objects (a parked car on the pavement side, a pole and
    a wall) stay fixed outside the corridor.
    """
    rows = []
    for k in range(frames):
        frame_id = (k + 1) * every
        sim_time = frame_id * dt_s
        gap = start_gap_m - k * every * dt_s * closing_mps
        target_x = gap + front_bumper_m + 2.2  # centre is half a 4.4 m car behind its rear face
        boxes = (
            SceneBox(target_x, 0.0, 4.4, 1.9, 1.5, label="target"),
            SceneBox(14.0, 5.2, 4.4, 1.9, 1.5, base_z_m=0.15, label="parked_car_pavement"),
            SceneBox(9.0, -4.0, 0.3, 0.3, 3.0, base_z_m=0.15, label="pole"),
            SceneBox(20.0, 8.0, 30.0, 0.4, 2.0, base_z_m=0.15, label="wall"),
        )
        rows.append((frame_id, sim_time, Scene(boxes=boxes), gap))
    return rows


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def build_fixture(
    directory, *, seed=7, episode_id="lidar-synthetic", dt_s=0.05, front_bumper_m=3.8
):
    """Write a small synthetic sweep set and manifest to ``directory``.

    The sequence is an approaching car with roadside clutter, followed by one
    degraded (95 % dropout) sweep and one empty sweep so sparse-data invalidity is
    exercised. ``depth_gap_m`` rows simulate the RGB-D channel for fusion replay:
    agreement, a far-off disagreement, a nearer disagreement and a dropped depth
    frame. These depth values are fixture inputs, not sensor measurements.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = fixture_config(dt_s)
    extrinsics = default_sensor_to_ego()
    rows = approach_sequence(dt_s=dt_s, front_bumper_m=front_bumper_m)
    depth_channel = [
        (None, False, "depth_dropped"),
        (23.9, True, "agree"),
        (26.0, True, "depth_far_disagreement"),
        (20.5, True, "depth_nearer_disagreement"),
        (19.9, True, "agree"),
        (19.0, True, "agree"),
        (18.0, True, "agree"),
        (17.0, True, "agree"),
    ]
    sweeps = []
    for (frame_id, sim_time, scene, gap), (depth_gap, depth_valid, note) in zip(
        rows, depth_channel
    ):
        sweep = synthetic_sweep(
            scene,
            episode_id=episode_id,
            frame_id=frame_id,
            sim_time_s=sim_time,
            config=config,
            sensor_to_ego=extrinsics,
            seed=seed,
        )
        name = f"sweep_{frame_id:04d}.npy"
        np.save(directory / name, sweep.points_xyzi)
        sweeps.append(
            dict(
                frame_id=frame_id,
                sim_time_s=sim_time,
                npy=name,
                sha256=sha256_bytes((directory / name).read_bytes()),
                point_count=sweep.point_count,
                density_fraction=sweep.density_fraction,
                expected_gap_m=gap,
                objects=[asdict(b) for b in scene.boxes],
                depth_gap_m=depth_gap,
                depth_valid=depth_valid,
                note=note,
            )
        )
    last_frame, last_time, last_scene, last_gap = rows[-1]
    for offset, dropout, note in ((1, 0.95, "sparse_dropout"), (2, 1.0, "empty_sweep")):
        frame_id = last_frame + offset * 5
        sim_time = frame_id * dt_s
        sweep = synthetic_sweep(
            last_scene,
            episode_id=episode_id,
            frame_id=frame_id,
            sim_time_s=sim_time,
            config=config,
            sensor_to_ego=extrinsics,
            seed=seed,
            dropout=dropout,
        )
        name = f"sweep_{frame_id:04d}.npy"
        np.save(directory / name, sweep.points_xyzi)
        sweeps.append(
            dict(
                frame_id=frame_id,
                sim_time_s=sim_time,
                npy=name,
                sha256=sha256_bytes((directory / name).read_bytes()),
                point_count=sweep.point_count,
                density_fraction=sweep.density_fraction,
                expected_gap_m=last_gap,
                objects=[asdict(b) for b in last_scene.boxes],
                depth_gap_m=17.0,
                depth_valid=True,
                note=note,
            )
        )
    manifest = dict(
        schema_version=FIXTURE_SCHEMA_VERSION,
        origin="synthetic",
        generator="obstacle_perception.lidar_synthetic.build_fixture",
        seed=seed,
        episode_id=episode_id,
        fixed_delta_seconds=dt_s,
        front_bumper_from_rear_axle_m=front_bumper_m,
        config=config.to_dict(),
        sensor_to_ego=[[float(v) for v in row] for row in extrinsics],
        route_xy_m=[[0.0, 0.0], [60.0, 0.0]],
        scene=dict(road_half_width_m=3.5, kerb_height_m=0.15, pavement_extent_m=12.0),
        notes="Synthetic ray-cast sweeps; expected gaps and depth channel are fixture inputs for replay scoring, not sensor truth for control.",
        sweeps=sweeps,
    )
    (directory / "sweeps.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_fixture_sweep(manifest, row, directory, *, verify=True):
    """Rebuild one :class:`LidarSweep` from a manifest row, verifying the file hash."""
    directory = Path(directory)
    data = (directory / row["npy"]).read_bytes()
    if verify and sha256_bytes(data) != row["sha256"]:
        raise ValueError(f"Fixture {row['npy']} hash mismatch")
    points = np.load(directory / row["npy"])
    return LidarSweep(
        manifest["episode_id"],
        int(row["frame_id"]),
        float(row["sim_time_s"]),
        0.0,
        points,
        tuple(map(tuple, manifest["sensor_to_ego"])),
        LidarConfig(**manifest["config"]),
    )


__all__ = [
    "FIXTURE_SCHEMA_VERSION",
    "Scene",
    "SceneBox",
    "approach_sequence",
    "build_fixture",
    "cast_sweep",
    "default_sensor_to_ego",
    "fixture_config",
    "load_fixture_sweep",
    "ray_directions",
    "synthetic_sweep",
]
