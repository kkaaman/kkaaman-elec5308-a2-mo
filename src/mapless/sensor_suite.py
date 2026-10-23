"""What is mounted on the car, checked against the CARLA Leaderboard 2.0 SENSORS track.

A suite file (``configs/sensor_suites/*.json``) lists every sensor:
- Mounts are in the vehicle frame (+x forward, +y left, +z up, metres), relative to the
  CARLA actor origin.
- ``yaw_deg`` is positive to the left; ``pitch_deg`` is positive nose-up. Roll is not
  supported.
- Noise settings sit on each sensor. The simulator adapter applies them; the car never
  sees noise-free values.

Only these types may be mounted:

| Type | Limit |
|---|---|
| camera (RGB) | 8 |
| lidar (ray cast) | 2 |
| radar | 4 |
| gnss | 1 |
| imu | 1 |
| speedometer | 1 |

The speedometer stands for a wheel encoder: true speed with a seeded scale error and
noise added. Any other sensor type is refused, and ground-truth types are refused with
the reason. Collision and lane-invasion sensors belong to the evaluator, never to the car.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

SCHEMA_VERSION = 1

LIMITS = {"camera": 8, "lidar": 2, "radar": 4, "gnss": 1, "imu": 1, "speedometer": 1}

BLUEPRINTS = {
    "camera": "sensor.camera.rgb",
    "lidar": "sensor.lidar.ray_cast",
    "radar": "sensor.other.radar",
    "gnss": "sensor.other.gnss",
    "imu": "sensor.other.imu",
}

# Why each simulator sensor is not allowed on the car.
BANNED = {
    "depth": "sensor.camera.depth gives exact per-pixel range; no real sensor does",
    "semantic_segmentation": "segmentation cameras give ground-truth labels (training labels only)",
    "instance_segmentation": "segmentation cameras give ground-truth labels (training labels only)",
    "semantic_lidar": "semantic LiDAR labels every point with its true class and actor",
    "obstacle": "the obstacle detector reports true actors ahead",
    "optical_flow": "optical flow is rendered from true motion",
    "normals": "surface normals are rendered from true geometry",
    "collision": "collision events are for the evaluator",
    "lane_invasion": "lane-invasion events come from the map; evaluator only",
}

BANNED_BLUEPRINTS = frozenset(
    {
        "sensor.camera.depth",
        "sensor.camera.semantic_segmentation",
        "sensor.camera.instance_segmentation",
        "sensor.camera.optical_flow",
        "sensor.camera.normals",
        "sensor.camera.dvs",
        "sensor.lidar.ray_cast_semantic",
        "sensor.other.obstacle",
        "sensor.other.collision",
        "sensor.other.lane_invasion",
        "sensor.other.rss",
        "sensor.other.v2x",
        "sensor.other.v2x_custom",
    }
)


def _finite(name, value, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class Mount:
    x_m: float
    y_m: float
    z_m: float
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0

    def __post_init__(self):
        for item in fields(self):
            _finite(item.name, getattr(self, item.name))
        if not -180.0 <= self.yaw_deg <= 180.0 or not -90.0 < self.pitch_deg < 90.0:
            raise ValueError("Mount yaw must be in [-180, 180] and pitch in (-90, 90)")


@dataclass(frozen=True)
class CameraSpec:
    name: str
    mount: Mount
    width: int
    height: int
    fov_deg: float
    kind: str = "camera"

    def __post_init__(self):
        if type(self.width) is not int or type(self.height) is not int:
            raise ValueError("Camera size must be integers")
        if not 16 <= self.width <= 4096 or not 16 <= self.height <= 4096:
            raise ValueError("Camera size out of range")
        _finite("fov_deg", self.fov_deg, positive=True)
        if self.fov_deg >= 180:
            raise ValueError("fov_deg must be below 180")


@dataclass(frozen=True)
class LidarSpec:
    name: str
    mount: Mount
    channels: int = 32
    range_m: float = 70.0
    rotation_frequency_hz: float = 20.0
    points_per_second: int = 640000
    upper_fov_deg: float = 10.0
    lower_fov_deg: float = -25.0
    horizontal_fov_deg: float = 360.0
    noise_stddev_m: float = 0.02
    dropoff_general_rate: float = 0.45
    dropoff_intensity_limit: float = 0.8
    dropoff_zero_intensity: float = 0.4
    atmosphere_attenuation_rate: float = 0.004
    kind: str = "lidar"

    def __post_init__(self):
        if type(self.channels) is not int or not 1 <= self.channels <= 128:
            raise ValueError("channels must be an integer in 1..128")
        if type(self.points_per_second) is not int or self.points_per_second <= 0:
            raise ValueError("points_per_second must be a positive integer")
        for name in ("range_m", "rotation_frequency_hz", "horizontal_fov_deg"):
            _finite(name, getattr(self, name), positive=True)
        for name in (
            "noise_stddev_m",
            "dropoff_general_rate",
            "dropoff_intensity_limit",
            "dropoff_zero_intensity",
            "atmosphere_attenuation_rate",
        ):
            _finite(name, getattr(self, name), nonnegative=True)
        for name in ("dropoff_general_rate", "dropoff_intensity_limit", "dropoff_zero_intensity"):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must be at most 1")
        _finite("upper_fov_deg", self.upper_fov_deg)
        _finite("lower_fov_deg", self.lower_fov_deg)
        if not -90 <= self.lower_fov_deg < self.upper_fov_deg <= 90:
            raise ValueError("LiDAR vertical field of view is invalid")
        if self.horizontal_fov_deg > 360:
            raise ValueError("horizontal_fov_deg must be at most 360")

    def full_sweep_per_tick(self, dt_s):
        """True when one tick of ``dt_s`` covers exactly one rotation."""
        return math.isclose(self.rotation_frequency_hz * dt_s, 1.0, rel_tol=1e-9)


@dataclass(frozen=True)
class RadarSpec:
    name: str
    mount: Mount
    horizontal_fov_deg: float = 30.0
    vertical_fov_deg: float = 10.0
    range_m: float = 100.0
    points_per_second: int = 1500
    kind: str = "radar"

    def __post_init__(self):
        for name in ("horizontal_fov_deg", "vertical_fov_deg", "range_m"):
            _finite(name, getattr(self, name), positive=True)
        if self.horizontal_fov_deg > 180 or self.vertical_fov_deg > 90:
            raise ValueError("Radar field of view out of range")
        if type(self.points_per_second) is not int or self.points_per_second <= 0:
            raise ValueError("points_per_second must be a positive integer")


@dataclass(frozen=True)
class GnssSpec:
    name: str
    mount: Mount
    period_s: float = 0.2
    white_m: float = 0.5
    bias_sigma_m: float = 1.0  # constant per-axis offset, drawn once per run
    # +1: latitude increases towards the compass's north, as on Earth. -1: it increases
    # towards the compass's south, as measured on CARLA 0.9.16 (Town10HD, Town05: the GNSS
    # sensor's latitude grows with CARLA +y while the compass's north is CARLA -y). The
    # local plane uses it so that the route frame is right-handed with the heading; a
    # wrong value mirrors the route and the live curve run drove away from its goal.
    north_sign: int = 1
    kind: str = "gnss"

    def __post_init__(self):
        _finite("period_s", self.period_s, positive=True)
        _finite("white_m", self.white_m, nonnegative=True)
        _finite("bias_sigma_m", self.bias_sigma_m, nonnegative=True)
        if self.north_sign not in (1, -1):
            raise ValueError("north_sign must be 1 or -1")


@dataclass(frozen=True)
class ImuSpec:
    name: str
    mount: Mount
    gyro_white_rps: float = 0.005
    gyro_bias_sigma_rps: float = 0.002  # constant z offset, drawn once per run
    accel_white_mps2: float = 0.05
    compass_white_rad: float = 0.02
    kind: str = "imu"

    def __post_init__(self):
        for name in (
            "gyro_white_rps",
            "gyro_bias_sigma_rps",
            "accel_white_mps2",
            "compass_white_rad",
        ):
            _finite(name, getattr(self, name), nonnegative=True)


@dataclass(frozen=True)
class SpeedometerSpec:
    name: str
    white_mps: float = 0.05
    scale_sigma: float = 0.01  # constant scale error, drawn once per run
    kind: str = "speedometer"

    def __post_init__(self):
        _finite("white_mps", self.white_mps, nonnegative=True)
        _finite("scale_sigma", self.scale_sigma, nonnegative=True)


_SPECS = {
    "camera": CameraSpec,
    "lidar": LidarSpec,
    "radar": RadarSpec,
    "gnss": GnssSpec,
    "imu": ImuSpec,
    "speedometer": SpeedometerSpec,
}

# Fields a noise-free copy zeroes (rig checks verify geometry and signs, not noise).
_NOISE_FIELDS = {
    "lidar": ("noise_stddev_m", "dropoff_general_rate", "dropoff_zero_intensity"),
    "gnss": ("white_m", "bias_sigma_m"),
    "imu": ("gyro_white_rps", "gyro_bias_sigma_rps", "accel_white_mps2", "compass_white_rad"),
    "speedometer": ("white_mps", "scale_sigma"),
}


@dataclass(frozen=True)
class SensorSuite:
    name: str
    sensors: tuple = field(default_factory=tuple)
    description: str = ""
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unsupported sensor-suite schema version")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Sensor suite needs a name")
        sensors = tuple(self.sensors)
        names = [s.name for s in sensors]
        for s in sensors:
            if type(s) not in _SPECS.values():
                raise TypeError(f"Unsupported sensor specification {type(s).__name__}")
            if not isinstance(s.name, str) or not s.name.strip() or not s.name.isidentifier():
                raise ValueError(f"Sensor name {s.name!r} must be a plain identifier")
        if len(set(names)) != len(names):
            raise ValueError("Sensor names must be unique")
        for kind, limit in LIMITS.items():
            count = sum(s.kind == kind for s in sensors)
            if count > limit:
                raise ValueError(f"At most {limit} {kind} sensor(s) allowed; got {count}")
        for kind in ("camera", "imu", "speedometer"):
            if not any(s.kind == kind for s in sensors):
                raise ValueError(f"A suite needs at least one {kind}")
        object.__setattr__(self, "sensors", sensors)

    def of_kind(self, kind):
        return tuple(s for s in self.sensors if s.kind == kind)

    @property
    def cameras(self):
        return self.of_kind("camera")

    @property
    def lidars(self):
        return self.of_kind("lidar")

    @property
    def radars(self):
        return self.of_kind("radar")

    @property
    def gnss(self):
        found = self.of_kind("gnss")
        return found[0] if found else None

    @property
    def imu(self):
        return self.of_kind("imu")[0]

    @property
    def speedometer(self):
        return self.of_kind("speedometer")[0]

    def as_dict(self):
        return dict(
            schema_version=self.schema_version,
            name=self.name,
            description=self.description,
            sensors=[_sensor_dict(s) for s in self.sensors],
        )

    @property
    def sha256(self):
        text = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode()).hexdigest()

    def without_noise(self):
        """Copy with every noise term zeroed, for checks of geometry and sign conventions."""
        sensors = []
        for s in self.sensors:
            zeroed = {name: 0.0 for name in _NOISE_FIELDS.get(s.kind, ())}
            sensors.append(replace(s, **zeroed) if zeroed else s)
        return replace(self, name=f"{self.name}-noise-free", sensors=tuple(sensors))

    def with_overrides(self, overrides):
        """Copy with ``{"sensor.field": value}`` or ``{"sensor.mount.field": value}`` applied.

        Used for rig experiments (for example radar density against FOV). Every value
        passes the same validation as the suite file. The result records the overrides
        in its name and description, so its hash differs from the base suite's.
        """
        if not overrides:
            return self
        by_name = {s.name: s for s in self.sensors}
        changed = dict(by_name)
        for key, value in sorted(overrides.items()):
            parts = key.split(".")
            if len(parts) not in (2, 3) or parts[0] not in by_name:
                raise ValueError(f"Override {key!r} must be <sensor>.<field> or <sensor>.mount.<field>")
            spec = changed[parts[0]]
            if len(parts) == 3:
                if parts[1] != "mount" or not hasattr(spec, "mount") or parts[2] not in {f.name for f in fields(Mount)}:
                    raise ValueError(f"Unknown mount override {key!r}")
                changed[parts[0]] = replace(spec, mount=replace(spec.mount, **{parts[2]: value}))
            else:
                if parts[1] in ("name", "kind", "mount") or parts[1] not in {f.name for f in fields(spec)}:
                    raise ValueError(f"Unknown override field {key!r}")
                changed[parts[0]] = replace(spec, **{parts[1]: value})
        text = ", ".join(f"{k}={v}" for k, v in sorted(overrides.items()))
        return replace(
            self,
            name=f"{self.name}+overrides",
            description=f"{self.description} Overrides: {text}".strip(),
            sensors=tuple(changed[s.name] for s in self.sensors),
        )

    def validate_for_timestep(self, dt_s):
        """Problems that only show against the simulation step (empty when none)."""
        problems = []
        for lidar in self.lidars:
            if not lidar.full_sweep_per_tick(dt_s):
                problems.append(
                    f"{lidar.name}: rotation {lidar.rotation_frequency_hz} Hz does not give one"
                    f" full sweep per {dt_s} s tick"
                )
        gnss = self.gnss
        if gnss is not None and gnss.period_s < dt_s - 1e-12:
            problems.append(f"{gnss.name}: GNSS period is shorter than the simulation step")
        return problems


def _sensor_dict(spec):
    result = {"type": spec.kind}
    for item in fields(spec):
        if item.name == "kind":
            continue
        value = getattr(spec, item.name)
        result[item.name] = asdict(value) if isinstance(value, Mount) else value
    return result


def sensor_from_dict(entry):
    if not isinstance(entry, dict):
        raise ValueError("Each sensor entry must be an object")
    kind = entry.get("type")
    if kind in BANNED:
        raise ValueError(f"Sensor type {kind!r} is not allowed on the car: {BANNED[kind]}")
    if kind not in _SPECS:
        raise ValueError(f"Unknown sensor type {kind!r}; allowed: {', '.join(_SPECS)}")
    spec = _SPECS[kind]
    allowed = {item.name for item in fields(spec)} - {"kind"}
    values = {k: v for k, v in entry.items() if k != "type"}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {kind} fields: {', '.join(sorted(unknown))}")
    if "mount" in allowed:
        if not isinstance(values.get("mount"), dict):
            raise ValueError(f"{kind} needs a mount")
        mount = dict(values["mount"])
        unknown_mount = set(mount) - {item.name for item in fields(Mount)}
        if unknown_mount:
            raise ValueError(f"Unknown mount fields: {', '.join(sorted(unknown_mount))}")
        values["mount"] = Mount(**mount)
    return spec(**values)


def suite_from_dict(document):
    if not isinstance(document, dict):
        raise ValueError("Sensor suite must be a JSON object")
    unknown = set(document) - {"schema_version", "name", "description", "sensors"}
    if unknown:
        raise ValueError(f"Unknown sensor-suite fields: {', '.join(sorted(unknown))}")
    if not isinstance(document.get("sensors"), list):
        raise ValueError("Sensor suite needs a sensors list")
    return SensorSuite(
        name=document.get("name", ""),
        sensors=tuple(sensor_from_dict(entry) for entry in document["sensors"]),
        description=document.get("description", ""),
        schema_version=document.get("schema_version", SCHEMA_VERSION),
    )


def _reject_duplicates(pairs):
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate key in sensor-suite file")
    return dict(pairs)


def load_sensor_suite(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    return suite_from_dict(json.loads(text, object_pairs_hook=_reject_duplicates))


def parse_overrides(items):
    """``["radar_front.points_per_second=3000", ...]`` to a typed dict (int, float, else str)."""
    result = {}
    for item in items or ():
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"Override {item!r} must look like sensor.field=value")
        raw = raw.strip()
        for cast in (int, float):
            try:
                value = cast(raw)
                break
            except ValueError:
                continue
        else:
            value = raw
        result[key.strip()] = value
    return result
