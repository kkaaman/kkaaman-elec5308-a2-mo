"""Synthetic kerbed road for offline closed-loop tests of the mapless car (G1b).

Not a simulator replacement: it checks that the pieces fit together and behave sensibly
before a live run. The world is:

- a road bounded by a right kerb line made of straight and arc pieces, ``road_width_m``
  wide, with pavement (``kerb_height_m`` up) on both sides;
- parked cars as boxes;
- a kinematic bicycle car (rear-axle state, first-order steering, throttle and brake as
  accelerations).

It produces ``SensorFrame``s for a ``SensorSuite``: LiDAR by ray casting each beam against
the road, pavement and boxes; GNSS through a tangent plane with the suite's bias and white
noise; IMU gyro, compass and gravity with the suite's noise; and the speedometer with its
scale error. Cameras and radars are reported NOT_DUE (not simulated here).

Truth (lane-centre offset, heading error, bumper gap, collisions) is available from the
world for scoring and never enters a frame.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from mapless.frame import GnssFix, ImuSample, LidarSweep, SensorFrame, SensorState, SensorStatus
from mapless.geometry import sensor_to_ego
from mapless.satnav import Command, LocalTangentPlane, RoutePoint, SatNavRoute
from navigation_control.types import ControlCommand

REFERENCE_FIX = (-33.8885, 151.1873)  # an arbitrary geodetic origin for the synthetic plane


@dataclass(frozen=True)
class Straight:
    length_m: float


@dataclass(frozen=True)
class Arc:
    radius_m: float  # positive turns left
    angle_rad: float


class KerbLine:
    """A planar polyline-free curve: straight and arc pieces from a start pose."""

    def __init__(self, pieces, start=(0.0, 0.0, 0.0)):
        self.pieces = tuple(pieces)
        if not self.pieces:
            raise ValueError("A kerb line needs at least one piece")
        self.starts = []
        x, y, h = start
        s = 0.0
        for piece in self.pieces:
            self.starts.append((x, y, h, s))
            if isinstance(piece, Straight):
                x, y = x + piece.length_m * math.cos(h), y + piece.length_m * math.sin(h)
                s += piece.length_m
            else:
                r, a = piece.radius_m, piece.angle_rad
                cx, cy = x - r * math.sin(h), y + r * math.cos(h)
                h2 = h + math.copysign(a, r)
                x, y = cx + r * math.sin(h2), cy - r * math.cos(h2)
                h = h2
                s += abs(r) * a
        self.end = (x, y, h)
        self.length_m = s

    def pose_at(self, s_m):
        """(x, y, heading) at arc length s; straight extrapolation beyond either end."""
        if s_m < 0:
            x, y, h, _ = self.starts[0]
            return x + s_m * math.cos(h), y + s_m * math.sin(h), h
        for piece, (x, y, h, s0) in zip(self.pieces, self.starts):
            length = piece.length_m if isinstance(piece, Straight) else abs(piece.radius_m) * piece.angle_rad
            if s_m <= s0 + length:
                u = s_m - s0
                if isinstance(piece, Straight):
                    return x + u * math.cos(h), y + u * math.sin(h), h
                r = piece.radius_m
                cx, cy = x - r * math.sin(h), y + r * math.cos(h)
                h2 = h + u / r
                return cx + r * math.sin(h2), cy - r * math.cos(h2), h2
        x, y, h = self.end
        u = s_m - self.length_m
        return x + u * math.cos(h), y + u * math.sin(h), h

    def locate(self, xy):
        """Arc length and signed offset (+left) of points from the line."""
        p = np.asarray(xy, dtype=float).reshape(-1, 2)
        best_d = np.full(len(p), np.inf)
        best_s = np.zeros(len(p))
        count = len(self.pieces)
        for index, (piece, (x, y, h, s0)) in enumerate(zip(self.pieces, self.starts)):
            if isinstance(piece, Straight):
                rel = p - (x, y)
                u = rel[:, 0] * math.cos(h) + rel[:, 1] * math.sin(h)
                d = -rel[:, 0] * math.sin(h) + rel[:, 1] * math.cos(h)
                length = piece.length_m
            else:
                r = piece.radius_m
                cx, cy = x - r * math.sin(h), y + r * math.cos(h)
                rel = p - (cx, cy)
                rho = np.hypot(rel[:, 0], rel[:, 1])
                start_angle = math.atan2(y - cy, x - cx)
                phi = np.arctan2(rel[:, 1], rel[:, 0]) - start_angle
                phi = np.mod(phi if r > 0 else -phi, 2 * math.pi)
                phi = np.where(phi > math.pi + piece.angle_rad / 2, phi - 2 * math.pi, phi)
                u = abs(r) * phi
                d = (r - rho) if r > 0 else (rho - abs(r))
                length = abs(r) * piece.angle_rad
            lo = -np.inf if index == 0 else 0.0
            hi = np.inf if index == count - 1 else length
            ok = (u >= lo) & (u <= hi) & (np.abs(d) < np.abs(best_d))
            best_d = np.where(ok, d, best_d)
            best_s = np.where(ok, s0 + u, best_s)
        return best_s, best_d


@dataclass(frozen=True)
class ParkedCar:
    s_m: float  # along the kerb line
    offset_m: float  # centre's offset left of the kerb line
    length_m: float = 4.2
    width_m: float = 1.8
    height_m: float = 1.5


@dataclass(frozen=True)
class RoadScene:
    kerb: KerbLine
    road_width_m: float = 7.0
    lane_centre_m: float = 1.75  # ego lane centre, left of the right kerb
    kerb_height_m: float = 0.15
    parked: tuple = ()
    openings: tuple = ()  # (s_start, s_end): side roads on the right, road surface out to 15 m
    crown_m: float = 0.0  # road centre this much higher than its edges (drainage camber)

    def on_road(self, s, d):
        road = (d >= 0) & (d <= self.road_width_m)
        for s0, s1 in self.openings:
            road |= (s >= s0) & (s <= s1) & (d < 0) & (d > -15.0)
        return road

    def box(self, car):
        x, y, h = self.kerb.pose_at(car.s_m)
        cx, cy = x - car.offset_m * math.sin(h), y + car.offset_m * math.cos(h)
        return (cx, cy, h, car.length_m / 2, car.width_m / 2, car.height_m)


@dataclass(frozen=True)
class CarModel:
    wheelbase_m: float = 3.00465
    max_steer_rad: float = math.radians(20)
    steer_tau_s: float = 0.15
    throttle_accel_mps2: float = 2.5
    brake_decel_mps2: float = 6.0
    drag_mps2: float = 0.3
    front_m: float = 3.78
    rear_m: float = -1.01
    half_width_m: float = 1.05


@dataclass
class CarState:
    x: float
    y: float
    yaw: float
    speed: float = 0.0
    steer: float = 0.0
    yaw_rate: float = 0.0


@dataclass(frozen=True)
class TruthSample:
    time_s: float
    s_m: float
    lane_offset_m: float  # rear axle's offset from the lane centre, +left
    heading_error_rad: float
    speed_mps: float
    front_gap_m: float  # bumper to the nearest parked car ahead in the ego lane (inf if none)
    collision: bool


@dataclass
class _Noise:
    gnss_bias: np.ndarray
    gyro_bias: float
    wheel_scale: float


def _ray_box(origin, directions, box):
    """Distance along each unit ray to an oriented box (inf if missed)."""
    cx, cy, h, hx, hy, top = box
    c, s = math.cos(h), math.sin(h)
    o = origin - (cx, cy, 0.0)
    ox, oy = c * o[0] + s * o[1], -s * o[0] + c * o[1]
    dx = c * directions[:, 0] + s * directions[:, 1]
    dy = -s * directions[:, 0] + c * directions[:, 1]
    dz = directions[:, 2]
    t_lo = np.full(len(directions), -np.inf)
    t_hi = np.full(len(directions), np.inf)
    for o_axis, d_axis, lo, hi in ((ox, dx, -hx, hx), (oy, dy, -hy, hy), (o[2], dz, 0.0, top)):
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (lo - o_axis) / d_axis, (hi - o_axis) / d_axis
        parallel = np.abs(d_axis) < 1e-12
        inside = (o_axis >= lo) & (o_axis <= hi)
        t1 = np.where(parallel, np.where(inside, -np.inf, np.inf), t1)
        t2 = np.where(parallel, np.where(inside, np.inf, -np.inf), t2)
        t_lo = np.maximum(t_lo, np.minimum(t1, t2))
        t_hi = np.minimum(t_hi, np.maximum(t1, t2))
    hit = (t_hi >= t_lo) & (t_hi > 0)
    return np.where(hit, np.maximum(t_lo, 0.0), np.inf)


@dataclass
class MaplessRoadWorld:
    scene: RoadScene
    suite: object
    start_s_m: float = 10.0
    start_offset_m: float = 0.0  # from the lane centre, +left
    start_heading_error_rad: float = 0.0
    dt_s: float = 0.05
    seed: int = 5308
    car: CarModel = field(default_factory=CarModel)
    rear_axle_offset_m: float = 1.38641
    azimuth_step_deg: float = 0.72
    noise: bool = True

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        x, y, h = self.scene.kerb.pose_at(self.start_s_m)
        off = self.scene.lane_centre_m + self.start_offset_m
        self.state = CarState(x - off * math.sin(h), y + off * math.cos(h), h + self.start_heading_error_rad)
        self.gnss = next(s for s in self.suite.sensors if s.kind == "gnss")
        self.plane = LocalTangentPlane(*REFERENCE_FIX, north_sign=self.gnss.north_sign)
        self.time_s = 0.0
        self.frame_id = 0
        self.lidar = self.suite.lidars[0]
        self.imu = next(s for s in self.suite.sensors if s.kind == "imu")
        self.wheel = next(s for s in self.suite.sensors if s.kind == "speedometer")
        scale = 1.0 if self.noise else 0.0
        self._noise = _Noise(
            gnss_bias=self.rng.normal(0.0, self.gnss.bias_sigma_m * scale, 2),
            gyro_bias=float(self.rng.normal(0.0, self.imu.gyro_bias_sigma_rps * scale)),
            wheel_scale=1.0 + float(self.rng.normal(0.0, self.wheel.scale_sigma * scale)),
        )
        self._next_fix_s = 0.0
        self.boxes = [self.scene.box(car) for car in self.scene.parked]
        self.lidar_to_ego = sensor_to_ego(self.lidar.mount, self.rear_axle_offset_m)
        channels = np.linspace(self.lidar.lower_fov_deg, self.lidar.upper_fov_deg, self.lidar.channels)
        azimuths = np.arange(0.0, 360.0, self.azimuth_step_deg)
        el, az = np.meshgrid(np.radians(channels), np.radians(azimuths), indexing="ij")
        self._beams = np.column_stack(
            (np.cos(el).ravel() * np.cos(az).ravel(), np.cos(el).ravel() * np.sin(az).ravel(), np.sin(el).ravel())
        )

    # -- routes and truth ------------------------------------------------------------
    def route(self, goal_s_m, spacing_m=50.0, name="synthetic"):
        """A LANEFOLLOW sat-nav route along the ego lane centre from the start to ``goal_s_m``."""
        stations = list(np.arange(self.start_s_m, goal_s_m, spacing_m)) + [goal_s_m]
        points = []
        for s in stations:
            x, y, h = self.scene.kerb.pose_at(s)
            off = self.scene.lane_centre_m
            lat, lon = self.plane.to_geodetic(x - off * math.sin(h), y + off * math.cos(h))
            points.append(RoutePoint(lat, lon, Command.LANEFOLLOW))
        return SatNavRoute(name, tuple(points))

    def truth(self):
        st, car = self.state, self.car
        s, d = self.scene.kerb.locate(np.array([[st.x, st.y]]))
        _, _, h = self.scene.kerb.pose_at(float(s[0]))
        heading_error = (st.yaw - h + math.pi) % (2 * math.pi) - math.pi
        gap, collision = math.inf, False
        corners = self._body_corners()
        for box in self.boxes:
            if _overlap(corners, box):
                collision = True
            bs, bd = self.scene.kerb.locate(np.array(box[:2])[None])
            lateral = abs(float(bd[0]) - float(d[0]))
            if lateral < box[4] + car.half_width_m:
                front_s = float(s[0]) + car.front_m
                gap = min(gap, float(bs[0]) - box[3] - front_s) if float(bs[0]) > float(s[0]) else gap
        return TruthSample(
            time_s=self.time_s,
            s_m=float(s[0]),
            lane_offset_m=float(d[0]) - self.scene.lane_centre_m,
            heading_error_rad=heading_error,
            speed_mps=st.speed,
            front_gap_m=gap,
            collision=collision,
        )

    def _body_corners(self):
        st, car = self.state, self.car
        c, s = math.cos(st.yaw), math.sin(st.yaw)
        local = [(car.front_m, car.half_width_m), (car.front_m, -car.half_width_m), (car.rear_m, -car.half_width_m), (car.rear_m, car.half_width_m)]
        return np.array([(st.x + c * u - s * v, st.y + s * u + c * v) for u, v in local])

    # -- sensing -----------------------------------------------------------------------
    def _lidar_points(self):
        st = self.state
        origin_ego = self.lidar_to_ego[:3, 3]
        directions_ego = self._beams @ self.lidar_to_ego[:3, :3].T
        c, s = math.cos(st.yaw), math.sin(st.yaw)
        to_world = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        origin = np.array([st.x, st.y, 0.0]) + to_world @ origin_ego
        directions = directions_ego @ to_world.T
        r_max = self.lidar.range_m
        t = np.full(len(directions), np.inf)
        down = directions[:, 2] < -1e-6
        dz = np.where(down, directions[:, 2], -1.0)
        t_ground = np.where(down, -origin[2] / dz, 0.0)  # 0 for beams that never come down (masked below)
        t_kerb = np.where(down, (self.scene.kerb_height_m - origin[2]) / dz, 0.0)
        width = self.scene.road_width_m
        ground = origin[:2] + t_ground[:, None] * directions[:, :2]
        top = origin[:2] + t_kerb[:, None] * directions[:, :2]
        s_ground, d_ground = self.scene.kerb.locate(ground)
        s_top, d_top = self.scene.kerb.locate(top)
        on_road = self.scene.on_road(s_ground, d_ground)
        top_on_pavement = ~self.scene.on_road(s_top, d_top)
        t = np.where(down & on_road, t_ground, t)
        t = np.where(down & ~on_road & top_on_pavement, t_kerb, t)
        # Over the road at kerb height but on the pavement at road level: the beam hits the
        # kerb face. Interpolate to where it crosses the kerb line.
        face = down & ~on_road & ~top_on_pavement
        boundary = np.where(d_ground < 0, 0.0, width)
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.clip((boundary - d_top) / (d_ground - d_top), 0.0, 1.0)
        t = np.where(face, t_kerb + f * (t_ground - t_kerb), t)
        for box in self.boxes:
            t = np.minimum(t, _ray_box(origin, directions, box))
        hit = np.isfinite(t) & (t <= r_max)
        world = origin + t[hit, None] * directions[hit]
        if self.scene.crown_m:
            _, d = self.scene.kerb.locate(world[:, :2])
            road = (d >= 0) & (d <= width) & (world[:, 2] < 0.05)
            world[road, 2] += self.scene.crown_m * (1 - np.abs(2 * d[road] / width - 1))
        if self.noise:
            world += directions[hit] * self.rng.normal(0.0, self.lidar.noise_stddev_m, (int(hit.sum()), 1))
        rel = world - np.array([st.x, st.y, 0.0])
        ego = rel @ to_world  # world -> ego (inverse rotation)
        intensity = np.full((len(ego), 1), 0.5)
        return np.hstack((ego, intensity)).astype(np.float32)

    def frame(self):
        st, n = self.state, self._noise
        status = []
        lidars = ()
        for spec in self.suite.sensors:
            if spec.kind == "lidar":
                lidars = (LidarSweep(spec.name, self._lidar_points(), self.lidar_to_ego),)
                status.append(SensorStatus(spec.name, spec.kind, SensorState.OK, self.time_s))
            elif spec.kind in ("camera", "radar"):
                status.append(SensorStatus(spec.name, spec.kind, SensorState.NOT_DUE, detail="not simulated"))
        gnss = None
        if self.time_s + 1e-9 >= self._next_fix_s:
            self._next_fix_s += self.gnss.period_s
            white = self.rng.normal(0.0, self.gnss.white_m if self.noise else 0.0, 2)
            lat, lon = self.plane.to_geodetic(*(np.array([st.x, st.y]) + n.gnss_bias + white))
            gnss = GnssFix(lat, lon, 0.0)
            status.append(SensorStatus(self.gnss.name, "gnss", SensorState.OK, self.time_s))
        else:
            status.append(SensorStatus(self.gnss.name, "gnss", SensorState.NOT_DUE))
        k = 1.0 if self.noise else 0.0
        gyro = st.yaw_rate + n.gyro_bias + float(self.rng.normal(0.0, self.imu.gyro_white_rps * k))
        compass = (math.pi / 2 - st.yaw + float(self.rng.normal(0.0, self.imu.compass_white_rad * k))) % (2 * math.pi)
        imu = ImuSample((0.0, 0.0, 9.81), (0.0, 0.0, gyro), compass)
        status.append(SensorStatus(self.imu.name, "imu", SensorState.OK, self.time_s))
        speed = n.wheel_scale * st.speed + float(self.rng.normal(0.0, self.wheel.white_mps * k))
        status.append(SensorStatus(self.wheel.name, "speedometer", SensorState.OK, self.time_s))
        frame = SensorFrame(
            episode_id="synthetic",
            frame_id=self.frame_id,
            sim_time_s=self.time_s,
            received_monotonic_s=self.time_s,
            cameras=(),
            lidars=lidars,
            radars=(),
            gnss=gnss,
            imu=imu,
            speed_mps=speed,
            status=tuple(status),
            calibration_id="synthetic",
        )
        return frame

    # -- motion --------------------------------------------------------------------------
    def apply(self, command):
        if not isinstance(command, ControlCommand):
            raise TypeError("command must be a ControlCommand")
        st, car, dt = self.state, self.car, self.dt_s
        target = max(-car.max_steer_rad, min(car.max_steer_rad, command.steering_angle_rad))
        st.steer += (target - st.steer) * min(1.0, dt / car.steer_tau_s)
        accel = command.throttle * car.throttle_accel_mps2 - command.brake * car.brake_decel_mps2 - car.drag_mps2 * (st.speed > 0)
        speed = max(0.0, st.speed + accel * dt)
        mean = 0.5 * (st.speed + speed)
        st.yaw_rate = mean * math.tan(st.steer) / car.wheelbase_m
        heading = st.yaw + 0.5 * st.yaw_rate * dt
        st.x += mean * dt * math.cos(heading)
        st.y += mean * dt * math.sin(heading)
        st.yaw += st.yaw_rate * dt
        st.speed = speed
        self.time_s += dt
        self.frame_id += 1


def _overlap(corners, box):
    """Separating-axis test between the car's corners and an oriented box (plan view)."""
    cx, cy, h, hx, hy, _ = box
    c, s = math.cos(h), math.sin(h)
    box_corners = np.array([(cx + c * u - s * v, cy + s * u + c * v) for u, v in ((hx, hy), (hx, -hy), (-hx, -hy), (-hx, hy))])
    for poly in (corners, box_corners):
        for i in range(4):
            edge = poly[(i + 1) % 4] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            a, b = corners @ axis, box_corners @ axis
            if a.max() < b.min() or b.max() < a.min():
                return False
    return True
