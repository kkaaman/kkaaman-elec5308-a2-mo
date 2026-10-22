"""End-to-end smoke run of the live calibration gate against a small fake CARLA world.

The fake world is a straight road with a lane on each side of the ego lane. It simulates
simple kinematics, including constant-velocity actors, and emits sensor data in CARLA's
own conventions, computed independently of the rig's conversion code:
- left-handed axes;
- radar azimuth positive to the right;
- radar velocity is the range rate, positive when receding, as measured live on
  22 Oct (``RADAR_VELOCITY_IS_RANGE_RATE = True``), including the sensor's lever arm;
- compass clockwise from north, where north is -y;
- instance masks drawn from box silhouettes, projected in the camera's UE frame.

This cannot prove CARLA's conventions; only the live run can. It proves two things:
- the gate runs from start to finish;
- a simulator whose data follows these conventions gets MEASURED_PASS.
"""

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import carla_adapter.sensor_rig_check as rig_check
from evaluation.satnav_route import carla_geolocation

ROOT = Path(__file__).resolve().parents[2]
GEO = carla_geolocation(49.0, 8.0)
DT = 0.05
LANE = 3.5


class Vector:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)


class Rotation:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch, self.yaw, self.roll = float(pitch), float(yaw), float(roll)


class Transform:
    def __init__(self, location=None, rotation=None):
        self.location = location or Vector()
        self.rotation = rotation or Rotation()

    def get_forward_vector(self):
        yaw = math.radians(self.rotation.yaw)
        return Vector(math.cos(yaw), math.sin(yaw), 0.0)

    def copy(self):
        loc, rot = self.location, self.rotation
        return Transform(Vector(loc.x, loc.y, loc.z), Rotation(rot.pitch, rot.yaw, rot.roll))


class VehicleControl:
    def __init__(self, throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False):
        self.throttle, self.steer, self.brake = throttle, steer, brake


def fake_carla_module():
    return SimpleNamespace(
        Client=Client,
        Location=Vector,
        Vector3D=Vector,
        Rotation=Rotation,
        Transform=Transform,
        VehicleControl=VehicleControl,
        WeatherParameters=SimpleNamespace(ClearNoon="ClearNoon"),
        LaneType=SimpleNamespace(Driving="Driving"),
        AttachmentType=SimpleNamespace(Rigid="Rigid"),
    )


EXTENTS = {"vehicle.tesla.model3": (2.396, 1.082, 0.75), "vehicle.audi.a2": (1.86, 0.88, 0.77), "static.prop.box01": (0.4, 0.4, 0.4)}


class Box:
    def __init__(self, extent):
        self.extent = Vector(*extent)
        self.location = Vector(0.0, 0.0, extent[2])

    def get_world_vertices(self, transform):
        yaw = math.radians(transform.rotation.yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        out = []
        for i in (-1, 1):
            for j in (-1, 1):
                for k in (-1, 1):
                    lx, ly = i * self.extent.x, j * self.extent.y
                    out.append(
                        Vector(
                            transform.location.x + c * lx - s * ly,
                            transform.location.y + s * lx + c * ly,
                            transform.location.z + self.location.z + k * self.extent.z,
                        )
                    )
        return out


class Actor:
    def __init__(self, actor_id, type_id, transform):
        self.id, self.type_id, self.is_alive = actor_id, type_id, True
        t = transform.copy()
        t.location.z = 0.0
        self._t = t
        self.speed, self.yaw_rate, self.accel = 0.0, 0.0, 0.0  # yaw rate in UE (rad/s)
        self.control, self.physics, self.const = VehicleControl(brake=1.0), True, None
        self.bounding_box = Box(EXTENTS[type_id])

    def get_transform(self):
        return self._t.copy()

    def set_transform(self, transform):
        t = transform.copy()
        t.location.z = 0.0
        self._t = t

    def get_velocity(self):
        f = self._t.get_forward_vector()
        return Vector(self.speed * f.x, self.speed * f.y, 0.0)

    def get_angular_velocity(self):
        return Vector(0.0, 0.0, math.degrees(self.yaw_rate))

    def get_acceleration(self):
        f = self._t.get_forward_vector()
        return Vector(self.accel * f.x, self.accel * f.y, 0.0)

    def apply_control(self, control):
        self.control = control

    def set_simulate_physics(self, enabled):
        self.physics = enabled

    def enable_constant_velocity(self, velocity):
        self.const = velocity.x

    def disable_constant_velocity(self):
        self.const = None

    def destroy(self):
        self.is_alive = False
        return True

    def step(self):
        if self.const is not None:
            self.accel, self.speed, self.yaw_rate = 0.0, self.const, 0.0
        elif self.physics and self.type_id == "vehicle.tesla.model3":
            c = self.control
            new = max(0.0, self.speed + (4.0 * c.throttle - 8.0 * c.brake) * DT)
            self.accel, self.speed = (new - self.speed) / DT, new
            self.yaw_rate = self.speed * math.tan(0.6 * c.steer) / 2.9  # CARLA steer < 0 turns left
        else:
            self.speed, self.accel, self.yaw_rate = 0.0, 0.0, 0.0
        yaw = math.radians(self._t.rotation.yaw) + self.yaw_rate * DT
        self._t.rotation.yaw = math.degrees(yaw)
        self._t.location.x += self.speed * math.cos(yaw) * DT
        self._t.location.y += self.speed * math.sin(yaw) * DT


class Blueprint:
    def __init__(self, blueprint_id):
        self.id, self.values = blueprint_id, {}

    def has_attribute(self, name):
        return True

    def set_attribute(self, name, value):
        self.values[name] = value


class Library:
    def find(self, blueprint_id):
        return Blueprint(blueprint_id)

    def filter(self, pattern):
        return [Blueprint(pattern)] if pattern in EXTENTS else []


class Sensor:
    def __init__(self, actor_id, blueprint, transform, parent):
        self.id, self.type_id, self.blueprint = actor_id, blueprint.id, blueprint
        self.mount, self.parent, self.is_alive, self.callback = transform, parent, True, None
        self.last_fire = None

    @property
    def attributes(self):
        return dict(self.blueprint.values)

    def listen(self, callback):
        self.callback = callback

    def stop(self):
        self.callback = None

    def destroy(self):
        self.is_alive = False
        return True

    def world_pose(self):
        """Sensor position (UE), yaw (deg) and position relative to the parent (UE, world axes)."""
        p = self.parent.get_transform()
        yaw = math.radians(p.rotation.yaw)
        m = self.mount.location
        offset = np.array([math.cos(yaw) * m.x - math.sin(yaw) * m.y, math.sin(yaw) * m.x + math.cos(yaw) * m.y, m.z])
        return np.array([p.location.x, p.location.y, 0.0]) + offset, p.rotation.yaw + self.mount.rotation.yaw, offset

    def to_sensor(self, points_ue):
        origin, yaw, _ = self.world_pose()
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        d = np.asarray(points_ue, float) - origin
        return np.column_stack((c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], d[:, 2]))


def box_surface(actor, n=12):
    corners = np.array([(v.x, v.y, v.z) for v in actor.bounding_box.get_world_vertices(actor.get_transform())])
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    grid = np.linspace(0, 1, n)
    pts = [(x, y, z) for x in np.linspace(lo[0], hi[0], n) for y in (lo[1], hi[1]) for z in np.linspace(lo[2], hi[2], 4)]
    pts += [(x, y, z) for x in (lo[0], hi[0]) for y in lo[1] + grid * (hi[1] - lo[1]) for z in np.linspace(lo[2], hi[2], 4)]
    return np.array(pts)


def first_hit(origin, point, boxes):
    """True if no axis-aligned UE box blocks the segment origin->point before the point."""
    d = point - origin
    r = float(np.linalg.norm(d))
    d = d / r
    for lo, hi in boxes:
        t_near, t_far = -math.inf, math.inf
        for a in range(3):
            if abs(d[a]) < 1e-12:
                if origin[a] < lo[a] or origin[a] > hi[a]:
                    break
                continue
            t1, t2 = (lo[a] - origin[a]) / d[a], (hi[a] - origin[a]) / d[a]
            t_near, t_far = max(t_near, min(t1, t2)), min(t_far, max(t1, t2))
        else:
            if t_near <= t_far and 0 < t_near < r - 0.05:
                return False
    return True


class Map:
    name = "Carla/Maps/FakeStraight"

    def get_spawn_points(self):
        return [Transform(Vector(0.0, 0.0, 0.5), Rotation(yaw=0.0))]

    def get_waypoint(self, location, project_to_road=True):
        return Waypoint(location.x, round(location.y / LANE) * LANE)

    def transform_to_geolocation(self, location):
        lat, lon = GEO(location.x, location.y)
        return SimpleNamespace(latitude=lat, longitude=lon, altitude=location.z)


class Waypoint:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.lane_id = -1 - int(round(y / LANE))
        self.road_id, self.lane_type, self.is_junction = 1, "Driving", False

    @property
    def transform(self):
        return Transform(Vector(self.x, self.y, 0.0), Rotation(yaw=0.0))

    def get_left_lane(self):
        return Waypoint(self.x, self.y - LANE) if self.y > -LANE else None

    def get_right_lane(self):
        return Waypoint(self.x, self.y + LANE) if self.y < LANE else None

    def next(self, d):
        return [Waypoint(self.x + d, self.y)]

    def previous(self, d):
        return [Waypoint(self.x - d, self.y)]


class World:
    def __init__(self):
        self.settings = SimpleNamespace(
            synchronous_mode=False, fixed_delta_seconds=None, substepping=True, max_substep_delta_time=0.01, max_substeps=10, no_rendering_mode=False
        )
        self.actors, self.frame, self.time, self.next_id = [], 100, 0.0, 1000
        self.map = Map()

    def get_settings(self):
        return SimpleNamespace(**vars(self.settings))

    def apply_settings(self, settings):
        self.settings = SimpleNamespace(**vars(settings))

    def get_map(self):
        return self.map

    def get_actors(self):
        return [a for a in self.actors if a.is_alive]

    def set_weather(self, weather):
        pass

    def get_blueprint_library(self):
        return Library()

    def ground_projection(self, location, distance):
        return SimpleNamespace(location=Vector(location.x, location.y, 0.0))

    def try_spawn_actor(self, blueprint, transform):
        self.next_id += 1
        actor = Actor(self.next_id, blueprint.id, transform)
        self.actors.append(actor)
        return actor

    def spawn_actor(self, blueprint, transform, attach_to=None, attachment_type=None):
        self.next_id += 1
        sensor = Sensor(self.next_id, blueprint, transform, attach_to)
        self.actors.append(sensor)
        return sensor

    def get_snapshot(self):
        actors = {a.id: a for a in self.get_actors()}
        return SimpleNamespace(frame=self.frame, timestamp=SimpleNamespace(elapsed_seconds=self.time), find=lambda i: actors.get(i))

    def tick(self, timeout=None):
        self.frame += 1
        self.time += DT
        for actor in self.get_actors():
            if isinstance(actor, Actor):
                actor.step()
        for actor in self.get_actors():
            if isinstance(actor, Sensor) and actor.callback is not None:
                data = self.measure(actor)
                if data is not None:
                    actor.callback(data)
        return self.frame

    def others(self, ego):
        return [a for a in self.get_actors() if isinstance(a, Actor) and a is not ego]

    def measure(self, sensor):
        bp, values, ego = sensor.blueprint.id, sensor.blueprint.values, sensor.parent
        base = dict(frame=self.frame, timestamp=self.time)
        targets = self.others(ego)
        if bp in ("sensor.camera.rgb", "sensor.camera.instance_segmentation"):
            w, h = int(values["image_size_x"]), int(values["image_size_y"])
            image = np.zeros((h, w, 4), np.uint8)
            if bp.endswith("instance_segmentation"):
                self.draw_instances(sensor, image, targets, w, h, float(values["fov"]))
            return SimpleNamespace(**base, width=w, height=h, raw_data=image.tobytes())
        if bp == "sensor.lidar.ray_cast":
            angles = np.radians(np.arange(0, 360, 0.5))
            origin, _, _ = sensor.world_pose()
            ring = np.column_stack((origin[0] + 15 * np.cos(angles), origin[1] + 15 * np.sin(angles), np.zeros_like(angles)))
            local = sensor.to_sensor(np.vstack([ring] + [box_surface(t) for t in targets]))
            raw = np.column_stack((local, np.ones(len(local)))).astype(np.float32)
            return SimpleNamespace(**base, raw_data=raw.tobytes())
        if bp == "sensor.other.radar":
            return RadarData(self.radar(sensor, ego, targets, float(values["horizontal_fov"]), float(values["range"])), **base)
        if bp == "sensor.other.imu":
            yaw = math.radians(ego.get_transform().rotation.yaw)
            return SimpleNamespace(
                **base,
                accelerometer=Vector(ego.accel, ego.speed * ego.yaw_rate, 9.81),
                gyroscope=Vector(0.0, 0.0, ego.yaw_rate),
                compass=(yaw + math.pi / 2) % (2 * math.pi),
            )
        if bp == "sensor.other.gnss":
            period = float(values["sensor_tick"])
            if sensor.last_fire is not None and self.time - sensor.last_fire < period - 1e-9:
                return None
            sensor.last_fire = self.time
            origin, _, _ = sensor.world_pose()
            lat, lon = GEO(origin[0], origin[1])
            return SimpleNamespace(**base, latitude=lat, longitude=lon, altitude=0.0)
        raise AssertionError(bp)

    def radar(self, sensor, ego, targets, hfov, rng_max):
        origin, _, lever = sensor.world_pose()
        v_ego = ego.get_velocity()
        omega = ego.yaw_rate  # about UE +z
        # Sensor velocity in UE: v_ego + omega x r (z-axis rotation, same algebra in either handedness).
        v_sensor = np.array([v_ego.x - omega * lever[1], v_ego.y + omega * lever[0], 0.0])
        static = np.array([[origin[0] + d, origin[1] + o, 0.3] for d in (8.0, 12.0, 16.0) for o in (-1.0, 0.0, 1.0)])
        static = np.vstack((static, [[origin[0] - d, origin[1] + o, 0.3] for d in (8.0, 12.0) for o in (-4.0, 4.0)]))
        boxes = []
        for t in targets:
            c = np.array([(v.x, v.y, v.z) for v in t.bounding_box.get_world_vertices(t.get_transform())])
            boxes.append((c.min(axis=0), c.max(axis=0)))
        sources = [(static, np.zeros(3))] + [
            (box_surface(t, 4), np.array([t.get_velocity().x, t.get_velocity().y, 0.0])) for t in targets
        ]
        detections = []
        for points, v_point in sources:
            for (x, y, z), p in zip(sensor.to_sensor(points), points):
                r = math.sqrt(x * x + y * y + z * z)
                az, el = math.atan2(y, x), math.asin(z / r)
                if abs(math.degrees(az)) > hfov / 2 or r > rng_max or not first_hit(origin, p, boxes):
                    continue
                u = (p - origin) / r
                range_rate = float(u @ (v_point - v_sensor))
                detections.append(SimpleNamespace(depth=r, azimuth=az, altitude=el, velocity=range_rate))
        return detections

    def draw_instances(self, sensor, image, targets, w, h, fov):
        import cv2

        f = w / (2 * math.tan(math.radians(fov) / 2))
        drawn = []
        for t in targets:
            corners = np.array([(v.x, v.y, v.z) for v in t.bounding_box.get_world_vertices(t.get_transform())])
            local = sensor.to_sensor(corners)
            if (local[:, 0] <= 0.1).any():
                continue
            u = w / 2 + f * local[:, 1] / local[:, 0]
            v = h / 2 - f * local[:, 2] / local[:, 0]
            drawn.append((float(local[:, 0].mean()), t, np.column_stack((u, v))))
        for _, t, pixels in sorted(drawn, key=lambda item: -item[0]):  # far first
            hull = cv2.convexHull(np.round(pixels).astype(np.int32))
            mask = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(mask, hull, 1)
            # As measured live in CARLA 0.9.16: runtime vehicles carry instance code 0 (tag 14
            # Car); props carry their own code (tag 20 Static).
            vehicle = t.type_id.startswith("vehicle.")
            ident, tag = (0, 14) if vehicle else (t.id % 65536, 20)
            image[mask == 1] = (ident % 256, ident // 256, tag, 255)  # BGRA: B, G = code; R = tag


class RadarData(list):
    def __init__(self, detections, frame, timestamp):
        super().__init__(detections)
        self.frame, self.timestamp = frame, timestamp


class Client:
    def __init__(self, host, port):
        self.world = WORLD[0]

    def set_timeout(self, seconds):
        pass

    def get_server_version(self):
        return "0.9.16"

    def get_client_version(self):
        return "0.9.16"

    def get_world(self):
        return self.world


WORLD = []


def run_check(monkeypatch, tmp_path, *extra):
    pytest.importorskip("cv2")
    WORLD[:] = [World()]
    monkeypatch.setitem(sys.modules, "carla", fake_carla_module())
    monkeypatch.setattr(
        rig_check, "FixedVehicleAdapter", lambda v: SimpleNamespace(geometry=SimpleNamespace(rear_axle_offset_m=1.38641))
    )
    out = tmp_path / "check"
    rig_check.main(["--suite", str(ROOT / "configs/sensor_suites/mapless_v2.json"), "--output", str(out), "--static-ticks", "12", *extra])
    return out, json.loads((out / "rig_check.json").read_text(encoding="utf-8"))


def test_gate_runs_end_to_end_and_passes_on_a_consistent_world(monkeypatch, tmp_path):
    out, report = run_check(monkeypatch, tmp_path, "--save-images", "--settle-ticks", "4")
    failed = [c for c in report["checks"] if c["status"] not in ("PASS", "REPORT")]
    assert not failed, json.dumps(failed, indent=2)
    assert report["status"] == "MEASURED_PASS"
    names = {c["check"] for c in report["checks"]}
    assert {
        "lidar_extrinsics",
        "camera_mask_consistency",
        "camera_reprojection_props",
        "radar_range_accuracy",
        "radar_sign_moving_target",
        "radar_equal_speed_lead",
        "radar_velocity_sign",
        "radar_range_rate_accuracy",
        "gnss_displacement",
        "gyro_yaw_rate",
        "cleanup",
    } <= names
    by_name = {c["check"]: c for c in report["checks"]}
    # The lever arm matters on the turn: leaving it out is visibly worse.
    rate = by_name["radar_range_rate_accuracy"]
    assert rate["p95_turn_without_lever_arm_mps"] > rate["p95_turn_with_lever_arm_mps"]
    coverage = report["coverage"]["cars"]
    assert {r["placement"] for r in coverage} >= {"ahead_5m", "left_alongside", "right_behind_10m", "behind_15m"}
    assert report["rig"]["effective_attributes"]["radar_front"]["points_per_second"] == "6000"
    assert (out / "coverage_top_down.png").exists() and (out / "samples" / "lidar_top_down.png").exists()
    assert (out / "placements" / "ahead_10m_front.jpg").exists()
    ahead = next(r for r in coverage if r["placement"] == "ahead_10m")["cameras"]["front"]
    assert ahead["instance_code"] == 0 and ahead["instance_tag"] == 14
    assert ahead["rect_contents"]["tags"][0][0] == 14  # the car fills its own box
    lag = by_name["camera_render_lag"]
    assert lag["status"] == "PASS" and lag["camera_minus_lidar_ticks"] == [0, 0, 0, 0]
    assert next(r for r in coverage if r["placement"] == "ahead_10m")["first_visible_tick"]["front"] == 1


def test_overrides_are_recorded(monkeypatch, tmp_path):
    _, report = run_check(
        monkeypatch, tmp_path, "--skip-placements", "--skip-radar-motion", "--set", "radar_front.points_per_second=1500"
    )
    assert report["overrides"] == {"radar_front.points_per_second": 1500}
    assert report["suite_name"].endswith("+overrides")
    assert report["rig"]["attributes"]["radar_front"]["points_per_second"] == 1500


def test_the_gate_catches_a_wrong_radar_sign(monkeypatch, tmp_path):
    import carla_adapter.sensor_rig as rig_module

    monkeypatch.setattr(rig_module, "RADAR_VELOCITY_IS_RANGE_RATE", False)
    _, report = run_check(monkeypatch, tmp_path, "--skip-placements")
    by_name = {c["check"]: c["status"] for c in report["checks"]}
    assert by_name["radar_velocity_sign"] == "FAIL" and by_name["radar_sign_moving_target"] == "FAIL"
    assert report["status"] == "MEASURED_FAIL"


def test_the_gate_catches_a_lidar_axis_error(monkeypatch, tmp_path):
    import carla_adapter.sensor_rig as rig_module

    def unflipped(raw, sensor_to_ego):
        points = np.frombuffer(raw, dtype=np.float32).reshape(-1, 4).astype(float)
        m = np.asarray(sensor_to_ego)
        ego = points[:, :3] @ m[:3, :3].T + m[:3, 3]  # forgot CARLA's y flip
        return np.column_stack((ego, points[:, 3])).astype(np.float32)

    monkeypatch.setattr(rig_module, "lidar_points_to_ego", unflipped)
    _, report = run_check(monkeypatch, tmp_path, "--skip-radar-motion")
    by_name = {c["check"]: c for c in report["checks"]}
    assert by_name["lidar_extrinsics"]["status"] == "FAIL"
    # Without trustworthy LiDAR box points, camera consistency cannot be judged either.
    assert by_name["camera_mask_consistency"]["status"] in ("FAIL", "INCOMPLETE")
    assert report["status"] == "MEASURED_FAIL"
