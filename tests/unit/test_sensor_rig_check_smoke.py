"""End-to-end smoke run of the live rig check against a small fake CARLA world.

The fake world is a straight two-lane road. It simulates simple car kinematics and emits
sensor data in CARLA's own conventions:
- left-handed axes;
- radar azimuth positive to the right;
- radar velocity positive when approaching (``RADAR_VELOCITY_IS_RANGE_RATE = False``);
- compass clockwise from north, where north is -y.

This cannot prove CARLA's conventions; only the live run can. It proves two things:
- the check script runs from start to finish;
- the check's own geometry is consistent with the rig's conversions, so a correctly
  converting simulator gives MEASURED_PASS.
"""

import json
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import carla_adapter.sensor_rig_check as rig_check
from evaluation.satnav_route import carla_geolocation

GEO = carla_geolocation(49.0, 8.0)
DT = 0.05
NEIGHBOUR_Y = -3.5  # CARLA y; negative = left of travel along +x


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
        Client=None,
        Location=Vector,
        Rotation=Rotation,
        Transform=Transform,
        VehicleControl=VehicleControl,
        WeatherParameters=SimpleNamespace(ClearNoon="ClearNoon"),
        LaneType=SimpleNamespace(Driving="Driving"),
        AttachmentType=SimpleNamespace(Rigid="Rigid"),
    )


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


class Vehicle:
    def __init__(self, actor_id, type_id, transform):
        self.id, self.type_id, self.is_alive = actor_id, type_id, True
        transform = transform.copy()
        transform.location.z = 0.0
        self._t = transform
        self.speed, self.yaw_rate, self.accel = 0.0, 0.0, 0.0  # UE yaw rate, rad/s
        self.control = VehicleControl(brake=1.0)
        self.physics = True
        extent = (2.396, 1.082, 0.75) if "tesla" in type_id else (1.86, 0.88, 0.77)
        self.bounding_box = Box(extent)

    def get_transform(self):
        return self._t.copy()

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

    def destroy(self):
        self.is_alive = False
        return True

    def step(self):
        c = self.control
        if self.physics and "tesla" in self.type_id:
            self.accel = 4.0 * c.throttle - 8.0 * c.brake - (0.0 if c.throttle else 0.0)
            new = max(0.0, self.speed + self.accel * DT)
            self.accel = (new - self.speed) / DT
            self.speed = new
            # CARLA steer < 0 turns left, which is a negative UE yaw rate.
            self.yaw_rate = self.speed * math.tan(0.6 * c.steer) / 2.9
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


class Sensor:
    def __init__(self, actor_id, blueprint, transform, parent):
        self.id, self.type_id, self.blueprint = actor_id, blueprint.id, blueprint
        self.mount, self.parent, self.is_alive, self.callback = transform, parent, True, None
        self.last_fire = None

    def listen(self, callback):
        self.callback = callback

    def stop(self):
        self.callback = None

    def destroy(self):
        self.is_alive = False
        return True

    def world_pose(self):
        """Sensor position (UE) and yaw (deg)."""
        p = self.parent.get_transform()
        yaw = math.radians(p.rotation.yaw)
        m = self.mount.location
        return (
            np.array([p.location.x + math.cos(yaw) * m.x - math.sin(yaw) * m.y, p.location.y + math.sin(yaw) * m.x + math.cos(yaw) * m.y, m.z]),
            p.rotation.yaw + self.mount.rotation.yaw,
        )

    def to_sensor(self, points_ue):
        origin, yaw = self.world_pose()
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        d = np.asarray(points_ue, float) - origin
        return np.column_stack((c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], d[:, 2]))


def box_surface(vehicle, n=12):
    t = vehicle.get_transform()
    corners = np.array([(v.x, v.y, v.z) for v in vehicle.bounding_box.get_world_vertices(t)])
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    grid = np.linspace(0, 1, n)
    pts = [(x, y, z) for x in np.linspace(lo[0], hi[0], n) for y in (lo[1], hi[1]) for z in np.linspace(lo[2], hi[2], 4)]
    pts += [(x, y, z) for x in (lo[0], hi[0]) for y in lo[1] + grid * (hi[1] - lo[1]) for z in np.linspace(lo[2], hi[2], 4)]
    return np.array(pts)


class Map:
    name = "Carla/Maps/FakeStraight"

    def get_spawn_points(self):
        return [Transform(Vector(0.0, 0.0, 0.5), Rotation(yaw=0.0))]

    def get_waypoint(self, location, project_to_road=True):
        return Waypoint(location.x, 0.0 if abs(location.y) < 1.75 else NEIGHBOUR_Y)

    def transform_to_geolocation(self, location):
        lat, lon = GEO(location.x, location.y)
        return SimpleNamespace(latitude=lat, longitude=lon, altitude=location.z)


class Waypoint:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.lane_id = -1 if y == 0.0 else -2
        self.road_id, self.lane_type, self.is_junction = 1, "Driving", False

    @property
    def transform(self):
        return Transform(Vector(self.x, self.y, 0.0), Rotation(yaw=0.0))

    def get_left_lane(self):
        return Waypoint(self.x, NEIGHBOUR_Y) if self.y == 0.0 else None

    def get_right_lane(self):
        return None

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
        return SimpleNamespace(find=Blueprint)

    def try_spawn_actor(self, blueprint, transform):
        self.next_id += 1
        vehicle = Vehicle(self.next_id, blueprint.id, transform)
        self.actors.append(vehicle)
        return vehicle

    def spawn_actor(self, blueprint, transform, attach_to=None, attachment_type=None):
        self.next_id += 1
        sensor = Sensor(self.next_id, blueprint, transform, attach_to)
        self.actors.append(sensor)
        return sensor

    def get_snapshot(self):
        actors = {a.id: a for a in self.get_actors()}
        return SimpleNamespace(
            frame=self.frame, timestamp=SimpleNamespace(elapsed_seconds=self.time), find=lambda i: actors.get(i)
        )

    def tick(self, timeout=None):
        self.frame += 1
        self.time += DT
        for actor in self.get_actors():
            if isinstance(actor, Vehicle):
                actor.step()
        for actor in self.get_actors():
            if isinstance(actor, Sensor) and actor.callback is not None:
                data = self.measure(actor)
                if data is not None:
                    actor.callback(data)
        return self.frame

    def measure(self, sensor):
        bp, values = sensor.blueprint.id, sensor.blueprint.values
        ego = sensor.parent
        base = dict(frame=self.frame, timestamp=self.time)
        targets = [a for a in self.get_actors() if isinstance(a, Vehicle) and a is not ego]
        if bp == "sensor.camera.rgb":
            w, h = int(values["image_size_x"]), int(values["image_size_y"])
            return SimpleNamespace(**base, width=w, height=h, raw_data=np.zeros((h, w, 4), np.uint8).tobytes())
        if bp == "sensor.lidar.ray_cast":
            angles = np.radians(np.arange(0, 360, 0.5))
            origin, _ = sensor.world_pose()
            ring = np.column_stack((origin[0] + 15 * np.cos(angles), origin[1] + 15 * np.sin(angles), np.zeros_like(angles)))
            world_pts = [ring] + [box_surface(t) for t in targets]
            local = sensor.to_sensor(np.vstack(world_pts))
            raw = np.column_stack((local, np.ones(len(local)))).astype(np.float32)
            return SimpleNamespace(**base, raw_data=raw.tobytes())
        if bp == "sensor.other.radar":
            hfov, rng_max = float(values["horizontal_fov"]), float(values["range"])
            origin, _ = sensor.world_pose()
            static = np.array([[origin[0] + d, origin[1] + o, 0.3] for d in (8.0, 12.0, 16.0) for o in (-1.0, 0.0, 1.0)])
            static = np.vstack((static, [[origin[0] - d, origin[1] + o, 0.3] for d in (8.0, 12.0) for o in (-4.0, 4.0)]))
            world_points = np.vstack([static] + [box_surface(t, 4) for t in targets])
            v = ego.get_velocity()
            detections = []
            for (x, y, z), world_point in zip(sensor.to_sensor(world_points), world_points):
                r = math.sqrt(x * x + y * y + z * z)
                az, el = math.atan2(y, x), math.asin(z / r)
                if abs(math.degrees(az)) > hfov / 2 or r > rng_max:
                    continue
                d = world_point - origin
                # Everything else is parked: positive = approaching (assumed CARLA sign).
                approaching = (v.x * d[0] + v.y * d[1] + v.z * d[2]) / r
                detections.append(SimpleNamespace(depth=r, azimuth=az, altitude=el, velocity=approaching))
            return RadarData(detections, **base)
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
            origin, _ = sensor.world_pose()
            lat, lon = GEO(origin[0], origin[1])
            return SimpleNamespace(**base, latitude=lat, longitude=lon, altitude=0.0)
        raise AssertionError(bp)


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


def test_rig_check_runs_end_to_end_and_passes_on_a_consistent_world(monkeypatch, tmp_path):
    pytest.importorskip("cv2")
    fake = fake_carla_module()
    fake.Client = Client
    WORLD[:] = [World()]
    monkeypatch.setitem(sys.modules, "carla", fake)
    monkeypatch.setattr(rig_check, "FixedVehicleAdapter", lambda v: SimpleNamespace(geometry=SimpleNamespace(rear_axle_offset_m=1.38641)))
    out = tmp_path / "check"
    rig_check.main(["--output", str(out), "--static-ticks", "12"])
    report = json.loads((out / "rig_check.json").read_text(encoding="utf-8"))
    failed = [c for c in report["checks"] if c["status"] != "PASS"]
    assert not failed, json.dumps(failed, indent=2)
    assert report["status"] == "MEASURED_PASS"
    names = {c["check"] for c in report["checks"]}
    assert {"lidar_extrinsics_roof_ahead", "camera_lidar_agree_front", "radar_velocity_sign", "gnss_displacement", "gyro_yaw_rate", "cleanup"} <= names
    assert (out / "samples" / "lidar_top_down.png").exists()
