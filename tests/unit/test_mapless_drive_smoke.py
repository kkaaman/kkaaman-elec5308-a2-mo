"""End-to-end smoke run of the live G1b harness against a fake CARLA built on the synthetic road.

The fake map is one straight lane with a pavement on its right. The ego's motion, its
sensors and collisions come from ``sim_harness.mapless_road.MaplessRoadWorld``, so the
real ``MaplessDriver`` drives in closed loop. What is under test is the harness: site
search, route making, the parked car, the tick loop, end conditions, scoring and the
output files. The CARLA sensor plumbing is the rig's, already checked live (G0).
"""

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import carla_adapter.mapless_drive as drive
from navigation_control.types import ControlCommand
from sim_harness.mapless_road import KerbLine, MaplessRoadWorld, RoadScene, Straight

ROOT = Path(__file__).resolve().parents[2]
DT = 0.05
LANE_WIDTH = 3.5
ROAD_M = 400.0
OFFSET = 1.38641
EXTENTS = {"vehicle.tesla.model3": (2.396, 1.082, 0.75), "vehicle.audi.a2": (1.86, 0.88, 0.77)}
SCENE = RoadScene(KerbLine([Straight(ROAD_M)], start=(-50.0, 0.0, 0.0)), lane_centre_m=LANE_WIDTH / 2)


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


class VehicleControl:
    def __init__(self, throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False):
        self.throttle, self.steer, self.brake = throttle, steer, brake


LANE_TYPE = SimpleNamespace(Driving="Driving", Sidewalk="Sidewalk", Shoulder="Shoulder")


def fake_carla_module():
    return SimpleNamespace(
        Client=Client,
        Location=Vector,
        Rotation=Rotation,
        Transform=Transform,
        VehicleControl=VehicleControl,
        WeatherParameters=SimpleNamespace(ClearNoon="ClearNoon"),
        LaneType=LANE_TYPE,
        AttachmentType=SimpleNamespace(Rigid="Rigid"),
    )


class Waypoint:
    """Ego lane centre at core y = +1.75 (CARLA y = -1.75), heading east; pavement to its right."""

    end_s = ROAD_M - 60

    def __init__(self, s, lane_type="Driving"):
        self.s, self.lane_type = s, lane_type
        self.lane_width = LANE_WIDTH if lane_type == "Driving" else 3.0
        self.is_junction = False
        self.road_id, self.section_id, self.lane_id = 7, 0, -1 if lane_type == "Driving" else -2

    @property
    def transform(self):
        y_core = LANE_WIDTH / 2 if self.lane_type == "Driving" else -1.5
        return Transform(Vector(self.s, -y_core, 0.0), Rotation(yaw=0.0))

    def get_right_lane(self):
        return Waypoint(self.s, "Sidewalk") if self.lane_type == "Driving" else None

    def get_left_lane(self):
        return None

    def next(self, d):
        return [type(self)(self.s + d)] if self.s + d < self.end_s else []


class Map:
    name = "Carla/Maps/FakeStraight"

    def __init__(self, road_world_plane):
        self.plane = road_world_plane

    def get_spawn_points(self):
        return [Transform(Vector(0.0, -LANE_WIDTH / 2, 0.3), Rotation(yaw=0.0))]

    def get_topology(self):
        # One junction-free lane run, plus the junction it ends in (skipped).
        junction = Waypoint(Waypoint.end_s)
        junction.is_junction = True
        return [(Waypoint(-40.0), Waypoint(Waypoint.end_s)), (junction, Waypoint(Waypoint.end_s + 10))]

    def get_waypoint(self, location, project_to_road=True, lane_type=None):
        return Waypoint(location.x)

    def transform_to_geolocation(self, location):
        lat, lon = self.plane.to_geodetic(location.x, -location.y)
        return SimpleNamespace(latitude=lat, longitude=lon, altitude=0.0)


class Actor:
    def __init__(self, world, actor_id, type_id, transform):
        self.world, self.id, self.type_id = world, actor_id, type_id
        self.is_alive = True
        self._transform = transform
        e = EXTENTS[type_id]
        self.bounding_box = SimpleNamespace(location=Vector(0.0, 0.0, e[2]), extent=Vector(*e))
        self.pending = ControlCommand(0.0, 0.0, 1.0)

    def get_transform(self):
        if self is self.world.ego:
            st = self.world.road.state
            x = st.x + OFFSET * math.cos(st.yaw)
            y = st.y + OFFSET * math.sin(st.yaw)
            return Transform(Vector(x, -y, 0.0), Rotation(yaw=-math.degrees(st.yaw)))
        return self._transform

    def get_velocity(self):
        if self is self.world.ego:
            st = self.world.road.state
            return Vector(st.speed * math.cos(st.yaw), -st.speed * math.sin(st.yaw), 0.0)
        return Vector()

    def apply_control(self, control):
        self.pending = ControlCommand(0.0, 0.0, float(control.brake)) if control.brake else self.pending

    def set_simulate_physics(self, enabled):
        pass

    def destroy(self):
        self.is_alive = False
        return True


class CollisionSensor:
    def __init__(self, actor_id, parent):
        self.id, self.parent, self.is_alive, self.callback = actor_id, parent, True, None
        self.type_id = "sensor.other.collision"

    def listen(self, callback):
        self.callback = callback

    def stop(self):
        self.callback = None

    def destroy(self):
        self.is_alive = False
        return True


class World:
    def __init__(self):
        self.settings = SimpleNamespace(
            synchronous_mode=False, fixed_delta_seconds=None, substepping=True, max_substep_delta_time=0.01, max_substeps=10, no_rendering_mode=False
        )
        self.actors, self.frame, self.time, self.next_id = [], 100, 0.0, 1000
        self.ego = None
        self.suite = None
        self.road = None
        self.map = Map(MaplessRoadWorld(SCENE, _suite(), noise=False).plane)

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
        return SimpleNamespace(find=lambda blueprint_id: SimpleNamespace(id=blueprint_id))

    def try_spawn_actor(self, blueprint, transform):
        self.next_id += 1
        actor = Actor(self, self.next_id, blueprint.id, transform)
        self.actors.append(actor)
        if blueprint.id == "vehicle.tesla.model3":
            self.ego = actor
            # Rear axle at the lane centre where the actor origin was placed.
            start_s = transform.location.x - OFFSET + 50.0
            self.road = MaplessRoadWorld(SCENE, _suite(), start_s_m=start_s, seed=11)
        else:
            e = EXTENTS[blueprint.id]
            loc = transform.location
            self.road.boxes.append((loc.x, -loc.y, -math.radians(transform.rotation.yaw), e[0], e[1], 2 * e[2]))
        return actor

    def spawn_actor(self, blueprint, transform, attach_to=None, attachment_type=None):
        self.next_id += 1
        sensor = CollisionSensor(self.next_id, attach_to)
        self.actors.append(sensor)
        return sensor

    def get_snapshot(self):
        actors = {a.id: a for a in self.get_actors()}
        return SimpleNamespace(frame=self.frame, timestamp=SimpleNamespace(elapsed_seconds=self.time), find=lambda i: actors.get(i))

    def tick(self, timeout=None):
        self.frame += 1
        self.time += DT
        if self.road is not None:
            self.road.apply(self.ego.pending)
            if self.road.truth().collision:
                for sensor in self.get_actors():
                    if isinstance(sensor, CollisionSensor) and sensor.callback:
                        sensor.callback(SimpleNamespace(frame=self.frame, other_actor=SimpleNamespace(type_id="vehicle.audi.a2")))
        return self.frame


def _suite():
    from mapless.sensor_suite import load_sensor_suite

    return load_sensor_suite(ROOT / "configs/sensor_suites/mapless_v2.json")


class FakeRig:
    """Frames from the synthetic road, which owns the ego's motion."""

    def __init__(self, owner, vehicle, suite, *, episode_id, rear_axle_offset_m, dt_s, seed, capacity=8):
        self.world = owner.world
        self.calibration_id, self.episode_id = "fake-calibration", episode_id
        self.extrinsics = {s.name: self.world.road.lidar_to_ego for s in suite.lidars}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, frame, true_speed_mps, *, sim_time_s=None, grace_s=0.5, strict=False):
        return self.world.road.frame()

    def record(self):
        return dict(fake=True)


class FakeAdapter:
    def __init__(self, vehicle):
        # Live CARLA reports wheel positions that do not match the car until physics has
        # run (first G1b attempt, 22 Oct). Insist on settled ticks, as the real check would fail.
        if vehicle.world.frame - 100 < 20:
            raise ValueError("Wheel pairs do not define aligned axles")
        self.vehicle = vehicle
        self.geometry = SimpleNamespace(wheelbase_m=3.00465, rear_axle_offset_m=OFFSET, controller_steering_limit_rad=math.radians(20))
        self.roll_back_ticks = 0

    def apply(self, command):
        self.vehicle.pending = command

    def stop(self):
        self.vehicle.pending = ControlCommand(0.0, 0.0, 1.0)


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


def run_drive(monkeypatch, tmp_path, *extra):
    pytest.importorskip("cv2")
    WORLD[:] = [World()]
    monkeypatch.setitem(sys.modules, "carla", fake_carla_module())
    monkeypatch.setattr(drive, "FixedVehicleAdapter", FakeAdapter)
    monkeypatch.setattr(drive, "HoldingVehicleAdapter", FakeAdapter)  # the harness's subclass of the baseline adapter (3c6a457)
    monkeypatch.setattr(drive, "SensorRig", FakeRig)
    out = tmp_path / "drive"
    drive.main(["--suite", str(ROOT / "configs/sensor_suites/mapless_v2.json"), "--output", str(out), *extra])
    return out, json.loads((out / "drive_report.json").read_text(encoding="utf-8"))


def test_straight_drive_reaches_the_goal_and_passes(monkeypatch, tmp_path):
    out, report = run_drive(monkeypatch, tmp_path, "--scenario", "straight", "--goal-m", "60", "--snapshot-every", "100", "--record-frames", "all")
    by_name = {c["check"]: c for c in report["checks"]}
    failed = [c for c in report["checks"] if c["status"] not in ("PASS", "REPORT")]
    assert not failed, json.dumps(failed, indent=2)
    assert report["status"] == "MEASURED_PASS" and report["end"] == "finished"
    assert by_name["goal"]["goal_distance_m"] <= 3.0
    calibration = by_name["calibration"]
    assert calibration["map_lane_centre_to_kerb_m"] == pytest.approx(1.75)
    assert calibration["source"].startswith("measured at a standstill")
    assert abs(calibration["keep_right_used_m"] - 1.75) <= 0.2 and calibration["calibrating_ticks"] >= 0
    assert abs(by_name["kerb_bias"]["detected_kerb_bias_m"]) < 0.25
    assert by_name["lane_footprint"]["status"] == "PASS"
    assert by_name["startup"]["route_pose_error_m"] < 3.0
    assert list((out / "sweeps").glob("*.npz"))
    route = json.loads((out / "route.json").read_text(encoding="utf-8"))
    assert {p["command"] for p in route["points"]} == {"LANEFOLLOW"}
    assert (out / "drive_plot.png").exists() and list((out / "snapshots").glob("*.png"))
    rows = (out / "drive_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(rows[0])["mode"] == "startup"
    # The recording replays as the frames the driver saw (LiDAR to the centimetre).
    from carla_adapter.frame_record import iter_frames  # noqa: PLC0415

    replayed = list(iter_frames(out / "frames"))
    assert report["frames_recorded"] == len(replayed) == len(rows)
    tick, frame = replayed[-1]
    assert tick == json.loads(rows[-1])["tick"] and frame.lidars and frame.imu is not None
    assert any(f.gnss is not None for _, f in replayed)  # GNSS at its own rate
    # The offline replay reproduces the live driver tick for tick (same frames, same code).
    from carla_adapter.mapless_replay import replay  # noqa: PLC0415

    replayed_rows = replay(out, {}, out=tmp_path / "replay.jsonl", suite=None, console_every=0, until_tick=200)
    live_rows = [json.loads(r) for r in rows[: len(replayed_rows)]]
    assert [r["mode"] for r in replayed_rows] == [r["mode"] for r in live_rows]
    assert np.allclose([r["driver"]["odometry"] for r in replayed_rows[50:]], [r["driver"]["odometry"] for r in live_rows[50:]], atol=1e-3)
    assert frame.lidars[0].points.shape[1] == 4 and abs(frame.sim_time_s - json.loads(rows[-1])["driver"]["time_s"]) < 1e-3


def test_stop_scenario_holds_short_of_the_parked_car(monkeypatch, tmp_path):
    _, report = run_drive(monkeypatch, tmp_path, "--scenario", "stop", "--goal-m", "90", "--obstacle-m", "40", "--hold-s", "3", "--snapshot-every", "0")
    by_name = {c["check"]: c for c in report["checks"]}
    assert report["end"] == "held at the obstacle"
    assert by_name["obstacle_stop"]["status"] == "PASS"
    assert 1.5 <= by_name["obstacle_stop"]["gap_m"] <= 5.0
    assert by_name["no_collision"]["status"] == "PASS"


def test_curve_scenario_refuses_a_map_without_curves(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="Only 0 curve sites"):
        run_drive(monkeypatch, tmp_path, "--scenario", "curve")


def test_site_search_explains_why_nothing_fits():
    class Short(Waypoint):
        end_s = 50.0

    carla = fake_carla_module()
    fake_map = SimpleNamespace(get_topology=lambda: [(Short(0.0), Short(50.0))])
    sites, census = drive.find_sites(carla, fake_map, "straight", 75.0)
    assert sites == [] and census["too_short"] == 1
    assert census["longest_rightmost_runs_m"][0]["length_m"] == 41  # 50 m run, 8 m clearance, 1 m steps
    sites, census = drive.find_sites(carla, fake_map, "straight", 30.0)
    assert len(sites) == 1 and sites[0]["kerb_source"] == "sidewalk" and sites[0]["kerb_distance_m"] == LANE_WIDTH / 2


# Third external review (e93154f), section 6: p95, the worst case and each stage are first-class.
def test_step_time_report_gives_the_tail_the_deadline_share_and_each_stage():
    times = [0.03] * 90 + [0.08] * 9 + [0.2]
    log = [dict(driver=dict(stack_timing_ms=dict(drive=20.0, track=float(k % 10)), driver_timing_ms=dict(classify=5.0))) for k in range(100)]
    from carla_adapter.mapless_drive import step_time_report

    out = step_time_report(times, log)
    assert out["p50_ms"] == pytest.approx(30.0) and out["max_ms"] == pytest.approx(200.0)
    assert out["over_target_share"] == pytest.approx(0.10) and out["target_ms"] == 50.0
    assert out["stages_ms"]["track"]["p95"] == pytest.approx(9.0) and out["stages_ms"]["driver.classify"]["p50"] == pytest.approx(5.0)
