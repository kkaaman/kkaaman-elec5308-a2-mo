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
EXTENTS = {"vehicle.tesla.model3": (2.396, 1.082, 0.75), "vehicle.audi.a2": (1.86, 0.88, 0.77), "walker.pedestrian.0001": (0.25, 0.25, 0.9)}
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
    def __init__(self, throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False, manual_gear_shift=False):
        self.throttle, self.steer, self.brake, self.reverse = throttle, steer, brake, reverse


LANE_TYPE = SimpleNamespace(Driving="Driving", Sidewalk="Sidewalk", Shoulder="Shoulder", Any="Any")


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
        TrafficLightState=SimpleNamespace(Red="Red", Yellow="Yellow", Green="Green"),
        WalkerControl=lambda direction=None, speed=0.0: SimpleNamespace(direction=direction, speed=speed, brake=0.0),
        Vector3D=Vector,
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

    def get_traffic_light_state(self):
        light = self.world.light  # one light governs the car's lane, when a test sets it
        return "Green" if light is None else light.state

    def destroy(self):
        self.is_alive = False
        box = getattr(self, "box", None)
        if box is not None and box in self.world.road.boxes:
            self.world.road.boxes.remove(box)  # a removed obstacle leaves the LiDAR's world too
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


class FakeLight:
    type_id = "traffic.traffic_light"

    def __init__(self):
        self.state, self.frozen, self.states = "Green", False, []

    def set_state(self, state):
        self.state = state
        self.states.append(state)

    def freeze(self, frozen):
        self.frozen = frozen

    def get_state(self):
        return self.state


class ActorList(list):
    def __init__(self, items, lights=()):
        super().__init__(items)
        self.lights = list(lights)

    def filter(self, pattern):
        return [a for a in self.lights if a.type_id == pattern] + [a for a in self if a.type_id == pattern]


class World:
    def __init__(self):
        self.settings = SimpleNamespace(
            synchronous_mode=False, fixed_delta_seconds=None, substepping=True, max_substep_delta_time=0.01, max_substeps=10, no_rendering_mode=False
        )
        self.actors, self.frame, self.time, self.next_id = [], 100, 0.0, 1000
        self.ego = None
        self.suite = None
        self.road = None
        self.light = None
        self.map = Map(MaplessRoadWorld(SCENE, _suite(), noise=False).plane)

    def get_settings(self):
        return SimpleNamespace(**vars(self.settings))

    def apply_settings(self, settings):
        self.settings = SimpleNamespace(**vars(settings))

    def get_map(self):
        return self.map

    def get_actors(self):
        return ActorList([a for a in self.actors if a.is_alive], () if self.light is None else (self.light,))

    def get_traffic_lights_from_waypoint(self, waypoint, distance):
        return [] if self.light is None else [self.light]

    def set_weather(self, weather):
        pass

    def get_blueprint_library(self):
        return SimpleNamespace(find=lambda blueprint_id: SimpleNamespace(id=blueprint_id, has_attribute=lambda name: False))

    def try_spawn_actor(self, blueprint, transform):
        self.next_id += 1
        actor = Actor(self, self.next_id, blueprint.id, transform)
        self.actors.append(actor)
        if blueprint.id == "vehicle.tesla.model3" and self.ego is None:  # the first Tesla is the ego; later ones are parked cars
            self.ego = actor
            # Rear axle at the lane centre where the actor origin was placed.
            start_s = transform.location.x - OFFSET + 50.0
            self.road = MaplessRoadWorld(SCENE, _suite(), start_s_m=start_s, seed=11)
        else:
            e = EXTENTS[blueprint.id]
            loc = transform.location
            actor.box = (loc.x, -loc.y, -math.radians(transform.rotation.yaw), e[0], e[1], 2 * e[2])
            self.road.boxes.append(actor.box)
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
            pending = self.ego.pending
            if hasattr(pending, "reverse"):  # the parking adapter's ParkCommand
                self.road.apply(ControlCommand(pending.steering_angle_rad, pending.throttle, pending.brake), reverse=pending.reverse)
            else:
                self.road.apply(pending)
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


def run_drive(monkeypatch, tmp_path, *extra, stack=False, light=False):
    pytest.importorskip("cv2")
    WORLD[:] = [World()]
    if light:
        WORLD[0].light = FakeLight()
    monkeypatch.setitem(sys.modules, "carla", fake_carla_module())
    monkeypatch.setattr(drive, "FixedVehicleAdapter", FakeAdapter)
    monkeypatch.setattr(drive, "HoldingVehicleAdapter", FakeAdapter)  # the harness's subclass of the baseline adapter (3c6a457)
    monkeypatch.setattr(drive, "SensorRig", FakeRig)
    out = tmp_path / "drive"
    drive.main(["--suite", str(ROOT / "configs/sensor_suites/mapless_v2.json"), "--output", str(out), *extra], stack=stack)
    return out, json.loads((out / "drive_report.json").read_text(encoding="utf-8"))


def test_straight_drive_reaches_the_goal_and_passes(monkeypatch, tmp_path):
    out, report = run_drive(monkeypatch, tmp_path, "--scenario", "straight", "--goal-m", "60", "--snapshot-every", "100", "--record-frames", "all")
    by_name = {c["check"]: c for c in report["checks"]}
    failed = [c for c in report["checks"] if c["status"] not in ("PASS", "REPORT", drive.OVER_BUDGET)]
    assert not failed, json.dumps(failed, indent=2)
    assert report["status"] in ("MEASURED_PASS", "MEASURED_PASS_OVER_BUDGET") and report["end"] == "finished"  # this machine's steps may miss 50 ms
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


def _fake_mission_site(length_m=90.0, obstacle_s_m=None, junctions=()):
    from carla_adapter.mapless_mission import MissionSite  # noqa: PLC0415

    xs = np.arange(0.0, length_m + 0.5, 1.0)
    pts = np.column_stack((xs, np.full(len(xs), LANE_WIDTH / 2)))
    arcs = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))))
    return MissionSite("fake_mission", pts, arcs, tuple(junctions), length_m - 10.0, LANE_WIDTH, LANE_WIDTH / 2, "sidewalk", [], ("7:0:-1",), obstacle_s_m)


def run_mission(monkeypatch, tmp_path, acceptance, *extra, obstacle_s_m=None, junctions=(), stack=False, light=False):
    monkeypatch.setattr(drive, "resolve_mission", lambda carla, carla_map, path: _fake_mission_site(obstacle_s_m=obstacle_s_m, junctions=junctions))
    return run_drive(monkeypatch, tmp_path, "--scenario", "mission", "--acceptance", acceptance, "--hold-s", "3", "--snapshot-every", "0", *extra, stack=stack, light=light)


def test_mission_s1_holds_behind_the_car_resumes_when_it_leaves_and_finishes(monkeypatch, tmp_path):
    """G2's gate on the harness: the baseline route as a sat-nav route, the S1 car at the
    scenario's s, removed after the 2 s hold; the car resumes and reaches the goal."""
    out, report = run_mission(monkeypatch, tmp_path, "S1", obstacle_s_m=40.0)
    by_name = {c["check"]: c for c in report["checks"]}
    assert report["acceptance"] == dict(scenario="S1", seed=5308, mission="fake_mission", route_noise_m=0.0)
    assert report["obstacle"]["s_m"] == 40.0 and report["obstacle"]["remove_after_hold_s"] == 2.0
    assert by_name["obstacle_stop"]["status"] == "PASS" and 1.5 <= by_name["obstacle_stop"]["gap_m"] <= 5.0
    assert by_name["resume"]["status"] == "PASS" and by_name["resume"]["moved_off_s"] < 6.0
    assert report["end"] == "finished" and report["status"] in ("MEASURED_PASS", "MEASURED_PASS_OVER_BUDGET")
    route = json.loads((out / "route.json").read_text(encoding="utf-8"))
    assert route["provenance"]["source"] == "baseline mission lane chain" and [p["command"] for p in route["points"]][-1] == "LANEFOLLOW"


def test_mission_s4_holds_for_good_and_s0_drives_the_route(monkeypatch, tmp_path):
    _, report = run_mission(monkeypatch, tmp_path, "S4", obstacle_s_m=40.0)
    by_name = {c["check"]: c for c in report["checks"]}
    assert report["end"] == "held at the obstacle" and by_name["held"]["status"] == "PASS" and by_name["obstacle_stop"]["status"] == "PASS"
    assert "resume" not in by_name
    _, report = run_mission(monkeypatch, tmp_path / "s0", "S0")
    by_name = {c["check"]: c for c in report["checks"]}
    assert "obstacle" not in report and by_name["goal"]["status"] == "PASS" and report["end"] == "finished"
    assert "junction_lane_keeping" not in by_name


def test_mission_lane_keeping_is_judged_outside_the_junction_box(monkeypatch, tmp_path):
    """r13: through a junction the car dead-reckons on its latched line and the truth lane
    is the map's connecting lane; those ticks are a REPORT of their own, the lane gate
    judges the rest, and the footprint gate still covers the whole run."""
    _, report = run_mission(monkeypatch, tmp_path, "S0", junctions=((30.0, 50.0, 468),))
    by_name = {c["check"]: c for c in report["checks"]}
    assert by_name["junction_lane_keeping"]["status"] == "REPORT" and by_name["junction_lane_keeping"]["ticks"] > 0
    assert by_name["lane_keeping"]["ticks"] + by_name["junction_lane_keeping"]["ticks"] <= by_name["lane_footprint"]["ticks"]
    assert by_name["junction_crossing"]["junctions"] == [dict(entry_s_m=30.0, exit_s_m=50.0, junction_id=468)]


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
    # The control path without the optional semantic work (fifth review, F5-05).
    log = [dict(driver=dict(stack_timing_ms=dict(drive=20.0, detect=10.0, track=8.0, plan=4.0, paint=30.0 if k % 2 else 0.0, **({"road_model": 6.0} if k % 4 == 1 else {})))) for k in range(100)]
    out = step_time_report(times, log)
    assert out["control_path_ms"]["p50"] == pytest.approx(42.0) and out["control_path_ms"]["over_target_share"] == 0.0
    assert out["optional_ms"]["max"] == pytest.approx(36.0)
    # The reaction budget: a tick, the whole step's p99 and the brake lag within response_s.
    # The command goes out when the step returns, so the whole step governs actuation in
    # both modes (seventh review, F7-02); the control stages are a decomposition.
    whole = 0.05 + float(np.percentile(np.array(times) * 1000, 99)) / 1000 + 0.25
    for mode in ("thread", None):
        out = step_time_report(times, log, response_s=0.5, semantic_mode=mode)
        assert out["actuation_governed_by"].startswith("the whole step") and out["response_budget"]["governing"] == "whole step"
        assert out["response_budget"]["worst_s"] == pytest.approx(whole, abs=1e-3)
        assert out["response_budget"]["control_path_p99_s"] == pytest.approx(0.042)
    assert out["semantic_mode"] == "inline"
    # What the stage timers do not see: the step's time less every stage's.
    staged = np.array([sum(r["driver"]["stack_timing_ms"].values()) for r in log])
    assert out["uninstrumented_ms"]["p50"] == pytest.approx(float(np.median(np.array(times) * 1000 - staged)), abs=0.06)
    slow_times = [0.25] * 100
    slow = [dict(driver=dict(stack_timing_ms=dict(drive=150.0, detect=50.0, track=20.0, plan=10.0))) for _ in range(100)]
    out = step_time_report(slow_times, slow, response_s=0.5, semantic_mode="thread")
    assert out["response_budget"]["within"] is False and out["uninstrumented_ms"]["p50"] == pytest.approx(20.0)


def test_a_run_that_passes_every_gate_without_exercising_its_target_is_not_a_pass():
    """Fifth review, testing: outcomes are PASS, FAIL or NOT_EXERCISED. Every gate passing
    on a cut-in run whose car drove off is NOT_EXERCISED, not MEASURED_PASS; a failed gate
    still fails the run, and an unmeasured one leaves it INCOMPLETE."""
    from carla_adapter.mapless_drive import (  # noqa: PLC0415
        FAIL,
        INCOMPLETE,
        NOT_EXERCISED,
        PASS,
        REPORT,
        run_status,
    )

    def c(name, status):
        return dict(check=name, status=status)

    base = [c("no_collision", PASS), c("behaviour", REPORT)]
    assert run_status(base + [c("exercised", PASS)]) == "MEASURED_PASS"
    assert run_status(base + [c("exercised", NOT_EXERCISED)]) == NOT_EXERCISED
    assert run_status(base + [c("exercised", REPORT)]) == "MEASURED_PASS"
    assert run_status(base + [c("separation", FAIL), c("exercised", NOT_EXERCISED)]) == "MEASURED_FAIL"
    assert run_status(base + [c("separation", INCOMPLETE), c("exercised", NOT_EXERCISED)]) == "INCOMPLETE"
    assert run_status([c("behaviour", REPORT), c("exercised", NOT_EXERCISED)]) == "INCOMPLETE"
    # The deadline missed is its own verdict, not a behaviour failure (review of 519b8fe, F-04).
    assert run_status(base + [c("real_time", drive.OVER_BUDGET)]) == "MEASURED_PASS_OVER_BUDGET"
    assert run_status(base + [c("real_time", drive.OVER_BUDGET), c("exercised", NOT_EXERCISED)]) == NOT_EXERCISED
    assert run_status(base + [c("real_time", drive.OVER_BUDGET), c("separation", FAIL)]) == "MEASURED_FAIL"
    assert run_status(base + [c("real_time", PASS)]) == "MEASURED_PASS"


def test_semantic_shadow_report_summarises_the_agreement_and_is_none_without_it():
    from carla_adapter.mapless_drive import semantic_shadow_report  # noqa: PLC0415

    log = [dict(driver=dict(semantic_shadow=dict(cells=100, with_evidence=80, road_share=0.9 if k % 10 else 0.2, pavement_share=0.1 if k % 10 else 0.8))) for k in range(40)]
    log += [dict(driver=dict(semantic_shadow=None)) for _ in range(5)]
    out = semantic_shadow_report(log)
    assert out["ticks"] == 40 and out["road_share_median"] == pytest.approx(0.9) and out["mostly_pavement_ticks"] == 4
    assert semantic_shadow_report([dict(driver={})]) is None
    # Sixth review, ML: coverage, disagreement and the right edge against the site's kerb.
    log = [dict(driver=dict(semantic_shadow=dict(cells=100, with_evidence=80, coverage=0.8, road_share=0.9, pavement_share=0.1, right_edge_m=1.9 + 0.1 * (k % 3)))) for k in range(30)]
    out = semantic_shadow_report(log, kerb_distance_m=1.75)
    assert out["coverage_median"] == pytest.approx(0.8) and out["pavement_share_mean"] == pytest.approx(0.1) and out["right_edge_ticks"] == 30
    assert out["right_edge_error_m"]["median"] == pytest.approx(0.25) and out["right_edge_error_m"]["mae"] == pytest.approx(0.25)
    assert "right_edge_error_m" not in semantic_shadow_report(log)
    # Seventh review, F7-06: by section of the path ahead and by the driver's mode.
    near = dict(coverage=0.9, right_edge_m=1.8, edge_stations=18, missed=2)
    far = dict(coverage=0.3, right_edge_m=2.6, edge_stations=2, missed=8)
    log = [dict(mode="turning" if k < 10 else "driving", driver=dict(semantic_shadow=dict(cells=100, with_evidence=60, coverage=0.6, road_share=1.0, pavement_share=0.0, right_edge_m=2.0, right_edge_stations=20, sections=dict(near=near, far=far)))) for k in range(30)]
    out = semantic_shadow_report(log, kerb_distance_m=1.75)
    assert out["by_section"]["near"]["missed_share"] == pytest.approx(0.1) and out["by_section"]["far"]["missed_share"] == pytest.approx(0.8)
    assert out["by_section"]["near"]["edge_error_median_m"] == pytest.approx(0.05) and out["by_section"]["far"]["edge_error_median_m"] == pytest.approx(0.85)
    assert out["by_mode"]["turning"]["ticks"] == 10 and out["by_mode"]["driving"]["ticks"] == 20


def test_f7_04_the_run_report_summarises_the_workers_takes():
    from carla_adapter.mapless_drive import semantic_worker_report  # noqa: PLC0415

    stats = dict(submitted=10, dropped=1, segmented=6, built=4, taken=4, stale_epoch=0, stale_age=0, stale_wall=0, late_wall=1, errors=0, dropped_share=0.1, error_share=0.0, stale_share=0.0, late_share=0.25)
    log = [dict(driver=dict(semantic_worker=stats, semantic_take=dict(sim_age_s=0.05, wall_age_ms=40.0 + 10 * k, queue_wait_ms=1.0, run_ms=30.0) if k % 2 else None)) for k in range(8)]
    out = semantic_worker_report(log)
    assert out["takes"] == 4 and out["dropped_share"] == 0.1 and out["late_share"] == 0.25
    assert out["wall_age_ms"]["max"] == 110.0 and out["run_ms"]["p50"] == 30.0
    assert semantic_worker_report([dict(driver=dict(semantic_worker=None))]) is None


def test_f7_05_the_stop_scenario_is_long_enough_for_lane_keepings_sample():
    """Seventh review, F7-05: at 4 m/s the 35 m stop left lane keeping 94 of its 100
    settled ticks (live, PC Temp). The default distance now grows with the speed so the
    sample is collected; the requirement itself stays at 100 ticks."""
    from carla_adapter.mapless_drive import stop_obstacle_m  # noqa: PLC0415

    assert stop_obstacle_m(2.0) == 35.0  # unchanged where it was enough
    assert stop_obstacle_m(4.0) == 45.0
    for v in (2.0, 3.0, 4.0, 5.0):
        settled_m = stop_obstacle_m(v) - 15.0 - 8.0
        assert settled_m / v / 0.05 >= 100


def test_sensor_to_actuator_is_reported_from_the_world_tick_to_the_applied_command():
    """Eighth review, section 5: the step's time is the computation only. Per tick the
    harness marks the world tick's return, the frame's assembly, the step and the command's
    hand-over; the report splits the whole path and its share over the 50 ms target."""
    from carla_adapter.mapless_drive import sensor_to_actuator_report  # noqa: PLC0415

    marks = []
    for k in range(100):
        t0 = 10.0 + k * 0.1
        step = 0.030 if k < 90 else 0.060  # ten slow steps
        marks.append(dict(ticked=t0, assembled=t0 + 0.004, begun=t0 + 0.005, ended=t0 + 0.005 + step, applied=t0 + 0.006 + step))
    out = sensor_to_actuator_report(marks)
    assert out["delivery_ms"]["p50"] == pytest.approx(4.0) and out["record_ms"]["p50"] == pytest.approx(1.0)
    assert out["step_ms"]["p50"] == pytest.approx(30.0) and out["step_ms"]["max"] == pytest.approx(60.0)
    assert out["apply_ms"]["p50"] == pytest.approx(1.0)
    assert out["sensor_to_actuator_ms"]["p50"] == pytest.approx(36.0) and out["sensor_to_actuator_ms"]["max"] == pytest.approx(66.0)
    assert out["over_target_share"] == pytest.approx(0.1) and out["ticks"] == 100
    assert sensor_to_actuator_report([]) is None


def test_frame_recordings_keep_the_radars_and_read_old_ones_without(tmp_path):
    """Schema 2 stores each radar's returns, so a stack replay sees what the live stack saw
    (PC1, r9_left_oncoming: the oncoming car was first tracked 26 m away, and with no radar
    in the recording the replay could not say why not earlier). Schema 1 reads as before."""
    from carla_adapter.frame_record import FrameRecorder, iter_frames  # noqa: PLC0415
    from mapless.frame import (  # noqa: PLC0415
        ImuSample,
        LidarSweep,
        RadarReturns,
        SensorFrame,
        SensorState,
        SensorStatus,
    )
    from mapless.sensor_suite import load_sensor_suite  # noqa: PLC0415

    suite = load_sensor_suite(ROOT / "configs/sensor_suites/mapless_v2.json")
    lidar, radar = suite.lidars[0].name, suite.radars[0].name
    lidar_to_ego, radar_to_ego = np.eye(4), np.eye(4)
    radar_to_ego[0, 3] = 2.3
    returns = np.array([[40.0, 0.05, 0.01, -7.6], [41.0, 0.06, 0.0, -7.5]], dtype=np.float32)
    status = (SensorStatus(lidar, "lidar", SensorState.OK, 1.0), SensorStatus(radar, "radar", SensorState.OK, 1.0))
    frame = SensorFrame(
        episode_id="e", frame_id=7, sim_time_s=1.0, received_monotonic_s=1.0, cameras=(),
        lidars=(LidarSweep(lidar, np.array([[5.0, 0.0, -2.0, 0.0]], dtype=np.float32), lidar_to_ego),),
        radars=(RadarReturns(radar, returns, radar_to_ego),),
        gnss=None, imu=ImuSample((0.0, 0.0, 9.8), (0.0, 0.0, 0.0), 0.0), speed_mps=2.0, status=status, calibration_id="c",
    )
    with FrameRecorder(tmp_path / "frames", suite, "c", "e", {lidar: lidar_to_ego, radar: radar_to_ego}) as rec:
        rec.record(0, frame)
    ((_, back),) = list(iter_frames(tmp_path / "frames"))
    assert len(back.radars) == 1 and back.radars[0].name == radar
    assert np.allclose(back.radars[0].returns, returns) and np.allclose(np.asarray(back.radars[0].sensor_to_ego), radar_to_ego)
    assert {s.name for s in back.status} == {lidar, radar}
    # A schema 1 recording: no radar_to_ego, no radar names; the radar's status is dropped.
    meta = json.loads((tmp_path / "frames/meta.json").read_text(encoding="utf-8"))
    meta.pop("radar_to_ego")
    (tmp_path / "frames/meta.json").write_text(json.dumps(meta), encoding="utf-8")
    ((_, old),) = list(iter_frames(tmp_path / "frames"))
    assert old.radars == () and {s.name for s in old.status} == {lidar} and len(old.lidars) == 1


def test_camera_images_are_recorded_on_the_stacks_segmenting_ticks_and_replayed(tmp_path):
    """G5 (2 Nov): with ``cameras_every`` 2 each OK camera's image is stored as JPEG on the
    ticks the stack segments (its tick count, from 1, even), with its calibration, and the
    replay gives them back as CameraImages with their status; other ticks carry no camera
    and no camera status; a camera missing that tick is not there either. Without it nothing changes."""
    from autonomy.interfaces import CameraCalibration  # noqa: PLC0415
    from carla_adapter.frame_record import FrameRecorder, iter_frames  # noqa: PLC0415
    from mapless.frame import CameraImage, LidarSweep, SensorFrame, SensorState, SensorStatus  # noqa: PLC0415
    from mapless.sensor_suite import load_sensor_suite  # noqa: PLC0415

    suite = load_sensor_suite(ROOT / "configs/sensor_suites/mapless_v2.json")
    lidar = suite.lidars[0].name
    cal = CameraCalibration(640.0, 640.0, 640.0, 360.0, ((0.0, 0.0, 1.0, 1.5), (-1.0, 0.0, 0.0, 0.0), (0.0, -1.0, 0.0, 2.0), (0.0, 0.0, 0.0, 1.0)))
    yy, xx = np.mgrid[0:72, 0:128]
    rgb = np.stack(((xx * 2) % 256, (yy * 3) % 256, np.full_like(xx, 90)), axis=2).astype(np.uint8)

    def frame(tick, side_ok=True):
        status = (
            SensorStatus(lidar, "lidar", SensorState.OK, tick * 0.05),
            SensorStatus("front", "camera", SensorState.OK, tick * 0.05),
            SensorStatus("front_right", "camera", SensorState.OK, tick * 0.05) if side_ok else SensorStatus("front_right", "camera", SensorState.MISSING, None),
        )
        return SensorFrame(
            episode_id="e", frame_id=tick, sim_time_s=tick * 0.05, received_monotonic_s=tick * 0.05,
            cameras=(CameraImage("front", rgb, cal),) + ((CameraImage("front_right", rgb[:, ::-1].copy(), cal),) if side_ok else ()),
            lidars=(LidarSweep(lidar, np.array([[5.0, 0.0, -2.0, 0.0]], dtype=np.float32), np.eye(4)),),
            radars=(), gnss=None, imu=None, speed_mps=0.0, status=status, calibration_id="c",
        )

    with FrameRecorder(tmp_path / "frames", suite, "c", "e", {lidar: np.eye(4)}, cameras_every=2) as rec:
        for tick in range(4):
            rec.record(tick, frame(tick, side_ok=tick != 3))
    back = dict(iter_frames(tmp_path / "frames"))
    assert [len(back[t].cameras) for t in range(4)] == [0, 2, 0, 1]  # ticks 1 and 3: the stack's 2nd and 4th
    assert {s.kind for s in back[0].status} == {"lidar"} and {s.name for s in back[3].status} == {lidar, "front"}
    image = back[1].cameras[0]
    assert image.name == "front" and image.rgb.shape == rgb.shape and np.abs(image.rgb.astype(int) - rgb.astype(int)).mean() < 3.0  # JPEG: close, not exact
    assert np.allclose(np.asarray(image.calibration.camera_to_ego), np.asarray(cal.camera_to_ego)) and image.calibration.fx_px == 640.0
    assert back[1].status_of("front_right").state is SensorState.OK
    # Without images, as before: no camera, no camera status, no cameras.json.
    with FrameRecorder(tmp_path / "plain", suite, "c", "e", {lidar: np.eye(4)}) as rec:
        rec.record(1, frame(1))
    ((_, plain),) = list(iter_frames(tmp_path / "plain"))
    assert plain.cameras == () and {s.kind for s in plain.status} == {"lidar"} and not (tmp_path / "plain/cameras.json").exists()
    with pytest.raises(ValueError):
        FrameRecorder(tmp_path / "bad", suite, "c", "e", {lidar: np.eye(4)}, cameras_every=-1)


def test_an_injected_sensor_outage_is_driven_through_and_recorded(monkeypatch, tmp_path):
    """Ninth review, T4: ``--drop-sensor NAME:START:END`` takes a sensor out of the frames
    for that window (status MISSING, "injected outage"), as recorded, and says so in the
    report. Half a second without the LiDAR on a straight: the driver brakes for the
    stale view and drives on when it returns."""
    out, report = run_drive(monkeypatch, tmp_path, "--scenario", "straight", "--goal-m", "40", "--snapshot-every", "0", "--record-frames", "all", "--drop-sensor", "roof:4.0:4.5")
    assert report["injected_outages"] == [dict(sensor="roof", start_s=4.0, end_s=4.5)]
    rows = [json.loads(r) for r in (out / "drive_log.jsonl").read_text(encoding="utf-8").splitlines()]
    from carla_adapter.frame_record import iter_frames  # noqa: PLC0415

    frames = list(iter_frames(out / "frames"))
    start = frames[0][1].sim_time_s
    dark = [tick for tick, f in frames if 4.0 <= f.sim_time_s - start - 1e-6 < 4.5]
    assert len(dark) == 10 and all(not f.lidars and f.status_of("roof").state.value == "missing" for tick, f in frames if tick in dark)
    by_tick = {r["tick"]: r for r in rows}
    assert any(by_tick[k]["mode"] == "fault" and "LiDAR" in by_tick[k]["reason"] for k in dark)  # the stale view is braked for
    assert report["end"] == "finished"  # ... and driven on from
    with pytest.raises(ValueError):
        drive.parse_outages(["roof:3:2"])


def test_the_oncoming_gate_report_counts_clear_verdicts_on_a_short_view():
    rows = [dict(driver=dict(turn_gate="oncoming lane clear for 40 m", turn_gate_required_s=11.1, turn_gate_view_s=4.7)) for _ in range(5)]
    rows += [dict(driver=dict(turn_gate="waiting: track 241 in the oncoming lane", turn_gate_required_s=10.0, turn_gate_view_s=4.7))]
    rows += [dict(driver=dict(turn_gate="oncoming lane clear for 40 m", turn_gate_required_s=4.0, turn_gate_view_s=4.7))]
    rows += [dict(driver=dict(turn_gate="in the opposing lane: clearing it", turn_gate_required_s=7.9, turn_gate_view_s=4.7)), dict(driver={})]
    rows[5]["driver"]["oncoming"] = "track 241 blocked 0.4 s ago and is no longer observed, not seen to leave: still not clear"
    out = drive.gate_view_report(rows)
    assert out["evaluated_ticks"] == 8 and out["clear_ticks"] == 6 and out["clear_with_view_short_ticks"] == 5
    assert out["lost_blocker_held_ticks"] == 1
    assert out["worst_shortfall_s"] == pytest.approx(6.4) and out["at_entry"]["required_s"] == 7.9
    assert drive.gate_view_report([dict(driver={})]) is None


def test_t10_the_sight_bound_report_counts_ticks_over_the_observability_bound():
    """Tenth review, 5: did the car keep to the speed it could stop from within the path seen
    clear? Over by more than the speed loop's lag (0.3 m/s) is counted, with the excess."""
    log = [dict(truth=dict(speed_mps=v), driver=dict(sight_speed_mps=b)) for v, b in ((3.0, 4.0), (3.0, 4.0), (3.0, 4.0), (3.0, 4.0), (3.0, 4.0), (3.2, 3.0), (3.6, 3.0), (2.0, None), (3.5, 2.8), (3.3, 2.8), (3.1, 2.8), (2.9, 2.8), (2.7, 2.8), (2.5, 2.8))]
    out = drive.sight_bound_report(log)
    assert out["ticks"] == 13 and out["over_ticks"] == 0 and out["over_brief_ticks"] == 4 and out["max_excess_mps"] == pytest.approx(0.7) and out["min_bound_mps"] == 2.8
    assert out["longest_over_s"] == 0.2 and out["over_distance_m"] == pytest.approx(0.7, abs=0.06)
    blind = [dict(truth=dict(speed_mps=4.0), driver=dict(sight_speed_mps=3.0)) for _ in range(30)]
    assert drive.sight_bound_report(blind)["over_ticks"] == 20  # over for 1.5 s and never slowing: from the 11th tick on
    assert drive.sight_bound_report([dict(truth=dict(speed_mps=1.0), driver={})]) is None


def test_t12_slowing_excuses_a_tick_over_the_bound_only_when_the_bound_is_in_reach():
    """Twelfth review, 3 (reproduced there): a car at 5 m/s shedding 0.02 m/s a tick against
    a 1 m/s bound was "slowing" for 75 ticks and nothing counted. Now a slowing tick is
    excused only when the present deceleration brings the car under the bound within a
    second; a real brake (1 m/s² against a 0.7 m/s excess) still is."""
    crawl = [dict(truth=dict(speed_mps=5.0 - 0.02 * k), driver=dict(sight_speed_mps=1.0)) for k in range(80)]
    out = drive.sight_bound_report(crawl)
    assert out["over_ticks"] == 70 and out["over_brief_ticks"] == 10 and out["over_while_slowing_ticks"] == 0
    assert out["longest_over_s"] == 4.0 and out["over_distance_m"] == pytest.approx(16.8, abs=0.2) and out["max_excess_mps"] == 4.0
    # The bound steps from 4 to 1.3 m/s under a car at 2.5 m/s braking at 1 m/s²: 0.5 s of
    # response, then the excess (0.35-0.7 m/s) is within a second's braking: nothing counts.
    brake = [dict(truth=dict(speed_mps=2.5), driver=dict(sight_speed_mps=4.0)) for _ in range(6)]
    brake += [dict(truth=dict(speed_mps=max(0.0, 2.5 - 0.05 * k)), driver=dict(sight_speed_mps=1.3)) for k in range(40)]
    out = drive.sight_bound_report(brake)
    assert out["over_ticks"] == 0 and out["over_brief_ticks"] == 10 and out["over_while_slowing_ticks"] == 9  # 1.6 - 1.3 is a hair over 0.3 in floating point


def test_t10_the_load_generator_spins_and_stops_and_says_what_ran():
    """Tenth review, 4: timing under deliberate contention. ``--cpu-load N`` runs N spinning
    processes for the run and stops them after; the report says what ran (no CUDA here:
    the GPU load says why it did not start)."""
    with drive.LoadGenerator(cpu=1, gpu=True) as load:
        procs = list(load._procs)
        assert len(procs) == 1 and procs[0].is_alive()
        assert load.note["cpu_processes"] == 1 and load.note["gpu"] != "off"
    assert not procs[0].is_alive()
    with drive.LoadGenerator() as idle:
        assert idle.note == dict(cpu_processes=0, gpu="off")


def test_t12_route_noise_leaves_the_start_and_the_destination_exact():
    """Twelfth review, P2: the goal gate's reach had grown with --route-noise because the
    goal point itself was moved. The destination is what a sat-nav has exactly: it stays,
    as the first point (the plane's origin) does, and the waypoints between move."""
    from mapless.satnav import Command, LocalTangentPlane, RoutePoint, SatNavRoute

    plane = LocalTangentPlane(-33.9, 151.2)
    points = tuple(RoutePoint(*plane.to_geodetic(x, 0.0), Command.LANEFOLLOW) for x in (0.0, 25.0, 50.0, 75.0))
    noisy = drive.perturb_route(SatNavRoute("r", points), 2.0, seed=5308)
    assert noisy.points[0] == points[0] and noisy.points[-1] == points[-1] and noisy.provenance["noise_sigma_m"] == 2.0
    moved = [math.dist(plane.to_local(p.latitude_deg, p.longitude_deg), (x, 0.0)) for p, x in zip(noisy.points[1:-1], (25.0, 50.0))]
    assert all(0.05 < m < 10.0 for m in moved)
    assert drive.perturb_route(SatNavRoute("r", points), 0.0, seed=1).points == points


def test_weather_presets_copy_before_overriding_and_are_recorded():
    """G3 (2 Nov): ``--weather`` names a CARLA preset; one with overrides (fog) is a copy,
    so the shared preset is left as it was; the report carries the preset and its fields."""

    class W:
        def __init__(self, **kw):
            for k in drive.WEATHER_FIELDS:
                setattr(self, k, kw.get(k, 0.0))

    W.ClearNoon, W.CloudyNoon = W(cloudiness=5.0, sun_altitude_angle=75.0), W(cloudiness=60.0, sun_altitude_angle=75.0)
    fake = SimpleNamespace(WeatherParameters=W)
    assert drive.weather_parameters(fake, "clear_noon") is W.ClearNoon
    fog = drive.weather_parameters(fake, "light_fog")
    assert fog is not W.CloudyNoon and fog.fog_density == 35.0 and fog.cloudiness == 60.0 and W.CloudyNoon.fog_density == 0.0
    record = drive.weather_record("light_fog", fog)
    assert record["preset"] == "light_fog" and record["fog_density"] == 35.0 and record["cloudiness"] == 60.0
    assert drive.weather_record("clear_noon", "ClearNoon") == {"preset": "clear_noon"}  # the smoke tests' stand-in
    assert set(drive.WEATHER_PRESETS) >= {"clear_noon", "moderate_rain", "light_fog"}


def test_a_scripted_red_the_car_cannot_see_is_run_and_fails_the_red_light_check(monkeypatch, tmp_path):
    """G4 M07 (2 Nov): ``--lights scripted`` freezes the lights and turns the route
    junction's light red as the car's front comes within 35 m of the entry. The driver
    alone has no light perception: it drives on, enters on red, and the run FAILs
    ``red_light`` (the light was tested). The light's truth is logged per tick."""
    out, report = run_mission(monkeypatch, tmp_path, "S0", "--lights", "scripted", junctions=((30.0, 50.0, 468),), light=True)
    by_name = {c["check"]: c for c in report["checks"]}
    light = by_name["red_light"]
    assert light["status"] == "FAIL" and light["red_entries"] == 1 and light["mode"] == "scripted"
    assert light["junctions"][0]["state_at_entry"] == "Red" and not light["junctions"][0]["stopped_on_red"]
    assert light["red_line"] == 1 and light["junctions"][0]["past_line_on_red_m"] > 0.0 and light["junctions"][0]["decided_red_at_front_s_m"] is None
    assert WORLD[0].light.frozen and WORLD[0].light.states[:2] == ["Green", "Red"]
    rows = [json.loads(r) for r in (out / "drive_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r.get("light_truth") for r in rows} >= {"Red"}


def test_the_stack_reads_the_scripted_red_stops_at_the_line_and_goes_on_green(monkeypatch, tmp_path):
    """... and with a light classifier on the stack (here one that reads the fake light,
    standing in for the trained network on the front image), the car stops short of the
    entry, stands until the script turns the light green, and crosses on green:
    ``red_light`` PASSes, exercised, with the approach readings counted against truth."""
    import dataclasses  # noqa: PLC0415

    from autonomy.interfaces import CameraCalibration  # noqa: PLC0415
    from mapless import traffic_light  # noqa: PLC0415
    from mapless.frame import CameraImage, SensorState, SensorStatus  # noqa: PLC0415

    cal = CameraCalibration(64.0, 64.0, 64.0, 36.0, ((0.0, 0.0, 1.0, 1.5), (-1.0, 0.0, 0.0, 0.0), (0.0, -1.0, 0.0, 2.0), (0.0, 0.0, 0.0, 1.0)))
    rgb = np.zeros((72, 128, 3), dtype=np.uint8)
    plain = FakeRig.read
    seen = []

    def read(self, frame, true_speed_mps, **kw):
        f = plain(self, frame, true_speed_mps, **kw)
        status = tuple(x for x in f.status if x.name != "front") + (SensorStatus("front", "camera", SensorState.OK, f.sim_time_s),)
        return dataclasses.replace(f, cameras=(CameraImage("front", rgb, cal),), status=status)

    class ReadsTheLight:
        def __init__(self, checkpoint, device=None, band_scale=None):
            self.provenance = {}

        def classify(self, image, time_s):
            seen.append(time_s)
            return traffic_light.LightReading(WORLD[0].light.state.lower(), 0.97, time_s)

    monkeypatch.setattr(FakeRig, "read", read)
    monkeypatch.setattr(traffic_light, "LightClassifier", ReadsTheLight)
    checkpoint = tmp_path / "light.pt"
    checkpoint.write_bytes(b"stand-in")
    out, report = run_mission(
        monkeypatch, tmp_path, "S0", "--lights", "scripted", "--light-classifier", str(checkpoint), "--semantic", "off",
        junctions=((30.0, 50.0, 468),), stack=True, light=True,
    )
    by_name = {c["check"]: c for c in report["checks"]}
    light = by_name["red_light"]
    assert light["status"] == "PASS" and light["exercised"] and light["classifier"], json.dumps(light, indent=2)
    entry = light["junctions"][0]
    assert entry["state_at_entry"] == "Green" and entry["stood_s"] >= 4.0 and light["red_line"] == 0 and not entry["moved_on_red"]
    assert entry["red_at_front_s_m"] < entry["decided_red_at_front_s_m"] < entry["stood_front_s_m"] < 30.0 and entry["past_line_on_red_m"] == 0.0
    assert entry["readings"].get("red->red", 0) > 0 and not entry["readings"].get("red->green")
    assert report["stack"]["light_classifier"] == str(checkpoint) and report["end"] == "finished"
    assert 3.0 <= np.median(np.diff(seen)) / DT <= 5.0  # every 4th tick
    rows = [json.loads(r) for r in (out / "drive_log.jsonl").read_text(encoding="utf-8").splitlines()]
    stopped = [r for r in rows if r["driver"].get("light_hold_m") is not None and r["truth"]["speed_mps"] < 0.05]
    assert stopped and max(r["truth"]["s_m"] for r in stopped) + 2.4 < 30.0  # the front short of the entry


def test_a_scripted_walker_is_placed_on_the_mission_route_by_route_s(monkeypatch, tmp_path):
    """The showcase (2 Nov): ``--mission-actor crossing_pedestrian`` places the walker by
    route s on the mission's chain (sampled every metre), steps it each tick and scores
    it as the actor scenarios are; the mission's own gates stand. (The fake walker does
    not walk: what is under test is the placement and the plumbing.)"""
    _, report = run_mission(monkeypatch, tmp_path, "S0", "--mission-actor", "crossing_pedestrian", "--actor-param", "brake_at_m=40")
    by_name = {c["check"]: c for c in report["checks"]}
    assert report["actor"]["on"] == "mission" and report["actor"]["name"] == "crossing_pedestrian" and report["actor"]["brake_at_m"] == 40.0
    assert by_name["goal"]["status"] == "PASS" and by_name["separation"]["status"] in ("PASS", "INCOMPLETE")
    assert "behaviour" in by_name and "exercised" in by_name
    with pytest.raises(SystemExit):
        run_drive(monkeypatch, tmp_path / "bad", "--scenario", "straight", "--mission-actor", "crossing_pedestrian")


def test_parallel_parking_finds_the_gap_between_two_parked_cars_and_parks(monkeypatch, tmp_path):
    """The showcase's parking clip on the harness (2 Nov): two cars of the ego's model in
    the same-direction lane to the right of the mission's chain, 9 m apart; the parker
    finds the gap with the LiDAR, reverses into it and centres; the evaluator measures
    the gaps to both cars, the offset from their line and the heading, box to box."""
    import functools
    import sys as _sys

    from sim_harness.mapless_road import CarModel

    this = _sys.modules[__name__]
    monkeypatch.setattr(this, "MaplessRoadWorld", functools.partial(MaplessRoadWorld, car=CarModel(max_steer_rad=math.radians(40))))
    monkeypatch.setattr(drive, "ParkingAdapter", FakeAdapter)

    class RightLane(Waypoint):
        @property
        def transform(self):
            return Transform(Vector(self.s, LANE_WIDTH / 2, 0.0), Rotation(yaw=0.0))  # core y -1.75: one lane to the right

    def right_lane(self):
        lane = RightLane(self.s)
        lane.lane_id = -2  # same direction as the ego's -1
        return lane

    monkeypatch.setattr(Waypoint, "get_right_lane", right_lane)
    monkeypatch.setattr(drive, "resolve_mission", lambda carla, carla_map, path: _fake_mission_site(length_m=160.0))
    out, report = run_drive(monkeypatch, tmp_path, "--scenario", "parallel_park", "--park-at-m", "60", "--park-start-m", "25", "--max-s", "90", "--snapshot-every", "0")
    by_name = {c["check"]: c for c in report["checks"]}
    park = by_name["parallel_park"]
    assert park["status"] == "PASS", json.dumps(park, indent=1)
    assert park["front_gap_m"] >= 0.2 and park["rear_gap_m"] >= 0.2 and abs(park["lateral_from_line_m"]) <= 0.5 and abs(park["heading_deg"]) <= 5.0
    assert report["end"] == "finished" and report["parking"]["gap_m"] == 9.0 and by_name["no_collision"]["status"] == "PASS"
    with pytest.raises(SystemExit):
        drive.main(["--scenario", "parallel_park", "--output", str(tmp_path / "s")], stack=True)  # the parker runs alone


def test_the_parking_adapter_reports_a_reverse_speed_and_sends_the_reverse_gear(monkeypatch):
    """park_t1 (PC 1, 2 Nov): the parking adapter built its state as a VehicleState, whose
    speed must be non-negative, so the first roll-back refused the run. Its state carries
    the signed speed; its commands carry the gear and the steering to the physical lock."""
    from mapless.parking import ParkCommand

    sent = []
    vehicle = SimpleNamespace(
        is_alive=True,
        get_transform=lambda: Transform(Vector(10.0, -2.0, 0.0), Rotation(yaw=0.0)),
        get_velocity=lambda: Vector(-0.6, 0.0, 0.0),
        apply_control=sent.append,
    )
    adapter = object.__new__(drive.ParkingAdapter)
    adapter.vehicle = vehicle
    adapter.geometry = SimpleNamespace(rear_axle_offset_m=OFFSET, physical_steering_limit_rad=math.radians(70))
    adapter.roll_back_ticks = 0
    monkeypatch.setitem(sys.modules, "carla", fake_carla_module())
    state = adapter.state()
    assert state.speed_mps == pytest.approx(-0.6) and state.x_m == pytest.approx(10.0 - OFFSET)
    control = adapter.apply(ParkCommand(math.radians(-35.0), 0.3, 0.0, True))
    assert sent and control.reverse is True and control.steer == pytest.approx(0.5) and control.throttle == 0.3
