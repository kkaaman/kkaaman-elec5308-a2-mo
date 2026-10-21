"""BL-B-06/07 adapter geometry and live-check arithmetic, without a CARLA server.

A minimal fake ``carla`` module reproduces CARLA's left-handed transforms (yaw from
+X toward +Y; core Y = -CARLA Y), so placement, footprint, truth and the
known-distance check are checked for sign and offset errors before a live run.
"""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from carla_adapter.vehicle_geometry import VehicleGeometry, forward_state
from evaluation.scenarios import load_scenarios, placement_pose
from navigation_control.path import ReferencePath
from planning.baseline_mission import load_frozen_mission

ROOT = Path(__file__).resolve().parents[2]
TESLA = ((0.0, 0.0, 0.75), (2.396, 1.082, 0.75))
WALKER = ((0.0, 0.0, 0.0), (0.19, 0.19, 0.93))
GEOMETRY = VehicleGeometry(3.00465, 1.38641, math.radians(70), math.radians(20))


class Location:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z

    def distance(self, other):
        return math.dist((self.x, self.y, self.z), (other.x, other.y, other.z))


class Rotation:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch, self.yaw, self.roll = pitch, yaw, roll


class Transform:
    def __init__(self, location=None, rotation=None):
        self.location = location or Location()
        self.rotation = rotation or Rotation()

    def transform(self, point):
        yaw = math.radians(self.rotation.yaw)
        x = self.location.x + math.cos(yaw) * point.x - math.sin(yaw) * point.y
        y = self.location.y + math.sin(yaw) * point.x + math.cos(yaw) * point.y
        point.x, point.y, point.z = x, y, self.location.z + point.z


class Box:
    def __init__(self, dims):
        location, extent = dims
        self.location, self.extent, self.rotation = (
            Location(*location),
            Location(*extent),
            Rotation(),
        )

    def get_world_vertices(self, transform):
        return [
            Location(z=transform.location.z + self.location.z + dz * self.extent.z)
            for dz in (-1, 1)
        ]


class Actor:
    _next = 100

    def __init__(self, type_id, transform):
        Actor._next += 1
        self.id, self.type_id, self.is_alive = Actor._next, type_id, True
        self.bounding_box = Box(WALKER if type_id.startswith("walker.") else TESLA)
        self._transform = transform

    def get_transform(self):
        t = self._transform
        return Transform(
            Location(t.location.x, t.location.y, t.location.z), Rotation(yaw=t.rotation.yaw)
        )

    def set_transform(self, transform):
        self._transform = transform

    def set_simulate_physics(self, enabled):
        self.physics = enabled

    def destroy(self):
        self.is_alive = False
        return True


class Blueprint:
    def __init__(self, name):
        self.id, self.attributes = name, {}

    def has_attribute(self, name):
        return name == "role_name"

    def set_attribute(self, name, value):
        self.attributes[name] = value


class Sensor(Actor):
    def listen(self, callback):
        self.callback = callback

    def stop(self):
        pass


class World:
    def __init__(self, road_z=0.0):
        self.road_z = road_z
        self.sensors = []

    def ground_projection(self, location, distance):
        return SimpleNamespace(location=Location(location.x, location.y, self.road_z))

    def get_blueprint_library(self):
        return SimpleNamespace(find=Blueprint)

    def get_map(self):
        return SimpleNamespace(
            get_waypoint=lambda location, project_to_road: SimpleNamespace(
                road_id=20, lane_id=-1, is_junction=False, lane_type="Driving"
            )
        )

    def spawn_actor(self, blueprint, transform, attach_to=None, attachment_type=None):
        sensor = Sensor(blueprint.id, transform)
        self.sensors.append(sensor)
        return sensor


class Owner:
    def __init__(self):
        self.world, self._actors, self.ticks = World(), [], 0

    def try_spawn(self, blueprint, transform):
        actor = Actor(blueprint.id, transform)
        self._actors.append(actor)
        return actor

    def tick(self):
        self.ticks += 1


@pytest.fixture
def carla(monkeypatch):
    fake = SimpleNamespace(
        Location=Location,
        Rotation=Rotation,
        Transform=Transform,
        AttachmentType=SimpleNamespace(Rigid="rigid"),
    )
    monkeypatch.setitem(sys.modules, "carla", fake)
    return fake


@pytest.fixture(scope="module")
def mission():
    return load_frozen_mission(ROOT / "configs/mission_baseline.json")


@pytest.fixture(scope="module")
def scenarios():
    return load_scenarios(ROOT / "configs/scenarios")


def obstacle_for(mission, definition, owner=None):
    from carla_adapter.scenario_world import ScenarioObstacle

    return ScenarioObstacle(
        owner or Owner(),
        definition,
        ReferencePath(mission.waypoints),
        [tuple(p) for p in mission.route["points_xyz_m"]],
    )


def ego_at_route_start(mission):
    spawn = mission.spawn_pose
    transform = Transform(
        Location(spawn["x_m"], -spawn["y_m"], spawn["z_m"]),
        Rotation(yaw=-math.degrees(spawn["yaw_rad"])),
    )
    ego = Actor("vehicle.tesla.model3", transform)

    def state():
        pose = ego.get_transform()
        return forward_state(
            pose.location.x, pose.location.y, pose.rotation.yaw, 0.0, 0.0, GEOMETRY
        )

    return ego, SimpleNamespace(state=state)


def test_placement_is_the_scenario_pose_in_carla_coordinates(carla, mission, scenarios):
    for sid in ("S1", "S3", "S5"):
        obstacle = obstacle_for(mission, scenarios[sid])
        record = obstacle.spawn()
        x, y, yaw = placement_pose(ReferencePath(mission.waypoints), scenarios[sid].obstacle)
        assert record["pose_core"] == pytest.approx({"x_m": x, "y_m": y, "yaw_rad": yaw})
        assert record["carla_transform"]["x"] == pytest.approx(x)
        assert record["carla_transform"]["y"] == pytest.approx(-y)
        assert record["carla_transform"]["yaw_deg"] == pytest.approx(-math.degrees(yaw))
        assert obstacle.actor.physics is False
        # Box bottom 1 cm above the ground; footprint centred on the pose.
        assert record["carla_transform"]["z"] == pytest.approx(0.01)
        cx = sum(p[0] for p in record["footprint_core_m"]) / 4
        cy = sum(p[1] for p in record["footprint_core_m"]) / 4
        assert (cx, cy) == pytest.approx((x, y))
        assert record["scenario_sha256"] == scenarios[sid].sha256
        again = obstacle_for(mission, scenarios[sid]).spawn()
        for key in ("pose_core", "carla_transform", "footprint_core_m"):
            assert again[key] == record[key]


def test_truth_from_the_mission_start(carla, mission, scenarios):
    from carla_adapter.scenario_world import ego_truth

    path = ReferencePath(mission.waypoints)
    _, adapter = ego_at_route_start(mission)
    state = adapter.state()
    front = 1.38641 + TESLA[1][0]
    s1 = obstacle_for(mission, scenarios["S1"])
    s1.spawn()
    truth = ego_truth(path, state, front, TESLA[1][1], s1)
    ego_s = path.project(state.x_m, state.y_m).progress_m
    assert truth["truth_in_corridor"] is True
    assert truth["truth_front_gap_m"] == pytest.approx(40.0 - TESLA[1][0] - ego_s - front, abs=2e-3)
    s3 = obstacle_for(mission, scenarios["S3"])
    s3.spawn()
    assert ego_truth(path, state, front, TESLA[1][1], s3)["truth_in_corridor"] is False
    s1.remove(frame=7, sim_time_s=3.5, reason="after_steady_hold")
    gone = ego_truth(path, state, front, TESLA[1][1], s1)
    assert gone["truth_front_gap_m"] is None and gone["truth_obstacle_present"] is False
    assert s1.removal == {
        "frame": 7,
        "simulation_time_s": 3.5,
        "reason": "after_steady_hold",
        "destroyed": True,
    }
    with pytest.raises(RuntimeError):
        s1.remove(frame=8, sim_time_s=3.6, reason="again")


def test_event_sensors_keep_raw_callbacks_and_build_records(carla):
    from carla_adapter.scenario_world import ScenarioEventSensors, event_records

    owner = Owner()
    sensors = ScenarioEventSensors(owner, Actor("vehicle.tesla.model3", Transform()))
    collision, lane = owner.world.sensors
    assert owner._actors == [collision, lane]
    hit = SimpleNamespace(
        frame=10,
        timestamp=0.5,
        other_actor=SimpleNamespace(id=55, type_id="vehicle.tesla.model3"),
        normal_impulse=SimpleNamespace(x=1.0, y=0.0, z=0.0),
    )
    collision.callback(hit)
    collision.callback(SimpleNamespace(**(hit.__dict__ | {"frame": 12, "timestamp": 0.6})))
    lane.callback(
        SimpleNamespace(
            frame=11, timestamp=0.55, crossed_lane_markings=[SimpleNamespace(type="Broken")]
        )
    )
    collision.callback(SimpleNamespace(frame=13))  # malformed: recorded as an error, never raised
    raw = sensors.snapshot()
    assert [r["channel"] for r in raw] == ["collision", "collision", "lane_invasion"]
    assert len(sensors.callback_errors) == 1
    records = event_records(raw, {9: 20.0, 11: 21.0}, ())
    assert [r["kind"] for r in records] == ["collision", "lane_invasion", "collision"]
    assert records[0]["contact_id"] == records[2]["contact_id"] == "55:10"
    assert records[1]["route_s_m"] == 21.0 and records[1]["predeclared_legal"] is False
    sensors.close()
    assert not collision.is_alive and not lane.is_alive


def test_known_distance_check_arithmetic(carla, mission, scenarios):
    from carla_adapter.scenario_verification import Verification

    owner = Owner()
    run = Verification(owner, mission, scenarios)
    run.now = lambda: 0.0
    ego, adapter = ego_at_route_start(mission)
    box = ego.bounding_box
    front = 1.38641 + box.location.x + box.extent.x
    results = [
        run.gap_case(sid, distance, lateral, ego, adapter, front, box)
        for sid, distance, lateral in (
            ("S1", 6.0, 0.0),
            ("S1", 6.0, 0.8),
            ("S1", 8.0, 3.5),
            ("S2", 3.0, 0.0),
        )
    ]
    assert [r["status"] for r in results] == ["PASS"] * 4
    for result in (results[0], results[1], results[3]):
        assert abs(result["error_m"]) < 0.01
        assert result["face_to_face_s_m"] == pytest.approx(
            float(result["check"].split()[3]), abs=0.01
        )
    assert results[2]["expected_in_corridor"] is False


def test_placement_check_passes_for_a_stationary_obstacle(carla, mission, scenarios):
    from carla_adapter.scenario_verification import Verification

    run = Verification(Owner(), mission, scenarios)
    ego, adapter = ego_at_route_start(mission)
    box = ego.bounding_box
    front = 1.38641 + box.location.x + box.extent.x
    for sid in ("S1", "S3"):
        result = run.placement_case(sid, ego, adapter, front, box)
        assert result["status"] == "PASS", result
        assert len(result["placements"]) == 2
