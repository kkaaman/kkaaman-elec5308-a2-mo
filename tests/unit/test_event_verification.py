"""BL-B-07 scoring rules and live-fixture geometry, without a CARLA server.

A minimal fake ``carla`` module reproduces CARLA's left-handed transforms
(yaw from +X toward +Y, core Y = -CARLA Y) so the known-distance placement and
truth gap arithmetic are checked for sign and offset errors before a live run.
"""

import math
import sys
from types import SimpleNamespace

import pytest

from evaluation.event_checks import (
    annotate_invasion,
    check_gap,
    count_contacts,
    count_unplanned_invasions,
    load_declarations,
    verification_status,
)
from navigation_control.types import VehicleState


def test_contacts_merge_until_one_second_gap():
    events = [dict(type="collision", other_actor_id=7, sim_time_s=t) for t in (1.0, 1.05, 1.4, 3.0)]
    events.append(dict(type="collision", other_actor_id=8, sim_time_s=1.1))
    assert count_contacts(events) == 3


def test_invasions_need_route_and_marking_specific_declaration(tmp_path):
    path = tmp_path / "crossings.json"
    path.write_text(
        '{"schema_version": 1, "crossings": [{"from_s_m": 60, "to_s_m": 80,'
        ' "markings": ["Broken"], "reason": "planned turn"}]}'
    )
    declarations = load_declarations(path)
    event = dict(type="lane_invasion", markings=["Broken"])
    assert annotate_invasion(event, 70, declarations)["declared_legal"]
    assert not annotate_invasion(event, 90, declarations)["declared_legal"]
    solid = dict(type="lane_invasion", markings=["Solid"])
    assert not annotate_invasion(solid, 70, declarations)["declared_legal"]
    rows = [annotate_invasion(e, 70, declarations) for e in (event, solid)]
    assert count_unplanned_invasions(rows) == 1
    assert load_declarations(None) == ()
    path.write_text(
        '{"schema_version": 1, "crossings": [{"from_s_m": 1, "to_s_m": 0,'
        ' "markings": ["Broken"], "reason": "x"}]}'
    )
    with pytest.raises(ValueError):
        load_declarations(path)


def test_verification_status_requires_every_part():
    good = dict(status="PASS")
    assert verification_status([good], good, good) == "MEASURED_PASS"
    assert verification_status([good], good, dict(status="FAIL")) == "MEASURED_FAIL"
    assert verification_status([], good, good) == "INCOMPLETE"
    assert verification_status([good], None, good) == "INCOMPLETE"
    assert check_gap(5.0, 5.03)["status"] == "PASS"
    assert check_gap(5.0, 5.2)["status"] == "FAIL"
    assert check_gap(5.0, None)["status"] == "FAIL"


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


BOXES = {
    "vehicle.tesla.model3": ((0.0, 0.0, 0.75), (2.396, 1.082, 0.75)),
    "walker.pedestrian.0001": ((0.0, 0.0, 0.0), (0.19, 0.19, 0.93)),
}


class Box:
    def __init__(self, type_id):
        location, extent = BOXES[type_id]
        self.location = Location(*location)
        self.extent = Location(*extent)
        self.rotation = Rotation()

    def get_world_vertices(self, transform):
        return [
            Location(z=transform.location.z + self.location.z + dz * self.extent.z)
            for dz in (-1, 1)
        ]


class Actor:
    def __init__(self, type_id, transform):
        self.type_id, self.id, self.is_alive = type_id, id(self), True
        self.bounding_box = Box(type_id)
        self._transform = transform

    def get_transform(self):
        t = self._transform
        return Transform(
            Location(t.location.x, t.location.y, t.location.z), Rotation(yaw=t.rotation.yaw)
        )

    def set_transform(self, transform):
        self._transform = transform

    def set_simulate_physics(self, enabled):
        pass

    def destroy(self):
        self.is_alive = False
        return True


@pytest.fixture
def fixture(monkeypatch):
    fake = SimpleNamespace(Location=Location, Rotation=Rotation, Transform=Transform)
    monkeypatch.setitem(sys.modules, "carla", fake)
    from carla_adapter import event_verification as ev
    from planning.corridor import RouteGeometry

    # Straight mission heading +X in core coordinates (CARLA yaw 0).
    route = RouteGeometry(tuple((float(x), 0.0) for x in range(0, 80)))
    world = SimpleNamespace(
        ground_projection=lambda location, distance: SimpleNamespace(location=Location(z=0.0))
    )
    library = SimpleNamespace(find=lambda name: SimpleNamespace(id=name))
    instance = ev.Fixture.__new__(ev.Fixture)
    instance.route, instance.world, instance.library, instance.follow = route, world, library, False
    instance.tick = lambda owner, count=1, vehicle=None: None
    owner = SimpleNamespace(try_spawn=lambda bp, transform: Actor(bp.id, transform))
    ego = Actor("vehicle.tesla.model3", Transform(Location(10.0, 0.0, 0.0)))

    class StubAdapter:
        def __init__(self, vehicle):
            self.vehicle = vehicle

        def state(self):
            location = self.vehicle.get_transform().location
            return VehicleState(location.x - 1.38641, -location.y, 0.0, 0.0)

    monkeypatch.setattr(ev, "FixedVehicleAdapter", StubAdapter)
    return instance, owner, ego


@pytest.mark.parametrize("kind,distance", [("car", 3.0), ("car", 10.0), ("person", 3.0)])
def test_known_distance_case_places_near_face_at_requested_gap(fixture, kind, distance):
    instance, owner, ego = fixture
    result = instance.gap_case(owner, ego, kind, distance, 0.0)
    assert result["status"] == "PASS", result
    assert result["expected_m"] == pytest.approx(distance, abs=1e-6)
    assert result["truth_in_corridor"]


def test_roadside_case_must_be_outside_corridor(fixture):
    instance, owner, ego = fixture
    result = instance.gap_case(owner, ego, "car", 8.0, 4.5)
    assert result["status"] == "PASS", result
    assert not result["truth_in_corridor"]


def test_partial_overlap_case_counts_as_in_corridor(fixture):
    instance, owner, ego = fixture
    result = instance.gap_case(owner, ego, "car", 6.0, 0.8)
    assert result["status"] == "PASS", result
    assert result["truth_in_corridor"]


class UntickedActor(Actor):
    """Synchronous CARLA: a pose set this tick is not reported until the next."""

    def get_transform(self):
        return Transform()


@pytest.mark.parametrize(("scenario", "expected"), [("S1", True), ("S3", False)])
def test_obstacle_placement_record_uses_applied_pose_before_a_tick(monkeypatch, scenario, expected):
    fake = SimpleNamespace(
        Location=Location,
        Rotation=Rotation,
        Transform=Transform,
        LaneType=SimpleNamespace(Any="Any"),
    )
    monkeypatch.setitem(sys.modules, "carla", fake)
    from autonomy.baseline import BaselineConfig
    from carla_adapter.baseline_world import CarlaWorld
    from planning.corridor import RouteGeometry
    from sim_harness.baseline import scenario_definition

    # Mission far from the CARLA origin, heading -X like the live Town10HD start.
    points = tuple((27.0 - float(i), -66.3) for i in range(0, 120))
    world = CarlaWorld.__new__(CarlaWorld)
    world.scenario = scenario_definition(scenario)
    world.config = BaselineConfig()
    world.route = RouteGeometry(points)
    world.mission = SimpleNamespace(
        turn_exit_s_m=70.0, points_xyz_m=tuple((x, y, 0.0) for x, y in points)
    )
    world.events = []
    lane = SimpleNamespace(road_id=12, lane_id=-1, lane_type="Driving", is_junction=False)
    world._world = SimpleNamespace(
        ground_projection=lambda location, distance: SimpleNamespace(location=Location(z=0.0)),
        get_blueprint_library=lambda: SimpleNamespace(
            find=lambda name: SimpleNamespace(
                id=name, has_attribute=lambda key: False, set_attribute=lambda *a: None
            )
        ),
        get_map=lambda: SimpleNamespace(get_waypoint=lambda *a, **k: lane),
        get_snapshot=lambda: SimpleNamespace(timestamp=SimpleNamespace(elapsed_seconds=1.0)),
    )
    world.owner = SimpleNamespace(try_spawn=lambda bp, transform: UntickedActor(bp.id, transform))
    world._spawn_obstacle()
    (event,) = world.events
    assert event["in_corridor"] is expected
    centre = [sum(c[i] for c in event["corners_world_m"]) / 4 for i in (0, 1)]
    x, y, _ = world.route.at(40.0, world.scenario["lateral_m"])
    assert centre == pytest.approx([x, y], abs=1e-6)
