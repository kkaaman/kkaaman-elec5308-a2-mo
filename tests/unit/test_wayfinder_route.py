"""E3-02: named destinations produce continuous, legal routes for the route manager."""

import json
import math
from pathlib import Path

import pytest

from autonomy.mission import MissionEpisode
from navigation_control import ControllerConfig, VehicleState, Waypoint
from planning.baseline_mission import load_frozen_mission, validate_open_polyline
from planning.lane_graph import LaneSegment, LaneTopology
from planning.route_manager import stitch_route
from planning.wayfinder import (
    PointOfInterest,
    PoiRegistry,
    RoutePlanningError,
    check_destination,
    plan_named_route,
    plan_route,
)

ROOT = Path(__file__).resolve().parents[2]
TOPOLOGY_PATH = ROOT / "tests/fixtures/Town10HD.lane_graph.json"
POI_PATH = ROOT / "configs/wayfinder/pois_town10hd.json"
PAIRS_PATH = ROOT / "configs/wayfinder/fixture_pairs_town10hd.json"


@pytest.fixture(scope="module")
def town():
    topology = LaneTopology.load(TOPOLOGY_PATH)
    registry = PoiRegistry.load(POI_PATH)
    pairs = json.loads(PAIRS_PATH.read_text(encoding="utf-8"))
    return topology, topology.graph, registry, pairs


def segment(road, points, successors=(), junction=None, lane=-1):
    return LaneSegment(road, 0, lane, 0, 10, tuple(points), tuple(f"{x}:0:-1" for x in successors), junction)


def synthetic(exit_angle_deg=90.0):
    """Approach lane 1 -> junction lane 2 -> exit lane 3 leaving at ``exit_angle_deg``."""
    angle = math.radians(exit_angle_deg)
    ex, ey = 4.0 + 4.0 * math.cos(angle), 4.0 * math.sin(angle)
    lanes = [
        segment(1, [(-10, 0, 0), (0, 0, 0)], (2,)),
        segment(2, [(0.1, 0, 0), (2, 0, 0), (4 - 1.1 * math.cos(angle), -1.1 * math.sin(angle), 0),
                    (4 - 0.1 * math.cos(angle), -0.1 * math.sin(angle), 0)], (3,), 7),
        segment(3, [(4, 0, 0), (ex, ey, 0), (ex + 10 * math.cos(angle), ey + 10 * math.sin(angle), 0)]),
    ]
    return LaneTopology("fixture", "0" * 64, 1, 3, tuple(lanes))


def test_registry_validates_against_the_fixture_topology(town):
    topology, _graph, registry, pairs = town
    registry.validate(topology)
    assert pairs["topology_sha256"] == registry.topology_sha256 == topology.sha256
    assert len(registry.names) >= 5
    assert registry.get("baseline_start").node_id == "20:0:-1:entry"
    with pytest.raises(KeyError):
        registry.get("nowhere")


def test_registry_rejects_junction_nodes_bad_snaps_and_wrong_topology(town):
    topology, _graph, registry, _pairs = town
    junction_lane = next(lane for lane in topology.lanes if lane.junction_id is not None)
    bad = PointOfInterest("inside", junction_lane.entry_node_id, junction_lane.points_xyz_m[0])
    with pytest.raises(ValueError, match="junction"):
        bad.validate(topology)
    good = registry.get("baseline_start")
    moved = PointOfInterest(good.name, good.node_id, (good.xyz_m[0] + 1.0, good.xyz_m[1], good.xyz_m[2]))
    with pytest.raises(ValueError, match="from node"):
        moved.validate(topology)
    with pytest.raises(ValueError, match="fingerprint"):
        PoiRegistry(registry.map_name, "f" * 64, registry.pois).validate(topology)
    with pytest.raises(ValueError, match="duplicate"):
        PoiRegistry(registry.map_name, registry.topology_sha256, (good, good))
    with pytest.raises(ValueError):
        PoiRegistry.from_json('{"map_name": "Town10HD", "map_name": "x", "topology_sha256": "0", "pois": []}')
    assert PoiRegistry.from_json(registry.to_json()) == registry


def test_destination_checks_report_reasons_without_raising(town):
    topology, graph, registry, pairs = town
    start = registry.get("baseline_start").node_id
    assert check_destination(topology, start, start, graph).reason == "same_node"
    assert check_destination(topology, "9:9:9:entry", start, graph).reason == "unknown_node"
    assert check_destination(topology, start, "20:0:-1:middle", graph).reason == "unknown_node"
    junction_lane = next(lane for lane in topology.lanes if lane.junction_id is not None)
    assert check_destination(topology, start, junction_lane.exit_node_id, graph).reason == "junction_node"
    for pair in pairs["unreachable_pairs"]:
        origin, destination = registry.get(pair["origin"]).node_id, registry.get(pair["destination"]).node_id
        check = check_destination(topology, origin, destination, graph)
        assert (check.legal, check.reason) == (False, pair["reason"])
        with pytest.raises(RoutePlanningError) as error:
            plan_route(topology, origin, destination, graph=graph)
        assert error.value.reason == "unreachable"
    # The outer circuit is a legal destination: one-way reachability, as documented in BL-A-02.
    assert check_destination(topology, start, registry.get("outer_circuit").node_id, graph).legal


def assert_continuous(topology, route):
    by_id = {lane.segment_id: lane for lane in topology.lanes}
    # Node path edges chain end-to-start with no endpoint reset.
    for a, b in zip(route.node_ids, route.node_ids[1:]):
        assert a != b
    assert route.node_ids[0] == route.origin_node_id and route.node_ids[-1] == route.destination_node_id
    # Every lane transition is an explicit legal successor; no lane is repeated.
    assert len(set(route.segment_ids)) == len(route.segment_ids)
    for a, b in zip(route.segment_ids, route.segment_ids[1:]):
        assert b in by_id[a].successor_ids, (a, b)
    assert all(by_id[key].junction_id is None for key in (route.segment_ids[0], route.segment_ids[-1]))
    # The polyline is one continuous slice of the route manager's stitched geometry.
    stitched = stitch_route(topology, route.segment_ids)
    start, stop = route.point_slice
    assert route.points_xyz_m == stitched[start:stop]
    gaps = [math.dist(a[:2], b[:2]) for a, b in zip(route.points_xyz_m, route.points_xyz_m[1:])]
    assert min(gaps) > 0 and max(gaps) <= topology.sampling_distance_m + topology.max_connection_gap_m + 0.5
    assert route.points_xyz_m[-1] == route.goal_xyz_m
    # Manoeuvres are announcements at increasing route positions, ending with arrival.
    positions = [m.route_s_m for m in route.maneuvers]
    assert positions == sorted(positions) and route.maneuvers[-1].kind == "arrive"
    assert route.maneuvers[-1].route_s_m == pytest.approx(route.length_m)
    for maneuver in route.maneuvers[:-1]:
        assert maneuver.junction_id is not None
        assert maneuver.exit_segment_id in route.segment_ids and maneuver.approach_segment_id in route.segment_ids
        assert 0 <= maneuver.route_s_m < route.length_m


def test_five_fixture_pairs_yield_continuous_legal_routes(town):
    topology, graph, registry, pairs = town
    assert len(pairs["pairs"]) == 5
    for pair in pairs["pairs"]:
        route = plan_named_route(topology, registry, pair["origin"], pair["destination"], graph)
        expected = pair["expected"]
        assert len(route.segment_ids) == expected["lane_count"]
        assert route.length_m == pytest.approx(expected["length_m"], abs=1e-3)
        assert [m.kind for m in route.maneuvers] == expected["maneuvers"]
        assert (route.segment_ids[0], route.segment_ids[-1]) == (expected["first_segment"], expected["last_segment"])
        assert_continuous(topology, route)
        validate_open_polyline(route.points_xyz_m)
        document = route.to_document()
        assert document["segment_ids"] == list(route.segment_ids)
        assert document["route_sha256"] == route.to_document()["route_sha256"]


def test_baseline_pair_reproduces_the_frozen_lane_sequence(town):
    topology, graph, registry, _pairs = town
    route = plan_named_route(topology, registry, "baseline_start", "baseline_goal", graph)
    mission = load_frozen_mission(ROOT / "configs/mission_baseline.json")
    frozen = mission.route
    assert list(route.segment_ids) == frozen["segment_ids"]
    # The frozen route is a shorter slice of the same stitched geometry.
    assert route.point_slice[0] <= frozen["point_slice"][0] and route.point_slice[1] >= frozen["point_slice"][1]
    assert tuple(tuple(p) for p in frozen["points_xyz_m"]) == route.points_xyz_m[frozen["point_slice"][0]:frozen["point_slice"][1]]
    assert route.turn_count == 1 and [m.kind for m in route.maneuvers] == ["straight", "left", "arrive"]


def test_route_installs_once_in_a_mission_episode_across_maneuvers(town):
    topology, graph, registry, _pairs = town
    route = plan_named_route(topology, registry, "north_end", "west_end", graph)
    assert route.turn_count >= 2
    waypoints = tuple(Waypoint(p[0], p[1]) for p in route.points_xyz_m)
    episode = MissionEpisode(ControllerConfig(wheelbase_m=2.8, max_steering_angle_rad=0.5), waypoints, route.to_document()["route_sha256"])
    progress = []
    for index in range(0, len(route.points_xyz_m) - 1, 3):
        a, b = route.points_xyz_m[index], route.points_xyz_m[index + 1]
        episode.step(VehicleState(a[0], a[1], math.atan2(b[1] - a[1], b[0] - a[0]), 3.0), 3.0, 0.05)
        progress.append(episode.progress_m)
    assert episode.route_install_count == 1
    assert progress == sorted(progress) and progress[-1] > max(m.route_s_m for m in route.maneuvers[:-1])
    assert [event.event for event in episode.events] == ["route_installed"]


def test_origin_at_a_lane_exit_and_destination_at_an_entry(town):
    topology, graph, _registry, _pairs = town
    route = plan_route(topology, "20:0:-1:exit", "21:0:-1:entry", graph=graph)
    assert route.segment_ids == ("20:0:-1", "875:0:-1", "21:0:-1")
    by_id = {lane.segment_id: lane for lane in topology.lanes}
    assert route.points_xyz_m[0] == by_id["20:0:-1"].points_xyz_m[-1]
    assert route.points_xyz_m[-1] == by_id["21:0:-1"].points_xyz_m[0]
    assert [m.kind for m in route.maneuvers] == ["straight", "arrive"]
    assert route.maneuvers[0].route_s_m == 0.0
    assert_continuous(topology, route)


@pytest.mark.parametrize("angle, kind", [(0, "straight"), (90, "left"), (-90, "right")])
def test_maneuver_kind_follows_heading_change(angle, kind):
    topology = synthetic(angle)
    route = plan_route(topology, "1:0:-1:entry", "3:0:-1:exit")
    assert [m.kind for m in route.maneuvers] == [kind, "arrive"]
    assert route.maneuvers[0].junction_id == 7
    assert route.maneuvers[0].heading_change_rad == pytest.approx(math.radians(angle), abs=1e-6)
    assert route.maneuvers[0].route_s_m == pytest.approx(10.0)


def test_ambiguous_heading_change_is_refused_explicitly():
    with pytest.raises(RoutePlanningError) as error:
        plan_route(synthetic(35), "1:0:-1:entry", "3:0:-1:exit")
    assert error.value.reason == "ambiguous_maneuver"


def test_junction_nodes_are_not_legal_endpoints():
    topology = synthetic()
    for origin, destination in (("2:0:-1:entry", "3:0:-1:exit"), ("1:0:-1:entry", "2:0:-1:exit")):
        with pytest.raises(RoutePlanningError) as error:
            plan_route(topology, origin, destination)
        assert error.value.reason == "junction_node"
