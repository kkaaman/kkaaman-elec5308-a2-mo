"""A2 on the roads A1 drove in TEST-01 (configs/missions/a1_test01.json), no CARLA.

The explicit route is held to the same legality rules as the automatic mission,
and the baseline mission (and so its frozen evidence) must not change.
"""

import csv
import json
import math
from dataclasses import replace

import pytest

from autonomy.baseline import BaselineConfig
from autonomy.paths import FIXTURES_DIR, REPO_ROOT
from evaluation.runner import run_episode
from planning.corridor import RouteGeometry
from planning.lane_graph import LaneTopology
from planning.mission import (
    load_route_definition,
    mission_along_lanes,
    mission_from_definition,
    scenario_obstacle_s,
)
from sim_harness.baseline import BaselineWorld, RasterDetector, mission_route

ROUTE = REPO_ROOT / "configs/missions/a1_test01.json"
A1_SAMPLES = REPO_ROOT / "docs/evidence/TEST-01-samples.csv"
BASELINE_MISSION_SHA256 = "bfa104861dc58a7039bae4a71d9161e31728fa6515f83792fcf5e6b60c4741f3"


@pytest.fixture(scope="module")
def topology():
    return LaneTopology.load(FIXTURES_DIR / "Town10HD.lane_graph.json")


@pytest.fixture(scope="module")
def mission(topology):
    return mission_from_definition(topology, load_route_definition(ROUTE))


def manifest(scenario):
    return dict(
        engine="harness",
        scenario=scenario,
        seed=1,
        step_wall_budget_s=10.0,
        event_sensors_verified=True,
        truth_channel_complete=True,
    )


def test_baseline_mission_is_unchanged():
    assert mission_route().sha256 == BASELINE_MISSION_SHA256


def test_route_starts_and_ends_where_a1_did(mission):
    rows = list(csv.DictReader(A1_SAMPLES.open()))
    first = (float(rows[0]["actor_x_m"]), float(rows[0]["actor_y_m"]))
    last = (float(rows[-1]["actor_x_m"]), float(rows[-1]["actor_y_m"]))
    route = RouteGeometry(mission.points_xy_m)
    # A1 spawned on the lane centre; its run ended across the lane boundary, so
    # the goal is A1's end point projected onto A2's lane.
    assert math.dist(mission.points_xy_m[0], first) < 0.05
    assert route.project(*last).s_m == pytest.approx(route.length_m, abs=0.05)
    assert route.length_m == pytest.approx(228.9, abs=0.5)
    a1_length = sum(
        math.dist(
            (float(a["actor_x_m"]), float(a["actor_y_m"])),
            (float(b["actor_x_m"]), float(b["actor_y_m"])),
        )
        for a, b in zip(rows, rows[1:])
    )
    assert abs(route.length_m - a1_length) < 5


def test_route_is_the_legal_lane_chain(mission):
    assert mission.segment_ids == ("1:0:1", "8:0:1", "4:0:-1", "515:0:-1", "5:0:-1")
    assert mission.manoeuvres == ("straight",)
    route = RouteGeometry(mission.points_xy_m)
    # Junction 468 exit, with room after it for the S5 obstacle and a finish.
    assert 190 < mission.turn_exit_s_m < route.length_m - 25
    assert mission.map_validation["status"] == "NOT_VALIDATED_LIVE"
    assert mission.map_validation["route_definition"] == dict(name="a1_test01", obstacle_s_m=120.0)


def test_route_obstacles_sit_on_the_straight(mission):
    route = RouteGeometry(mission.points_xy_m)
    s = scenario_obstacle_s(mission, dict(obstacle_s_m=40.0, after_turn_m=None))
    assert s == 120.0
    headings = [route.at(s + d)[2] for d in (-30, -15, 0, 15)]
    assert max(headings) - min(headings) < math.radians(2)
    # S5 keeps its turn-relative placement; a mission without a route keeps the YAML value.
    assert scenario_obstacle_s(mission, dict(obstacle_s_m=40.0, after_turn_m=6.0)) == (
        mission.turn_exit_s_m + 6.0
    )
    assert scenario_obstacle_s(mission_route(), dict(obstacle_s_m=40.0, after_turn_m=None)) == 40


@pytest.mark.parametrize(
    "change, message",
    [
        (dict(manoeuvres=["right"]), "not the legal right exit|Expected exactly one"),
        (dict(manoeuvres=["straight", "left"]), "match the junctions"),
        (dict(segment_ids=["1:0:1", "4:0:-1", "515:0:-1", "5:0:-1"]), "Illegal route transition"),
        (dict(segment_ids=["515:0:-1", "5:0:-1"]), "outside a junction"),
        (dict(end_xy_m=[102.2, -95.0]), "at least 50 m"),
    ],
)
def test_illegal_or_degenerate_routes_are_refused(topology, change, message):
    definition = dict(load_route_definition(ROUTE), **change)
    with pytest.raises(ValueError, match=message):
        mission_from_definition(topology, definition)


def test_definition_needs_name_and_lists(tmp_path):
    path = tmp_path / "route.json"
    path.write_text(json.dumps(dict(schema_version=1, name="x", segment_ids=[], manoeuvres=[])))
    with pytest.raises(ValueError, match="segment_ids"):
        load_route_definition(path)


def test_untrimmed_chain_uses_whole_lanes(topology):
    mission = mission_along_lanes(
        topology, ("1:0:1", "8:0:1", "4:0:-1", "515:0:-1", "5:0:-1"), ("straight",)
    )
    assert RouteGeometry(mission.points_xy_m).length_m > 300
    assert mission.map_validation is None


@pytest.mark.parametrize("scenario", ["S1", "S3"])
def test_full_stack_on_a1_route(tmp_path, mission, scenario):
    world = BaselineWorld(scenario, route="mission", mission=mission)
    result = run_episode(
        world,
        RasterDetector(),
        world.points,
        BaselineConfig(),
        tmp_path / scenario,
        manifest(scenario),
        max_sim_s=120,
    )
    assert result["collection_status"] == "COLLECTED", result["error"]
    metrics = result["metrics"]
    assert metrics["route_completion"] == 1
    assert metrics["collisions"] == 0
    if scenario == "S1":
        assert 2.5 <= metrics["hold_margin_min_m"] <= metrics["hold_margin_max_m"] <= 6.0
        assert metrics["resume_delay_s"] is not None
    else:
        assert metrics["false_stops"] == 0


@pytest.mark.xfail(
    strict=True,
    reason="Known limitation: the track filter's world-axis footprint box widens a "
    "roadside car on a diagonal road section (here the 45-degree bend at s=40 m) into "
    "the corridor, so A2 holds. Remove this xfail when footprints follow the road.",
)
def test_roadside_car_on_the_diagonal_bend_does_not_stop(tmp_path, mission):
    bend = replace(
        mission,
        map_validation=dict(
            mission.map_validation,
            route_definition=dict(name="a1_test01", obstacle_s_m=40.0),
        ),
    )
    world = BaselineWorld("S3", route="mission", mission=bend)
    result = run_episode(
        world,
        RasterDetector(),
        world.points,
        BaselineConfig(),
        tmp_path / "S3-bend",
        manifest("S3"),
        max_sim_s=45,
    )
    assert result["metrics"]["false_stops"] == 0
    assert result["collection_status"] == "COLLECTED"
