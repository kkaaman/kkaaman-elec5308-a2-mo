"""BL-B-06: committed S1-S5 scenario files, deterministic placement and scripted removal."""

import json
import math

import pytest

from evaluation.scenarios import (
    RemovalSchedule,
    load_scenario,
    load_scenarios,
    placement_pose,
    route_heading,
)
from navigation_control.path import ReferencePath
from navigation_control.types import Waypoint
from planning.baseline_mission import load_frozen_mission

ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
SCENARIOS = ROOT / "configs" / "scenarios"


@pytest.fixture(scope="module")
def mission_path():
    mission = load_frozen_mission(ROOT / "configs" / "mission_baseline.json")
    return ReferencePath(mission.waypoints)


def test_committed_scenarios_load_and_refer_to_the_baseline_mission():
    scenarios = load_scenarios(SCENARIOS)
    assert list(scenarios) == ["S1", "S2", "S3", "S4", "S5"]
    for sid, definition in scenarios.items():
        assert (SCENARIOS / definition.mission_config).resolve() == (
            ROOT / "configs" / "mission_baseline.json"
        ).resolve()
        assert definition.seeds == (5308, 5309, 5310, 5311, 5312)
        assert len(definition.sha256) == 64
    assert scenarios["S2"].obstacle.kind == "pedestrian"
    assert scenarios["S3"].expected_behaviour == "no_stop"
    assert scenarios["S4"].removal.rule == "never"
    assert {scenarios[s].removal.rule for s in ("S1", "S2", "S5")} == {"after_steady_hold"}


def test_scenario_hash_is_line_ending_independent(tmp_path):
    text = (SCENARIOS / "S1.json").read_text(encoding="utf-8")
    crlf = tmp_path / "S1.json"
    crlf.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    assert load_scenario(crlf).sha256 == load_scenario(SCENARIOS / "S1.json").sha256


def test_placements_sit_where_the_scenarios_say(mission_path):
    scenarios = load_scenarios(SCENARIOS)
    # Road 20 runs to s = 49.2 m (junction 841 follows); junction 532 ends at s = 151.6 m.
    for sid in ("S1", "S2", "S3", "S4"):
        assert scenarios[sid].obstacle.route_s_m < 49.2 - 3.0
    assert scenarios["S5"].obstacle.route_s_m > 151.6
    s3 = scenarios["S3"].obstacle
    x, y, _ = placement_pose(mission_path, s3)
    projection = mission_path.project(x, y)
    assert projection.distance_m == pytest.approx(3.5, abs=1e-6)
    heading = route_heading(mission_path, s3.route_s_m)
    left = (-math.sin(heading), math.cos(heading))
    assert (x - projection.point.x_m) * left[0] + (y - projection.point.y_m) * left[1] > 0
    first = [placement_pose(mission_path, scenarios[s].obstacle) for s in scenarios]
    again = [
        placement_pose(mission_path, load_scenario(SCENARIOS / f"{s}.json").obstacle)
        for s in scenarios
    ]
    assert first == again


def test_placement_pose_on_a_bent_route():
    path = ReferencePath(Waypoint(*p) for p in ((0, 0), (10, 0), (10, 20)))
    obstacle = load_scenario(SCENARIOS / "S1.json").obstacle
    import dataclasses

    left = dataclasses.replace(obstacle, route_s_m=15.0, lateral_m=1.0, yaw_offset_deg=90.0)
    x, y, yaw = placement_pose(path, left)
    assert (x, y) == pytest.approx((9.0, 5.0))
    assert yaw == pytest.approx(math.pi)
    with pytest.raises(ValueError):
        placement_pose(path, dataclasses.replace(obstacle, route_s_m=31.0))


def _write(tmp_path, name, change):
    document = json.loads((SCENARIOS / "S1.json").read_text(encoding="utf-8"))
    change(document)
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(schema_version=2),
        lambda d: d.update(extra=1),
        lambda d: d["obstacle"].update(blueprint="walker.pedestrian.0001"),
        lambda d: d["obstacle"].update(route_s_m=float("nan")),
        lambda d: d["obstacle"].update(lateral_m=True),
        lambda d: d.update(removal={"rule": "never"}),
        lambda d: d.update(removal={"rule": "after_steady_hold"}),
        lambda d: d.update(seeds=[5308, 5308]),
        lambda d: d.update(seeds=[]),
        lambda d: d.update(expected_behaviour="maybe"),
    ],
)
def test_invalid_scenarios_are_refused(tmp_path, change):
    with pytest.raises(ValueError):
        load_scenario(_write(tmp_path, "S1", change))


def test_file_name_must_match_the_id(tmp_path):
    with pytest.raises(ValueError):
        load_scenario(_write(tmp_path, "S2", lambda d: None))


def test_duplicate_keys_are_refused(tmp_path):
    path = tmp_path / "S1.json"
    text = (SCENARIOS / "S1.json").read_text(encoding="utf-8")
    path.write_text(
        text.replace('"max_episode_s"', '"seeds": [1],\n  "max_episode_s"', 1), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Duplicate"):
        load_scenario(path)


def _drive(schedule, speeds, dt=0.05, start=10.0, skip=()):
    removals = []
    for i, speed in enumerate(speeds):
        if i in skip:
            continue
        t = start + i * dt
        if schedule.update(t, speed):
            removals.append(round(t, 6))
    return removals


def test_removal_after_a_continuous_steady_hold():
    removal = load_scenario(SCENARIOS / "S1.json").removal
    speeds = [2.0] * 10 + [0.05] * 60
    # Stopped from tick 10 (t = 10.5 s); 2.0 s later is tick 50 (t = 12.5 s); once only.
    assert _drive(RemovalSchedule(removal, 0.05), speeds) == [12.5]


def test_motion_or_a_dropped_tick_restarts_the_hold():
    removal = load_scenario(SCENARIOS / "S1.json").removal
    creep = [0.0] * 30 + [0.3] + [0.0] * 50
    assert _drive(RemovalSchedule(removal, 0.05), creep) == [round(10.0 + 71 * 0.05, 6)]
    gap = [0.0] * 80
    assert _drive(RemovalSchedule(removal, 0.05), gap, skip={20, 21}) == [
        round(10.0 + 62 * 0.05, 6)
    ]


def test_never_rule_keeps_the_obstacle():
    removal = load_scenario(SCENARIOS / "S4.json").removal
    assert _drive(RemovalSchedule(removal, 0.05), [0.0] * 2000) == []
