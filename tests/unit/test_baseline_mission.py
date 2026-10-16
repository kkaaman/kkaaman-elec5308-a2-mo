"""Frozen route geometry, identity and deterministic loading checks."""

import copy
import json
from pathlib import Path

import pytest

from planning.baseline_mission import load_frozen_mission, validate_open_polyline

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("points", [
    [(0, 0, 0), (10, 10, 0), (0, 10, 0), (10, 0, 0)],
    [(0, 0, 0), (10, 0, 0), (5, 0, 0), (15, 0, 0)],
    [(0, 0, 0), (10, 0, 0), (5, 0, 0)],
    [(0, 0, 0), (0, 0, 0), (10, 0, 0)],
    [(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 0, 0)],
])
def test_rejects_crossing_overlapping_closed_or_repeated_geometry(points):
    with pytest.raises(ValueError):
        validate_open_polyline(points)


def test_valid_turn_and_repeated_loading_are_deterministic():
    validate_open_polyline([(0, 0, 0), (10, 0, 0), (10, 10, 0)])
    first = load_frozen_mission(ROOT / "configs/mission_baseline.json")
    second = load_frozen_mission(ROOT / "configs/mission_baseline.json")
    assert first.route_sha256 == second.route_sha256
    assert first.config_sha256 == second.config_sha256
    assert first.waypoints == second.waypoints
    assert first.length_m > 100


def test_caller_cannot_change_frozen_route_or_config_identity():
    mission = load_frozen_mission(ROOT / "configs/mission_baseline.json")
    original_points, original_hash = mission.waypoints, mission.route_sha256
    original_speed = mission.config["cruise_speed_mps"]
    mission.route["points_xyz_m"][0][0] += 10
    mission.config["cruise_speed_mps"] = 99
    assert mission.waypoints == original_points
    assert mission.route_sha256 == original_hash
    assert mission.config["cruise_speed_mps"] == original_speed


@pytest.mark.parametrize("mutation", ["points", "fingerprint", "spawn", "goal"])
def test_tampered_route_is_rejected(tmp_path, mutation):
    config = json.loads((ROOT / "configs/mission_baseline.json").read_text())
    route = json.loads((ROOT / "configs" / config["route_file"]).read_text())
    config["topology_file"] = str(ROOT / "tests/fixtures/Town10HD.lane_graph.json")
    config["route_file"] = "route.json"
    route = copy.deepcopy(route)
    if mutation == "points":
        route["points_xyz_m"][5][0] += 0.1
    elif mutation == "fingerprint":
        route["topology_sha256"] = "0" * 64
    elif mutation == "spawn":
        route["spawn_pose"]["x_m"] += 1
    else:
        route["goal_xyz_m"][0] += 1
    (tmp_path / "route.json").write_text(json.dumps(route))
    path = tmp_path / "mission.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_frozen_mission(path)


@pytest.mark.parametrize(("key", "value"), [
    ("map_layers", "Buildings"), ("rgb_enabled", 0), ("fixed_delta_seconds", 0.1),
])
def test_runtime_contract_rejects_unsupported_scene_or_timing(tmp_path, key, value):
    config = json.loads((ROOT / "configs/mission_baseline.json").read_text())
    config["route_file"] = str(ROOT / "configs" / config["route_file"])
    config["topology_file"] = str(ROOT / "tests/fixtures/Town10HD.lane_graph.json")
    config["runtime"][key] = value
    path = tmp_path / "mission.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_frozen_mission(path)
