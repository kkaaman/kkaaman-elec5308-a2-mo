"""Lane-aided localisation, replayed offline on harness runs with synthetic lane offsets.

The live tool runs LaneUNet on recorded video; here the lane monitor's measurement is
synthesised from the logged true pose (8 m look-ahead offset from the route), with
white noise and occasional 3 m outliers (a lane-marking confusion), so the estimator,
the replay and the report can be checked without CARLA or the LaneUNet weights.
"""

import importlib.util
import json
import math
import sys

import numpy as np
import pytest

from autonomy.baseline import BaselineConfig
from autonomy.lane_localisation import LaneAidedEstimator, LaneAiding
from autonomy.localisation import FilterTuning, LocalisationConfig, LocalisationInputs
from autonomy.paths import FIXTURES_DIR, REPO_ROOT
from evaluation.lane_localisation_replay import lane_measurements, replay, summarise
from evaluation.runner import run_episode
from planning.corridor import RouteGeometry
from planning.lane_graph import LaneTopology
from planning.mission import load_route_definition, mission_from_definition
from sim_harness.baseline import BaselineWorld, RasterDetector
from visualisation.lane_monitor import map_lookahead_offset

CONSUMER = LocalisationConfig.load(REPO_ROOT / "configs/localisation_consumer_gnss.json")
AIDING = LaneAiding.load(REPO_ROOT / "configs/lane_localisation_consumer.json")
SEED = 5308  # draws a (1.24, -1.48) m consumer-GNSS offset


@pytest.fixture(scope="module")
def route_mission():
    topology = LaneTopology.load(FIXTURES_DIR / "Town10HD.lane_graph.json")
    definition = load_route_definition(REPO_ROOT / "configs/missions/a1_test01.json")
    return mission_from_definition(topology, definition)


def harness_run(folder, mission, *, drive):
    world = BaselineWorld("S0", route="mission", mission=mission)
    world.enable_localisation(CONSUMER, SEED)
    manifest = dict(
        engine="harness",
        scenario="S0",
        seed=SEED,
        step_wall_budget_s=10.0,
        event_sensors_verified=True,
        truth_channel_complete=True,
    )
    if drive:
        manifest["pose_source"] = "estimate"
    run_episode(
        world,
        RasterDetector(),
        world.points,
        BaselineConfig(),
        folder,
        manifest,
        max_sim_s=150,
        navigation_only=True,
    )
    rows = [json.loads(x) for x in (folder / "samples.jsonl").read_text().splitlines()]
    return rows, json.loads((folder / "manifest.json").read_text()), world


def monitor_records(rows, route, *, every=2, noise_m=0.08, outlier_rate=0.02, seed=1):
    """Lane-monitor records in replay_lane_monitor's format, from the true pose."""
    rng = np.random.default_rng(seed)
    records = []
    for index, row in enumerate(rows):
        if index % every:
            continue
        ego = row["ego"]
        truth = map_lookahead_offset(route, ego["x_m"], ego["y_m"], ego["yaw_rad"])
        measured = truth + rng.normal(0.0, noise_m)
        if rng.random() < outlier_rate:
            measured += rng.choice([-3.0, 3.0])
        records.append(
            dict(
                index=index,
                frame_id=row["frame_id"],
                map_offset_m=truth,
                departure=dict(signed_offset_m=measured, valid=True, confidence=0.8),
            )
        )
    return records


@pytest.fixture(scope="module")
def shadow(tmp_path_factory, route_mission):
    folder = tmp_path_factory.mktemp("lane") / "shadow"
    rows, manifest, world = harness_run(folder, route_mission, drive=False)
    route = RouteGeometry(world.points)
    records = monitor_records(rows, route)
    trajectory, info = replay(
        rows, route, CONSUMER, AIDING, lane_measurements(records, 0.3), drive_mode=False
    )
    return dict(
        folder=folder,
        rows=rows,
        manifest=manifest,
        route=route,
        records=records,
        trajectory=trajectory,
        info=info,
        true_offset=world.localisation_record()["gnss_bias_m"],
    )


def test_replayed_gnss_filter_reproduces_the_logged_estimate(shadow):
    assert shadow["info"]["replay_max_difference"] < 1e-9


def test_gnss_only_estimate_carries_the_offset(shadow):
    gnss = summarise(shadow["trajectory"]["gnss"])
    assert gnss["lateral_error_m"]["mean"] > 1.0
    assert abs(gnss["final_along_error_m"]) > 1.0


def test_lane_aiding_with_offset_states_removes_it(shadow):
    aided = summarise(shadow["trajectory"]["gnss_lane_offset"])
    assert aided["lateral_error_m"]["mean"] < 0.05
    assert aided["lateral_error_m"]["p95"] < 0.15
    assert aided["yaw_error_deg"]["p95"] < 3.0
    # Along-route error is only removed once a bend has made the offset observable:
    # at the goal it is small, where the GNSS-only filter is off by the offset.
    assert aided["goal_approach_along_error_m"]["mean"] < 0.3
    estimated = shadow["info"]["final_gnss_offset_m"]["gnss_lane_offset"]
    assert estimated == pytest.approx(shadow["true_offset"], abs=0.15)


def test_lane_without_offset_states_fixes_lateral_only(shadow):
    lane = summarise(shadow["trajectory"]["gnss_lane"])
    aided = summarise(shadow["trajectory"]["gnss_lane_offset"])
    assert lane["lateral_error_m"]["mean"] < 0.15
    assert (
        lane["goal_approach_along_error_m"]["mean"] > aided["goal_approach_along_error_m"]["mean"]
    )


def test_outliers_are_gated_once_converged(shadow):
    outliers = {
        r["frame_id"]
        for r in shadow["records"]
        if abs(r["departure"]["signed_offset_m"] - r["map_offset_m"]) > 1
    }
    assert outliers
    status = {r["frame_id"]: r for r in shadow["trajectory"]["gnss_lane_offset"]}
    start = shadow["trajectory"]["gnss_lane_offset"][0]["sim_time_s"]
    late = [f for f in outliers if status[f]["sim_time_s"] - start > 10.0]
    # Never applied: gated, or past the end of the route ("route_end").
    assert late and all(status[f]["lane"] in ("gated", "route_end") for f in late)
    counts = shadow["info"]["lane_counts"]["gnss_lane_offset"]
    assert counts["lane_used"] > 0.9 * len(shadow["records"])


def test_drive_mode_replay_starts_at_the_mission_start(tmp_path, route_mission):
    rows, _, world = harness_run(tmp_path / "drive", route_mission, drive=True)
    route = RouteGeometry(world.points)
    _, info = replay(
        rows,
        route,
        CONSUMER,
        AIDING,
        lane_measurements(monitor_records(rows, route), 0.3),
        drive_mode=True,
    )
    assert info["replay_max_difference"] < 1e-9


def test_lane_measurement_jacobian_matches_finite_differences(route_mission):
    route = RouteGeometry(route_mission.points_xy_m)
    estimator = LaneAidedEstimator(FilterTuning(), AIDING)
    x, y, yaw = route.at(30.0, 0.4)  # inside the opening bend, 0.4 m left of the lane
    estimator.initialise_at(x, y, yaw + 0.05, 0.0, route)
    value, theta = estimator.predicted_lane_offset(route)
    analytic = [
        -math.sin(theta),
        math.cos(theta),
        AIDING.reference_x_m * math.cos(estimator.x[2] - theta),
    ]
    for state, expected in zip(range(3), analytic):
        nudged = LaneAidedEstimator(FilterTuning(), AIDING)
        nudged.initialise_at(x, y, yaw + 0.05, 0.0, route)
        nudged.x[state] += 1e-4
        numeric = (nudged.predicted_lane_offset(route)[0] - value) / 1e-4
        assert numeric == pytest.approx(expected, abs=0.02)


def test_lane_update_gates_a_jump_after_convergence(route_mission):
    route = RouteGeometry(route_mission.points_xy_m)
    estimator = LaneAidedEstimator(FilterTuning(), AIDING)
    x, y, yaw = route.at(100.0)
    estimator.initialise_at(x, y, yaw, 0.0, route)
    for k in range(1, 40):
        estimator.step(LocalisationInputs(k * 0.05, 0.0, 0.0, None), route, (0.0, 1.0))
    assert estimator.update_lane(3.0, route) == "gated"
    assert estimator.update_lane(0.05, route) == "used"


def test_configs_load_and_reject_bad_values(tmp_path):
    assert AIDING.estimate_gnss_offset and AIDING.estimate_wheel_scale
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(dict(schema_version=1, lane=dict(lane_sigma_m=0))))
    with pytest.raises(ValueError):
        LaneAiding.load(bad)
    assert LocalisationConfig.from_dict(CONSUMER.as_dict()) == CONSUMER


def test_replay_script_reports_every_variant(shadow):
    spec = importlib.util.spec_from_file_location(
        "replay_lane_localisation", REPO_ROOT / "scripts/replay_lane_localisation.py"
    )
    script = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        spec.loader.exec_module(script)
    finally:
        sys.path.remove(str(REPO_ROOT / "scripts"))
    result, trajectory = script.evaluate_run(
        shadow["folder"],
        shadow["manifest"],
        shadow["rows"],
        shadow["route"],
        AIDING,
        shadow["records"],
    )
    assert set(result["variants"]) == {"gnss", "gnss_lane", "gnss_lane_offset"}
    assert result["true_gnss_offset_m"] == pytest.approx(shadow["true_offset"])
    assert result["lane_monitor"]["usable"] == len(shadow["records"])
    assert result["lane_monitor"]["over_0_5_m"] > 0  # the injected outliers
    script.print_run(result)
    script.plot({result["run"]: trajectory}, shadow["folder"] / "lane_localisation.png")
    assert (shadow["folder"] / "lane_localisation.png").stat().st_size > 10_000
