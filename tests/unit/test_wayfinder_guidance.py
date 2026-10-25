"""E3-03: guidance state, ETA policy (paused, never divided by zero), mini-map and replay."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from planning.lane_graph import LaneTopology
from planning.wayfinder import (
    GuidanceState,
    PoiRegistry,
    RouteTracker,
    SpeedPolicy,
    format_distance,
    format_eta,
    plan_named_route,
)
from planning.wayfinder.minimap import MiniMapStyle, render_minimap, save_png
from scripts import wayfinder_replay

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def baseline_route():
    topology = LaneTopology.load(ROOT / "tests/fixtures/Town10HD.lane_graph.json")
    registry = PoiRegistry.load(ROOT / "configs/wayfinder/pois_town10hd.json")
    return topology, plan_named_route(topology, registry, "baseline_start", "baseline_goal")


def test_speed_policy_states_and_no_division_by_zero():
    policy = SpeedPolicy(cruise_speed_mps=4.0, paused_threshold_mps=0.1, arrival_radius_m=1.0)
    assert policy.eta_s(100.0, 0.0) == ("paused", None)
    assert policy.eta_s(100.0, 0.1) == ("paused", None)
    assert policy.eta_s(100.0, 2.0) == ("moving", pytest.approx(50.0))
    assert policy.eta_s(100.0, 9.0) == ("moving", pytest.approx(25.0))  # capped at cruise
    assert policy.eta_s(0.5, 0.0) == ("arrived", 0.0)
    cruise = SpeedPolicy(reference="cruise")
    assert cruise.eta_s(100.0, 1.0) == ("moving", pytest.approx(25.0))
    assert cruise.eta_s(100.0, 0.0) == ("paused", None)
    for bad in (dict(paused_threshold_mps=0.0), dict(cruise_speed_mps=0.05), dict(reference="guess"), dict(arrival_radius_m=-1)):
        with pytest.raises(ValueError):
            SpeedPolicy(**bad)
    with pytest.raises(ValueError):
        policy.eta_s(-1.0, 1.0)
    with pytest.raises(ValueError):
        policy.eta_s(10.0, math.nan)


def test_formatting_units():
    assert format_distance(0.0) == "0 m" and format_distance(999.4) == "999 m" and format_distance(1234.0) == "1.23 km"
    assert format_eta("paused", None) == "ETA paused"
    assert format_eta("arrived", 0.0) == "arrived"
    assert format_eta("moving", 125.0) == "ETA 2:05"
    with pytest.raises(ValueError):
        format_eta("moving", None)
    with pytest.raises(ValueError):
        GuidanceState(0, 0, 0, 1, 0, 10, 0, "arrive", 10, "moving", None)


def test_tracker_progress_is_monotonic_and_announces_turns(baseline_route):
    _topology, route = baseline_route
    tracker = RouteTracker(route, SpeedPolicy(cruise_speed_mps=4.0))
    first = tracker.update(0.0, *route.points_xyz_m[0][:2], 4.0)
    assert first.progress_m == 0.0 and first.remaining_m == pytest.approx(route.length_m)
    assert first.next_maneuver == "straight" and first.next_maneuver_m == pytest.approx(route.maneuvers[0].route_s_m)
    assert first.eta_state == "moving" and first.eta_s == pytest.approx(route.length_m / 4.0)
    assert first.units == "m, m/s, s"
    states = [first]
    for index, point in enumerate(route.points_xyz_m[1:], start=1):
        states.append(tracker.update(float(index), point[0], point[1], 4.0))
    progress = [s.progress_m for s in states]
    assert progress == sorted(progress) and progress[-1] == pytest.approx(route.length_m)
    assert states[-1].eta_state == "arrived" and states[-1].next_maneuver == "arrive"
    kinds = [s.next_maneuver for s in states]
    assert kinds.index("left") > 0 and kinds.index("arrive") > kinds.index("left")
    # After passing the straight-through junction the announced turn is the left turn.
    passed = next(s for s in states if s.progress_m > route.maneuvers[0].route_s_m + 1)
    assert passed.next_maneuver == "left"
    assert passed.next_maneuver_m == pytest.approx(route.maneuvers[1].route_s_m - passed.progress_m)
    with pytest.raises(ValueError):
        tracker.update(0.0, 0.0, 0.0, 1.0)  # time runs backwards


def test_tracker_stopped_mid_route_reports_paused_not_zero_division(baseline_route):
    _topology, route = baseline_route
    tracker = RouteTracker(route)
    x, y, _z = route.points_xyz_m[20]
    moving = tracker.update(0.0, x, y, 3.0)
    stopped = tracker.update(1.0, x, y, 0.0)
    crawling = tracker.update(2.0, x, y, 0.05)
    assert moving.eta_state == "moving" and moving.eta_s == pytest.approx(moving.remaining_m / 3.0)
    assert stopped.eta_state == "paused" and stopped.eta_s is None
    assert crawling.eta_state == "paused" and crawling.eta_s is None
    assert stopped.progress_m == moving.progress_m == crawling.progress_m
    assert format_eta(stopped.eta_state, stopped.eta_s) == "ETA paused"


def test_tracker_is_not_pulled_backwards_by_a_nearby_earlier_segment(baseline_route):
    _topology, route = baseline_route
    tracker = RouteTracker(route, search_window_m=30.0)
    # Drive to 60 % of the route, then report a pose 0.5 m beside the route start.
    target = 0.6 * route.length_m
    arc = 0.0
    for a, b in zip(route.points_xyz_m, route.points_xyz_m[1:]):
        arc += math.dist(a[:2], b[:2])
        tracker.update(arc, b[0], b[1], 4.0)
        if arc >= target:
            break
    before = tracker.progress_m
    state = tracker.update(arc + 1.0, route.points_xyz_m[0][0], route.points_xyz_m[0][1] + 0.5, 4.0)
    assert state.progress_m == before
    assert state.off_route_m > 10.0


def test_minimap_renders_route_pose_and_next_turn(baseline_route, tmp_path):
    topology, route = baseline_route
    tracker = RouteTracker(route)
    x, y, _z = route.points_xyz_m[30]
    state = tracker.update(0.0, x, y, 0.0)
    image = render_minimap(route, state, 0.0, topology, MiniMapStyle(size_px=320, panel_px=80))
    assert image.shape == (320, 320, 3) and image.dtype == np.uint8
    plain = render_minimap(route, state, 0.0, None, MiniMapStyle(size_px=320, panel_px=80))
    assert not np.array_equal(image, plain)
    later = render_minimap(route, tracker.update(1.0, *route.points_xyz_m[60][:2], 4.0), 1.0, None, MiniMapStyle(size_px=320, panel_px=80))
    assert not np.array_equal(plain, later)
    save_png(image, tmp_path / "frame.png")
    assert (tmp_path / "frame.png").stat().st_size > 1000
    with pytest.raises(ValueError):
        MiniMapStyle(size_px=100, panel_px=90)
    with pytest.raises(ValueError):
        render_minimap(route, state, math.nan)


def test_replay_script_writes_log_and_frames(tmp_path):
    output = tmp_path / "replay"
    assert wayfinder_replay.main(["--origin", "baseline_start", "--destination", "baseline_goal",
                                 "--output", str(output), "--frames", "3", "--dt", "1.0", "--pause-s", "3"]) == 0
    log = json.loads((output / "guidance_log.json").read_text(encoding="utf-8"))
    summary, steps = log["summary"], log["steps"]
    assert summary["progress_monotonic"] and summary["paused_steps"] >= 3
    assert steps[-1]["eta_state"] == "arrived" and steps[0]["eta_state"] == "moving"
    paused = [row for row in steps if row["eta_state"] == "paused"]
    assert paused and all(row["eta_s"] is None and row["speed_mps"] == 0.0 for row in paused)
    assert all(row["eta_s"] is not None for row in steps if row["eta_state"] == "moving")
    assert {row["next_maneuver"] for row in steps} == {"straight", "left", "arrive"}
    frames = sorted(output.glob("frame_*.png"))
    assert len(frames) == len(summary["frames"]) >= 3
    assert any(frame["eta_state"] == "paused" for frame in summary["frames"])
    route = json.loads((output / "route.json").read_text(encoding="utf-8"))
    assert route["origin"]["name"] == "baseline_start" and len(route["maneuvers"]) == 3
