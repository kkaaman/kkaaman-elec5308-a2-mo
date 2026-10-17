"""Regression checks on the real Town10HD mission geometry (no CARLA).

The fixed mission turns right with a radius near 10 m. Through that turn the
inside corridor edge lies outside the 90-degree camera's field of view, which
previously made the visible-route horizon zero and held the car in the junction
forever on every full-stack run, even without an obstacle.
"""

import json
import math
from dataclasses import replace

import numpy as np
import pytest

from autonomy.baseline import BaselineConfig
from autonomy.interfaces import EgoPose, FrameBundle
from evaluation.runner import run_episode
from navigation_control.types import VehicleState
from obstacle_perception.types import DetectionObservation, PixelBox
from obstacle_perception.visibility import DepthVisibility, RouteClearanceMemory
from planning.corridor import RouteGeometry
from sim_harness.baseline import BaselineWorld, RasterDetector, flat_ground_depth, mission_route

FRONT = 3.81152
HALF_WIDTH = 1.08173 + 0.15


def manifest(scenario):
    return dict(
        engine="harness",
        scenario=scenario,
        seed=1,
        step_wall_budget_s=10.0,
        event_sensors_verified=True,
        truth_channel_complete=True,
    )


def mission_frames():
    world = BaselineWorld("S0", route="mission")
    route = RouteGeometry(world.points)
    depth = flat_ground_depth(world.calibration, 640, 360)
    rgb = np.zeros((360, 640, 3), np.uint8)
    for frame_id, s in enumerate(np.arange(0.0, route.length_m - 5.0, 0.5), start=1):
        x, y, yaw = route.at(s)
        yield (
            s,
            route,
            FrameBundle(
                "e",
                frame_id,
                frame_id * 0.1,
                0.0,
                EgoPose(VehicleState(x, y, yaw, 0.0)),
                rgb,
                world.calibration,
                depth,
                frame_id,
            ),
        )


def test_mission_is_the_selected_right_turn():
    mission = mission_route()
    assert mission.manoeuvres == ("right", "straight")
    route = RouteGeometry(mission.points_xy_m)
    assert 120 <= route.length_m <= 350


def test_turn_leaves_camera_view_without_memory():
    horizons = {
        round(s): DepthVisibility(frame).route_horizon(route, s, FRONT, HALF_WIDTH, 35.0)
        for s, route, frame in mission_frames()
    }
    # Documents the geometry: an instantaneous view cannot see the inside edge.
    assert min(horizons.values()) < 8.0


def test_route_memory_keeps_the_turn_observable():
    memory = RouteClearanceMemory(15.0)
    lowest = math.inf
    for s, route, frame in mission_frames():
        horizon = DepthVisibility(frame).route_horizon(
            route, s, FRONT, HALF_WIDTH, 35.0, memory=memory, time_s=frame.sim_time_s
        )
        lowest = min(lowest, horizon)
    assert lowest >= 8.0


def test_memory_never_overrides_in_view_occlusion():
    s, route, frame = next(iter(mission_frames()))
    memory = RouteClearanceMemory(15.0)
    vis = DepthVisibility(frame)
    assert vis.route_horizon(route, s, FRONT, HALF_WIDTH, 35.0, memory=memory, time_s=0.1) >= 30
    # A surface 3 m from the camera is nearer than the first probe (~3.9 m):
    # previously remembered clear cells cannot outvote a current in-view occluder.
    blocked = replace(frame, depth_m=np.full((360, 640), 3.0, np.float32))
    assert (
        DepthVisibility(blocked).route_horizon(
            route, s, FRONT, HALF_WIDTH, 35.0, memory=memory, time_s=0.2
        )
        == 0.0
    )
    # A wall 6 m from the camera leaves only the nearer cells visible.
    wall = replace(frame, depth_m=np.full((360, 640), 6.0, np.float32))
    assert (
        DepthVisibility(wall).route_horizon(
            route, s, FRONT, HALF_WIDTH, 35.0, memory=memory, time_s=0.3
        )
        == 5.0
    )
    stale = DepthVisibility(frame)
    memory_old = RouteClearanceMemory(1.0)
    stale.route_horizon(route, s, FRONT, HALF_WIDTH, 35.0, memory=memory_old, time_s=0.0)
    assert not memory_old.recent(next(iter(memory_old._seen)), 5.0)


@pytest.mark.parametrize("scenario", ["S0", "S5"])
def test_full_stack_completes_the_mission(tmp_path, scenario):
    world = BaselineWorld(scenario, route="mission")
    result = run_episode(
        world,
        RasterDetector(),
        world.points,
        BaselineConfig(),
        tmp_path / scenario,
        manifest(scenario),
        max_sim_s=150,
    )
    assert result["collection_status"] == "COLLECTED", result["error"]
    metrics = result["metrics"]
    assert metrics["route_completion"] == 1
    assert metrics["collisions"] == 0
    if scenario == "S0":
        assert metrics["false_stops"] == 0
        rows = [json.loads(x) for x in (tmp_path / "S0" / "samples.jsonl").read_text().splitlines()]
        assert not any(r["state"] in {"STOPPING", "HOLD"} for r in rows)
    else:
        assert 2.5 <= metrics["hold_margin_min_m"] <= metrics["hold_margin_max_m"] <= 6.0
        assert metrics["resume_delay_s"] is not None


class InjectingDetector(RasterDetector):
    """Adds an untracked box on the road surface at chosen frames."""

    def __init__(self, frames):
        self.frames = set(frames)

    def detect(self, frame):
        batch = super().detect(frame)
        if frame.frame_id in self.frames:
            box = DetectionObservation("car", 0.4, PixelBox(300, 200, 340, 215), None)
            batch = replace(batch, detections=batch.detections + (box,))
        return batch


def straight_s0(tmp_path, frames):
    world = BaselineWorld("S0", ground=True)
    result = run_episode(
        world,
        InjectingDetector(frames),
        world.points,
        BaselineConfig(),
        tmp_path / "fp",
        manifest("S0"),
        max_sim_s=60,
    )
    rows = [json.loads(x) for x in (tmp_path / "fp" / "samples.jsonl").read_text().splitlines()]
    return result, rows


def test_single_false_positive_frame_does_not_force_a_stop(tmp_path):
    result, rows = straight_s0(tmp_path, {200})
    after = rows[200:300]
    assert result["metrics"]["route_completion"] == 1
    assert not any(r["state"] in {"STOPPING", "HOLD"} for r in after)
    assert min(r["speed_mps"] for r in after) > 4.0


def test_persistent_unknown_depth_box_still_stops(tmp_path):
    _, rows = straight_s0(tmp_path, set(range(200, 400)))
    window = rows[200:400]
    assert any(r["state"] in {"STOPPING", "HOLD"} for r in window)
    assert window[-1]["speed_mps"] <= 0.1


# Live S0 (17 Oct): a Town10HD map-layer SUV parked at the kerb, its near side
# ~2.0 m right of the lane centre, stopped every full-stack run at s ~ 13 m.
# Road pixels inside its oblique detection box were treated as car surface.
@pytest.mark.parametrize(
    ("scenario", "lateral"),
    [("S0", -3.0), ("S0", -2.75), ("S1", -2.75)],
)
def test_kerbside_parked_car_does_not_stop_the_mission(tmp_path, scenario, lateral):
    # lateral -2.75: near side exactly on the 1.75 m lane edge (0.52 m outside
    # the swept corridor); -3.0 matches the live SUV.
    world = BaselineWorld(scenario, route="mission", parked=[(23.0, lateral)])
    result = run_episode(
        world,
        RasterDetector(),
        world.points,
        BaselineConfig(),
        tmp_path / scenario,
        manifest(scenario),
        max_sim_s=150,
    )
    assert result["collection_status"] == "COLLECTED", result["error"]
    metrics = result["metrics"]
    assert metrics["route_completion"] == 1
    assert metrics["collisions"] == 0
    rows = [json.loads(x) for x in (tmp_path / scenario / "samples.jsonl").read_text().splitlines()]
    passing = [r for r in rows if r["progress_m"] < 25.0]
    assert passing and not any(r["state"] in {"STOPPING", "HOLD"} for r in passing)
    assert not any(r["unknown_in_route"] for r in passing)
    if scenario == "S0":
        assert min(r["speed_mps"] for r in passing if r["progress_m"] > 10.0) > 4.5
        assert metrics["false_stops"] == 0
    else:
        assert 2.5 <= metrics["hold_margin_min_m"] <= metrics["hold_margin_max_m"] <= 6.0
