"""Straight through a junction (``mapless.crossing``): the STRAIGHT command on the route, for
the baseline route's one junction (G2) and the mixed route (M13). On the synthetic world
kerb following alone followed the right kerb into the side road's fillet and stopped for
its hedge; the crossing latches the lane line before the mouth and follows it across."""

import math
from pathlib import Path

import numpy as np
import pytest

from mapless.corridor import CorridorConfig
from mapless.crossing import CrossingConfig, StraightCrossing
from mapless.driver import DriveMode, DriverConfig, MaplessDriver
from mapless.ego_motion import Pose2D
from mapless.junction import JunctionConfig
from mapless.planner import PlannerConfig
from mapless.satnav import Command, RoutePoint, SatNavRoute
from mapless.sensor_suite import load_sensor_suite
from mapless.terrain import VehicleBody
from navigation_control.local import LocalControlConfig
from navigation_control.types import ControllerConfig
from sim_harness.mapless_road import (
    Arc,
    KerbLine,
    MaplessRoadWorld,
    RoadScene,
    SideRoad,
    Straight,
    side_road_hedges,
)

ROOT = Path(__file__).resolve().parents[2]
SUITE = load_sensor_suite(ROOT / "configs" / "sensor_suites" / "mapless_v2.json")
BODY = VehicleBody()
DT = 0.05
S_JUNCTION = 70.0


def control():
    return LocalControlConfig(
        ControllerConfig(wheelbase_m=3.00465, max_steering_angle_rad=math.radians(20), lookahead_m=4.0, kp=0.45, ki=0.15, stopping_deceleration_mps2=1.0, hold_brake=0.5)
    )


def crossroads(sides=(-1, 1), width_m=14.0, fillet_m=6.0, seed=5308):
    roads = tuple(SideRoad(S_JUNCTION, side, width_m=width_m, fillet_m=fillet_m) for side in sides)
    scene = RoadScene(KerbLine([Straight(160)]), side_roads=roads, hedges=sum((side_road_hedges(r) for r in roads), ()))
    return MaplessRoadWorld(scene, SUITE, start_s_m=15.0, seed=seed), width_m


def straight_route(world, width_m, goal_s=140.0, entry_shift_m=0.0):
    scene = world.scene

    def fix(s, cmd, d=0.0):
        x, y, h = scene.kerb.pose_at(s)
        off = scene.lane_centre_m + d
        return RoutePoint(*world.plane.to_geodetic(x - off * math.sin(h), y + off * math.cos(h)), cmd)

    entry, exit_ = S_JUNCTION - width_m / 2 + entry_shift_m, S_JUNCTION + width_m / 2
    return SatNavRoute("cross", (fix(15.0, Command.LANEFOLLOW), fix(40.0, Command.LANEFOLLOW), fix(entry, Command.STRAIGHT), fix(exit_, Command.LANEFOLLOW), fix(110.0, Command.LANEFOLLOW), fix(goal_s, Command.LANEFOLLOW)))


def drive(world, route, speed=4.0, seconds=90, crossing=None):
    config = DriverConfig(body=BODY, control=control(), corridor=CorridorConfig(keep_right_m=1.75), junction=JunctionConfig(), planner=PlannerConfig(cruise_mps=speed), crossing=crossing or CrossingConfig())
    driver = MaplessDriver(SUITE, route, config, DT)
    log = []
    for _ in range(int(seconds / DT)):
        step = driver.step(world.frame())
        log.append((world.truth(), step))
        world.apply(step.command)
        if step.mode is DriveMode.FINISHED and world.state.speed == 0:
            break
    return log


def _crossed(log, goal_s=140.0):
    truth, last = log[-1]
    return last.mode is DriveMode.FINISHED and abs(truth.s_m - goal_s) < 3.0 and not any(t.collision for t, _ in log)


@pytest.mark.parametrize("sides", [pytest.param((-1, 1), id="crossroads"), pytest.param((-1,), id="t-right"), pytest.param((1,), id="t-left")])
def test_a_straight_command_crosses_the_junction_on_the_latched_lane_line(sides):
    world, width = crossroads(sides)
    log = drive(world, straight_route(world, width))
    assert _crossed(log)
    in_box = [t for t, _ in log if S_JUNCTION - width / 2 <= t.s_m <= S_JUNCTION + width / 2]
    assert in_box and min(t.speed_mps for t in in_box) > 3.0  # crossed at speed, not crept
    assert max(abs(t.lane_offset_m) for t, _ in log) < 0.6
    notes = [s.diagnostics.get("crossing") for _, s in log]
    assert "lane line latched" in notes and "crossing on the latched lane line" in notes and "handed back to the kerb corridor" in notes
    exit_leg = [t for t, _ in log if t.s_m > S_JUNCTION + width / 2 + 20.0]
    assert exit_leg and np.percentile(np.abs([t.lane_offset_m for t in exit_leg]), 95) < 0.3


def test_a_wide_junction_with_long_fillets_is_crossed_too():
    world, width = crossroads(width_m=20.0, fillet_m=10.0)
    log = drive(world, straight_route(world, width))
    assert _crossed(log) and max(abs(t.lane_offset_m) for t, _ in log) < 0.7


def test_the_route_entry_point_may_sit_metres_from_the_box():
    """A sat-nav's entry point lies beside the junction, not on its edge (review of
    519b8fe, F-07): 4 m late or early, the car still crosses on the line."""
    for shift in (4.0, -4.0):
        world, width = crossroads()
        log = drive(world, straight_route(world, width, entry_shift_m=shift))
        assert _crossed(log), shift


def bent_junction(bend_deg=25.0, length_m=43.0, seed=5308, noise_m=2.0, curve=False):
    """Town10HD junction 468 as r12 showed it: 43 m long, its exit road bending right from
    2 m before the exit; the sat-nav points moved by ``noise_m`` as --route-noise moves them.
    ``curve``: as r13 showed it — the exit lane a curve of ``bend_deg`` (50) at radius 35 m
    from 2 m before the exit point on, the route's points 5 m and 23 m into it and the goal
    30 m in (the baseline's chain ends there)."""
    bend = math.radians(bend_deg)
    entry_s, exit_s = 100.0, 100.0 + length_m
    if curve:
        kerb = KerbLine([Straight(exit_s - 2.0), Arc(-35.0, bend) if bend > 0 else Arc(35.0, -bend), Straight(80.0)])
    else:
        kerb = KerbLine([Straight(exit_s - 2.0), Arc(-40.0, bend), Straight(80.0)]) if abs(bend) > 1e-6 else KerbLine([Straight(exit_s + 80.0)])
    mid = entry_s + length_m / 2
    roads = (SideRoad(mid, -1, width_m=length_m, fillet_m=6.0, length_m=60.0), SideRoad(mid, 1, width_m=length_m, fillet_m=6.0, length_m=60.0))
    scene = RoadScene(kerb, side_roads=roads, hedges=sum((side_road_hedges(r) for r in roads), ()))
    world = MaplessRoadWorld(scene, SUITE, start_s_m=15.0, seed=seed)
    rng = np.random.default_rng(seed)

    def fix(s, cmd, noise=noise_m):
        x, y, h = kerb.pose_at(s)
        off = scene.lane_centre_m
        px, py = x - off * math.sin(h), y + off * math.cos(h)
        if noise:
            px, py = px + rng.normal(0, noise), py + rng.normal(0, noise)
        return RoutePoint(*world.plane.to_geodetic(px, py), cmd)

    approach = (fix(15.0, Command.LANEFOLLOW, 0), fix(40.0, Command.LANEFOLLOW), fix(65.0, Command.LANEFOLLOW), fix(entry_s, Command.STRAIGHT), fix(exit_s + 5.0, Command.LANEFOLLOW))
    goal = exit_s + (30.0 if curve else 55.0)
    rest = (fix(exit_s + 23.0, Command.LANEFOLLOW),) if curve else (fix(exit_s + 30.0, Command.LANEFOLLOW),)
    route = SatNavRoute("bend", approach + rest + (fix(goal, Command.LANEFOLLOW, 0),))
    return world, route, goal  # the goal point exact: the test measures the crossing, not the goal's 2 m of noise


@pytest.mark.parametrize("seed", [5308, 5309, 5310])
def test_a_long_junction_whose_exit_road_bends_is_crossed_on_the_routes_exit_line(seed):
    """r12 on Town10HD: junction 468 is 43 m long and its exit road bends 25 degrees right;
    the straight line ran over the centre line, and over 40 m with no kerb the line
    itself drifted 0.7-1.5 m. The path now blends onto the exit lane's line (fitted through
    the route points after the exit), and into the junction the kerb corridor is sought
    beside the path and re-anchors it; with the sat-nav points 2 m off, the car stays in
    its lane to the goal."""
    world, route, goal = bent_junction(seed=seed)
    log = drive(world, route, speed=2.0, seconds=140)
    truth, last = log[-1]
    assert last.mode is DriveMode.FINISHED and abs(truth.s_m - goal) < 3.0 and not any(t.collision for t, _ in log)
    assert max(abs(t.lane_offset_m) for t, _ in log) < 1.75  # never over the lane's edges
    exit_leg = [t for t, _ in log if t.s_m > 143.0 + 25.0]
    assert exit_leg and np.percentile(np.abs([t.lane_offset_m for t in exit_leg]), 95) < 0.3
    assert any(s.diagnostics.get("crossing_bend") not in (None, 0.0) for _, s in log)


@pytest.mark.parametrize("seed", [5308, 5311, 5314])
def test_a_junction_whose_exit_lane_curves_is_handed_to_the_kerb_corridor_before_the_exit(seed):
    """r13 on Town10HD: junction 468's exit lane is a curve of 50 degrees over 30 m from
    the exit point on. A line fitted through the three sat-nav points past the exit was
    40 degrees off and 3 m to the side, and the kerb re-anchoring turned it further into
    the kerb. Now the path past the exit is an arc fitted to those points, and the kerb
    corridor is handed the car as soon as a kerb fit agrees with the path at the car and
    runs on towards the route: it follows the curve as no line does."""
    world, route, goal = bent_junction(bend_deg=50.0, seed=seed, curve=True)
    log = drive(world, route, speed=2.0, seconds=110)
    truth, last = log[-1]
    assert last.mode is DriveMode.FINISHED and abs(truth.s_m - goal) < 3.0 and not any(t.collision for t, _ in log)
    assert max(abs(t.lane_offset_m) for t, _ in log) < 1.0
    handed = [t.s_m for t, s in log if s.diagnostics.get("crossing") == "handed back to the kerb corridor"]
    assert handed and handed[0] < 143.0 + 12.0  # the kerb corridor has the car by 12 m past the exit point
    curve = [t for t, _ in log if t.s_m > 150.0]
    assert curve and np.percentile(np.abs([t.lane_offset_m for t in curve]), 95) < 0.5


def test_without_the_crossing_kerb_following_follows_the_fillet_into_the_mouth():
    """The failure the crossing is for, kept as the record: with the crossing off, the kerb
    fit turns into the right side road and the car stops for its hedge."""
    world, width = crossroads()
    off = CrossingConfig(enabled=False)
    log = drive(world, straight_route(world, width), seconds=40, crossing=off)
    truth, last = log[-1]
    assert not _crossed(log) and truth.s_m < S_JUNCTION + width / 2
    assert max(abs(t.lane_offset_m) for t, _ in log) > 0.8


def test_the_line_is_the_cars_own_track_and_the_corridor_along_it_is_straight():
    c = CrossingConfig(history_m=20.0, history_min_m=10.0)
    x = StraightCrossing(c)
    for k in range(80):  # 20 m straight along +x, 0.25 m apart, offset 0.3 m left
        x.observe(Pose2D(k * 0.25, 0.3, 0.0), moving=True)
    assert x._track_heading() == pytest.approx(0.0, abs=1e-6)
    x.observe(Pose2D(19.75, 0.3, 0.0), moving=False)  # standing still adds nothing
    corridor = type("C", (), dict(ok=True, heading_near=lambda self: 0.2, lateral_offset_at=lambda self, _x: -0.3))()
    pose = Pose2D(19.75, 0.3, 0.1)  # yawed 0.1 rad: the track, not the yaw or the kerb slope, gives the line
    x.latch(corridor, pose, entry_body=np.array([20.0, 0.0]))
    assert x.line is not None and x.line[2] == pytest.approx(0.0, abs=1e-6)
    cx, cy, _ = x.line
    assert (cx, cy) == pytest.approx(pose.to_world(np.array([[0.0, -0.3]]))[0].tolist(), abs=1e-6)
    pts = x.line_body(pose)
    assert pts[0, 0] == pytest.approx(c.start_m * math.cos(0.1), abs=0.05) and len(pts) == int((c.look_ahead_m - c.start_m) / c.step_m) + 1
    heading = math.atan2(pts[-1, 1] - pts[0, 1], pts[-1, 0] - pts[0, 0])
    assert heading == pytest.approx(-0.1, abs=1e-6)  # the line in the yawed car's frame
    cor = x.corridor(None, pose, 1.0, support=lambda grid, p, body: 25.0)
    assert cor.ok and cor.valid_length_m == 25.0 and cor.support_m == 25.0 and np.isnan(cor.right_edge_m).all()
    assert cor.heading_at(0.0) == pytest.approx(-0.1, abs=1e-6)
    # Not latched beyond the window, nor inside the freeze, nor with the entry far off the line.
    y = StraightCrossing(c)
    y.latch(corridor, pose, np.array([c.latch_from_m + 1.0, 0.0]))
    y.latch(corridor, pose, np.array([c.freeze_m - 1.0, 0.0]))
    y.latch(corridor, pose, np.array([20.0, c.max_route_offset_m + 1.0]))
    assert y.line is None
    with pytest.raises(ValueError):
        CrossingConfig(freeze_m=40.0)


def test_the_heading_hold_turns_the_line_by_the_odometrys_drift_from_the_compass():
    """The odometry frame turns with the gyro's bias residual; the compass does not. With the
    odometry's yaw climbing 0.001 rad a tick against a steady compass, the line latched at
    heading 0 turns with it (smoothed, 5 s), about the car's foot on it, so the car's true
    heading holds; without drift it does not move."""
    x = StraightCrossing(CrossingConfig(history_m=20.0, history_min_m=10.0))
    corridor = type("C", (), dict(ok=True, heading_near=lambda self: 0.0, lateral_offset_at=lambda self, _x: 0.0))()
    for k in range(100):
        x.observe(Pose2D(k * 0.25, 0.0, 0.0), moving=True, compass_yaw_rad=0.0)
    x.latch(corridor, Pose2D(24.75, 0.0, 0.0), entry_body=np.array([20.0, 0.0]))
    assert x.line[2] == pytest.approx(0.0, abs=1e-6) and x.hold_heading(Pose2D(25.0, 0.0, 0.0)) == 0.0
    for k in range(1, 801):  # 40 s: the odometry's yaw drifts 0.8 rad against the compass
        x.observe(Pose2D(25.0 + k * 0.1, 0.0, 0.001 * k), moving=True, compass_yaw_rad=0.0)
    turned = x.hold_heading(Pose2D(105.0, 0.0, 0.8))
    assert 0.6 < turned < 0.8 and x.line[2] == pytest.approx(turned) and x.line[0] == pytest.approx(105.0) and abs(x.line[1]) < 1e-9
    steady = StraightCrossing(CrossingConfig(history_m=20.0, history_min_m=10.0))
    for k in range(100):
        steady.observe(Pose2D(k * 0.25, 0.0, 0.0), moving=True, compass_yaw_rad=0.3)  # a constant compass offset is no drift
    steady.latch(corridor, Pose2D(24.75, 0.0, 0.0), entry_body=np.array([20.0, 0.0]))
    for k in range(1, 201):
        steady.observe(Pose2D(25.0 + k * 0.1, 0.0, 0.0), moving=True, compass_yaw_rad=0.3)
    assert steady.hold_heading(Pose2D(45.0, 0.0, 0.0)) == pytest.approx(0.0, abs=1e-6)
