"""G1c offline: commanded junction turns in the synthetic world, and the ways they must fail safely.

The world gets a side road (``SideRoad``) with rounded pavement corners; the route
carries the junction's entry point (LEFT or RIGHT) and exit point as a CARLA route would,
at GNSS accuracy. The car never sees the side road's geometry except through its LiDAR.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from mapless.corridor import CorridorConfig
from mapless.driver import DriveMode, DriverConfig, MaplessDriver
from mapless.junction import JunctionConfig
from mapless.satnav import Command, RoutePoint, SatNavRoute
from mapless.sensor_suite import load_sensor_suite
from mapless.terrain import VehicleBody
from navigation_control.local import LocalControlConfig
from navigation_control.types import ControllerConfig
from sim_harness.mapless_road import (
    KerbLine,
    MaplessRoadWorld,
    ParkedCar,
    RoadScene,
    SideRoad,
    Straight,
)

ROOT = Path(__file__).resolve().parents[2]
SUITE = load_sensor_suite(ROOT / "configs" / "sensor_suites" / "mapless_v2.json")
BODY = VehicleBody()
DT = 0.05


def control():
    return LocalControlConfig(
        ControllerConfig(
            wheelbase_m=3.00465,
            max_steering_angle_rad=math.radians(20),
            lookahead_m=4.0,
            kp=0.45,
            ki=0.15,
            stopping_deceleration_mps2=1.0,
            hold_brake=0.5,
        )
    )


def junction_world(side, fillet_m=10.0, parked=(), start_s_m=15.0, **scene_args):
    side_road = SideRoad(60.0, side, fillet_m=fillet_m)
    scene = RoadScene(KerbLine([Straight(150)]), side_roads=(side_road,), parked=parked, **scene_args)
    world = MaplessRoadWorld(scene, SUITE, turn=side_road, start_s_m=start_s_m)
    return world, side_road


def drive(world, route, seconds=70, junction=None):
    config = DriverConfig(
        body=BODY, control=control(), corridor=CorridorConfig(keep_right_m=1.75), junction=junction or JunctionConfig()
    )
    driver = MaplessDriver(SUITE, route, config, DT)
    log = []
    for _ in range(int(seconds / DT)):
        step = driver.step(world.frame())
        log.append((world.truth(), step))
        world.apply(step.command)
        if step.mode is DriveMode.FINISHED and world.state.speed == 0:
            break
    return log


def goal_after_turn(world, extra_m=25.0):
    return world.path.starts[2][3] + extra_m


@pytest.mark.parametrize("side, max_offset", [pytest.param(-1, 0.7, id="right"), pytest.param(1, 0.6, id="left")])
def test_commanded_turn_reaches_the_exit_lane(side, max_offset):
    world, _ = junction_world(side)
    log = drive(world, world.route(goal_after_turn(world)))
    truth, last = log[-1]
    assert last.mode is DriveMode.FINISHED and abs(truth.s_m - goal_after_turn(world)) < 3.0
    assert not any(t.collision for t, _ in log)
    assert any(s.mode is DriveMode.TURNING for _, s in log)
    assert max(abs(t.lane_offset_m) for t, _ in log) < max_offset  # the turn path differs from the truth arc
    exit_leg = [t for t, _ in log if t.s_m > world.path.starts[2][3] + 8]  # on the exit lane
    assert exit_leg and np.percentile(np.abs([t.lane_offset_m for t in exit_leg]), 95) < 0.3
    handed = [s for _, s in log if s.diagnostics.get("turn") == "handed back to the kerb corridor"]
    assert handed
    assert max(t.speed_mps for t, _ in log) < 2.3
    assert not any(s.mode is DriveMode.FAULT for _, s in log)


def test_the_turn_is_latched_once_begun_and_the_radius_clears_the_corner():
    world, side_road = junction_world(-1)
    log = drive(world, world.route(goal_after_turn(world)))
    radii = [s.diagnostics["turn_radius_m"] for _, s in log if s.mode is DriveMode.TURNING and s.diagnostics.get("turn_radius_m")]
    committed = [s.diagnostics for _, s in log if s.diagnostics.get("rebuilding") is False and "turn_exit" not in s.diagnostics and s.mode is DriveMode.TURNING]
    assert radii and min(radii) >= JunctionConfig().min_radius_m
    assert committed  # the latched phase: followed, not re-planned
    assert len({round(r, 1) for r in radii[-len(committed) :]}) == 1  # one radius through the arc


def test_route_without_a_real_junction_holds_before_the_entry():
    # The route says RIGHT at 60 m, the kerb runs on unbroken: the car stops short.
    scene = RoadScene(KerbLine([Straight(150)]))
    world = MaplessRoadWorld(scene, SUITE, start_s_m=15.0)
    plane = world.plane

    def point(s, d, command):
        x, y, h = scene.kerb.pose_at(s)
        return RoutePoint(*plane.to_geodetic(x - d * math.sin(h), y + d * math.cos(h)), command)

    route = SatNavRoute("fake", (point(15.0, 1.75, Command.LANEFOLLOW), point(56.5, 1.75, Command.RIGHT), point(60.0, -5.0, Command.LANEFOLLOW), point(60.0, -30.0, Command.LANEFOLLOW)))
    log = drive(world, route, seconds=45)
    truth, last = log[-1]
    assert truth.speed_mps == 0 and truth.s_m < 56.5 - BODY.front_m
    assert any(s.diagnostics.get("turn") == "junction not confirmed by the kerb" for _, s in log)
    assert not any(s.mode is DriveMode.TURNING for _, s in log)


def test_exit_point_off_by_three_metres_is_corrected_by_the_kerb():
    world, _ = junction_world(-1)
    route = world.route(goal_after_turn(world))
    plane = world.plane
    shifted = []
    for p in route.points:
        if p.command is Command.LANEFOLLOW and shifted and shifted[-1].command is Command.RIGHT:
            x, y = plane.to_local(p.latitude_deg, p.longitude_deg)
            p = RoutePoint(*plane.to_geodetic(x + 3.0, y), p.command)  # 3 m onto the far side
        shifted.append(p)
    log = drive(world, SatNavRoute(route.name, tuple(shifted)))
    truth, last = log[-1]
    assert last.mode is DriveMode.FINISHED and not any(t.collision for t, _ in log)
    exit_leg = [t for t, _ in log if t.s_m > world.path.starts[2][3] + 8]
    assert exit_leg and np.percentile(np.abs([t.lane_offset_m for t in exit_leg]), 95) < 0.3
    measured = [s.diagnostics["turn_exit_fit"] for _, s in log if s.diagnostics.get("turn_exit") == "kerb"]
    assert measured and abs(measured[-1]["kerb_from_probe_m"]) > 1.0  # the kerb, not the route point, placed the lane


def test_obstacle_on_the_inner_corner_rejects_the_turn():
    # A car parked across the exit lane just past the corner: no radius clears it, so the
    # car waits at the junction mouth instead of turning into it.
    world, side_road = junction_world(-1)
    x, y, h = world.path.pose_at(world.path.starts[2][3] + 4.0)  # on the exit lane, 4 m past the arc
    parked = world.scene.kerb.locate(np.array([[x, y]]))
    car = ParkedCar(float(parked[0][0]), float(parked[1][0]))
    world, _ = junction_world(-1, parked=(car,))
    log = drive(world, world.route(goal_after_turn(world)), seconds=50)
    truth, last = log[-1]
    assert truth.speed_mps == 0 and not any(t.collision for t, _ in log)
    assert truth.s_m < side_road.s_m - side_road.width_m / 2 + 2.0  # did not enter the junction box
    assert any(s.diagnostics.get("turn") == "no radius clears the terrain" or s.reason.startswith("obstacle") for _, s in log)


def _degrade():
    # tests run in importlib mode (no sibling imports): load the G1b helper by path.
    import importlib.util

    spec = importlib.util.spec_from_file_location("g1b_review", Path(__file__).with_name("test_mapless_g1b_review.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.degrade


def test_sensor_loss_in_the_turn_stops_with_steering_held():
    degrade = _degrade()
    world, _ = junction_world(-1)
    route = world.route(goal_after_turn(world))
    config = DriverConfig(body=BODY, control=control(), corridor=CorridorConfig(keep_right_m=1.75))
    driver = MaplessDriver(SUITE, route, config, DT)
    log, dropped = [], None
    for k in range(int(60 / DT)):
        frame = world.frame()
        if dropped is None and log and log[-1][1].mode is DriveMode.TURNING and abs(world.state.yaw) > math.radians(25):
            dropped = k
        if dropped is not None:
            frame = degrade(frame, drop=("lidar",))
        step = driver.step(frame)
        log.append((world.truth(), step))
        world.apply(step.command)
    assert dropped is not None
    after = [s for _, s in log[dropped + 3 :]]
    assert all(s.mode is DriveMode.FAULT and s.command.throttle == 0 for s in after)
    assert after[0].command.steering_angle_rad < -0.05  # still steering into the right turn while braking
    assert world.state.speed == 0 and not any(t.collision for t, _ in log)
