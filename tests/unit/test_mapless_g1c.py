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
    side_road_hedges,
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


def junction_world(side, fillet_m=10.0, parked=(), start_s_m=15.0, furnished=True, **scene_args):
    side_road = SideRoad(60.0, side, fillet_m=fillet_m)
    hedges = side_road_hedges(side_road) if furnished else ()
    scene = RoadScene(KerbLine([Straight(150)]), side_roads=(side_road,), parked=parked, hedges=hedges, **scene_args)
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


# The in-box offset is the planned arc against the synthetic truth arc (a different radius
# is a different path, not an error); the permitted-area gate in the harness is the safety
# measure. The exit-leg p95 below is the quality measure.
# The left turn reads its exit kerb across the junction from ~38 m, where the synthetic verge's
# raised returns lie up to a metre behind the kerb (the live kerb rule was chosen on the
# Town10HD recording, where raised returns sit at the kerb line): the synthetic left turn
# commits on a line ~1.3 m right of truth and swings wide in the box before the refinement
# and the hand-back correct it. The harness's footprint gate, on CARLA, is the real measure.
@pytest.mark.parametrize("side, max_offset", [pytest.param(-1, 1.2, id="right"), pytest.param(1, 2.6, id="left")])
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
    # One radius through the arc, except where the path was deliberately re-planned (the
    # exit line refined from closer by, or a blocked path repaired): each such event may
    # change it once, and never tick to tick.
    replans = sum(1 for _, s in log if s.mode is DriveMode.TURNING and (s.diagnostics.get("turn") or "").startswith(("exit line refined", "latched path blocked: re-planned")))
    assert len({round(r, 1) for r in radii[-len(committed) :]}) <= 1 + replans


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
    # Held before the entry: stopped to within numerical noise, and no creep over the last
    # 10 s (external review of 405b056: exact zero failed at 1.2e-5 m/s on another machine).
    assert truth.speed_mps < 1e-3 and truth.s_m < 56.5 - BODY.front_m
    held = [t.s_m for t, _ in log if t.time_s >= log[-1][0].time_s - 10.0]
    assert max(held) - min(held) < 0.05
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


@pytest.mark.parametrize("side, dx, dy", [pytest.param(-1, 3.0, 1.5, id="right-late-wide"), pytest.param(1, -3.0, -1.5, id="left-early-inside")])
def test_entry_point_a_satnav_distance_off_still_turns_at_the_kerbs_junction(side, dx, dy):
    """Review of 519b8fe, F-07: the routes were exact lane-centre points with the entry on
    the lane's last waypoint, and nothing moved the entry point. A sat-nav's point sits
    metres from the junction box: 3 m along the road and 1.5 m across, either way. The car
    still turns where the kerb says the junction is, and reaches the exit lane."""
    world, _ = junction_world(side)
    route = world.route(goal_after_turn(world))
    plane = world.plane
    moved = []
    for p in route.points:
        if p.command in (Command.LEFT, Command.RIGHT):
            x, y = plane.to_local(p.latitude_deg, p.longitude_deg)
            p = RoutePoint(*plane.to_geodetic(x + dx, y + dy), p.command)
        moved.append(p)
    log = drive(world, SatNavRoute(route.name, tuple(moved)))
    truth, last = log[-1]
    assert last.mode is DriveMode.FINISHED and abs(truth.s_m - goal_after_turn(world)) < 3.0
    assert not any(t.collision for t, _ in log) and any(s.mode is DriveMode.TURNING for _, s in log)
    exit_leg = [t for t, _ in log if t.s_m > world.path.starts[2][3] + 8]
    assert exit_leg and np.percentile(np.abs([t.lane_offset_m for t in exit_leg]), 95) < 0.3


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


def _drive_into_the_arc(side=-1, turned_deg=25.0, world=None):
    """A driver part way round the commanded turn (the car has turned ``turned_deg``)."""
    from mapless.geometry import wrap  # noqa: PLC0415

    world = world or junction_world(side)[0]
    route = world.route(goal_after_turn(world))
    driver = MaplessDriver(SUITE, route, DriverConfig(body=BODY, control=control(), corridor=CorridorConfig(keep_right_m=1.75)), DT)
    for _ in range(int(60 / DT)):
        step = driver.step(world.frame())
        world.apply(step.command)
        if driver.turn.committed and driver._lane_heading_odo is not None and abs(wrap(driver.last_motion.odometry.yaw_rad - driver._lane_heading_odo)) > math.radians(turned_deg):
            return driver, world
    raise AssertionError("the car never got into the arc")


def test_a_mid_arc_replan_is_followable_and_a_rejected_one_leaves_the_latch_alone():
    # Fresh-eyes review of 58adaf3: the mid-arc re-plan put the tangent point at the car, so
    # the path began with a zero-length segment whose arbitrary heading swung the footprint
    # over the kerb; its own check passed, the follow after it failed, and the latch had
    # already been replaced, so every later follow failed too and the car stopped for good.
    driver, world = _drive_into_the_arc(-1)
    turn, pose, grid, t = driver.turn, driver.last_motion.odometry, driver.grid, world.time_s
    before = (turn._committed[0].copy(), turn._radius)
    plan = turn.replan_mid_arc(grid, pose, t)
    assert plan is not None and plan.arc_start_m == 0.0 and plan.verified_m > 10.0
    assert turn.follow(grid, pose, t + DT) is not None, "the re-latched path must be followable on the next tick"
    assert turn._entry[2] == turn._radius  # the entry frame moved with the new arc
    assert turn._radius != before[1] and len(turn._committed[0]) != len(before[0]) or not np.array_equal(turn._committed[0], before[0])
    # A re-plan whose follow fails keeps the old latch: force the follow to reject.
    latched = (turn._committed[0].copy(), turn._radius)
    original = turn.follow
    turn.follow = lambda *a, **k: None
    try:
        assert turn._relatch(turn._committed, turn._exit_odo, turn._exit_fit, 99.0, (pose, 0.0, 99.0), grid, pose, t) is None
    finally:
        turn.follow = original
    assert turn._radius == latched[1] and np.array_equal(turn._committed[0], latched[0])
    assert turn.follow(grid, pose, t + DT) is not None


def test_a_block_far_along_the_exit_straight_does_not_stop_the_car_at_the_mouth():
    # Review of 58adaf3: a 1.5 m wall along the approach pavement tilted the lane heading
    # prior by 4 degrees, the exit straight read a kerb cell 28 m out, and the car stood at
    # the mouth for the rest of the run with 20 m of verified road ahead. A hazard beyond
    # the stopping envelope now shortens the verified path instead of blocking it.
    side_road = SideRoad(60.0, -1, fillet_m=10.0)
    wall = (0.5 * (15.0 + 50.0), -1.0, 0.0, 0.5 * (50.0 - 15.0), 0.3, 1.5)
    scene = RoadScene(KerbLine([Straight(150)]), side_roads=(side_road,), hedges=side_road_hedges(side_road) + (wall,))
    world = MaplessRoadWorld(scene, SUITE, turn=side_road, start_s_m=15.0)
    log = drive(world, world.route(goal_after_turn(world)), seconds=70)
    truth, last = log[-1]
    assert last.mode is DriveMode.FINISHED and abs(truth.s_m - goal_after_turn(world)) < 3.0
    assert not any(t.collision for t, _ in log)
    notes = {s.diagnostics.get("turn") for _, s in log}
    assert "latched path no longer clears the terrain: stopping on it" not in notes


def test_a_parked_car_flush_with_the_exit_kerb_is_not_the_kerb():
    # Review of 58adaf3: raised evidence under a parked car's sill read as the exit kerb,
    # placing the lane 1.9 m into the road; the tall veto that labels() applies now applies
    # to the raised evidence the kerb rule reads.
    from mapless.terrain import PointClass, TerrainGrid  # noqa: PLC0415

    grid = TerrainGrid()
    grid.time_s = 0.0
    xy = np.array([[5.0, -1.8]])
    ij, inside = grid.cells(xy)
    i, j = ij[inside, 0], ij[inside, 1]
    grid.raised[i, j] = 3.0
    grid.seen_by_class[PointClass.RAISED][i, j] = 0.0
    assert grid.raised_evidence(xy, max_age_s=5.0)[0] == 3.0
    grid.tall[i, j] = 2.0  # a car body seen above the sill
    assert grid.raised_evidence(xy, max_age_s=5.0)[0] == 0.0
