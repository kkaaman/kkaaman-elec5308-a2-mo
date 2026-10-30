"""Stage 3, junction commitment: the full review of 73ae759, S5 (oncoming traffic on a left
turn), S6 (divided exit roads), S7 (standing at the junction mouth) and the exit-offset,
tube and evaluator items, on the synthetic world."""

import math
from pathlib import Path

import numpy as np
import pytest

from mapless.corridor import CorridorConfig
from mapless.driver import DriveMode, DriverConfig, MaplessDriver
from mapless.ego_motion import Pose2D
from mapless.junction import JunctionConfig, _arc_path, oncoming_gate
from mapless.obstacles import tube_bounds
from mapless.satnav import Command, RoutePoint, SatNavRoute
from mapless.sensor_suite import load_sensor_suite
from mapless.terrain import (
    CellLabel,
    ClassifiedSweep,
    PointClass,
    TerrainConfig,
    TerrainGrid,
    VehicleBody,
)
from navigation_control.local import LocalControlConfig
from navigation_control.types import ControlCommand, ControllerConfig
from sim_harness.mapless_road import (
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


def control():
    return LocalControlConfig(
        ControllerConfig(wheelbase_m=3.00465, max_steering_angle_rad=math.radians(20), lookahead_m=4.0, kp=0.45, ki=0.15, stopping_deceleration_mps2=1.0, hold_brake=0.5)
    )


def junction_world(side, fillet_m=10.0, extra_boxes=(), side_road=None, seed=5308):
    side_road = side_road or SideRoad(60.0, side, fillet_m=fillet_m)
    scene = RoadScene(KerbLine([Straight(150)]), side_roads=(side_road,), hedges=side_road_hedges(side_road) + tuple(extra_boxes))
    return MaplessRoadWorld(scene, SUITE, turn=side_road, start_s_m=15.0, seed=seed), side_road


def goal_after_turn(world, extra_m=25.0):
    return world.path.starts[2][3] + extra_m


def drive(world, route, seconds=80, per_tick=None):
    config = DriverConfig(body=BODY, control=control(), corridor=CorridorConfig(keep_right_m=1.75), junction=JunctionConfig())
    driver = MaplessDriver(SUITE, route, config, DT)
    log = []
    for k in range(int(seconds / DT)):
        step = driver.step(world.frame())
        log.append((world.truth(), step))
        if per_tick is not None:
            per_tick(k, world, driver, step)
        world.apply(step.command)
        if step.mode is DriveMode.FINISHED and world.state.speed == 0:
            break
    return log


def finished(world, log):
    truth, last = log[-1]
    return last.mode is DriveMode.FINISHED and abs(truth.s_m - goal_after_turn(world)) < 3.0 and not any(t.collision for t, _ in log)


# -- S7: standing at the mouth -------------------------------------------------------------


def test_cells_between_the_rings_keep_their_evidence_while_the_car_stands():
    """S7: standing still, the LiDAR's rings hit the same cells every sweep; the cells
    between them were seen while the car moved and keep that, instead of decaying to
    unknown (the verified distance fell from 22 m to 0 after 10 s at the mouth)."""
    grid = TerrainGrid(TerrainConfig())
    pose = Pose2D(0.0, 0.0, 0.0)

    def sweep(xs):
        xyz = np.column_stack((xs, np.zeros(len(xs)), np.zeros(len(xs))))
        return ClassifiedSweep(xyz, np.zeros(len(xs)), np.full(len(xs), PointClass.ROAD, dtype=np.int8), np.zeros(0), np.zeros(0))

    grid.add(sweep(np.array([10.0, 20.0])), pose, 0.0)  # both seen while moving
    for k in range(1, 201):  # 10 s standing: only the cell at 10 m is hit again
        grid.add(sweep(np.array([10.0])), pose, k * DT, stationary=True)
    far = grid.cells(np.array([[20.0, 0.0]]))[0][0]
    near = grid.cells(np.array([[10.0, 0.0]]))[0][0]
    assert grid.road[far[0], far[1]] == 1.0  # kept as it was
    assert 1.0 < grid.road[near[0], near[1]] < 1.0 / (1.0 - math.exp(-DT / 3.0)) + 1.0  # re-hit and decaying: bounded
    grid.add(sweep(np.array([10.0])), pose, 10.0 + 3.0, stationary=False)  # moving: everything decays
    assert grid.road[far[0], far[1]] < 0.4


def test_a_long_pause_before_the_mouth_does_not_leave_the_car_stuck():
    """S7: held on the brake for 12 s just before the junction mouth, the car went on to
    commit, saw its verified distance shrink to 0 and stood there for 55 s."""
    world, _ = junction_world(-1)
    route = world.route(goal_after_turn(world))
    state = dict(start=None)
    apply = world.apply

    def held(command):
        if state["start"] is not None and world.time_s - state["start"] < 12.0:
            command = ControlCommand(0.0, 0.0, 1.0)
        apply(command)

    def per_tick(k, w, driver, step):
        if state["start"] is None and w.state.x >= 44.0:
            state["start"] = w.time_s

    world.apply = held
    log = drive(world, route, seconds=60, per_tick=per_tick)
    assert state["start"] is not None and finished(world, log)


def test_a_walker_on_the_exit_lane_is_waited_for_and_the_turn_then_completes():
    """S7: a walker on the exit lane until the car has stood 3 s at the mouth (8 m kerb
    fillet). The car waits, then turns: it was left at the mouth for good."""
    walker = (58.25, -10.0, 0.0, 0.25, 0.25, 1.7)
    world, _ = junction_world(-1, fillet_m=8.0, extra_boxes=(walker,))
    state = dict(since=None, removed=False)

    def per_tick(k, w, driver, step):
        if state["removed"]:
            return
        if w.state.speed < 0.01 and w.state.x > 40.0:
            state["since"] = w.time_s if state["since"] is None else state["since"]
            if w.time_s - state["since"] >= 3.0:
                w.boxes = [b for b in w.boxes if b != walker]
                state["removed"] = True
        else:
            state["since"] = None

    log = drive(world, world.route(goal_after_turn(world)), seconds=80, per_tick=per_tick)
    assert state["removed"] and finished(world, log)


# -- the arc and the tube ------------------------------------------------------------------


def test_an_arc_begun_just_past_its_tangent_point_ends_on_the_exit_line():
    """The tangent point just behind the car (up to 1.5 m) was clamped to the car without
    moving the arc, and the exit straight lay up to 1.5 m off the measured line."""
    for theta in (-math.pi / 2, math.pi / 2):
        exit_point = (10.0, 12.0 * math.copysign(1.0, theta))
        for radius in (9.5, 10.5, 11.5):
            points, arc_start, _ = _arc_path(0.0, exit_point, theta, radius, 1.0, 15.0)
            rel = points[-1] - np.array(exit_point)
            assert abs(-rel[0] * math.sin(theta) + rel[1] * math.cos(theta)) < 1e-6
            assert arc_start >= 0.0 and np.hypot(*points[0]) < 0.15  # entered where it passes the car


def test_the_tube_widens_only_as_the_bend_takes_hold():
    """The corner over a point belongs to the rear axle a car's length behind it: where a
    bend begins the body has not swung out. The full widening from the bend's first metre
    put the kerb beside a left turn's arc start in the tube as a low obstacle, whose held
    region, on the kerb, never saw ground."""
    straight = np.column_stack((np.arange(0.0, 10.01, 0.5), np.zeros(21)))
    th = np.linspace(0.0, math.pi / 2, 40)[1:]
    arc = np.column_stack((10.0 + 10.0 * np.sin(th), 10.0 * (1 - np.cos(th))))
    line = np.vstack((straight, arc))
    right, _ = tube_bounds(line, np.array([5.0, 10.5, 16.0]), 1.4, BODY)
    full = BODY.front_m**2 * 0.1 / 2
    assert right[0] == -1.4  # straight
    assert -1.4 - right[1] < 0.3 * full  # half a metre into the bend
    assert -1.4 - right[2] > 0.8 * full  # a car's length in


# -- S5: oncoming traffic on a left turn ---------------------------------------------------


def _gate_grid(occupied_xy=()):
    grid = TerrainGrid(TerrainConfig())
    grid.time_s = 10.0
    for x, y in occupied_xy:
        ij = grid.cells(np.array([[x, y]]))[0][0]
        grid.obstacle[ij[0], ij[1]] = 3.0
        grid.seen[ij[0], ij[1]] = 10.0
    return grid


def _left_turn_path():
    entry = np.column_stack((np.arange(0.0, 6.0, 0.5), np.zeros(12)))
    th = np.linspace(0.0, math.pi / 2, 30)[1:]
    arc = np.column_stack((6.0 + 10.0 * np.sin(th), 10.0 * (1 - np.cos(th))))
    out = np.column_stack((np.full(20, 16.0), 10.0 + np.arange(0.5, 10.5, 0.5)))
    return np.vstack((entry, arc, out))


def test_the_gate_sees_the_opposing_lane_and_what_is_in_it():
    c = JunctionConfig()
    frame = Pose2D(0.0, 0.0, 0.0)  # the lane centre at the car, along the lane
    path = _left_turn_path()
    gate = oncoming_gate(_gate_grid(), frame, path, frame, BODY, c, 0.0, 2.0)
    assert gate is not None and gate.clear and gate.band == (1.75, 5.25)
    assert 0.0 < gate.entry_m < gate.exit_m < 20.0 and gate.required_s > 5.0
    car = [(30.0 + dx, 3.5 + dy) for dx in (0.0, 1.0, 2.0) for dy in (-0.5, 0.5)]  # a car in the opposing lane 15 m on
    blocked = oncoming_gate(_gate_grid(car), frame, path, frame, BODY, c, 0.0, 2.0)
    assert not blocked.clear and "occupied" in blocked.note
    excused = oncoming_gate(_gate_grid(car), frame, path, frame, BODY, c, 0.0, 2.0, excused=(((31.0, 3.5), 4.5, 2.0, 0.0),))
    assert excused.clear  # the stack knows it is parked
    beside = oncoming_gate(_gate_grid([(30.0, -1.0), (31.0, -1.0)]), frame, path, frame, BODY, c, 0.0, 2.0)
    assert beside.clear  # in the car's own lane: not oncoming traffic


def test_a_left_turn_waits_for_oncoming_traffic():
    """S5: an oncoming car at 8 m/s, 40 m out when the car reaches the junction: the car
    entered with a 13 m (1.6 s) gap, stopped in the opposing lane and was hit. It now
    waits and enters behind it."""
    world, _ = junction_world(1)
    route = world.route(goal_after_turn(world))
    length, width = 4.2, 1.8
    world.boxes.append((500.0, 5.25, math.pi, length / 2, width / 2, 1.5))
    state = dict(x=None, entered_gap=None)

    def per_tick(k, w, driver, step):
        if state["x"] is None and w.state.x >= 45.0:
            state["x"] = 85.0
        if state["x"] is None:
            return
        corners = w._body_corners()
        if state["entered_gap"] is None and corners[:, 1].max() > 3.5:
            state["entered_gap"] = (state["x"] - length / 2) - corners[:, 0].max()
        state["x"] -= 8.0 * w.dt_s
        w.boxes[-1] = (state["x"], 5.25, math.pi, length / 2, width / 2, 1.5)

    log = drive(world, route, seconds=70, per_tick=per_tick)
    assert not any(t.collision for t, _ in log)
    assert state["entered_gap"] is not None and state["entered_gap"] < 0.0  # it went behind the oncoming car
    assert any("waiting" in (s.diagnostics.get("turn_gate") or "") for _, s in log)
    assert finished(world, log)


# -- S6: a divided exit road ---------------------------------------------------------------


def test_a_divided_exit_road_is_entered_on_the_near_carriageway():
    """S6: with the route's exit point 4 m out, the exit road nearest it was the opposing
    carriageway beyond a 1 m median, and a 4.6 m refinement sent the car 20 m the wrong
    way. The median is skipped (road beyond a raised strip) and refinements are capped."""
    side_road = SideRoad(60.0, -1, width_m=8.0, fillet_m=10.0)  # carriageways [56, 59.5] and [60.5, 64]
    median = (60.0, -45.0, math.pi / 2, 35.0, 0.5, 0.15)
    world, _ = junction_world(-1, side_road=side_road, extra_boxes=(median,))
    route = world.route(goal_after_turn(world))
    plane, points = world.plane, []
    for p in route.points:
        if p.command is Command.LANEFOLLOW and points and points[-1].command is Command.RIGHT:
            x, y = plane.to_local(p.latitude_deg, p.longitude_deg)
            p = RoutePoint(*plane.to_geodetic(x + 4.0, y), p.command)
        points.append(p)
    log = drive(world, SatNavRoute(route.name, tuple(points)), seconds=60)
    assert finished(world, log)
    on_exit = [t for t, _ in log if t.s_m > goal_after_turn(world) - 20.0]
    assert on_exit and max(abs(t.lane_offset_m) for t in on_exit) < 0.5  # the near carriageway's lane


def test_labels_used_by_the_gate_are_the_grids():
    # The gate counts OCCUPIED cells only: kerbs and pavement in the region are not traffic.
    grid = _gate_grid()
    ij = grid.cells(np.array([[30.0, 4.0]]))[0][0]
    grid.raised[ij[0], ij[1]] = 5.0
    grid.seen[ij[0], ij[1]] = 10.0
    grid.seen_by_class[PointClass.RAISED][ij[0], ij[1]] = 10.0
    assert grid.labels(np.array([[30.0, 4.0]]), max_age_s=0.3)[0] == CellLabel.BARRIER
    frame = Pose2D(0.0, 0.0, 0.0)
    assert oncoming_gate(grid, frame, _left_turn_path(), frame, BODY, JunctionConfig(), 0.0, 2.0).clear


# -- Live left turns on PC Temp (Town03, 36b49ef and dd9344c): held at the mouth for good ----


def test_evidence_that_comes_and_goes_while_the_car_stands_still_decays():
    """A car waiting to turn left: an oncoming car's wheels passing in the opposing lane
    left raised hits there, kept as kerb for the whole standstill (static evidence keeps
    while the car stands, S7), and no turn ever cleared them. Evidence last added during
    the standstill in a cell no longer hit came from something that moved through: it
    decays as if the car were moving. What was seen before the standstill keeps (S7)."""
    grid = TerrainGrid(TerrainConfig())
    pose = Pose2D(0.0, 0.0, 0.0)

    def sweep(points):
        xyz = np.array([[x, y, 0.0] for x, y, _ in points], dtype=float)
        cls = np.array([c for *_, c in points], dtype=np.int8)
        return ClassifiedSweep(xyz, np.zeros(len(points)), cls, np.zeros(0), np.zeros(0))

    kerb, ring, lane = (15.0, -6.0, PointClass.RAISED), (10.0, 0.0, PointClass.ROAD), (15.0, 3.0, PointClass.ROAD)
    for k in range(10):  # moving: the kerb and the opposing lane's road are seen
        grid.add(sweep([kerb, ring, lane]), pose, k * DT)
    t = 10 * DT
    for k in range(40):  # standing; a passing car's wheel is in a cell 15 m ahead, 3 m left, for 0.3 s
        wheel = [(15.0, 3.0, PointClass.RAISED)] if k < 6 else []
        grid.add(sweep([ring] + wheel), pose, t, stationary=True)
        t += DT
    for _ in range(200):  # 10 s more standing: only the ring is hit
        grid.add(sweep([ring]), pose, t, stationary=True)
        t += DT
    label = lambda x, y: grid.labels(np.array([[x, y]]), max_age_s=30.0)[0]  # noqa: E731
    assert label(15.0, -6.0) == CellLabel.BARRIER  # seen before the standstill: kept
    assert label(15.0, 3.0) != CellLabel.BARRIER  # came and went during it: decayed
    i, j = grid.cells(np.array([[15.0, 3.0]]))[0][0]
    assert grid.raised[i, j] < TerrainConfig().barrier_min_evidence
    # A cell first observed during the standstill (the whole grid at start-up) keeps, as before.
    fresh = TerrainGrid(TerrainConfig())
    for k in range(6):
        fresh.add(sweep([(15.0, 3.0, PointClass.RAISED)]), pose, k * DT, stationary=True)
    for k in range(6, 200):
        fresh.add(sweep([ring]), pose, k * DT, stationary=True)
    i, j = fresh.cells(np.array([[15.0, 3.0]]))[0][0]
    assert fresh.raised[i, j] == pytest.approx(6.0, rel=0.05)


def test_the_exit_kerb_measured_on_the_approach_is_kept_while_the_car_stands_at_the_mouth():
    """From a left turn's mouth only the exit kerb's first three stations are in view: the
    car forgot the rest 2 s after it stopped there and never measured the exit again
    (held 25 s, "exit lane not measured"). Stations now age only while the car moves; and
    three stations a metre apart count as a 2 m span when re-projection reads 1.99 m."""
    from mapless.corridor import CorridorEstimator
    from mapless.junction import JunctionTurn

    config = JunctionConfig()
    turn = JunctionTurn(config, BODY, CorridorEstimator(CorridorConfig(keep_right_m=1.75)))
    grid = TerrainGrid(TerrainConfig())  # nothing seen now: only the memory can measure the exit
    pose = Pose2D(0.0, 0.0, 0.0)
    probe = Pose2D(20.0, 10.0, math.pi / 2)

    def remember(xs):
        turn._kerb = [(turn._now, turn._moving_s, probe.to_world(np.array([[x, -6.4] for x in xs])))]

    def source(now_s, standing_s=0.0):
        turn._now, turn.standing_s = now_s, standing_s
        return turn.exit_line(grid, pose, (20.0, 10.0), math.pi / 2)[1]

    stations = [11.0, 12.0, 13.0, 15.0, 18.0]
    source(0.0)
    remember(stations)
    assert source(1.0) == "kerb"
    assert source(1.5, standing_s=0.05) == "kerb"
    assert source(30.0, standing_s=28.5) == "kerb"  # stood still since: kept
    assert source(30.8) == "kerb"  # moving off: 1.8 s of motion since
    assert source(31.5) == "kerb"  # 2.5 s of motion: past memory_s, read as the fallback
    assert turn.exit_line(grid, pose, (20.0, 10.0), math.pi / 2)[3]["memory"] == "fallback"
    assert source(33.2) == "route"  # over 4 s of motion: forgotten, replaced from nearer
    remember([11.0, 12.0, 12.99])
    assert source(33.5) == "kerb"  # 1.99 m: three stations
    remember([11.0, 12.0])
    assert source(34.0) == "route"  # two stations are not a line


def test_the_measured_exit_is_kept_when_the_near_stations_go_out_of_view():
    """PC 1, Town03, r10 and r11 left turns, 14.2 s: approaching the mouth, the exit kerb's
    near stations went out of view and aged out of the 2 s memory; the stations still in
    view spanned 1.06 m, no line, and the measured exit was lost for 0.25 s: the car braked
    hard (0.95) for the mouth from 2 m/s. The stations kept exit_fallback_s longer now make
    the line when the last memory_s's do not; the recent ones still come first."""
    from mapless.corridor import CorridorEstimator
    from mapless.junction import JunctionTurn

    turn = JunctionTurn(JunctionConfig(), BODY, CorridorEstimator(CorridorConfig(keep_right_m=1.75)))
    grid = TerrainGrid(TerrainConfig())
    pose = Pose2D(0.0, 0.0, 0.0)
    probe = Pose2D(20.0, 10.0, math.pi / 2)

    def measure(now_s):
        turn._now, turn.standing_s = now_s, 0.0
        return turn.exit_line(grid, pose, (20.0, 10.0), math.pi / 2)

    def remember(xs):
        turn._kerb.append((turn._now, turn._moving_s, probe.to_world(np.array([[x, -6.4] for x in xs]))))

    measure(0.0)
    remember([5.0, 6.0, 7.0, 8.0, 11.0, 12.0])  # the approach's view: near and far stations
    _, source, _, fit = measure(1.0)
    assert source == "kerb" and "memory" not in fit
    measure(2.2)
    remember([11.0, 11.5, 12.06])  # only these in view now: 1.06 m
    _, source, _, fit = measure(2.4)  # the near ones are 2.4 s of motion old
    assert source == "kerb" and fit["memory"] == "fallback"
    remember([13.0, 14.0])  # a line from the last memory_s again: only its stations are read
    _, source, _, fit = measure(2.5)
    assert source == "kerb" and "memory" not in fit and fit["stations"] == 5
    measure(6.5)  # 4 s of motion on with nothing new: all of it forgotten
    turn._kerb = [k for k in turn._kerb if turn._moving_s - k[1] <= 4.0]
    assert measure(6.6)[1] == "route"


# -- an object on the latched path (PC1, Town03, r8_left_oncoming, b15b76f) ---------------


def test_an_object_on_the_latched_turn_is_waited_for_not_planned_around():
    """The oncoming car a left turn had yielded to swept through the latched path while the
    car stood; the latch was dropped at once, the re-plan from where the car stood (past
    the 14 m arc's tangent point) could only take 11.5 m, and the tighter arc cut 0.21 m
    into the exit road's far lane. An object (OCCUPIED, not a kerb) on the latched path is
    now waited for, the car held short of it, and the turn goes on on the same arc."""
    world, _ = junction_world(1)
    route = world.route(goal_after_turn(world))
    state = dict(box=None, placed_s=None, removed=False, radius=None, radii=set())

    def per_tick(k, w, driver, step):
        if driver.turn.committed and step.diagnostics.get("turn_radius_m"):
            state["radii"].add(step.diagnostics["turn_radius_m"])
        if state["removed"]:
            return
        if state["box"] is None and driver.turn.committed:
            # 6 m ahead along the heading, 1 m towards the turn: on the arc's swept path.
            h = w.state.yaw
            x = w.state.x + 6.0 * math.cos(h) - 1.0 * math.sin(h)
            y = w.state.y + 6.0 * math.sin(h) + 1.0 * math.cos(h)
            state["box"], state["placed_s"] = (x, y, h, 0.5, 0.5, 1.5), w.time_s
            state["radius"] = step.diagnostics.get("turn_radius_m")
            w.boxes.append(state["box"])
        elif state["box"] is not None and w.time_s - state["placed_s"] >= 2.5:
            w.boxes = [b for b in w.boxes if b != state["box"]]
            state["removed"] = True

    log = drive(world, route, seconds=80, per_tick=per_tick)
    assert state["removed"] and finished(world, log)
    notes = [s.diagnostics.get("turn") or "" for _, s in log]
    assert any(n.startswith("an object on the latched path") for n in notes)
    assert not any(n.startswith("latched path blocked") for n in notes)
    assert state["radius"] is not None and state["radii"] == {state["radius"]}  # one arc, committed to the end
    # A kerb is not waited for: the swept check says which it met.
    from mapless.junction import _swept_check  # noqa: PLC0415

    path = np.column_stack((np.arange(0.0, 20.0, 1.0), np.zeros(20)))
    for label in ("occupied", "barrier"):
        grid = _gate_grid([(6.0 + 0.25 * i, 0.25 * j) for i in range(3) for j in range(-2, 3)] if label == "occupied" else ())
        if label == "barrier":
            for i in range(3):
                for j in range(-2, 3):
                    ij = grid.cells(np.array([[6.0 + 0.25 * i, 0.25 * j]]))[0][0]
                    grid.raised[ij[0], ij[1]] = 5.0
                    grid.seen[ij[0], ij[1]] = 10.0
                    grid.seen_by_class[PointClass.RAISED][ij[0], ij[1]] = 10.0
        why = {}
        feasible, verified, _ = _swept_check(grid, Pose2D(0.0, 0.0, 0.0), path, BODY, JunctionConfig(), 1.0, margin_m=0.05, hazard_tolerated_beyond_m=8.0, tolerate_occupied=True, why=why)
        assert why.get("label") == label and why["near"], (label, why)
        assert feasible == (label == "occupied")


def test_f8_02_a_stale_oncoming_verdict_excuses_nothing():
    """Eighth review, F8-02: the stack's oncoming verdict (excused footprints, clear) was
    left as it was on a tick without perception, and the gate went on excusing the cells of
    a car that might have started moving meanwhile. Stamped with its time, it is used only
    while fresh (computed the tick before); older, its excuses lapse (every occupied cell
    counts again) and a "not clear" stands."""
    from mapless.driver import resolve_oncoming_override  # noqa: PLC0415

    c = JunctionConfig()
    frame = Pose2D(0.0, 0.0, 0.0)
    path = _left_turn_path()
    parked = (((31.0, 3.5), 4.5, 2.0, 0.0),)
    verdict = dict(clear=True, excused=parked, note="oncoming lane: 1 tracks excused", time_s=10.0, generation=0)
    # The car stands at 31 m, its cells all on the excused footprint: while the verdict is
    # fresh (read a tick old) they are excused; two ticks without perception later, when it
    # may have begun to pull away and the tracks no longer say, they are not: one missed
    # tick is enough.
    standing = _gate_grid([(30.0 + dx, 3.5 + dy) for dx in (0.0, 1.0, 2.0) for dy in (-0.5, 0.5)])
    fresh = resolve_oncoming_override(verdict, 10.05, c.oncoming_override_max_age_s)
    assert not fresh["stale"] and fresh["excused"] == parked
    assert oncoming_gate(standing, frame, path, frame, BODY, c, 0.0, 2.0, excused=fresh["excused"]).clear
    stale = resolve_oncoming_override(verdict, 10.10, c.oncoming_override_max_age_s)
    assert stale["stale"] and stale["excused"] == () and stale["age_s"] == pytest.approx(0.10)
    assert not oncoming_gate(standing, frame, path, frame, BODY, c, 0.0, 2.0, excused=stale["excused"]).clear
    # A stale "not clear" is the cautious reading and stands; no verdict, no time: as given.
    waiting = resolve_oncoming_override(dict(verdict, clear=False), 11.0, c.oncoming_override_max_age_s)
    assert waiting["stale"] and waiting["clear"] is False
    assert resolve_oncoming_override(None, 11.0, 0.1) is None
    assert resolve_oncoming_override(dict(clear=True, excused=parked), 11.0, 0.1)["excused"] == parked


# -- a refinement taken from a wrong far reading (PC1, Town03, r9_left_parked, b4280cd) ----
def test_a_refinement_may_return_to_the_first_commits_line_beyond_the_cap():
    """The first commit's line was right; a far reading 1.1 m off was taken; the right line,
    read from nearer, then lay +1.5 m from the latched and was refused as "another road" for
    good. A line within the cap of the first commit's is a correction back; one beyond both
    is still refused."""
    from mapless.corridor import CorridorEstimator
    from mapless.junction import JunctionTurn

    config = JunctionConfig()
    turn = JunctionTurn(config, BODY, CorridorEstimator(CorridorConfig(keep_right_m=1.75)))
    grid = TerrainGrid(TerrainConfig())
    pose = Pose2D(0.0, 0.0, 0.0)
    heading = math.pi / 2  # the exit runs along +y; its lateral offsets are along x
    turn._entry = (pose, 0.0, 10.0)
    turn._probe_odo = np.array([20.0, 10.0])
    turn._first_exit_odo = np.array([20.0, 10.0])
    turn._exit_odo = (np.array([21.2, 10.0]), heading)  # latched on the wrong reading
    readings = []
    turn.exit_line = lambda *a, **k: (readings[-1], "kerb", heading, dict(range_m=20.0))

    def refine(x, now_s):
        readings.append((x, 10.0))
        turn._now = now_s
        turn.replan_same_arc(grid, pose, now_s, min_shift_m=0.5)
        return turn.refine_note

    note = refine(19.6, 1.0)  # 1.6 m from the latched, 0.4 m from the first commit's
    assert "not taken" not in note and "from the first commit's" in note
    assert "for 0.0 s" in note
    note = refine(19.6, 1.0 + config.refine_persist_s + 0.05)
    assert "re-anchored" in note or "fresh arc" in note  # held: taken
    turn._refine_candidate = None
    assert "not taken" in refine(18.0, 2.0)  # 2.0 m from the first commit's and further from the latched: refused
