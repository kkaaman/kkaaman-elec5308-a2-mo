"""Stage 1, reliable stopping: regressions for the findings of the full review of 73ae759.

Planner holds:
- a farther obstacle seen while the held one is missed does not move the stop (S2);
- ground beside a narrow, or off-centre, held obstacle does not clear it (S1);
- the stop is worked out from the held obstacle on each tick's path, not from a point kept
  on an old path;
- holds survive a standstill frame rebuild (S3).

Detection:
- the tube widens on the outside of a bend, where the front corner sweeps;
- a few raised points far ahead are an obstacle (a brick under either wheel, S4);
- a person just ahead of the bumper is not masked as the car itself.

Corridor:
- a raised strip with road beyond it does not end the road's support;
- short kerb evidence that bends is fitted with a quadratic;
- standing still, the line stays within the refit tolerance of where it stopped.

Closed loop in the synthetic road: a brick under the left wheel, a pole on the outside of a
12 m bend, a held pedestrian whose returns go missing, and an IMU outage while held.
"""

import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from mapless.corridor import Corridor, CorridorConfig, CorridorEstimator, _shows_bend
from mapless.driver import DriveMode, DriverConfig, MaplessDriver
from mapless.ego_motion import Pose2D
from mapless.frame import LidarSweep, SensorState, SensorStatus
from mapless.obstacles import (
    ObstacleConfig,
    PathObservation,
    PathObstacle,
    observe_path,
    path_obstacle,
    tube_bounds,
)
from mapless.planner import LanePlanner, PlannerConfig
from mapless.sensor_suite import load_sensor_suite
from mapless.terrain import (
    ClassifiedSweep,
    PointClass,
    TerrainConfig,
    TerrainGrid,
    VehicleBody,
    classify_sweep,
)
from navigation_control.local import LocalControlConfig
from navigation_control.types import ControllerConfig
from sim_harness.mapless_road import Arc, KerbLine, MaplessRoadWorld, ParkedCar, RoadScene, Straight

ROOT = Path(__file__).resolve().parents[2]
SUITE = load_sensor_suite(ROOT / "configs" / "sensor_suites" / "mapless_v2.json")
BODY = VehicleBody()
DT = 0.05
TUBE_HALF = BODY.half_width_m + ObstacleConfig().margin_m


# -- helpers --------------------------------------------------------------------------
def line_corridor(centre_body, time_s=0.0, pose=None):
    """A valid corridor along ``centre_body`` (vehicle frame) seen from ``pose``."""
    pose = pose or Pose2D(0.0, 0.0, 0.0)
    centre_body = np.asarray(centre_body, dtype=float)
    stations = centre_body[:, 0]
    return Corridor(
        time_s=time_s,
        pose=pose,
        centre_body=centre_body,
        centre_odometry=pose.to_world(centre_body),
        valid_length_m=float(stations[-1]),
        coefficients=tuple(np.polyfit(stations, centre_body[:, 1], 2)),
        station_x_m=stations,
        reference_y_m=centre_body[:, 1],
        right_edge_m=np.full(len(stations), -1.75),
        left_edge_m=np.full(len(stations), np.nan),
        blocked=np.zeros(len(stations), dtype=bool),
        evidence_body=centre_body,
        evidence_time_s=time_s,
        support_m=float(stations[-1]),
    )


def straight(length=30.0):
    xs = np.arange(-2.0, length + 1e-9, 0.5)
    return line_corridor(np.column_stack((xs, np.zeros_like(xs))))


def right_bend(radius=12.0, length=14.0):
    """Rear axle at the origin heading +x, bending right (curvature -1/radius)."""
    a = np.arange(-2.0, length + 1e-9, 0.5) / radius
    return line_corridor(np.column_stack((radius * np.sin(a), -radius * (1.0 - np.cos(a)))))


def ground(along=(0.0, 20.0), lateral=(-TUBE_HALF, TUBE_HALF), hole=None, step=0.1):
    """In-tube ground points (along from the rear axle, lateral), minus ``hole`` = (a0, a1, l0, l1)."""
    a, lat = np.meshgrid(np.arange(along[0], along[1], step), np.arange(lateral[0], lateral[1] + 1e-9, step))
    pts = np.column_stack((a.ravel(), lat.ravel()))
    if hole is not None:
        a0, a1, l0, l1 = hole
        pts = pts[~((pts[:, 0] >= a0) & (pts[:, 0] <= a1) & (pts[:, 1] >= l0) & (pts[:, 1] <= l1))]
    return pts


def planner():
    return LanePlanner(PlannerConfig(), BODY, tube_half_m=TUBE_HALF)


def held_stop(obstacle_m):
    return obstacle_m - BODY.front_m - PlannerConfig().stop_gap_m


# -- planner holds ----------------------------------------------------------------------
def test_a_farther_obstacle_seen_while_the_held_one_is_missed_does_not_move_the_stop():
    corridor, p = straight(), planner()
    p.plan(corridor, PathObservation(PathObstacle(7.1, 0.0, 30, -0.2, 0.2), 7.1), None, 10.0)
    p.plan(corridor, PathObservation(None, 7.0), None, 10.05)
    plan = p.plan(corridor, PathObservation(PathObstacle(22.0, 0.3, 40, 0.1, 0.5), 7.0), None, 10.10)
    assert abs(plan.stop_distance_m - held_stop(7.1)) < 0.05
    assert plan.stop_reason == "obstacle_hold"
    plan = p.plan(corridor, PathObservation(None, 7.0), None, 10.15)
    assert abs(plan.stop_distance_m - held_stop(7.1)) < 0.05


def test_ground_beside_a_narrow_held_obstacle_does_not_clear_it():
    corridor, p = straight(), planner()
    p.plan(corridor, PathObservation(PathObstacle(7.1, 0.0, 30, -0.25, 0.25), 7.1), None, 0.0)
    # Missed from now on, but still standing: its own region (and its shadow) stays unseen,
    # while the road beside it and beyond is in view.
    shadow = ground(hole=(7.0, 12.0, -0.45, 0.45))
    for k in range(1, 60):
        plan = p.plan(corridor, PathObservation(None, 15.0, shadow), None, k * DT)
        assert plan.stop_reason == "obstacle_hold" and plan.hold_state == "unknown"
    # Gone: its region is seen, and after resume_clear_s of that the stop is released.
    released = None
    for k in range(60, 120):
        plan = p.plan(corridor, PathObservation(None, 15.0, ground()), None, k * DT)
        if plan.stop_reason == "" and released is None:
            released = k * DT
    assert released is not None and released - 60 * DT >= PlannerConfig().resume_clear_s - 1e-9


def test_ground_over_one_half_does_not_clear_an_obstacle_over_the_other_half():
    corridor, p = straight(), planner()
    p.plan(corridor, PathObservation(PathObstacle(7.1, -1.3, 80, -1.4, -0.2), 7.1), None, 0.0)
    left_only = ground(hole=(7.0, 12.0, -TUBE_HALF, -0.1))
    for k in range(1, 60):
        plan = p.plan(corridor, PathObservation(None, 15.0, left_only), None, k * DT)
        assert plan.stop_reason == "obstacle_hold"


def test_the_stop_follows_the_held_obstacle_when_the_path_changes_shape():
    bend, p = right_bend(), planner()
    p.plan(bend, PathObservation(PathObstacle(7.0, 1.6, 10, 1.5, 1.7), 7.0), None, 0.0)
    # The same scene refitted as a straight line (the old failure): the pole now projects
    # about 2 m left of the line, and its stop must stay in front of the car.
    plan = p.plan(straight(), PathObservation(None, 5.0, ground(along=(0.0, 5.0))), None, DT)
    assert plan.stop_reason == "obstacle_hold"
    assert plan.stop_distance_m > 0.3


def test_a_held_obstacle_past_the_lines_end_keeps_its_place():
    # Live stop run (27 Oct, PC 2): unseen holds beyond a shorter line clamped to its end,
    # and the clamped "lateral" (the distance to the end point) would drop a far one.
    p = planner()
    p.plan(straight(30.0), PathObservation(PathObstacle(15.0, 0.2, 30, 0.0, 0.4), 15.0), None, 0.0)
    plan = p.plan(straight(5.0), PathObservation(None, 5.0, ground(along=(0.0, 5.0))), None, DT)
    assert plan.stop_reason == "obstacle_hold"
    assert abs(plan.stop_distance_m - held_stop(15.0)) < 0.05


def test_holds_survive_a_standstill_frame_rebuild_and_not_a_moving_one():
    corridor, p = straight(), planner()
    p.plan(corridor, PathObservation(PathObstacle(7.1, 0.0, 30, -0.2, 0.2), 7.1), None, 0.0)
    p.reset(keep_holds=True)
    plan = p.plan(corridor, PathObservation(None, 7.0), None, DT)
    assert plan.stop_reason == "obstacle_hold" and plan.hold_state == "unknown"
    p.reset()
    assert not p.holding


# -- detection --------------------------------------------------------------------------
def test_the_tube_widens_on_the_outside_of_a_bend():
    bend = right_bend()
    right, left = tube_bounds(bend.centre_body, np.array([7.0]), TUBE_HALF, BODY)
    assert left[0] > TUBE_HALF + 0.4 and right[0] == -TUBE_HALF  # outside (left) widened only
    a = 9.0 / 12.0  # 7 m ahead of the rear axle (the line starts 2 m behind it)
    foot = np.array([12.0 * math.sin(a), -12.0 * (1.0 - math.cos(a))])
    normal = np.array([math.sin(a), math.cos(a)])  # +left of the tangent (cos a, -sin a)
    pole = foot + 1.6 * normal + np.random.default_rng(0).normal(0.0, 0.02, (30, 2))
    assert path_obstacle(pole, bend, BODY) is not None
    inner = foot - 1.6 * normal + np.random.default_rng(1).normal(0.0, 0.02, (30, 2))
    assert path_obstacle(inner, bend, BODY) is None


def raised_sweep(points):
    pts = np.asarray(points, dtype=float)
    xyz = np.column_stack((pts, np.full(len(pts), 0.2)))
    n = len(xyz)
    return ClassifiedSweep(xyz, np.full(n, 0.2), np.full(n, PointClass.RAISED), np.zeros(0), np.zeros(0))


def test_a_few_raised_points_far_ahead_are_an_obstacle_under_either_wheel():
    # The 32-channel roof LiDAR puts 3 raised points on a 0.25 m brick 11 m from the axle.
    corridor = straight()
    for lateral in (0.0, 0.85, -0.85):
        brick = [(11.4, lateral - 0.1), (11.5, lateral), (11.6, lateral + 0.1)]
        seen = observe_path(raised_sweep(brick), corridor, BODY).obstacle
        assert seen is not None and abs(seen.distance_m - 11.4) < 0.05, lateral
    # Two stray points are not, nor a kerb-high line just outside the body.
    assert observe_path(raised_sweep([(11.4, 0.0), (11.5, 0.1)]), corridor, BODY).obstacle is None
    kerb = [(x, -1.2) for x in np.arange(4.0, 20.0, 0.25)]
    assert observe_path(raised_sweep(kerb), corridor, BODY).obstacle is None


def test_a_person_just_ahead_of_the_bumper_is_not_masked_as_the_car():
    xs, ys = np.meshgrid(np.arange(-15, 40, 0.3), np.arange(-8, 8, 0.3))
    road = np.column_stack((xs.ravel(), ys.ravel(), np.zeros(xs.size)))
    person = np.array([[BODY.front_m + 0.12, dy, z] for dy in (-0.15, 0.0, 0.15) for z in np.arange(0.4, 1.7, 0.1)])
    sweep = classify_sweep(np.vstack((road, person)), BODY, TerrainConfig())
    kept = sweep.of(PointClass.OBSTACLE)
    assert np.sum(np.abs(kept[:, 0] - (BODY.front_m + 0.12)) < 0.01) == len(person)


# -- corridor ---------------------------------------------------------------------------
def test_a_raised_strip_with_road_beyond_does_not_end_the_support():
    grid, estimator = TerrainGrid(), CorridorEstimator(CorridorConfig(keep_right_m=1.75))
    xs, ys = np.meshgrid(np.arange(-5.0, 30.0, 0.2), np.arange(-1.6, 1.6, 0.2))
    pts = np.column_stack((xs.ravel(), ys.ravel()))
    strip = (pts[:, 0] >= 10.0) & (pts[:, 0] <= 10.4)  # a 0.4 m kerbstone across the lane
    cls = np.where(strip, PointClass.RAISED, PointClass.ROAD)
    xyz = np.column_stack((pts, np.where(strip, 0.2, 0.0)))
    sweep = ClassifiedSweep(xyz, xyz[:, 2], cls, np.zeros(0), np.zeros(0))
    pose = Pose2D(0.0, 0.0, 0.0)
    for k in range(5):
        grid.add(sweep, pose, k * DT)
    line = np.column_stack((np.arange(0.0, 25.0, 0.5), np.zeros(50)))
    assert estimator._support(grid, pose, line) > 20.0


def test_short_kerb_evidence_that_bends_is_fitted_with_a_quadratic():
    x = np.array([2.0, 3.0, 4.0, 7.9, 8.9, 9.6])  # the stations left on a 12 m bend
    assert _shows_bend(x, -x**2 / 24.0, CorridorConfig().bend_min_rms_m)
    noisy = np.random.default_rng(2).normal(0.0, 0.05, len(x))
    assert not _shows_bend(x, 0.1 * x + noisy, CorridorConfig().bend_min_rms_m)


def test_standing_still_the_corridor_line_is_kept():
    kerb = KerbLine([Straight(25), Arc(-10.25, math.pi / 2), Straight(100)])
    world = MaplessRoadWorld(RoadScene(kerb), SUITE, noise=False)
    grid, estimator = TerrainGrid(), CorridorEstimator(CorridorConfig(keep_right_m=1.75))
    for k in range(1, 211):  # slide 2 m/s along the lane centre, 6 m into the bend
        frame = world.frame()
        pose = Pose2D(world.state.x, world.state.y, world.state.yaw)
        grid.add(classify_sweep(frame.lidar("roof").xyz, BODY, TerrainConfig()), pose, world.time_s)
        corridor = estimator.update(grid, pose, world.time_s)
        x, y, h = kerb.pose_at(world.start_s_m + 0.1 * k)
        world.state.x, world.state.y, world.state.yaw = x - 1.75 * math.sin(h), y + 1.75 * math.cos(h), h
        world.time_s += DT
    assert corridor.ok
    stopped = estimator.previous.centre_odometry.copy()
    estimator.standing_s = DT
    for _ in range(80):  # 4 s standing: the near kerb ages out of the grid
        frame = world.frame()
        pose = Pose2D(world.state.x, world.state.y, world.state.yaw)
        grid.add(classify_sweep(frame.lidar("roof").xyz, BODY, TerrainConfig()), pose, world.time_s)
        corridor = estimator.update(grid, pose, world.time_s)
        world.time_s += DT
        estimator.standing_s += DT
        assert corridor.ok
        # Within the refit tolerance of the line it stopped on (a straight refit across the
        # corner was 0.5 m off it).
        seen = corridor.pose.to_body(stopped)
        line = corridor.centre_body
        x = np.arange(max(0.0, line[0, 0]), min(seen[-1, 0], line[-1, 0]), 0.5)
        assert len(x) > 10
        gap = np.abs(np.interp(x, line[:, 0], line[:, 1]) - np.interp(x, seen[:, 0], seen[:, 1]))
        assert gap.max() <= CorridorConfig().standing_refit_tol_m + 1e-9


# -- closed loop ------------------------------------------------------------------------
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


def run(scene, seconds, goal_s=120.0, tamper=None):
    world = MaplessRoadWorld(scene, SUITE)
    driver = MaplessDriver(SUITE, world.route(goal_s), DriverConfig(body=BODY, control=control(), corridor=CorridorConfig()), DT)
    log = []
    for k in range(int(seconds / DT)):
        frame = world.frame()
        if tamper is not None:
            frame = tamper(k, frame, world, log)
        step = driver.step(frame)
        log.append((world.truth(), step))
        world.apply(step.command)
    return log


def box_at(kerb, s, lateral, hx, hy, height):
    """A box ``lateral`` left of the lane centre (the lane centre is 1.75 m left of the kerb)."""
    x, y, h = kerb.pose_at(s)
    off = 1.75 + lateral
    return (x - off * math.sin(h), y + off * math.cos(h), h, hx, hy, height)


def held_at(log):
    return next((k for k, (t, s) in enumerate(log) if s.reason.startswith("holding for obstacle") and t.speed_mps == 0), None)


def without_box(frame, world, box):
    """``frame`` with the LiDAR returns of ``box`` removed (a missed detection)."""
    st = world.state
    cx, cy, h, hx, hy, _ = box

    def cut(p):
        c, s = math.cos(st.yaw), math.sin(st.yaw)
        wx, wy = st.x + c * p[:, 0] - s * p[:, 1], st.y + s * p[:, 0] + c * p[:, 1]
        u = (wx - cx) * math.cos(h) + (wy - cy) * math.sin(h)
        v = -(wx - cx) * math.sin(h) + (wy - cy) * math.cos(h)
        return p[~((np.abs(u) < hx + 0.15) & (np.abs(v) < hy + 0.15) & (p[:, 2] > 0.05))]

    sweep = frame.lidars[0]
    return replace(frame, lidars=(LidarSweep(sweep.name, cut(sweep.points).astype(np.float32), sweep.sensor_to_ego),))


def test_closed_loop_a_brick_under_the_left_wheel_is_stopped_for():
    kerb = KerbLine([Straight(250)])
    log = run(RoadScene(kerb, hedges=(box_at(kerb, 45.0, 0.85, 0.2, 0.2, 0.25),)), seconds=28)
    assert not any(t.collision for t, _ in log)
    assert log[-1][0].speed_mps == 0 and log[-1][1].reason.startswith("holding for obstacle")


def test_closed_loop_a_pole_outside_a_tight_bend_is_not_driven_into():
    kerb = KerbLine([Straight(25), Arc(-10.25, math.pi / 2), Straight(100)])
    pole = box_at(kerb, 25 + 10.25 * math.pi / 4, 1.5, 0.1, 0.1, 1.8)
    log = run(RoadScene(kerb, hedges=(pole,)), seconds=30, goal_s=80.0)
    assert not any(t.collision for t, _ in log)
    assert held_at(log) is not None


def test_closed_loop_a_held_pedestrian_whose_returns_go_missing_keeps_the_car_stopped():
    kerb = KerbLine([Straight(250)])
    person = box_at(kerb, 45.0, 0.0, 0.25, 0.25, 1.8)
    state = {"at": None}

    def tamper(k, frame, world, log):
        if state["at"] is None and log:
            state["at"] = held_at(log)
        return frame if state["at"] is None else without_box(frame, world, person)

    log = run(RoadScene(kerb, hedges=(person,)), seconds=24, tamper=tamper)
    at = state["at"]
    assert at is not None and not any(t.collision for t, _ in log)
    assert max(t.s_m for t, _ in log[at:]) - log[at][0].s_m < 0.05


def test_closed_loop_an_obstacle_that_leaves_while_far_ahead_does_not_stop_the_car():
    """Live stack runs on e93154f (PC 2): a walker that crossed out of the lane, and a lead
    that drove on, held the car almost to a stop (0.06 and 0.57 m/s), because one sweep put
    ground in the held 0.5 m strip only within about 6 m. The grid's hit history since the
    obstacle was last seen clears it as the rings sweep over it."""
    kerb = KerbLine([Straight(250)])
    person = box_at(kerb, 45.0, 0.0, 0.25, 0.25, 1.8)
    state = {"gone": None}

    def tamper(k, frame, world, log):
        if state["gone"] is None and log and log[-1][1].diagnostics.get("obstacle_m") is not None and log[-1][1].diagnostics["obstacle_m"] > 15.0:
            state["gone"] = k
            world.boxes = []  # the walker steps out of the lane
        return frame

    log = run(RoadScene(kerb, hedges=(person,)), seconds=30, tamper=tamper)
    gone = state["gone"]
    assert gone is not None and not any(t.collision for t, _ in log)
    after = log[gone:]
    assert min(t.speed_mps for t, _ in after[40:]) > 1.5  # no near stop once under way again
    assert max(t.s_m for t, _ in after) > 50.0


def test_closed_loop_an_imu_outage_while_held_keeps_the_hold():
    car = ParkedCar(45.0, 1.75)
    state = {"at": None}

    def tamper(k, frame, world, log):
        if state["at"] is None and log:
            state["at"] = held_at(log)
        if state["at"] is None:
            return frame
        points = frame.lidars[0].points
        keep = ~((points[:, 0] > 2.0) & (np.abs(points[:, 1]) < 2.5) & (points[:, 2] > 0.25))  # occluded from now on
        sweep = frame.lidars[0]
        frame = replace(frame, lidars=(LidarSweep(sweep.name, points[keep], sweep.sensor_to_ego),))
        if state["at"] + 60 <= k < state["at"] + 66:  # 0.3 s with no IMU, 3 s into the hold
            status = tuple(SensorStatus(s.name, s.kind, SensorState.MISSING) if s.kind == "imu" else s for s in frame.status)
            frame = replace(frame, imu=None, status=status)
        return frame

    log = run(RoadScene(KerbLine([Straight(250)]), parked=(car,)), seconds=24, tamper=tamper)
    at = state["at"]
    assert at is not None and len(log) > at + 100
    assert not any(t.collision for t, _ in log)
    assert max(t.s_m for t, _ in log[at:]) - log[at][0].s_m < 0.05
    assert all(s.mode is not DriveMode.DRIVING for _, s in log[at:])


# -- the first live stack turn (ff1910b): a hold just past a short corridor ----------------


def test_held_regions_are_judged_past_the_lines_end():
    """The tube continued straight past the line's end, for held regions only."""
    xs = np.arange(-2.0, 6.81, 0.5)
    corridor = line_corridor(np.column_stack((xs, np.zeros_like(xs))))
    pts = ground(along=(0.0, 12.0))
    xyz = np.column_stack((pts, np.zeros(len(pts))))
    sweep = ClassifiedSweep(xyz, np.zeros(len(pts)), np.full(len(pts), PointClass.ROAD, dtype=np.int8), np.zeros(0), np.zeros(0))
    obs = observe_path(sweep, corridor, BODY)
    assert obs.clear_to_m <= 7.0  # clearance stays on the line
    assert 10.0 < obs.ground[:, 0].max() <= 6.8 + ObstacleConfig().beyond_line_m + 0.1


def test_a_hold_just_past_a_short_corridor_clears_when_its_region_is_seen_empty():
    """Live: a one-tick detection 16 m out was held; the corridor shrank to 6.8 m (the kerb
    hidden by parked cars, the junction mouth ahead), the hold lay 0.8 m past its end, and
    "clear to 7.0 m of 9.6" could never be met: the car stood there for 45 s."""
    xs = np.arange(-2.0, 6.81, 0.5)
    corridor = line_corridor(np.column_stack((xs, np.zeros_like(xs))))
    p = planner()
    p.plan(straight(17.0), PathObservation(PathObstacle(7.6, 0.0, 6, -1.4, -1.1), 7.6), None, 0.0)  # seen on the long corridor
    shadow = ground(along=(0.0, 12.0), hole=(7.6, 12.0, -1.6, -0.9))  # still there: its strip in shadow
    for k in range(1, 40):
        plan = p.plan(corridor, PathObservation(None, 6.3, shadow), None, k * DT)
    assert plan.stop_reason == "obstacle_hold" and abs(plan.stop_distance_m - held_stop(7.6)) < 0.05
    for k in range(40, 80):  # gone: its strip shows ground, and the line is clear to its end
        plan = p.plan(corridor, PathObservation(None, 6.3, ground(along=(0.0, 12.0))), None, k * DT)
    assert plan.stop_reason != "obstacle_hold"


# -- third review (e93154f): hold identity and explainability (F2) ----------------------
def test_r2_two_obstacles_at_a_similar_distance_keep_their_own_holds():
    """Third review, R2: an obstacle on each side of the tube, 1 m apart along it. Matched
    by distance alone, the second was merged into the first's hold and the stop moved 1 m
    on, towards the first, which was only missed and still stood there."""
    corridor, p = straight(), planner()
    first = PathObstacle(8.0, 0.6, 30, 0.4, 0.8)
    second = PathObstacle(9.0, -0.6, 30, -0.8, -0.4)
    p.plan(corridor, PathObservation(first, 8.0), None, 0.0)
    # The first is missed (still there: its region in shadow), the second is seen.
    plan = p.plan(corridor, PathObservation(second, 8.0, ground(hole=(8.0, 14.0, 0.2, 1.0))), None, DT)
    assert len(plan.holds) == 2 and len({h["id"] for h in plan.holds}) == 2
    assert abs(plan.stop_distance_m - held_stop(8.0)) < 0.05 and plan.stop_reason == "obstacle_hold"
    held = next(h for h in plan.holds if h["near_m"] < 8.5)
    assert held["state"] == "unknown" and held["waiting_for"] != "detected this tick"
    # Both gone: each hold is cleared on its own region, and both are released.
    for k in range(2, 60):
        plan = p.plan(corridor, PathObservation(None, 15.0, ground()), None, k * DT)
    assert plan.holds == () and plan.stop_reason == ""


def test_r3_a_walker_gone_with_only_sparse_ground_in_view_stays_unknown():
    """Third review, R3: the walker has crossed out, but the LiDAR's rings fall either side
    of its region (3 m apart at range): no ground there is not clearance. The hold stays
    UNKNOWN, says it is waiting for ground across its region, and releases once a ring
    sweeps the region."""
    corridor, p = straight(), planner()
    p.plan(corridor, PathObservation(PathObstacle(12.0, 0.0, 20, -0.25, 0.25), 12.0), None, 0.0)
    rings = np.vstack([ground(along=(a, a + 0.1)) for a in (6.0, 9.0, 15.0, 18.0)])
    for k in range(1, 100):
        plan = p.plan(corridor, PathObservation(None, 20.0, rings), None, k * DT)
        assert plan.stop_reason == "obstacle_hold" and plan.hold_state == "unknown"
    report = plan.holds[0]
    assert report["waiting_for"] == "ground across its region" and report["region"].startswith("0/")
    swept = np.vstack((rings, ground(along=(12.0, 12.8))))
    released = None
    for k in range(100, 140):
        plan = p.plan(corridor, PathObservation(None, 20.0, swept), None, k * DT)
        if plan.stop_reason == "" and released is None:
            released = k * DT
    assert released is not None and released - 100 * DT >= PlannerConfig().resume_clear_s - 1e-9


def test_r7_a_refused_plan_still_limits_the_speed_of_the_kept_trajectory():
    """Third review, R7: when this tick's plan is refused (here: its path starts 2.5 m off
    the car), the controller keeps the last accepted trajectory, planned before whatever
    this tick's plan stops for. That one is now driven no faster than the refused plan
    allows where the car is: a refused stop brakes at once, not after the kept
    trajectory's 0.5 s of validity."""
    from mapless.planner import Plan
    from navigation_control.local import LocalTrajectory, TrajectorySample

    kerb = KerbLine([Straight(250)])
    world = MaplessRoadWorld(RoadScene(kerb), SUITE)
    driver = MaplessDriver(SUITE, world.route(120.0), DriverConfig(body=BODY, control=control(), corridor=CorridorConfig()), DT)
    plan_now = driver.planner.plan

    def refused_stop(*args, **kwargs):
        plan = plan_now(*args, **kwargs)
        if plan.trajectory is None:
            return plan
        moved = tuple(TrajectorySample(s.x_m, s.y_m + 2.5, 0.0) for s in plan.trajectory.samples)
        return Plan(LocalTrajectory(moved, plan.trajectory.created_s, plan.trajectory.valid_until_s), "obstacle", 3.0, plan.curve_speed_mps, "blocked")

    steps = []
    for k in range(560):
        if k == 500:
            assert steps[-1].mode is DriveMode.DRIVING and world.truth().speed_mps > 1.0
            driver.planner.plan = refused_stop
        step = driver.step(world.frame())
        steps.append(step)
        world.apply(step.command)
    after = steps[500:]
    assert all(not s.diagnostics.get("trajectory_accepted", True) for s in after if s.mode is DriveMode.DRIVING or s.mode is DriveMode.STOPPING)
    assert after[0].diagnostics["refused_plan_limit_mps"] == 0.0
    assert all(s.command.throttle == 0.0 for s in after)
    assert after[0].diagnostics["target_speed_mps"] == 0.0


# -- live r9 (b4280cd): holds that could never be released ----------------------------------
def _bumper_hold(departed_at=None, ticks=60):
    """A car-sized detection 1 m past the bumper, then nothing: the tube clear to 15 m, but
    no ground on its region (none within 2.5 m of the bumper, as the LiDAR sees it from
    here: its lowest unblocked ring meets the road beyond that). ``departed_at``: when the
    stack reports what stood there driven off."""
    near_m = BODY.front_m + 1.0
    corridor, p = straight(), planner()
    p.plan(corridor, PathObservation(PathObstacle(near_m, 0.0, 200, -0.9, 0.9, 1.4), near_m), None, 0.0)
    seen = ground(along=(BODY.front_m + 2.5, 20.0))
    (hold_id, span, seen_s), = p.held_regions()
    assert seen_s == 0.0 and span.shape == (2, 2)
    for k in range(1, ticks):
        if departed_at is not None and abs(k * DT - departed_at) < 1e-9:
            p.mark_departed(hold_id, k * DT)
        plan = p.plan(corridor, PathObservation(None, 15.0, seen), None, k * DT)
    return plan


def test_a_hold_whose_object_was_seen_to_leave_the_near_blind_zone_is_released():
    """PC1, Town03, r9_left_oncoming: a car crossed 1 m in front of the bumper; its region
    could never show ground from where the car stood, and the hold kept it there 130 s.
    The stack saw the car drive off: that is the evidence the ground cannot give."""
    plan = _bumper_hold(departed_at=0.2)
    assert plan.stop_reason != "obstacle_hold" and plan.holds == ()


def test_without_a_departure_a_hold_on_unseen_ground_stays():
    """Not seen is not gone: something under the rings (a person crouching there) is not
    seen either. No height rule releases it."""
    plan = _bumper_hold()
    assert plan.stop_reason == "obstacle_hold" and plan.holds[0]["state"] == "unknown"


def test_a_departure_before_the_last_detection_does_not_count():
    corridor, p = straight(), planner()
    near_m = BODY.front_m + 1.0
    p.plan(corridor, PathObservation(PathObstacle(near_m, 0.0, 200, -0.9, 0.9, 1.4), near_m), None, 0.0)
    (hold_id, _, _), = p.held_regions()
    p.mark_departed(hold_id, 0.05)
    p.plan(corridor, PathObservation(PathObstacle(near_m, 0.0, 200, -0.9, 0.9, 1.4), near_m), None, 0.1)  # seen again
    for k in range(3, 60):
        plan = p.plan(corridor, PathObservation(None, 15.0, ground(along=(BODY.front_m + 2.5, 20.0))), None, k * DT)
    assert plan.stop_reason == "obstacle_hold"


def test_a_hold_near_the_lines_end_needs_clearance_only_to_the_end():
    """PC1, Town03, r9_left_parked: a hold at 6.5 m on a 7.1 m corridor needed 8.5 m seen
    clear; the corridor showed 7.5 m and the car stood 115 s with the region's ground seen."""
    xs = np.arange(-2.0, 7.11, 0.5)
    corridor, p = line_corridor(np.column_stack((xs, np.zeros_like(xs)))), planner()
    p.plan(corridor, PathObservation(PathObstacle(6.5, 0.0, 40, -0.4, 0.4, 1.4), 6.5), None, 0.0)
    for k in range(1, 50):
        plan = p.plan(corridor, PathObservation(None, 7.0, ground(along=(0.0, 12.0))), None, k * DT)
    assert plan.stop_reason != "obstacle_hold" and plan.holds == ()


def test_path_obstacle_reports_the_tallest_point_of_its_cluster():
    pts = np.array([[8.0 + 0.05 * i, 0.1 * (i % 5 - 2), 0.2 + 0.1 * (i % 12)] for i in range(40)])
    obstacle = path_obstacle(pts, straight(), BODY, ObstacleConfig(), heights=pts[:, 2])
    assert obstacle is not None and abs(obstacle.height_m - pts[:, 2].max()) < 1e-9
    assert path_obstacle(pts, straight(), BODY, ObstacleConfig()).height_m is None


# -- eighth review, P1: observability-limited speed -----------------------------------------
def test_the_car_drives_no_faster_than_it_can_stop_within_the_road_it_sees_clear():
    """Nothing in view, but the tube seen clear only so far: the plan's speed at the car is
    the one from which it stops stop_gap_m short of the end of that view, with the brake lag."""
    from mapless.planner import sight_speed

    c = PlannerConfig(cruise_mps=6.0)
    p = LanePlanner(c, BODY, tube_half_m=TUBE_HALF)
    for clear_to in (6.0, 9.0, 12.0, 30.0):
        plan = p.plan(straight(), PathObservation(None, clear_to, ground(along=(0.0, clear_to))), None, 0.0)
        bound = sight_speed(clear_to - BODY.front_m, c.decel_mps2, c.sight_response_s, c.stop_gap_m)
        first = plan.trajectory.samples[0].speed_mps
        assert abs(plan.sight_speed_mps - bound) < 1e-9 and first <= bound + 1e-9
        v = bound
        assert v * c.sight_response_s + v * v / (2 * c.decel_mps2) <= max(0.0, clear_to - BODY.front_m - c.stop_gap_m) + 1e-6
    assert plan.trajectory.samples[0].speed_mps == 6.0  # 30 m seen clear supports more than the cruise
    assert LanePlanner(replace(c, sight_governed=False), BODY, tube_half_m=TUBE_HALF).plan(straight(), PathObservation(None, 6.0, ground(along=(0.0, 6.0))), None, 0.0).sight_speed_mps == math.inf


def test_the_observability_report_and_the_planner_agree_on_the_supported_speed():
    from evaluation.observability import supported_speed
    from mapless.planner import sight_speed

    for d in (0.5, 3.0, 8.0, 25.0):
        assert sight_speed(d, 1.0, 0.25, 2.5) == supported_speed(d, 1.0, 0.25, 2.5)


def test_a_low_detection_just_ahead_of_tall_points_is_as_tall_as_they_are():
    """The passing car's sills and wheels, kerb-high and nearer than its body's tall points,
    were the detection on some ticks, without a height: the hold had none and the blind
    zone could not release it. They are the same object; a low thing alone stays low."""
    from mapless.obstacles import observe_path
    from mapless.terrain import ClassifiedSweep

    def sweep(parts):
        xyz = np.vstack([p for p, _ in parts])
        cls = np.concatenate([np.full(len(p), c, dtype=np.int8) for p, c in parts])
        return ClassifiedSweep(xyz, xyz[:, 2].copy(), cls, np.zeros(0), np.zeros(0))

    ys = np.linspace(-0.8, 0.8, 30)
    sills = np.column_stack((np.full(30, 5.0), ys, np.full(30, 0.2)))
    body = np.vstack([np.column_stack((np.full(30, 5.4), ys, np.full(30, z))) for z in (0.5, 0.9, 1.4)])
    car = observe_path(sweep([(sills, PointClass.RAISED), (body, PointClass.OBSTACLE)]), straight(), BODY)
    assert car.obstacle is not None and abs(car.obstacle.distance_m - 5.0) < 0.2 and car.obstacle.height_m == 1.4
    brick = observe_path(sweep([(sills, PointClass.RAISED)]), straight(), BODY)
    assert brick.obstacle is not None and abs(brick.obstacle.height_m - 0.2) < 1e-6
