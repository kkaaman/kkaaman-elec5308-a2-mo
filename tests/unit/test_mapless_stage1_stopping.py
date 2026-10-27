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
