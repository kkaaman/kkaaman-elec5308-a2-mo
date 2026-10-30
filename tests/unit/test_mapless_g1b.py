"""G1b offline: ego motion, LiDAR terrain, corridor, obstacles, planner and the closed loop.

The closed-loop tests drive the real ``MaplessDriver`` in ``sim_harness.mapless_road``: a
ray-cast kerbed road with the suite's sensor noise. They check that the parts fit, not
that the live car will pass; that is the live G1b run.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from mapless.corridor import CorridorConfig, CorridorEstimator
from mapless.driver import DriveMode, DriverConfig, MaplessDriver
from mapless.ego_motion import EgoMotion, EgoMotionConfig, Pose2D
from mapless.frame import GnssFix, ImuSample, SensorFrame, SensorState, SensorStatus
from mapless.obstacles import PathObservation, PathObstacle, path_obstacle, project_onto_polyline
from mapless.planner import LanePlanner, PlannerConfig
from mapless.satnav import LocalTangentPlane
from mapless.sensor_suite import load_sensor_suite
from mapless.terrain import (
    CellLabel,
    PointClass,
    TerrainConfig,
    TerrainGrid,
    VehicleBody,
    classify_sweep,
)
from navigation_control.local import LocalControlConfig
from navigation_control.types import ControllerConfig
from sim_harness.mapless_road import (
    Arc,
    KerbLine,
    MaplessRoadWorld,
    ParkedCar,
    RoadScene,
    Straight,
)

ROOT = Path(__file__).resolve().parents[2]
SUITE = load_sensor_suite(ROOT / "configs" / "sensor_suites" / "mapless_v2.json")
BODY = VehicleBody()


def control():
    # The live harness's tuning (the baseline's measured controller settings).
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


# -- ego motion ---------------------------------------------------------------------------
def frame_at(t, speed, gyro, compass, fix=None):
    status = [
        SensorStatus(s.name, s.kind, SensorState.NOT_DUE)
        for s in SUITE.sensors
        if s.kind in ("camera", "radar", "lidar")
    ]
    status.append(SensorStatus("gnss", "gnss", SensorState.OK if fix else SensorState.NOT_DUE, t if fix else None))
    status.append(SensorStatus("imu", "imu", SensorState.OK, t))
    status.append(SensorStatus("wheel_speed", "speedometer", SensorState.OK, t))
    return SensorFrame(
        episode_id="t",
        frame_id=int(round(t / 0.05)),
        sim_time_s=t,
        received_monotonic_s=t,
        cameras=(),
        lidars=(),
        radars=(),
        gnss=None if fix is None else GnssFix(fix[0], fix[1], 0.0),
        imu=ImuSample((0.0, 0.0, 9.81), (0.0, 0.0, gyro), compass),
        speed_mps=speed,
        status=tuple(status),
        calibration_id="t",
    )


def test_route_pose_waits_for_a_stationary_average_and_takes_the_compass_heading():
    plane = LocalTangentPlane(-33.9, 151.2)
    ego = EgoMotion(plane, EgoMotionConfig(startup_s=1.0))
    fix = plane.to_geodetic(10.0, 20.0)
    compass = math.radians(30)  # clockwise from north: yaw = 60 deg anticlockwise from east
    states = [ego.update(frame_at(i * 0.05, 0.0, 0.003, compass, fix if i % 4 == 0 else None)) for i in range(15)]
    assert not any(s.localised for s in states)
    for i in range(15, 30):
        state = ego.update(frame_at(i * 0.05, 0.0, 0.003, compass, fix if i % 4 == 0 else None))
    assert state.localised
    assert state.route.x_m == pytest.approx(10.0, abs=1e-6) and state.route.y_m == pytest.approx(20.0, abs=1e-6)
    assert state.route.yaw_rad == pytest.approx(math.radians(60), abs=1e-6)
    assert state.gyro_bias_rps == pytest.approx(0.003)


def test_odometry_removes_the_standstill_gyro_bias_and_integrates_an_arc():
    plane = LocalTangentPlane(-33.9, 151.2)
    ego = EgoMotion(plane)
    bias, rate, speed, t = 0.004, 0.2, 2.0, 0.0
    for _ in range(40):
        ego.update(frame_at(t, 0.0, bias, 0.0))
        t += 0.05
    for _ in range(int(math.pi / rate / 0.05)):  # half a circle of radius 10 m
        state = ego.update(frame_at(t, speed, rate + bias, 0.0))
        t += 0.05
    assert state.odometry.yaw_rad == pytest.approx(math.pi, abs=0.03) or state.odometry.yaw_rad == pytest.approx(
        -math.pi, abs=0.03
    )
    assert state.odometry.x_m == pytest.approx(0.0, abs=0.3)
    assert state.odometry.y_m == pytest.approx(20.0, abs=0.3)


def test_speedometer_noise_at_a_standstill_does_not_creep_the_odometry():
    plane = LocalTangentPlane(-33.9, 151.2)
    ego = EgoMotion(plane)
    rng = np.random.default_rng(1)
    for i in range(200):
        state = ego.update(frame_at(i * 0.05, float(rng.normal(0.0, 0.05)), 0.0, 0.0))
    assert state.stationary
    assert math.hypot(state.odometry.x_m, state.odometry.y_m) < 0.05


def test_pose2d_round_trip():
    pose = Pose2D(3.0, -2.0, 0.7)
    pts = np.array([[1.0, 2.0], [-4.0, 0.5]])
    assert np.allclose(pose.to_body(pose.to_world(pts)), pts)


# -- terrain ------------------------------------------------------------------------------
def flat_scene_points(slope=0.0):
    xs, ys = np.meshgrid(np.arange(-15, 40, 0.3), np.arange(-8, 8, 0.3))
    x, y = xs.ravel(), ys.ravel()
    z = slope * x + np.where(y < -1.75, 0.15, 0.0)  # pavement right of the kerb
    return np.column_stack((x, y, z))


def test_classification_separates_road_kerb_obstacle_overhead_and_the_car_itself():
    road = flat_scene_points()
    car = np.array([[20.0 + dx, dy, h] for dx in (0, 0.5, 1.0) for dy in (-0.5, 0, 0.5) for h in (0.5, 1.0)])
    overhead = np.array([[10.0, 0.0, 3.5]])
    own_roof = np.array([[1.5, 0.0, 1.45], [0.5, 0.3, 1.4]])
    sweep = classify_sweep(np.vstack((road, car, overhead, own_roof)), BODY, TerrainConfig())
    assert len(sweep.xyz) == len(road) + len(car) + 1 - int(np.sum(
        (road[:, 0] > BODY.rear_m - 0.3) & (road[:, 0] < BODY.front_m + TerrainConfig().body_front_margin_m) & (np.abs(road[:, 1]) < BODY.half_width_m + 0.3)
    ))
    on_pavement = sweep.xyz[:, 1] < -1.9
    assert np.all(sweep.cls[on_pavement & (sweep.xyz[:, 2] < 0.2)] == PointClass.RAISED)
    assert np.all(sweep.cls[(np.abs(sweep.xyz[:, 1]) < 1.5) & (sweep.xyz[:, 2] < 0.05)] == PointClass.ROAD)
    assert np.sum(sweep.cls == PointClass.OBSTACLE) == len(car)
    assert np.sum(sweep.cls == PointClass.IGNORED) == 1


def test_road_profile_follows_a_hill():
    sweep = classify_sweep(flat_scene_points(slope=0.06), BODY, TerrainConfig())
    road_side = np.abs(sweep.xyz[:, 1]) < 1.5
    assert np.mean(sweep.cls[road_side] == PointClass.ROAD) > 0.98


def test_a_car_filling_a_band_does_not_become_the_road():
    road = flat_scene_points()
    road = road[~((road[:, 0] > 18) & (road[:, 0] < 26))]  # shadowed behind and under the car
    car = np.array([[x, y, z] for x in np.arange(18.5, 22, 0.2) for y in np.arange(-8, 8, 0.3) for z in (0.4, 0.8)])
    sweep = classify_sweep(np.vstack((road, car)), BODY, TerrainConfig())
    is_car = sweep.xyz[:, 2] >= 0.4
    assert np.all(sweep.cls[is_car] == PointClass.OBSTACLE)


def test_grid_keeps_evidence_when_it_recentres_and_obstacles_fade_faster():
    config = TerrainConfig()
    grid = TerrainGrid(config)
    points = np.array([[5.0, -2.0, 0.15], [6.0, 0.0, 0.0], [7.0, 0.0, 1.0]])
    sweep = classify_sweep(points, BODY, config)
    for i in range(3):
        grid.add(sweep, Pose2D(0.0, 0.0, 0.0), i * 0.05)
    world = np.array([[5.0, -2.0], [6.0, 0.0], [7.0, 0.0]])
    before = list(grid.labels(world))
    assert before == [CellLabel.BARRIER, CellLabel.ROAD, CellLabel.OCCUPIED]  # kerb, road, obstacle
    origin = grid.origin.copy()
    empty = classify_sweep(np.zeros((0, 3)), BODY, config)
    grid.add(empty, Pose2D(12.0, 0.0, 0.0), 0.15)  # beyond recentre_m: the grid moves
    assert not np.allclose(grid.origin, origin)
    assert list(grid.labels(world))[:2] == before[:2]
    grid.add(empty, Pose2D(12.0, 0.0, 0.0), 1.15)
    i, j = grid.cells(world)[0].T
    raised, obstacle = grid.raised[i[0], j[0]], grid.obstacle[i[2], j[2]]
    assert obstacle < 0.2 * raised  # obstacles (tau 0.5 s) fade faster than kerbs (tau 3 s)


# -- obstacles and planning -------------------------------------------------------------
def straight_corridor(ticks=60):
    """Corridor after sliding 2 m/s along a straight lane centre for ``ticks`` sweeps."""
    world = MaplessRoadWorld(RoadScene(KerbLine([Straight(200)])), SUITE, noise=False)
    grid, estimator = TerrainGrid(), CorridorEstimator(CorridorConfig(keep_right_m=1.75))
    corridor = None
    for _ in range(ticks):
        f = world.frame()
        pose = Pose2D(world.state.x - 10.0, world.state.y - 1.75, world.state.yaw)  # odometry frame
        grid.add(classify_sweep(f.lidar("roof").xyz, BODY, TerrainConfig()), pose, world.time_s)
        corridor = estimator.update(grid, pose, world.time_s)
        world.state.x += 0.1  # slide along the lane without dynamics
        world.time_s += 0.05
    return corridor


def test_corridor_on_a_straight_kerbed_road_finds_the_lane_centre_and_kerb():
    corridor = straight_corridor()
    assert corridor.ok and corridor.valid_length_m > 20
    assert abs(corridor.lateral_offset_at(0.0)) < 0.15
    assert abs(corridor.lateral_offset_at(15.0)) < 0.15
    assert corridor.right_edge_from_centre_m(8.0) == pytest.approx(-1.75, abs=0.2)


def test_projection_onto_a_polyline():
    line = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    s, lateral, outside = project_onto_polyline(np.array([[5.0, 1.0], [11.0, 5.0], [-2.0, 0.0]]), line)
    assert s[0] == pytest.approx(5.0) and lateral[0] == pytest.approx(1.0)
    assert s[1] == pytest.approx(15.0) and lateral[1] == pytest.approx(-1.0)
    assert outside[2] and not outside[0]


def test_path_obstacle_needs_a_dense_cluster_inside_the_tube():
    corridor = straight_corridor()
    in_tube = np.array([[12.0 + 0.1 * i, 0.3, 0.8] for i in range(8)])
    beside = np.array([[8.0 + 0.1 * i, 2.5, 0.8] for i in range(20)])  # outside the tube
    noise = np.array([[6.0, 0.0, 0.8], [6.5, 0.1, 0.9]])  # too few points
    found = path_obstacle(np.vstack((in_tube, beside, noise)), corridor, BODY)
    assert isinstance(found, PathObstacle)
    assert found.distance_m == pytest.approx(12.0, abs=0.2)
    assert path_obstacle(np.vstack((beside, noise)), corridor, BODY) is None


def test_planner_stops_the_bumper_short_of_an_obstacle_and_holds_until_the_path_is_seen_clear():
    corridor = straight_corridor()
    planner = LanePlanner(PlannerConfig(stop_gap_m=2.5, resume_clear_s=1.0, clear_beyond_m=2.0), BODY)
    plan = planner.plan(corridor, PathObservation(PathObstacle(15.0, 0.0, 10), 15.0), None, 10.0)
    assert (plan.stop_reason, plan.hold_state) == ("obstacle", "blocked")
    assert plan.stop_distance_m == pytest.approx(15.0 - BODY.front_m - 2.5, abs=0.05)
    speeds = np.array([s.speed_mps for s in plan.trajectory.samples])
    xy = np.array([(s.x_m, s.y_m) for s in plan.trajectory.samples])
    along = corridor.pose.to_body(xy)[:, 0]
    assert np.all(speeds[along > plan.stop_distance_m + 0.5] == 0)
    assert speeds.max() <= 2.0 + 1e-9

    def seen(clear_to):
        # The tube's ground out to clear_to, as the sweep sees it: the planner checks the
        # held obstacle's own region against it (Stage 1).
        a, lat = np.meshgrid(np.arange(0.0, clear_to, 0.1), np.arange(-1.4, 1.41, 0.1))
        return PathObservation(None, clear_to, np.column_stack((a.ravel(), lat.ravel())))

    def hold(observation, t):
        p = planner.plan(corridor, observation, None, t)
        return p.stop_reason, p.hold_state

    def clear_run(t0, seconds, clear_to=30.0):
        last = None
        for k in range(int(round(seconds / 0.05)) + 1):
            last = hold(seen(clear_to), t0 + 0.05 * k)
        return last

    # Not seeing the obstacle is not seeing the road clear: an occluded or missed obstacle
    # leaves the ground behind it unseen, and the stop holds however long that lasts.
    assert hold(seen(14.0), 10.05) == ("obstacle_hold", "unknown")
    assert hold(None, 10.1) == ("obstacle_hold", "unknown")  # no sweep this tick
    assert hold(seen(16.9), 30.0) == ("obstacle_hold", "unknown")  # needs 15 + 2 m
    # Seen clear past where it stood, every tick for resume_clear_s: released.
    assert clear_run(30.05, 0.5) == ("obstacle_hold", "clear")
    assert hold(seen(9.0), 30.6) == ("obstacle_hold", "unknown")  # interrupted
    assert clear_run(30.65, 0.9) == ("obstacle_hold", "clear")  # 0.95 s of continuous clearance
    assert hold(seen(30.0), 31.65) == ("", "clear")  # 1.0 s: released
    assert hold(seen(30.0), 31.7) == ("", "")
    # Clearance must be continuous (review of 2b7e14d): one clear tick, 3 s of no planner
    # calls (a fault), one more clear tick: still holding, the count restarted.
    planner.plan(corridor, PathObservation(PathObstacle(15.0, 0.0, 10), 15.0), None, 40.0)
    assert hold(seen(30.0), 40.05) == ("obstacle_hold", "clear")
    assert hold(seen(30.0), 43.05) == ("obstacle_hold", "clear")
    assert clear_run(43.1, 0.9) == ("obstacle_hold", "clear")
    assert hold(seen(30.0), 44.05) == ("", "clear")


def test_planner_goal_stop_is_a_final_goal_trajectory_ending_at_the_goal():
    corridor = straight_corridor()
    plan = LanePlanner(PlannerConfig(), BODY).plan(corridor, None, np.array([12.0, 0.1]), 3.0)
    assert plan.stop_reason == "goal" and plan.trajectory.final_goal
    last = plan.trajectory.samples[-1]
    end = corridor.pose.to_body(np.array([[last.x_m, last.y_m]]))[0]
    assert last.speed_mps == 0 and end[0] == pytest.approx(12.0, abs=0.2)


# -- closed loop --------------------------------------------------------------------------
def drive(scene, goal_s, seconds, keep_right_m=None, **world_args):
    world = MaplessRoadWorld(scene, SUITE, **world_args)
    config = DriverConfig(body=BODY, control=control(), corridor=CorridorConfig(keep_right_m=keep_right_m))
    driver = MaplessDriver(SUITE, world.route(goal_s), config, 0.05)
    log = []
    for _ in range(int(seconds / 0.05)):
        step = driver.step(world.frame())
        log.append((world.truth(), step))
        world.apply(step.command)
        if step.mode is DriveMode.FINISHED and world.state.speed == 0:
            break
    return log


def settled_lateral(log, after_s_m):
    start = log[0][0].s_m
    return np.array([t.lane_offset_m for t, _ in log if t.s_m > start + after_s_m and t.speed_mps > 0.5])


@pytest.mark.parametrize(
    "pieces, crown_m",
    [
        pytest.param([Straight(200)], 0.0, id="straight"),
        pytest.param([Straight(200)], 0.08, id="straight_crowned_road"),
        pytest.param([Straight(25), Arc(25, math.pi / 2), Straight(100)], 0.0, id="left_r25"),
        # The tightest bend the live curve site allows: 12 m lane radius (kerb at 10.25 m).
        pytest.param([Straight(25), Arc(-10.25, math.pi / 2), Straight(100)], 0.0, id="right_r12"),
    ],
)
def test_closed_loop_follows_the_lane_and_stops_at_the_goal(pieces, crown_m):
    # A configured kerb offset, so a car starting 0.4 m off centre must converge to the lane centre.
    scene = RoadScene(KerbLine(pieces), crown_m=crown_m)
    log = drive(scene, goal_s=75.0, seconds=60, keep_right_m=1.75, start_offset_m=0.4)
    truth, last = log[-1]
    assert last.mode is DriveMode.FINISHED
    assert abs(truth.s_m - 75.0) < 3.0  # GNSS-level goal accuracy
    assert not any(t.collision for t, _ in log)
    lateral = settled_lateral(log, after_s_m=15.0)
    assert np.percentile(np.abs(lateral), 95) < 0.3
    assert max(t.speed_mps for t, _ in log) < 2.3


def test_closed_loop_measures_the_kerb_offset_and_ignores_parked_cars_on_the_shoulder():
    # Town10HD-like: two shoulder lanes put the lane centre 4.4 m from the kerb, and cars
    # park on the shoulder. Parked cars must neither become the road edge nor stop the car.
    parked = tuple(ParkedCar(s, 1.2) for s in (28.0, 34.0, 52.0, 70.0))
    scene = RoadScene(KerbLine([Straight(200)]), road_width_m=11.0, lane_centre_m=4.4, parked=parked)
    log = drive(scene, goal_s=85.0, seconds=60)
    truth, last = log[-1]
    assert last.mode is DriveMode.FINISHED and abs(truth.s_m - 85.0) < 3.0
    assert not any(t.collision for t, _ in log)
    keep_right = [s.diagnostics.get("keep_right_m") for _, s in log if s.diagnostics.get("keep_right_m")]
    assert keep_right and keep_right[-1] == pytest.approx(4.4, abs=0.25)
    assert log[-1][1].diagnostics["keep_right_source"].startswith("measured at a standstill")
    assert np.percentile(np.abs(settled_lateral(log, after_s_m=15.0)), 95) < 0.3
    assert not any(s.reason.startswith("obstacle") for _, s in log)


def test_measured_kerb_offset_holds_the_start_position():
    # The start-up assumption, explicit: the car keeps the lateral position it started in.
    # The live harness starts it on the lane centre; camera lane evidence removes this.
    log = drive(RoadScene(KerbLine([Straight(200)])), goal_s=60.0, seconds=45, start_offset_m=0.4)
    assert np.median(settled_lateral(log, after_s_m=15.0)) == pytest.approx(0.4, abs=0.15)


def test_closed_loop_stops_for_a_parked_car_in_the_lane_and_holds():
    scene = RoadScene(KerbLine([Straight(200)]), parked=(ParkedCar(45.0, 1.75),))
    log = drive(scene, goal_s=120.0, seconds=40)
    truth, last = log[-1]
    assert not any(t.collision for t, _ in log)
    assert last.mode is DriveMode.STOPPING and truth.speed_mps == 0
    assert 1.5 <= truth.front_gap_m <= 5.0
    stopped = [t.s_m for t, s in log[-100:]]
    assert max(stopped) - min(stopped) < 0.05


def test_driver_brakes_when_the_lidar_goes_stale():
    world = MaplessRoadWorld(RoadScene(KerbLine([Straight(200)])), SUITE)
    driver = MaplessDriver(SUITE, world.route(150.0), DriverConfig(body=BODY, control=control()), 0.05)
    for _ in range(140):
        step = driver.step(world.frame())
        world.apply(step.command)
    assert step.mode is DriveMode.DRIVING and world.state.speed > 1.0
    good = world.frame()
    status = tuple(
        SensorStatus(s.name, s.kind, SensorState.MISSING) if s.kind == "lidar" else s for s in good.status
    )
    for _ in range(5):
        frame = SensorFrame(**{**good.__dict__, "lidars": (), "status": status})
        step = driver.step(frame)
        good = SensorFrame(**{**good.__dict__, "sim_time_s": good.sim_time_s + 0.05, "frame_id": good.frame_id + 1})
    assert step.mode is DriveMode.FAULT and step.command.brake > 0


def test_corridor_config_rejects_nonsense():
    with pytest.raises(ValueError):
        CorridorConfig(start_m=1.0)
    with pytest.raises(ValueError):
        TerrainConfig(raised_min_m=0.5, obstacle_min_m=0.3)
    with pytest.raises(ValueError):
        DriverConfig(body=BODY, control=control(), planner=PlannerConfig(decel_mps2=3.0))


def test_calibration_kerb_with_a_railing_is_kept_a_parked_car_is_not():
    """Town05 (26 Oct): a 1.5 m railing stands on the kerb for the whole window and the
    tall-beside veto (meant for a parked car's kerb-high sills) dropped every station:
    "kerb not seen". A continuous, thin tall face beside the kerb is a railing."""
    from mapless.corridor import CorridorConfig, CorridorEstimator
    from mapless.terrain import ClassifiedSweep, PointClass

    def sweep(tall_y, tall_x):
        pts, cls = [], []
        for x in np.arange(-8.0, 12.0, 0.1):
            rows = [(y, 0.0, PointClass.ROAD) for y in np.arange(-5.6, -1.0, 0.2)]
            rows += [(y, 0.15, PointClass.RAISED) for y in (-5.75, -5.85, -5.95)]
            if tall_x[0] <= x <= tall_x[1]:
                rows += [(y, 1.0, PointClass.OBSTACLE) for y in tall_y]
            pts += [(x, y, z) for y, z, _ in rows]
            cls += [k for _, _, k in rows]
        xyz = np.array(pts, dtype=float)
        return ClassifiedSweep(xyz, xyz[:, 2], np.array(cls), np.zeros(0), np.zeros(0))

    est = CorridorEstimator(CorridorConfig())
    _, railing = est._kerb_contour(sweep(tall_y=(-5.9, -6.0), tall_x=(-8.0, 12.0)))
    assert np.isfinite(railing).sum() > 30 and np.nanmedian(railing) == pytest.approx(-5.75)
    # A parked car beside the kerb from x = 0 to 4.5 m: wide (its roof is seen), short.
    _, car = est._kerb_contour(sweep(tall_y=(-4.2, -5.0, -5.8), tall_x=(0.0, 4.5)))
    assert np.all(np.isnan(car[(np.arange(-10.0, 15.01, 0.5) >= 0.0) & (np.arange(-10.0, 15.01, 0.5) <= 4.5)]))
    assert np.isfinite(car).sum() > 20  # the rest of the kerb is still read


def _turn_scene():
    """A left turn: the road straight ahead dips 3% beyond the junction, the exit road to the
    left is level with the car, its pavement 0.15 m up on the right (PC 1, Town03, r11)."""
    pts = []
    for x in np.arange(-8.0, 44.0, 0.25):
        for y in np.arange(-4.0, 4.0, 0.25):
            pts.append((x, y, -0.03 * max(0.0, x - 15.0)))
    for x in np.arange(15.0, 23.0, 0.25):
        for y in np.arange(4.0, 40.0, 0.25):
            pts.append((x, y, 0.0))
    for x in np.arange(23.0, 26.0, 0.25):
        for y in np.arange(6.0, 40.0, 0.25):
            pts.append((x, y, 0.15))
    return np.array(pts)


def _turn_path():
    th = np.linspace(0.0, math.pi / 2, 20)
    arc = np.column_stack((9.0 + 10.0 * np.sin(th), 10.0 * (1.0 - np.cos(th))))
    lead = np.column_stack((np.arange(0.0, 9.0, 1.0), np.zeros(9)))
    on = np.column_stack((np.full(30, 19.0), 10.0 + np.arange(1.0, 31.0)))
    return np.vstack((lead, arc, on))


def test_r11_far_points_are_referenced_along_the_path():
    """PC 1, r11_left_parked_lidargap (84fb071): mid-turn, the bands along x followed the
    street dipping beyond the junction, and the exit road beside it, level with the car,
    read as kerb and obstacle 20-30 m out; the exit line was read 1.2 m into the road and the
    latched turn re-planned twice. Given the path, those points are referenced along it: the
    exit road reads ROAD, its pavement still RAISED. A straight path changes nothing, nor
    does the near field."""
    scene, config = _turn_scene(), TerrainConfig()
    plain = classify_sweep(scene, BODY, config)
    along = classify_sweep(scene, BODY, config, path_xy=_turn_path())
    x, y = plain.xyz[:, 0], plain.xyz[:, 1]
    exit_road = (x > 15.5) & (x < 22.5) & (y > 12.0) & (y < 30.0)
    pavement = (x > 23.5) & (y > 12.0) & (y < 30.0)
    assert np.mean(plain.cls[exit_road] != PointClass.ROAD) > 0.5  # the fault: read against the dip
    assert np.all(along.cls[exit_road] == PointClass.ROAD)
    assert np.all(along.cls[pavement] == PointClass.RAISED)
    near = np.hypot(x, y) < config.path_profile_min_range_m
    assert np.array_equal(plain.cls[near], along.cls[near])
    ahead = np.abs(y) < config.road_window_m  # the street straight ahead: the bands', as before
    assert np.array_equal(plain.cls[ahead], along.cls[ahead])
    straight = np.column_stack((np.arange(-5.0, 45.0, 1.0), np.full(50, 0.3)))
    assert np.array_equal(classify_sweep(scene, BODY, config, path_xy=straight).cls, plain.cls)
    off = classify_sweep(scene, BODY, TerrainConfig(path_profile=False), path_xy=_turn_path())
    assert np.array_equal(off.cls, plain.cls)
