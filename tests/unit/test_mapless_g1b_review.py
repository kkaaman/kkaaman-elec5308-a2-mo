"""G1b fault handling and calibration, offline: the test sequence from the review of 94b5caa.

Each test drives the real ``MaplessDriver`` (or the part under test) in the synthetic
kerbed road, with sensors dropped or LiDAR returns removed mid-run:

- an empty LiDAR at start-up holds the car, and calibration fails instead of defaulting;
- calibrated drives (straight, skewed start, a kink like live run e's, a curve) keep the
  whole footprint in the lane from the first tick, calibration included;
- IMU or LiDAR loss while turning stops the car in its lane;
- loss of road evidence does not renew the corridor;
- an occluded obstacle keeps the stop; a removed one releases only on observed clearance.
"""

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mapless.corridor import CorridorConfig, CorridorEstimator
from mapless.driver import DriveMode, DriverConfig, MaplessDriver
from mapless.ego_motion import Pose2D
from mapless.frame import LidarSweep, SensorState, SensorStatus
from mapless.obstacles import observe_path
from mapless.sensor_suite import load_sensor_suite
from mapless.terrain import CellLabel, TerrainConfig, TerrainGrid, VehicleBody, classify_sweep
from navigation_control.local import LocalControlConfig
from navigation_control.types import ControllerConfig
from sim_harness.mapless_road import Arc, KerbLine, MaplessRoadWorld, ParkedCar, RoadScene, Straight

ROOT = Path(__file__).resolve().parents[2]
SUITE = load_sensor_suite(ROOT / "configs" / "sensor_suites" / "mapless_v2.json")
BODY = VehicleBody()
DT = 0.05
IN_LANE_M = 0.6  # |lane offset| that keeps the 2.1 m wide car inside a 3.5 m lane


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


def degrade(frame, drop=(), lidar_points=None):
    """``frame`` with sensors of the kinds in ``drop`` missing, or its LiDAR points replaced."""
    status, lidars, gnss, imu = [], list(frame.lidars), frame.gnss, frame.imu
    for s in frame.status:
        if s.kind in drop and s.state is SensorState.OK:
            s = SensorStatus(s.name, s.kind, SensorState.MISSING)
            lidars = [] if s.kind == "lidar" else lidars
            gnss = None if s.kind == "gnss" else gnss
            imu = None if s.kind == "imu" else imu
        status.append(s)
    if lidar_points is not None and lidars:
        sweep = lidars[0]
        lidars = [LidarSweep(sweep.name, lidar_points(sweep.points).astype(np.float32), sweep.sensor_to_ego)]
    return replace(frame, lidars=tuple(lidars), status=tuple(status), gnss=gnss, imu=imu)


def run(scene, seconds, goal_s=150.0, tamper=None, keep_right_m=None, **world_args):
    """Drive for ``seconds``; ``tamper(k, frame, world, log)`` may change each frame."""
    world = MaplessRoadWorld(scene, SUITE, **world_args)
    config = DriverConfig(body=BODY, control=control(), corridor=CorridorConfig(keep_right_m=keep_right_m))
    driver = MaplessDriver(SUITE, world.route(goal_s), config, DT)
    log = []
    for k in range(int(seconds / DT)):
        frame = world.frame()
        if tamper is not None:
            frame = tamper(k, frame, world, log)
        step = driver.step(frame)
        log.append((world.truth(), step))
        world.apply(step.command)
        if step.mode is DriveMode.FINISHED and world.state.speed == 0:
            break
    return log, world


STRAIGHT = RoadScene(KerbLine([Straight(250)]))
RIGHT_R12 = RoadScene(KerbLine([Straight(25), Arc(-10.25, math.pi / 2), Straight(100)]))


# -- calibration ---------------------------------------------------------------------------
def standstill_calibration(scene, ticks=80, **world_args):
    world = MaplessRoadWorld(scene, SUITE, **world_args)
    grid, estimator = TerrainGrid(), CorridorEstimator()
    pose = Pose2D(0.0, 0.0, 0.0)
    status = "calibrating"
    for _ in range(ticks):
        sweep = classify_sweep(world.frame().lidar("roof").xyz, BODY, TerrainConfig())
        grid.add(sweep, pose, world.time_s)
        status = estimator.calibrate(sweep, pose, world.time_s, True)
        world.time_s += DT
        if status != "calibrating":
            break
    return status, estimator


@pytest.mark.parametrize(
    "scene, world_args, truth_m",
    [
        pytest.param(STRAIGHT, {}, 1.75, id="straight"),
        pytest.param(
            RoadScene(KerbLine([Straight(200)]), road_width_m=11.0, lane_centre_m=4.4,
                      parked=tuple(ParkedCar(s, 1.2) for s in (3.0, 18.0, 28.0))),
            {}, 4.4, id="shoulder_with_parked_cars",
        ),
        pytest.param(STRAIGHT, dict(start_heading_error_rad=math.radians(3.5)), 1.75, id="skewed_left_3.5deg"),
        pytest.param(
            RoadScene(KerbLine([Straight(200)]), road_width_m=11.0, lane_centre_m=4.4),
            dict(start_heading_error_rad=math.radians(-3.5)), 4.4, id="skewed_right_3.5deg_shoulder",
        ),
        # Live run e: the lane runs a few degrees off the start heading soon after the start.
        pytest.param(RoadScene(KerbLine([Straight(12), Arc(30, math.radians(3.4)), Straight(150)])), {}, 1.75, id="kink"),
        pytest.param(STRAIGHT, dict(start_offset_m=0.4), 2.15, id="started_off_centre"),
        # The live "straight" site: a 58 m radius bend through the start. A line fit read
        # the chord, 0.4 m inside the kerb beside the car.
        pytest.param(RoadScene(KerbLine([Arc(58, 1.2), Straight(100)]), road_width_m=11.0, lane_centre_m=4.4),
                     dict(start_s_m=30.0), 4.4, id="bend_r58_through_the_start"),
    ],
)
def test_standstill_calibration_measures_the_kerb_across_the_kerb(scene, world_args, truth_m):
    status, estimator = standstill_calibration(scene, **world_args)
    assert status == "accepted"
    assert estimator.keep_right_m == pytest.approx(truth_m, abs=0.15)
    assert estimator.keep_right_source.startswith("measured at a standstill")
    assert estimator.previous is not None and estimator.previous.ok  # the first prior


def test_calibration_fails_with_a_reason_instead_of_defaulting():
    world = MaplessRoadWorld(STRAIGHT, SUITE)
    grid, estimator = TerrainGrid(), CorridorEstimator()
    pose = Pose2D(0.0, 0.0, 0.0)
    for _ in range(100):  # the LiDAR reports OK but returns nothing
        sweep = classify_sweep(np.zeros((0, 3)), BODY, TerrainConfig())
        grid.add(sweep, pose, world.time_s)
        status = estimator.calibrate(sweep, pose, world.time_s, True)
        world.time_s += DT
    assert status.startswith("failed: kerb not seen")
    assert estimator.keep_right_m is None
    corridor = estimator.update(grid, pose, world.time_s)
    assert not corridor.ok and corridor.reason.startswith("calibration failed")


def test_empty_lidar_at_startup_holds_the_car():
    empty = lambda k, f, w, log: degrade(f, lidar_points=lambda p: p[:0])  # noqa: E731
    log, world = run(STRAIGHT, seconds=8, tamper=empty)
    assert all(t.speed_mps == 0 for t, _ in log)
    assert all(s.command.throttle == 0 and s.command.brake > 0 for _, s in log)
    last = log[-1][1]
    assert last.mode is DriveMode.FAULT and "kerb calibration failed" in last.reason
    assert last.diagnostics["keep_right_m"] is None


# -- calibrated drives: the whole footprint, from the first tick -------------------------
@pytest.mark.parametrize(
    "scene, world_args",
    [
        pytest.param(STRAIGHT, {}, id="straight"),
        pytest.param(STRAIGHT, dict(start_heading_error_rad=math.radians(3.5)), id="skewed_start"),
        pytest.param(RoadScene(KerbLine([Straight(12), Arc(30, math.radians(3.4)), Straight(150)])), {}, id="kink"),
        pytest.param(RoadScene(KerbLine([Straight(25), Arc(25, math.pi / 2), Straight(100)])), {}, id="left_r25"),
        pytest.param(RIGHT_R12, {}, id="right_r12"),
    ],
)
def test_calibrated_drive_keeps_the_footprint_in_the_lane_and_stops_at_the_goal(scene, world_args):
    log, _ = run(scene, seconds=60, goal_s=80.0, **world_args)
    truth, last = log[-1]
    assert last.mode is DriveMode.FINISHED and abs(truth.s_m - 80.0) < 3.0
    assert not any(t.collision for t, _ in log)
    whole_run = np.abs([t.lane_offset_m for t, _ in log])  # calibration and start included
    assert whole_run.max() < IN_LANE_M
    assert not any(s.mode is DriveMode.FAULT for _, s in log)


# -- sensor loss --------------------------------------------------------------------------
def dropped_in_the_bend(kind):
    """Tamper that drops sensors of ``kind`` for good once the car has turned 20 degrees."""
    state = {"at": None}

    def tamper(k, frame, world, log):
        if state["at"] is None and abs(world.state.yaw) > math.radians(20):
            state["at"] = k
        return frame if state["at"] is None else degrade(frame, drop=(kind,))

    return tamper, state


@pytest.mark.parametrize("kind, steering", [("imu", "held"), ("lidar", "trajectory")])
def test_sensor_loss_in_a_bend_stops_the_car_in_its_lane(kind, steering):
    tamper, state = dropped_in_the_bend(kind)
    log, world = run(RIGHT_R12, seconds=25, tamper=tamper)
    at = state["at"]
    assert at is not None, "never reached the bend"
    after = log[at:]
    faulted = next(i for i, (_, s) in enumerate(after) if s.mode is DriveMode.FAULT)
    assert faulted <= 5  # within 0.25 s
    fault = after[faulted][1]
    assert fault.diagnostics["fault_steering"] == steering
    assert all(s.mode is DriveMode.FAULT and s.command.throttle == 0 for _, s in after[faulted:])
    if kind == "lidar":  # braking along the bend, not with the wheels straight
        assert fault.command.steering_angle_rad < -0.05
    assert world.state.speed == 0
    assert max(abs(t.lane_offset_m) for t, _ in log) < IN_LANE_M
    assert not any(t.collision for t, _ in log)


@pytest.mark.parametrize("kind, seconds_out", [("imu", 3.0), ("speedometer", 3.0), ("gnss", 5.0)])
def test_stale_ego_sensors_fault_and_recover(kind, seconds_out):
    start, end = int(10.0 / DT), int((10.0 + seconds_out) / DT)
    tamper = lambda k, f, w, log: degrade(f, drop=(kind,)) if start <= k < end else f  # noqa: E731
    log, _ = run(STRAIGHT, seconds=30, tamper=tamper)
    out = [s for _, s in log[start:end]]
    assert any(s.mode is DriveMode.FAULT and kind.replace("speedometer", "speedometer").lower() in s.reason.lower()
               or (kind == "gnss" and "GNSS" in s.reason) for s in out)
    assert any(s.mode is DriveMode.DRIVING for _, s in log[end:])  # recovered
    assert not any(t.collision for t, _ in log)
    if kind != "gnss":
        # The odometry integrated nothing during the outage, so the frame lags the car by
        # what it drove; the car stops and rebuilds grid, corridor, hold and trajectory
        # together before driving again (review of 2b7e14d), and ends up where it should.
        rebuilt = [i for i, (_, s) in enumerate(log) if s.diagnostics.get("frame_rebuilt")]
        assert rebuilt and log[rebuilt[0]][0].speed_mps < 0.1
        assert not any(s.diagnostics.get("odometry_trusted") for _, s in log[start + 5 : end])
        moved_during = log[end][0].s_m - log[start][0].s_m
        assert moved_during > 0.5  # it was moving when the sensor dropped
        after = [(t, s) for t, s in log[rebuilt[0] :] if s.mode is DriveMode.DRIVING]
        assert after and abs(after[-1][0].lane_offset_m) < IN_LANE_M
        truth, last = log[-1]
        assert abs(truth.lane_offset_m) < 0.3


def test_an_odometry_outage_does_not_poison_the_grid_or_the_hold():
    # Hold for a parked car, lose the speedometer for 3 s while a rolling start is forced by
    # the world (the car is pushed 2 m), regain it: the stop point must be re-measured, not
    # kept 2 m too far ahead in the old frame.
    pushed = {"done": False}

    def tamper(k, frame, world, log):
        holding = log and log[-1][1].reason.startswith("holding for obstacle")
        if holding and not pushed["done"]:
            pushed["done"] = True
            pushed["from"] = k
        if pushed["done"] and k < pushed["from"] + int(3.0 / DT):
            if k == pushed["from"] + 10:
                st = world.state
                st.x += 2.0 * math.cos(st.yaw)
                st.y += 2.0 * math.sin(st.yaw)
            return degrade(frame, drop=("speedometer",))
        return frame

    log, world = run(PARKED_IN_LANE, seconds=45, tamper=tamper)
    assert pushed["done"]
    after = log[pushed["from"] :]
    assert any(s.diagnostics.get("frame_rebuilt") for _, s in after)
    gaps = [t.front_gap_m for t, s in after if s.reason.startswith("holding for obstacle")]
    assert gaps and 1.0 <= min(gaps[-20:]) <= 5.0  # the hold was re-measured, 2 m closer
    assert not any(t.collision for t, _ in log)


def test_loss_of_road_evidence_does_not_renew_the_corridor():
    world = MaplessRoadWorld(STRAIGHT, SUITE, noise=False)
    grid, estimator = TerrainGrid(), CorridorEstimator(CorridorConfig(keep_right_m=1.75))
    for _ in range(40):  # slide 2 m/s along the lane, then stop
        pose = Pose2D(world.state.x - 10.0, world.state.y - 1.75, world.state.yaw)
        grid.add(classify_sweep(world.frame().lidar("roof").xyz, BODY, TerrainConfig()), pose, world.time_s)
        corridor = estimator.update(grid, pose, world.time_s)
        world.state.x += 0.1
        world.time_s += DT
    assert corridor.ok and corridor.evidence_age_s == 0
    last_fresh = corridor.evidence_time_s
    empty = classify_sweep(np.zeros((0, 3)), BODY, TerrainConfig())
    states = []
    for _ in range(60):  # standing still, the LiDAR returning nothing for 3 s
        grid.add(empty, pose, world.time_s)
        corridor = estimator.update(grid, pose, world.time_s)
        states.append((world.time_s, corridor.ok, corridor.evidence_time_s))
        world.time_s += DT
    assert all(evidence <= last_fresh for _, _, evidence in states)
    lost = next(t for t, ok, _ in states if not ok)
    assert lost - last_fresh <= estimator.config.support_max_age_s + 0.1
    assert "observed only" in corridor.reason or "used up" in corridor.reason
    assert np.all(grid.labels(pose.to_world(np.array([[8.0, 0.0]])), max_age_s=1.0) == CellLabel.UNKNOWN)


def test_empty_lidar_while_driving_stops_the_car():
    start = int(12.0 / DT)
    tamper = lambda k, f, w, log: degrade(f, lidar_points=lambda p: p[:0]) if k >= start else f  # noqa: E731
    log, world = run(STRAIGHT, seconds=20, tamper=tamper)
    assert log[start - 1][1].mode is DriveMode.DRIVING
    assert world.state.speed == 0
    faulted = [s for _, s in log[start:] if s.mode is DriveMode.FAULT]
    assert faulted and "no corridor" in faulted[-1].reason


# -- obstacle hold ------------------------------------------------------------------------
PARKED_IN_LANE = RoadScene(KerbLine([Straight(250)]), parked=(ParkedCar(45.0, 1.75),))


def hold_then(after_hold, seconds_after):
    """Tamper that applies ``after_hold(frame, world)`` from the tick the car holds for the obstacle."""
    state = {"at": None}

    def tamper(k, frame, world, log):
        if state["at"] is None and log and log[-1][1].reason.startswith("holding for obstacle"):
            if log[-1][0].speed_mps == 0:
                state["at"] = k
        return frame if state["at"] is None else after_hold(frame, world)

    return tamper, state


def without_obstacle_returns(points):
    """The parked car's returns gone (a missed or occluded obstacle); its shadow stays unseen."""
    return points[~((points[:, 0] > 2.0) & (np.abs(points[:, 1]) < 2.5) & (points[:, 2] > 0.25))]


def test_occluded_obstacle_keeps_the_stop():
    tamper, state = hold_then(lambda f, w: degrade(f, lidar_points=without_obstacle_returns), None)
    log, world = run(PARKED_IN_LANE, seconds=40, tamper=tamper)
    at = state["at"]
    assert at is not None and len(log) - at > int(10 / DT)
    after = log[at:]
    s_held = after[0][0].s_m
    assert max(t.s_m for t, _ in after) - s_held < 0.05  # 10+ s: no creep at all
    assert not any(s.mode is DriveMode.DRIVING for _, s in after)
    assert any(s.diagnostics.get("hold_state") == "unknown" for _, s in after)
    assert not any(t.collision for t, _ in log)


def test_removed_obstacle_resumes_only_after_observed_clearance():
    def remove(frame, world):
        world.boxes = []  # the parked car drives away: the LiDAR now sees the road behind it
        return frame

    tamper, state = hold_then(remove, None)
    log, world = run(PARKED_IN_LANE, seconds=45, tamper=tamper)
    at = state["at"]
    assert at is not None
    after = log[at:]
    moving = next(i for i, (t, _) in enumerate(after) if t.speed_mps > 0)
    assert moving * DT >= 1.0  # resume_clear_s of observed clearance first
    assert any(s.diagnostics.get("hold_state") == "clear" for _, s in after[:moving])
    assert max(t.s_m for t, _ in after) > after[0][0].s_m + 10  # and drives on
    assert not any(t.collision for t, _ in log)


def test_observe_path_counts_only_seen_ground_as_clear():
    world = MaplessRoadWorld(STRAIGHT, SUITE, noise=False)
    grid, estimator = TerrainGrid(), CorridorEstimator(CorridorConfig(keep_right_m=1.75))
    for _ in range(40):
        pose = Pose2D(world.state.x - 10.0, world.state.y - 1.75, world.state.yaw)
        grid.add(classify_sweep(world.frame().lidar("roof").xyz, BODY, TerrainConfig()), pose, world.time_s)
        corridor = estimator.update(grid, pose, world.time_s)
        world.state.x += 0.1
        world.time_s += DT
    assert corridor.ok
    xs, ys = np.meshgrid(np.arange(4.0, 30.0, 0.2), np.arange(-1.2, 1.2, 0.3))
    ground = np.column_stack((xs.ravel(), ys.ravel(), np.zeros(xs.size)))

    def observe(points):
        return observe_path(classify_sweep(points, BODY, TerrainConfig()), corridor, BODY)

    clear = observe(ground)
    assert clear.obstacle is None and clear.clear_to_m >= min(corridor.valid_length_m, 29.0) - 1.0
    car = np.array([[15.0 + 0.1 * i, 0.2 * j, 0.8] for i in range(10) for j in range(-3, 4)])
    shadowed = ground[ground[:, 0] < 15.0]
    blocked = observe(np.vstack((shadowed, car)))
    assert blocked.obstacle.distance_m == pytest.approx(15.0, abs=0.3) and blocked.clear_to_m <= 15.0
    hole = ground[(ground[:, 0] < 8.0) | (ground[:, 0] > 12.0)]
    assert observe(hole).clear_to_m == pytest.approx(8.0, abs=0.3)  # an unseen gap ends it
    assert observe(ground[ground[:, 0] > 10.0]).clear_to_m == 0.0  # must start near the car
    assert observe(np.zeros((0, 3))).clear_to_m == 0.0
    # Ground seen only along one edge of the tube is not clearance across the car's width
    # (review of 2b7e14d), and a kerb-high object in the lane is an obstacle.
    edge_only = ground[ground[:, 1] > 0.9]
    assert observe(edge_only).clear_to_m == 0.0
    low = np.array([[12.0 + 0.1 * i, 0.15 * j, 0.2] for i in range(10) for j in range(-3, 4)])
    found = observe(np.vstack((ground, low)))
    assert found.obstacle is not None and found.obstacle.distance_m == pytest.approx(12.0, abs=0.3)
