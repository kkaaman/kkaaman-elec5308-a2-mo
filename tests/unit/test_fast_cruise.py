"""Faster cruising (3 Nov): stops and the sight bound planned at a firmer deceleration
(``--brake-mps2``) and the steering point further ahead at speed (``--lookahead-s``). The car
still never drives faster than it can stop, at that deceleration, within the road it has
seen clear: on the synthetic road 25-30 m of it, ~9 m/s at 3 m/s² (~6 m/s at 1 m/s²)."""

import math

from mapless.corridor import CorridorConfig
from mapless.driver import DriveMode, DriverConfig, MaplessDriver
from mapless.planner import PlannerConfig
from mapless.terrain import VehicleBody
from navigation_control.local import (
    LocalControlConfig,
    LocalTrajectory,
    LocalTrajectoryController,
    TrajectorySample,
)
from navigation_control.types import ControllerConfig, VehicleState
from sim_harness.mapless_road import KerbLine, MaplessRoadWorld, ParkedCar, RoadScene, Straight
from tests.unit.test_overtake import DT, SUITE

BODY = VehicleBody()


def drive(brake, lookahead_s, parked=(), cruise=13.9, seconds=60.0, accel=None, two_stage=False):
    control = LocalControlConfig(
        ControllerConfig(wheelbase_m=3.00465, max_steering_angle_rad=math.radians(20), lookahead_m=4.0, kp=0.45, ki=0.15, stopping_deceleration_mps2=brake, hold_brake=0.5),
        lookahead_time_s=lookahead_s,
    )
    world = MaplessRoadWorld(RoadScene(KerbLine([Straight(320)]), road_width_m=7.5, lane_centre_m=1.75, parked=parked), SUITE, start_s_m=15.0, seed=5308)
    planner = PlannerConfig(cruise_mps=cruise, decel_mps2=brake, comfort_decel_mps2=min(brake, 2.0) if two_stage else None, final_decel_mps2=1.0 if two_stage else None)
    config = DriverConfig(body=BODY, control=control, corridor=CorridorConfig(keep_right_m=1.75), planner=planner, accel_mps2=accel)
    driver = MaplessDriver(SUITE, world.route(290.0), config, DT)
    log = []
    for _ in range(int(seconds / DT)):
        step = driver.step(world.frame())
        log.append((world.truth(), step))
        world.apply(step.command)
        if step.mode is DriveMode.FINISHED and world.state.speed == 0:
            break
    return log


def test_planned_at_3_mps2_the_car_cruises_faster_and_still_stops_for_a_parked_car():
    log = drive(3.0, 0.8, parked=(ParkedCar(220.0, 1.75),))
    top = max(t.speed_mps for t, _ in log)
    assert 8.0 < top < 13.9 and not any(t.collision for t, _ in log)  # its own bound, not the 13.9 asked
    assert max(abs(t.lane_offset_m) for t, _ in log if t.speed_mps > 5.0) < 0.3
    stood = log[-1][0]
    assert stood.speed_mps < 0.05 and 220.0 - 2.25 - (stood.s_m + BODY.front_m) > 1.5  # held short of its rear face


def test_the_steering_point_moves_out_with_speed_only_when_asked():
    """``lookahead_time_s``: at 10 m/s and 0.8 s the point is 8 m ahead, not the fixed 4 m;
    on a bend the wheel angle changes with it. 0 (the default) keeps the fixed point."""
    base = ControllerConfig(wheelbase_m=3.0, max_steering_angle_rad=0.6, lookahead_m=4.0)
    assert LocalControlConfig(base).lookahead_time_s == 0.0
    samples = tuple(TrajectorySample(float(x), 0.01 * float(x) ** 2, 10.0) for x in range(0, 60))
    angles = {}
    for look in (0.0, 0.8):
        controller = LocalTrajectoryController(LocalControlConfig(base, lookahead_time_s=look))
        state = VehicleState(0.0, 0.0, 0.0, 10.0)
        assert controller.set_local_trajectory(LocalTrajectory(samples=samples, created_s=0.0, valid_until_s=10.0), state, 0.0).accepted
        angles[look] = controller.update(state, 0.05, 0.05).command.steering_angle_rad
    assert angles[0.0] > 0.0 and angles[0.8] > 0.0 and angles[0.8] != angles[0.0]


def _accel(log, window_s=1.0):
    n = int(round(window_s / DT))
    v = [t.speed_mps for t, _ in log]
    return max(v[i + n] - v[i] for i in range(len(v) - n)) / window_s


def test_with_an_acceleration_limit_the_car_pulls_away_steadily_and_stops_in_two_stages():
    """``accel_mps2`` 1.5: no 1 s window gains more than ~1.5 m/s (without it, the speed loop
    gains more). Two-stage stops: at 2 m/s² to 3 m/s, then 1 m/s² over the last 4.5 m."""
    free = drive(3.0, 0.8, seconds=12.0)
    steady = drive(3.0, 0.8, parked=(ParkedCar(220.0, 1.75),), accel=1.5, two_stage=True)
    assert _accel(free) > 2.0 and _accel(steady) < 1.8
    assert max(t.speed_mps for t, _ in steady) > 8.0 and not any(t.collision for t, _ in steady)
    rear_face = 220.0 - 2.25
    near = [t.speed_mps for t, _ in steady if 2.5 <= rear_face - 2.5 - (t.s_m + BODY.front_m) <= 4.5]  # the last 4.5 m to the stop point
    stood = steady[-1][0]
    assert stood.speed_mps < 0.05 and rear_face - (stood.s_m + BODY.front_m) > 1.5
    assert near and max(near) < 3.4  # down to ~3 m/s before the last metres


def test_the_two_stage_stopping_speed():
    from mapless.planner import stopping_speed

    one = PlannerConfig(decel_mps2=3.0)
    two = PlannerConfig(decel_mps2=3.0, comfort_decel_mps2=2.0, final_decel_mps2=1.0, approach_mps=3.0)
    assert abs(float(stopping_speed(6.0, one)) - 6.0) < 1e-9
    assert abs(float(stopping_speed(4.5, two)) - 3.0) < 1e-9 and float(stopping_speed(0.0, two)) == 0.0
    assert abs(float(stopping_speed(4.5 + 10.0, two)) - 7.0) < 1e-9  # 3 m/s plus 10 m at 2 m/s²
    assert abs(float(stopping_speed(2.0, two)) - 2.0) < 1e-9  # the last metres at 1 m/s²
