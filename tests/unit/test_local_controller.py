"""LocalTrajectoryController: moving-horizon tracking without per-update resets (G1a, offline).

A simple kinematic bicycle with first-order pedal response and rolling drag stands in for
the car. A planner replans every 0.1 s, cutting a window of a longer reference path from
the car's current position.
"""

import math

import pytest

from navigation_control import ControllerConfig, PathFollower, VehicleState, Waypoint
from navigation_control.local import (
    LocalControlConfig,
    LocalStatus,
    LocalTrajectory,
    LocalTrajectoryController,
    TrajectorySample,
)

DT = 0.05
CONFIG = ControllerConfig(
    wheelbase_m=3.0, max_steering_angle_rad=math.radians(35), lookahead_m=5.0, stopping_deceleration_mps2=1.5
)
LOCAL = LocalControlConfig(CONFIG)


def reference(length=400.0, step=1.0):
    """Straight 100 m, then a gentle 60 m-radius left arc, then straight."""
    points = []
    s = 0.0
    while s <= length:
        if s <= 100:
            points.append((s, 0.0))
        elif s <= 100 + 60 * math.pi / 2:
            a = (s - 100) / 60
            points.append((100 + 60 * math.sin(a), 60 * (1 - math.cos(a))))
        else:
            d = s - 100 - 60 * math.pi / 2
            points.append((160.0, 60.0 + d))
        s += step
    return points


REF = reference()


class Car:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, speed=0.0):
        self.x, self.y, self.yaw, self.speed = x, y, yaw, speed

    def state(self):
        return VehicleState(self.x, self.y, self.yaw, max(0.0, self.speed))

    def step(self, command):
        accel = 3.0 * command.throttle - 7.0 * command.brake - 0.25 * self.speed - 0.15
        self.speed = max(0.0, self.speed + accel * DT)
        self.yaw += self.speed / CONFIG.wheelbase_m * math.tan(command.steering_angle_rad) * DT
        self.x += self.speed * math.cos(self.yaw) * DT
        self.y += self.speed * math.sin(self.yaw) * DT


def nearest_index(x, y):
    return min(range(len(REF)), key=lambda i: (REF[i][0] - x) ** 2 + (REF[i][1] - y) ** 2)


def window(car, now, horizon_m=25.0, speed=4.0, manoeuvre="lane_follow", final_goal_at=None, validity=0.5):
    i = max(0, nearest_index(car.x, car.y) - 1)  # start just behind the car, as a planner would
    pts = REF[i : i + int(horizon_m) + 1]
    samples = [TrajectorySample(x, y, speed) for x, y in pts]
    final = False
    if final_goal_at is not None and i + len(pts) - 1 >= final_goal_at:
        samples = [TrajectorySample(x, y, speed if k < final_goal_at - i else 0.0) for k, (x, y) in enumerate(REF[i : final_goal_at + 1])]
        final = True
    return LocalTrajectory(tuple(samples), now, now + validity, manoeuvre, final)


def drive(controller, seconds, *, replan_every=2, plan=window, car=None, start=0.0):
    car = car or Car()
    log = []
    for k in range(int(seconds / DT)):
        now = start + k * DT
        if plan is not None and k % replan_every == 0:
            result = controller.set_local_trajectory(plan(car, now), car.state(), now)
            assert result.accepted, result.reason
        out = controller.update(car.state(), now, DT)
        car.step(out.command)
        log.append((now, car.speed, out))
    return car, log


def test_replanning_every_100_ms_holds_speed_and_lane():
    controller = LocalTrajectoryController(LOCAL)
    car, log = drive(controller, 40.0)
    speeds = [speed for t, speed, _ in log if t > 3.0]
    # No dips at replanning: speed only rises towards the target, then holds it.
    assert all(b >= a - 0.005 for a, b in zip(speeds, speeds[1:]))
    assert 3.95 < speeds[-1] < 4.1
    assert all(out.status is LocalStatus.TRACKING for t, _, out in log if t > 8.0)
    assert max(out.diagnostics.cross_track_error_m for t, _, out in log if t > 8.0) < 0.35


def test_the_baseline_follower_loses_its_integral_on_every_update():
    """Same replanning with PathFollower.set_path: the PID restarts each time and speed sags."""
    follower = PathFollower(CONFIG)
    car = Car()
    speeds = []
    for k in range(int(40.0 / DT)):
        if k % 2 == 0:
            i = nearest_index(car.x, car.y)
            follower.set_path(Waypoint(x, y) for x, y in REF[i : i + 26])
        out = follower.update(car.state(), 4.0, DT)
        car.step(out.command)
        if k * DT > 8.0:
            speeds.append(car.speed)
    local_car, log = drive(LocalTrajectoryController(LOCAL), 40.0)
    local_mean = sum(s for t, s, _ in log if t > 8.0) / len(speeds)
    assert sum(speeds) / len(speeds) < 3.6 < local_mean  # steady-state error without the integral


def test_short_horizons_are_capped_and_the_minimum_horizon_is_published():
    assert LOCAL.minimum_horizon_m(4.0) == pytest.approx(16 / 3 + 1.0 + 1.0)
    controller = LocalTrajectoryController(LOCAL)
    _, log = drive(controller, 20.0, plan=lambda car, now: window(car, now, horizon_m=5.0))
    steady = [s for t, s, _ in log if t > 8.0]
    assert max(steady) < 3.5  # cannot outrun a 5 m plan
    assert any(out.status is LocalStatus.LIMITED for _, _, out in log)


def test_a_stale_trajectory_expires_and_brakes_along_the_last_path():
    controller = LocalTrajectoryController(LOCAL)
    car, _ = drive(controller, 15.0)
    assert car.speed > 3.5
    car, log = drive(controller, 6.0, plan=None, car=car, start=15.0)
    assert log[-1][1] == pytest.approx(0.0, abs=0.05)
    assert all(out.status is LocalStatus.EXPIRED for t, _, out in log if t > 15.6)
    assert max(out.diagnostics.cross_track_error_m for _, _, out in log) < 0.5


def test_final_goal_stops_and_latches_but_a_horizon_end_never_does():
    goal = 120  # reference index
    controller = LocalTrajectoryController(LOCAL)
    car, log = drive(controller, 60.0, plan=lambda c, now: window(c, now, final_goal_at=goal))
    assert log[-1][2].status is LocalStatus.FINISHED and car.speed < 0.1
    gx, gy = REF[goal]
    assert math.hypot(car.x - gx, car.y - gy) <= CONFIG.goal_tolerance_m + 0.2
    # Without final_goal the car never latches FINISHED, however long it drives.
    _, log = drive(LocalTrajectoryController(LOCAL), 30.0)
    assert not any(out.status is LocalStatus.FINISHED for _, _, out in log)


def test_manoeuvre_change_keeps_speed_control_continuity():
    controller = LocalTrajectoryController(LOCAL)
    car, _ = drive(controller, 15.0)
    integral = controller._speed._integral
    assert integral > 0
    now = 15.0
    assert controller.set_local_trajectory(window(car, now, manoeuvre="turn_left"), car.state(), now).accepted
    assert controller._speed._integral == integral
    car, log = drive(controller, 3.0, car=car, start=now, plan=lambda c, t: window(c, t, manoeuvre="turn_left"))
    assert min(s for _, s, _ in log) > 3.8


def test_implausible_trajectories_are_refused_and_the_current_one_kept():
    controller = LocalTrajectoryController(LOCAL)
    car = Car()
    good = window(car, 0.0)
    assert controller.set_local_trajectory(good, car.state(), 0.0).accepted
    far = LocalTrajectory(tuple(TrajectorySample(x, y + 5.0, 4.0) for x, y in REF[:20]), 0.0, 1.0)
    result = controller.set_local_trajectory(far, car.state(), 0.1)
    assert not result.accepted and "from the car" in result.reason
    stale = LocalTrajectory(good.samples, 0.0, 0.2)
    assert not controller.set_local_trajectory(stale, car.state(), 0.5).accepted
    assert controller.trajectory is good
    with pytest.raises(ValueError, match="zero speed"):
        LocalTrajectory(good.samples, 0.0, 1.0, final_goal=True)


def test_no_trajectory_brakes_and_external_limit_is_respected():
    controller = LocalTrajectoryController(LOCAL)
    out = controller.update(VehicleState(0, 0, 0, 2.0), 0.0, DT)
    assert out.status is LocalStatus.NO_TRAJECTORY and out.command.brake > 0
    car = Car()
    controller.set_local_trajectory(window(car, 0.0), car.state(), 0.0)
    for k in range(400):
        now = k * DT
        if k % 2 == 0:
            controller.set_local_trajectory(window(car, now), car.state(), now)
        out = controller.update(car.state(), now, DT, speed_limit_mps=2.0)
        car.step(out.command)
    assert car.speed == pytest.approx(2.0, abs=0.15) and out.status is LocalStatus.LIMITED


def test_a_small_speed_excess_is_shed_by_coasting_not_a_trace_of_brake():
    """The simulator stops a slowly moving car dead on any brake (the tyres stick below
    walking pace); the first full live stack run sawtoothed on brake commands of 0.006."""
    controller = LocalTrajectoryController(LOCAL)
    car = Car()
    car.speed = 0.75
    controller.set_local_trajectory(window(car, 0.0), car.state(), 0.0)
    brakes = []
    for k in range(40):
        now = k * DT
        if k % 2 == 0:
            controller.set_local_trajectory(window(car, now), car.state(), now)
        out = controller.update(car.state(), now, DT, speed_limit_mps=0.6)  # cap just below the speed
        brakes.append(out.command.brake)
        car.step(out.command)
    assert all(b == 0.0 or b >= LOCAL.brake_deadband for b in brakes)
    assert brakes[0] == 0.0  # a 0.15 m/s excess is coasted off
    # A commanded stop still brakes, whatever the effort.
    out = controller.update(car.state(), 2.0, DT, speed_limit_mps=0.0)
    assert out.command.brake > 0.0


def test_a_large_excess_at_low_speed_is_braked_not_coasted():
    """External review of 405b056: the coasting rule also covered a yield from 1.4 to
    0.8 m/s. Only an excess up to coast_max_excess_mps is coasted; more is braked."""
    controller = LocalTrajectoryController(LOCAL)
    car = Car()
    car.speed = 1.4
    controller.set_local_trajectory(window(car, 0.0), car.state(), 0.0)
    out = controller.update(car.state(), 0.0, DT, speed_limit_mps=0.8)
    assert out.command.brake > 0.0 and out.command.throttle == 0.0
