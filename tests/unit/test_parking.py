"""P1 contracts, continuous geometry bounds, depth coverage and terminal guards."""

import math
from dataclasses import replace

import numpy as np
import pytest

from autonomy.interfaces import CameraCalibration, EgoPose, FrameBundle
from autonomy.parking import ParkingEpisode
from autonomy.parking_contracts import (
    DriveMode,
    MotionState,
    ParkingConfig,
    ParkingPhase,
    angle_error,
)
from navigation_control import VehicleState
from obstacle_perception.parking_clearance import Clearance, DepthParkingClearance
from planning.parking import _leaves, plan_forward_parking
from planning.parking_geometry import (
    bay_polygon,
    contains,
    footprint,
    intersects,
    polygon,
    rectangle,
    swept_rectangle,
)
from sim_harness.parking import DepthBox, render_parking_depth, standard_case
from sim_harness.synthetic import SyntheticScenario


def observed(state=None, *, frame_id=0, boxes=(), unknown=False):
    state = VehicleState(0, 0, 0, 0) if state is None else state
    scenario = SyntheticScenario(image_width=256, image_height=256)
    rgb, depth = scenario.sensor_arrays(frame_id)
    frame = FrameBundle("p1-test", frame_id, frame_id*0.05, 100, EgoPose(state),
                        rgb, scenario.calibration, depth, frame_id)
    return render_parking_depth(frame, boxes, unknown=unknown)


def episode(frame=None):
    start, goal, vehicle, area, _ = standard_case("straight")
    frame = observed(start) if frame is None else frame
    return ParkingEpisode(frame, goal, vehicle, area)


def test_signed_api_does_not_weaken_a1_or_enable_park_by_default():
    signed = MotionState(0, 0, 0, -0.2)
    assert signed.mode == DriveMode.DRIVE
    with pytest.raises(ValueError, match="reverse"):
        signed.as_forward()
    with pytest.raises(ValueError):
        VehicleState(0, 0, 0, -0.2)
    state = VehicleState(1, 2, 3, 0.4)
    assert MotionState.from_forward(state).as_forward() == state
    assert angle_error(-math.pi + 0.01, math.pi - 0.01) == pytest.approx(0.02)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "1"])
def test_signed_api_rejects_invalid_numbers(value):
    with pytest.raises(ValueError):
        MotionState(0, 0, 0, value)


@pytest.mark.parametrize("changes", [
    {"max_speed_mps": 1.01}, {"dt_s": 0}, {"sample_step_m": 0.5},
    {"response_budget_s": 0.1}, {"ground_exclusion_m": 0.01},
    {"stopped_speed_mps": 0.9}, {"hold_brake": 2},
])
def test_parking_policy_rejects_unsafe_or_unsupported_configuration(changes):
    with pytest.raises(ValueError):
        ParkingConfig(**changes)


@pytest.mark.parametrize("changes", [
    {"position_tolerance_m": 0.31}, {"heading_tolerance_rad": math.radians(5.1)},
    {"clearance_m": 0},
])
def test_goal_cannot_relax_issue_acceptance(changes):
    goal = standard_case("straight")[1]
    with pytest.raises(ValueError):
        replace(goal, **changes)


@pytest.mark.parametrize("case", ["straight", "left", "right", "rotated"])
def test_plan_endpoints_tangents_curvature_and_full_body(case):
    start, goal, vehicle, area, _ = standard_case(case)
    plan = plan_forward_parking(start, goal, vehicle, area)
    assert (plan.waypoints[0].x_m, plan.waypoints[0].y_m) == (start.x_m, start.y_m)
    assert plan.waypoints[-1].x_m == pytest.approx(goal.x_m)
    assert plan.waypoints[-1].y_m == pytest.approx(goal.y_m)
    assert angle_error(plan.headings_rad[0], start.yaw_rad) == pytest.approx(0)
    assert angle_error(plan.headings_rad[-1], goal.yaw_rad) == pytest.approx(0, abs=1e-10)
    assert all(s.curvature_bound <= vehicle.curvature_limit for s in plan.segments)
    assert all(contains(area, s.sweep) for s in plan.segments)
    final = VehicleState(goal.x_m, goal.y_m, goal.yaw_rad, 0)
    assert contains(bay_polygon(goal.bay), footprint(final, vehicle, goal.clearance_m))


def test_continuous_bezier_bound_encloses_intermediate_body_corners():
    vehicle = standard_case("left")[2]
    controls = np.array([[0, 0], [5, 0], [10, 0], [20, 8], [20, 13], [20, 18]], dtype=float)
    leaves = _leaves(controls, vehicle.curvature_limit, 0.15)
    for p, bound in leaves[::7]:
        sweep = swept_rectangle(p, vehicle, 0.15)
        for t in np.linspace(0, 1, 13):
            def evaluate(c):
                n = len(c)-1
                return sum(math.comb(n, k) * (1-t)**(n-k) * t**k * c[k] for k in range(n+1))
            position = evaluate(p)
            velocity = evaluate(5*np.diff(p, axis=0))
            acceleration = evaluate(20*np.diff(p, n=2, axis=0))
            curvature = abs(velocity[0]*acceleration[1]-velocity[1]*acceleration[0]) / np.linalg.norm(velocity)**3
            assert curvature <= bound + 1e-8
            state = VehicleState(*position, math.atan2(velocity[1], velocity[0]), 0)
            assert contains(sweep, footprint(state, vehicle, 0.1))


def test_rejects_narrow_bay_unreachable_turn_and_body_overhang():
    start, goal, vehicle, area, _ = standard_case("straight")
    with pytest.raises(ValueError, match="fit"):
        plan_forward_parking(start, replace(goal, bay=replace(goal.bay, width_m=1.9)), vehicle, area)
    behind = replace(goal, x_m=-16, bay=replace(goal.bay, x_m=-14.5))
    with pytest.raises(ValueError, match="approach"):
        plan_forward_parking(start, behind, vehicle, area)
    narrow_area = ((-2, -0.95), (23, -0.95), (23, 4), (-2, 4))
    with pytest.raises(ValueError):
        plan_forward_parking(start, goal, vehicle, narrow_area)
    turn_goal = replace(goal, x_m=3, y_m=3, yaw_rad=math.pi/2,
                        bay=replace(goal.bay, x_m=3, y_m=4.5, yaw_rad=math.pi/2))
    with pytest.raises(ValueError, match="feasible"):
        plan_forward_parking(start, turn_goal, vehicle, area)


def test_convex_geometry_rejects_concavity_and_detects_edge_contact():
    with pytest.raises(ValueError):
        polygon(((0, 0), (2, 0), (1, 0.5), (2, 2), (0, 2)))
    a = rectangle(0, 0, 0, 1, 1, 1)
    assert intersects(a, rectangle(2, 0, 0, 1, 1, 1))
    assert not intersects(a, rectangle(2.1, 0, 0, 1, 1, 1))


def test_depth_requires_observed_volume_and_uses_full_body_not_axle():
    vehicle = standard_case("straight")[2]
    state = VehicleState(0, 0, 0, 0)
    body = footprint(state, vehicle)
    assert DepthParkingClearance(observed(), vehicle, ParkingConfig()).check(body) == Clearance.CLEAR
    nose_box = DepthBox(rectangle(3.8, 0, 0, 0.1, 0.1, 0.2))
    frame = observed(boxes=(nose_box,))
    assert DepthParkingClearance(frame, vehicle, ParkingConfig()).check(body) == Clearance.BLOCKED
    assert DepthParkingClearance(observed(unknown=True), vehicle, ParkingConfig()).check(body) == Clearance.UNKNOWN
    missing = replace(frame, depth_m=None, depth_frame_id=None)
    assert DepthParkingClearance(missing, vehicle, ParkingConfig()).check(body) == Clearance.UNKNOWN


def test_outside_fov_nan_and_max_range_are_not_free_space():
    frame = observed()
    vehicle = standard_case("straight")[2]
    far = rectangle(100, 0, 0, 1, 1, 1)
    assert DepthParkingClearance(frame, vehicle, ParkingConfig()).check(far) == Clearance.UNKNOWN
    for value in (np.nan, 100.0):
        depth = frame.depth_m.copy()
        depth[128, 128] = value
        query = DepthParkingClearance(replace(frame, depth_m=depth), vehicle, ParkingConfig())
        assert query.check(footprint(frame.ego.state, vehicle)) == Clearance.UNKNOWN
    # Valid empty front-camera images still do not see the body behind the lens.
    front = CameraCalibration(100, 100, 128, 128,
                              ((0, 0, 1, 0), (-1, 0, 0, 0), (0, -1, 0, 1.5), (0, 0, 0, 1)))
    assert DepthParkingClearance(replace(frame, calibration=front), vehicle, ParkingConfig()).check(
        footprint(frame.ego.state, vehicle)
    ) == Clearance.UNKNOWN


@pytest.mark.parametrize("kind", ["depth", "stale", "future", "capture_age"])
def test_invalid_observation_holds_then_valid_observation_can_resume(kind):
    original = observed()
    task = episode(original)
    if kind == "depth":
        frame = replace(original, depth_m=None, depth_frame_id=None)
    elif kind == "stale":
        frame = replace(original, received_monotonic_s=99)
    elif kind == "future":
        frame = replace(original, received_monotonic_s=101)
    else:
        frame = replace(original, capture_started_monotonic_s=99)
    result = task.step(frame, now_monotonic_s=100)
    assert result.phase == ParkingPhase.HOLD
    assert result.packet.command.throttle == 0 and result.packet.command.brake == 1
    assert task.mode == DriveMode.DRIVE
    assert not result.packet.diagnostics.observation_valid
    for frame_id in range(1, 10):
        resumed = task.step(observed(frame_id=frame_id), now_monotonic_s=100)
    assert resumed.packet.command.throttle > 0
    assert task.route_install_count == 1


@pytest.mark.parametrize("kind", ["duplicate", "episode", "time", "jump"])
def test_frame_identity_fault_latches_full_brake(kind):
    task = episode()
    task.step(observed(), now_monotonic_s=100)
    frame = observed(frame_id=1)
    if kind == "duplicate":
        frame = observed()
    elif kind == "episode":
        frame = replace(frame, episode_id="foreign")
    elif kind == "time":
        frame = replace(frame, sim_time_s=0.07)
    else:
        frame = observed(frame_id=2)
    result = task.step(frame, now_monotonic_s=100)
    assert result.phase == ParkingPhase.FAULT
    assert result.packet.command.brake == 1
    assert task.step(observed(frame_id=2), now_monotonic_s=100).phase == ParkingPhase.FAULT


def test_blocker_disappearance_requires_duration_and_frames():
    box = DepthBox(rectangle(3.5, 0, 0, 0.2, 0.2, 0.3))
    first = observed(boxes=(box,))
    task = episode(first)
    assert task.step(first, now_monotonic_s=100).phase == ParkingPhase.HOLD
    for frame_id in range(1, 9):
        result = task.step(observed(frame_id=frame_id), now_monotonic_s=100)
        assert result.phase == ParkingPhase.HOLD
        assert result.packet.command.throttle == 0
    assert task.step(observed(frame_id=9), now_monotonic_s=100).packet.command.throttle > 0


@pytest.mark.parametrize("kind", ["heading", "moving", "blocked", "unknown", "body"])
def test_near_goal_is_insufficient_for_park(kind):
    task = episode()
    task.step(observed(), now_monotonic_s=100)
    goal = task.goal
    state = VehicleState(goal.x_m, goal.y_m, math.radians(6) if kind == "heading" else 0,
                         0.2 if kind == "moving" else 0)
    if kind == "body":
        state = replace(state, y_m=2)
    boxes = (DepthBox(rectangle(goal.x_m+3.5, 0, 0, 0.2, 0.2, 0.2)),) if kind == "blocked" else ()
    frame = observed(state, frame_id=1, boxes=boxes, unknown=kind == "unknown")
    result = task.step(frame, now_monotonic_s=100)
    assert result.motion.mode == DriveMode.DRIVE
    assert not any(e.event == "drive_to_park" for e in task.events)


def test_park_requires_stop_latches_once_and_outputs_expiring_brake_packets():
    task = episode()
    task.step(observed(), now_monotonic_s=100)
    state = VehicleState(task.goal.x_m, task.goal.y_m, task.goal.yaw_rad, 0)
    for frame_id in range(1, 6):
        result = task.step(observed(state, frame_id=frame_id), now_monotonic_s=100)
        assert result.motion.mode == DriveMode.PARK
        assert result.packet.command.brake == 1 and result.packet.command.throttle == 0
        assert result.packet.is_current(task.episode_id, frame_id*0.05, monotonic_s=100)
        assert not result.packet.is_current(task.episode_id, frame_id*0.05, monotonic_s=101)
    assert sum(e.event == "drive_to_park" for e in task.events) == 1
    assert task.route_install_count == 1


def test_invalid_frame_interrupts_clear_confirmation_without_erasing_blocker():
    box = DepthBox(rectangle(3.5, 0, 0, 0.2, 0.2, 0.3))
    frame = observed(boxes=(box,))
    task = episode(frame)
    task.step(frame, now_monotonic_s=100)
    for frame_id in range(1, 7):
        task.step(observed(frame_id=frame_id), now_monotonic_s=100)
    invalid = task.step(observed(frame_id=7, unknown=True), now_monotonic_s=100)
    assert invalid.packet.command.brake == 1
    for frame_id in range(8, 16):
        assert task.step(observed(frame_id=frame_id), now_monotonic_s=100).packet.command.throttle == 0
    assert task.step(observed(frame_id=16), now_monotonic_s=100).packet.command.throttle > 0


@pytest.mark.parametrize(("times", "reason"), [
    ((100.0, 100.11), "processing_timeout"),
    ((100.24, 100.27), "observation_expired_during_step"),
])
def test_expiry_during_computation_cannot_publish_park(monkeypatch, times, reason):
    task = episode()
    task.step(observed(), now_monotonic_s=100)
    state = VehicleState(task.goal.x_m, task.goal.y_m, task.goal.yaw_rad, 0)
    clock = iter(times)
    monkeypatch.setattr("autonomy.parking.time.monotonic", lambda: next(clock))
    result = task.step(observed(state, frame_id=1))
    assert result.phase == ParkingPhase.FAULT and result.reason == reason
    assert task.mode == DriveMode.DRIVE and result.packet.command.brake == 1
    assert not any(event.event in ("drive_to_park", "parked") for event in task.events)
