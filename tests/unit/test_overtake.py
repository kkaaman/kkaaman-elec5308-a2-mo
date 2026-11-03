"""Passing a vehicle stopped in the lane (3 Nov), on the synthetic road: the car holds
behind it, waits, shifts left past it with the indicator on, comes back to its lane and
reaches the goal; it does not pass a walker, nor with something on the way past, nor
within a junction's reach."""

import math
from pathlib import Path

import numpy as np

from mapless.corridor import CorridorConfig
from mapless.driver import DriveMode, DriverConfig, MaplessDriver
from mapless.overtake import OvertakeConfig, ramp, shift_line
from mapless.planner import PlannerConfig
from mapless.sensor_suite import load_sensor_suite
from mapless.terrain import VehicleBody
from navigation_control.local import LocalControlConfig
from navigation_control.types import ControllerConfig
from sim_harness.mapless_road import KerbLine, MaplessRoadWorld, ParkedCar, RoadScene, Straight

ROOT = Path(__file__).resolve().parents[2]
SUITE = load_sensor_suite(ROOT / "configs" / "sensor_suites" / "mapless_v2.json")
BODY = VehicleBody()
DT = 0.05


def control():
    return LocalControlConfig(ControllerConfig(wheelbase_m=3.00465, max_steering_angle_rad=math.radians(20), lookahead_m=4.0, kp=0.45, ki=0.15, stopping_deceleration_mps2=1.0, hold_brake=0.5))


def run(parked, overtake=True, goal=130.0, seconds=90.0, width=7.5, seed=5308, veto=False, lane_centre=1.75):
    scene = RoadScene(KerbLine([Straight(170)]), road_width_m=width, lane_centre_m=lane_centre, parked=parked)
    world = MaplessRoadWorld(scene, SUITE, start_s_m=15.0, seed=seed)
    config = DriverConfig(
        body=BODY, control=control(), corridor=CorridorConfig(keep_right_m=lane_centre), planner=PlannerConfig(cruise_mps=3.0),
        overtake=OvertakeConfig() if overtake else None,
    )
    driver = MaplessDriver(SUITE, world.route(goal), config, DT)
    driver.overtake_veto = veto  # the stack's: moving traffic on the far side
    log = []
    for _ in range(int(seconds / DT)):
        step = driver.step(world.frame())
        log.append((world.truth(), step))
        world.apply(step.command)
        if step.mode is DriveMode.FINISHED and world.state.speed == 0:
            break
    return log, driver


def test_ramp_and_shift_are_smooth_and_leftwards():
    assert ramp(-1.0, 0.0, 12.0) == 0.0 and abs(ramp(6.0, 0.0, 12.0) - 0.5) < 1e-9 and ramp(20.0, 0.0, 12.0) == 1.0
    line = np.column_stack((np.arange(0.0, 10.0), np.zeros(10)))
    moved = shift_line(line, np.full(10, 2.0))
    assert np.allclose(moved[:, 1], 2.0) and np.allclose(moved[:, 0], line[:, 0])


def test_the_car_passes_a_stopped_car_in_its_lane_and_comes_back():
    log, driver = run((ParkedCar(60.0, 1.75),))
    truth, last = log[-1]
    assert last.mode is DriveMode.FINISHED and abs(truth.s_m - 130.0) < 3.0, driver.overtake.record()
    assert not any(t.collision for t, _ in log)
    assert driver.overtake.passes == 1
    beside = [t.lane_offset_m for t, _ in log if 57.5 <= t.s_m <= 63.0]  # the rear axle beside it
    assert beside and min(beside) > 2.3 and max(t.lane_offset_m for t, _ in log) < 3.6  # in the other lane beside it, no further
    after = [abs(t.lane_offset_m) for t, _ in log if t.s_m >= 100.0]
    assert after and max(after) < 0.35  # back in its lane
    signals = [(t.s_m, s.diagnostics.get("indicator")) for t, s in log]
    assert any(sig == "left" for s, sig in signals if s < 60.0) and any(sig == "right" for s, sig in signals if s > 60.0)
    held = [t for t, s in log if t.speed_mps < 0.05 and 30.0 < t.s_m < 60.0]
    assert held and len(held) * DT >= 3.0  # waited behind it first
    assert max(t.speed_mps for t, s in log if 50.0 <= t.s_m <= 74.0) <= 2.3  # slowly past it (back in its lane by 77 m)


def test_without_overtaking_the_car_holds_behind_it():
    log, driver = run((ParkedCar(60.0, 1.75),), overtake=False, seconds=40.0)
    truth, last = log[-1]
    assert truth.s_m < 60.0 - 2.1 - BODY.front_m and not any(t.collision for t, _ in log) and driver.overtake is None


def test_a_car_parked_on_the_far_side_beside_it_blocks_the_way_past():
    """A second car in the other lane beside the first: the way past is not clear; the car
    waits behind and never drives into either."""
    log, driver = run((ParkedCar(60.0, 1.75), ParkedCar(62.0, 5.5)), seconds=40.0)
    assert driver.overtake.passes == 0 and driver.overtake.state == "idle" and not any(t.collision for t, _ in log)
    assert driver.overtake.note.startswith("waiting")
    assert max(t.s_m for t, _ in log) < 60.0 - 2.1 - BODY.front_m


def test_traffic_on_the_far_side_vetoes_the_pass():
    log, driver = run((ParkedCar(60.0, 1.75),), seconds=30.0, veto=True)
    assert driver.overtake.passes == 0 and driver.overtake.state == "idle" and driver.overtake.note == "waiting: left: traffic on that side"
    assert max(t.s_m for t, _ in log) < 60.0 - 2.1 - BODY.front_m and not any(t.collision for t, _ in log)


def test_the_hold_gap_applies_only_with_overtaking_on():
    with_it = MaplessDriver(SUITE, MaplessRoadWorld(RoadScene(KerbLine([Straight(60)]), road_width_m=7.5, lane_centre_m=1.75), SUITE).route(40.0), DriverConfig(body=BODY, control=control(), overtake=OvertakeConfig()), DT)
    without = MaplessDriver(SUITE, MaplessRoadWorld(RoadScene(KerbLine([Straight(60)]), road_width_m=7.5, lane_centre_m=1.75), SUITE).route(40.0), DriverConfig(body=BODY, control=control()), DT)
    assert with_it.planner.config.hold_gap_m == OvertakeConfig().hold_gap_m and without.planner.config.hold_gap_m is None


def test_with_a_lane_on_the_right_the_car_passes_on_the_right():
    """Three lanes, the car in the middle one (5.25 m from the right kerb), a car stopped
    ahead of it: a lane between it and the kerb runs its way (traffic keeps right), so it
    passes there, not on the left (live S4_5308 on 7ba365b went into the oncoming lane),
    indicating right, then left as it comes back."""
    log, driver = run((ParkedCar(60.0, 5.25),), width=10.5, lane_centre=5.25)
    truth, last = log[-1]
    assert last.mode is DriveMode.FINISHED and abs(truth.s_m - 130.0) < 3.0, driver.overtake.record()
    assert not any(t.collision for t, _ in log) and driver.overtake.passes == 1 and driver.overtake.side == "right"
    beside = [t.lane_offset_m for t, _ in log if 57.5 <= t.s_m <= 63.0]
    assert beside and max(beside) < -2.3 and min(t.lane_offset_m for t, _ in log) > -3.6
    signals = [(t.s_m, s.diagnostics.get("indicator")) for t, s in log]
    assert any(sig == "right" for s, sig in signals if s < 60.0) and any(sig == "left" for s, sig in signals if s > 60.0)
    assert max(abs(t.lane_offset_m) for t, _ in log if t.s_m >= 100.0) < 0.35


def test_the_stack_names_the_sides_with_moving_traffic(monkeypatch):
    """The stack's veto (``overtake_traffic``): moving tracks beside the way past, by side;
    static ones and those beyond reach do not count."""
    from types import SimpleNamespace

    import mapless.stack as stack
    from mapless.ego_motion import Pose2D

    monkeypatch.setattr(stack, "motion_state", lambda track, config: track.state)
    tracks = [
        SimpleNamespace(position=np.array([20.0, 3.0]), state="moving"),
        SimpleNamespace(position=np.array([10.0, -3.5]), state="static"),
        SimpleNamespace(position=np.array([80.0, -3.0]), state="moving"),
        SimpleNamespace(position=np.array([-5.0, -3.0]), state="moving"),
    ]
    assert stack.overtake_traffic(tracks, Pose2D(0.0, 0.0, 0.0), None) == {"left", "right"}
    assert stack.overtake_traffic(tracks[1:3], Pose2D(0.0, 0.0, 0.0), None) == set()
