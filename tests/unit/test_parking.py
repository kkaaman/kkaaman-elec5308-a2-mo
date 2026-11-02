"""Parallel parking on the synthetic road (the showcase's parking clip, 2 Nov): the gap
found by the LiDAR, the two reversing arcs planned clear of both cars, centred by what
the LiDAR sees; a gap too short for the planned sweep is not tried."""

import math
from pathlib import Path

import pytest

from mapless.parking import ParallelParker
from mapless.sensor_suite import load_sensor_suite
from mapless.terrain import VehicleBody
from navigation_control.types import ControlCommand
from sim_harness.mapless_road import (
    CarModel,
    KerbLine,
    MaplessRoadWorld,
    ParkedCar,
    RoadScene,
    Straight,
)

ROOT = Path(__file__).resolve().parents[2]
SUITE = load_sensor_suite(ROOT / "configs" / "sensor_suites" / "mapless_v2.json")


def park(gap_m, seconds=70.0, seed=5308, behind=(), max_steer_deg=40.0):
    """Two cars 4.2 m long in the right lane (centres 1.75 m from the kerb), ``gap_m``
    apart; the car starts 30 m back in the left lane (5.25 m from the kerb)."""
    rear_c = 60.0
    front_c = rear_c + 4.2 + gap_m
    cars = tuple(ParkedCar(s, 1.75) for s in behind) + (ParkedCar(rear_c, 1.75), ParkedCar(front_c, 1.75))
    scene = RoadScene(KerbLine([Straight(160)]), road_width_m=11.0, lane_centre_m=5.25, parked=cars)
    world = MaplessRoadWorld(scene, SUITE, start_s_m=30.0, seed=seed, car=CarModel(max_steer_rad=math.radians(max_steer_deg)))
    parker = ParallelParker(VehicleBody(), 3.00465)
    collided = False
    for _ in range(int(seconds / 0.05)):
        cmd = parker.step(world.frame())
        world.apply(ControlCommand(cmd.steering_angle_rad, cmd.throttle, cmd.brake), reverse=cmd.reverse)
        truth = world.truth()
        collided = collided or truth.collision
        if parker.done and parker.state.phase_t > 0.5 or truth.s_m > front_c + 25.0:
            break
    return parker, world.truth(), collided, (rear_c + 2.1, front_c - 2.1)


def test_the_car_parks_between_two_cars_in_a_9_m_gap():
    parker, truth, collided, (gap_rear, gap_front) = park(9.0)
    body = VehicleBody()
    assert parker.state.phase == "parked" and not collided
    assert abs(math.degrees(truth.heading_error_rad)) < 5.0
    kerb = truth.lane_offset_m + 5.25  # the rear axle's distance from the kerb
    assert abs(kerb - 1.75) < 0.5 and kerb - body.half_width_m > 0.2  # in the parked cars' line, clear of the kerb
    rear, front = truth.s_m + body.rear_m, truth.s_m + body.front_m
    assert gap_rear + 0.3 < rear and front < gap_front - 0.3  # the whole body in the gap
    assert abs((rear - gap_rear) - (gap_front - front)) < 0.6  # centred


@pytest.mark.parametrize("gap_m", [6.5, 8.0])  # 8 m: the planned sweep comes within 0.4 m of the front car
def test_a_gap_the_planned_sweep_does_not_fit_is_passed_by(gap_m):
    parker, truth, collided, _ = park(gap_m, seconds=45.0)
    assert parker.state.phase == "search" and parker.state.slot is None and not collided


def test_a_gap_behind_the_start_is_not_the_one_the_car_was_sent_to():
    """park_t3 (PC 1): roadside objects behind the start made a gap and the car reversed
    towards it at once. Two cars 8 m apart behind the start: it parks in the 9 m gap ahead."""
    parker, truth, collided, (gap_rear, gap_front) = park(9.0, behind=(10.0, 22.2))
    assert parker.state.phase == "parked" and not collided and gap_rear < truth.s_m < gap_front


def test_a_car_that_turns_wider_than_planned_finishes_forwards():
    """park_t4 (PC 1): the second arc ran out of room behind with the nose still 23 deg out.
    A car whose wheels reach 29 deg where the plan asks 35 (radius 5.4 m, not 4.3) comes
    up against the rear car with its nose out, and finishes forwards with the wheels the
    other way: parked, no contact, heading under 5 deg."""
    parker, truth, collided, (gap_rear, gap_front) = park(9.0, max_steer_deg=29.0, seconds=90.0)
    body = VehicleBody()
    assert parker.state.phase == "parked" and not collided, parker.record()
    assert abs(math.degrees(truth.heading_error_rad)) < 5.0
    assert gap_rear < truth.s_m + body.rear_m and truth.s_m + body.front_m < gap_front
