"""Fixed-route episode lifecycle and explicit obstacle-hold regression tests."""

import math
from unittest.mock import patch

import pytest

from autonomy.mission import MissionEpisode
from navigation_control import (
    ControllerConfig,
    ControlStatus,
    PathFollower,
    VehicleState,
    Waypoint,
)

ROUTE_SHA256 = "a" * 64
STRAIGHT_ROUTE = (Waypoint(0, 0), Waypoint(50, 0))


def episode(waypoints=STRAIGHT_ROUTE):
    return MissionEpisode(
        ControllerConfig(wheelbase_m=2.8, max_steering_angle_rad=0.5),
        waypoints,
        ROUTE_SHA256,
    )


def test_hold_resume_installs_route_once_and_preserves_progress():
    original_set_path = PathFollower.set_path
    with patch.object(PathFollower, "set_path", autospec=True) as install:
        install.side_effect = original_set_path
        mission = episode()
        cruise = mission.step(VehicleState(10, 0, 0, 2), 3, 0.05)
        mission.step(VehicleState(12, 0, 0, 1), 3, 0.05, hold=True)
        held = mission.step(VehicleState(12, 0, 0, 0), 3, 0.05, hold=True)
        resumed = mission.step(VehicleState(11.9, 0, 0, 0), 3, 0.05)

    assert install.call_count == mission.route_install_count == 1
    assert mission.status == "running"
    assert mission.step_count == 4
    assert mission.progress_m == held.diagnostics.progress_m == 12
    assert mission.progress_fraction == pytest.approx(12 / 50)
    assert held.diagnostics.progress_m >= cruise.diagnostics.progress_m
    assert held.status == ControlStatus.STOPPING
    assert held.command.throttle == 0
    assert held.command.brake > 0
    assert resumed.command.throttle > 0
    assert [(event.step_index, event.event) for event in mission.events] == [
        (0, "route_installed"),
        (2, "hold_started"),
        (4, "hold_released"),
    ]
    assert all(event.route_sha256 == ROUTE_SHA256 for event in mission.events)


def test_hold_near_goal_does_not_latch_completion_and_can_resume():
    mission = episode()
    held = mission.step(VehicleState(49, 0, 0, 0), 3, 0.05, hold=True)
    assert held.diagnostics.distance_to_goal_m == 1
    assert held.status == ControlStatus.STOPPING
    assert mission.status == "holding"
    assert held.command.brake > 0

    # Small position noise outside the goal zone must not reveal a hidden latch.
    resumed = mission.step(VehicleState(48.9, 0, 0, 0), 3, 0.05)
    assert resumed.status != ControlStatus.FINISHED
    assert resumed.command.throttle > 0
    assert mission.progress_m == 49
    assert not any(event.event == "mission_finished" for event in mission.events)


def test_hold_at_endpoint_only_finishes_after_hold_is_released():
    mission = episode()
    for _ in range(3):
        held = mission.step(VehicleState(50, 0, 0, 0), 3, 0.05, hold=True)
        assert held.status == ControlStatus.STOPPING
        assert mission.status == "holding"
    released = mission.step(VehicleState(50, 0, 0, 0), 3, 0.05)
    assert released.status == ControlStatus.FINISHED
    assert mission.status == "finished"
    assert [event.event for event in mission.events] == [
        "route_installed",
        "hold_started",
        "hold_released",
        "mission_finished",
    ]


def test_hold_keeps_steering_on_a_curve_while_braking():
    mission = episode((Waypoint(0, 0), Waypoint(10, 0), Waypoint(20, 10)))
    held = mission.step(VehicleState(9, 0, 0, 3), 3, 0.05, hold=True)
    assert held.command.throttle == 0
    assert held.command.brake > 0
    assert held.command.steering_angle_rad > 0
    assert held.status == ControlStatus.STOPPING
    resumed = mission.step(VehicleState(9, 0, 0, 0), 3, 0.05)
    assert resumed.command.throttle > 0
    assert resumed.command.steering_angle_rad > 0
    assert mission.route_install_count == 1


def test_endpoint_braking_and_finished_latch_are_preserved():
    mission = episode()
    approaching = mission.step(VehicleState(49, 0, 0, 2), 3, 0.05)
    assert approaching.status == ControlStatus.STOPPING
    assert approaching.command.brake > 0
    assert 0 < approaching.diagnostics.internal_target_speed_mps < 3
    final = mission.step(VehicleState(50, 0, 0, 0), 3, 0.05)
    assert final.status == ControlStatus.FINISHED
    later = mission.step(VehicleState(40, 0, 0, 0), 3, 0.05, hold=True)
    assert later.status == ControlStatus.FINISHED
    assert mission.status == "finished"
    assert later.command.throttle == 0
    assert later.command.brake > 0
    assert [event.event for event in mission.events].count("mission_finished") == 1


def test_new_episode_has_fresh_completion_progress_and_event_history():
    first = episode()
    first.step(VehicleState(50, 0, 0, 0), 3, 0.05)
    second = episode()
    assert second.status == "ready"
    assert second.progress_m == second.progress_fraction == second.step_count == 0
    assert second.route_sha256 == ROUTE_SHA256
    assert len(second.events) == second.route_install_count == 1
    initial = second.step(VehicleState(0, 0, 0, 0), 3, 0.05)
    assert initial.status == ControlStatus.TRACKING
    assert initial.command.throttle > 0
    assert first.status == "finished"


@pytest.mark.parametrize("speed", [-1, math.nan, math.inf, True, "3"])
def test_invalid_speed_is_not_hidden_by_hold_or_recorded(speed):
    mission = episode()
    with pytest.raises(ValueError, match="target_speed_mps"):
        mission.step(VehicleState(20, 0, 0, 0), speed, 0.05, hold=True)
    assert mission.status == "ready"
    assert mission.step_count == mission.progress_m == 0
    assert len(mission.events) == 1


@pytest.mark.parametrize("dt_s", [0, -1, math.nan, True])
def test_invalid_timestep_does_not_advance_lifecycle(dt_s):
    mission = episode()
    with pytest.raises(ValueError):
        mission.step(VehicleState(20, 0, 0, 0), 3, dt_s, hold=True)
    result = mission.step(VehicleState(0, 0, 0, 0), 3, 0.05)
    assert result.diagnostics.progress_m == 0
    assert mission.step_count == 1
    assert len(mission.events) == 1


@pytest.mark.parametrize("hold", [0, 1, None, "hold"])
def test_hold_requires_explicit_bool(hold):
    mission = episode()
    with pytest.raises(TypeError, match="hold must be bool"):
        mission.step(VehicleState(0, 0, 0, 0), 3, 0.05, hold=hold)
    assert mission.step_count == 0


@pytest.mark.parametrize("digest", [None, "", "a" * 63, "g" * 64, "A" * 64])
def test_route_identity_requires_a_sha256_digest(digest):
    with pytest.raises(ValueError, match="route_sha256"):
        MissionEpisode(ControllerConfig(2.8, 0.5), STRAIGHT_ROUTE, digest)


def test_failed_episode_retains_first_reason_and_rejects_further_control():
    mission = episode()
    mission.step(VehicleState(10, 0, 0, 2), 3, 0.05)
    mission.fail("simulation_timeout")
    mission.fail("later_cleanup_error")
    assert mission.status == "failed"
    assert mission.failure_reason == "simulation_timeout"
    assert mission.progress_m == 10
    assert [event.event for event in mission.events].count("mission_failed") == 1
    with pytest.raises(RuntimeError, match="failed mission"):
        mission.step(VehicleState(20, 0, 0, 0), 3, 0.05)


def test_completion_permission_is_validated_without_changing_legacy_behavior():
    follower = PathFollower(ControllerConfig(2.8, 0.5))
    follower.set_path(STRAIGHT_ROUTE)
    with pytest.raises(TypeError, match="completion_allowed"):
        follower.update(VehicleState(50, 0, 0, 0), 3, 0.05, completion_allowed=0)
    held = follower.update(
        VehicleState(50, 0, 0, 0), 0, 0.05, completion_allowed=False
    )
    assert held.status == ControlStatus.STOPPING
    legacy = follower.update(VehicleState(50, 0, 0, 0), 0, 0.05)
    assert legacy.status == ControlStatus.FINISHED
    still_finished = follower.update(
        VehicleState(40, 0, 0, 0), 0, 0.05, completion_allowed=False
    )
    assert still_finished.status == ControlStatus.FINISHED
