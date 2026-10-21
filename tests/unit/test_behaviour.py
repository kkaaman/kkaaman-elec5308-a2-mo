"""Conservative clearance-guard tests for BL-D-05."""

import pytest

from planning.behaviour import ClearanceGuard, ClearanceGuardConfig


def test_fresh_clear_route_is_immediately_clear_before_any_blocker():
    guard = ClearanceGuard()

    assert guard.update(
        time_s=0.0,
        obstacle_present=False,
        observation_valid=True,
    )


def test_obstacle_sets_blocked_latch_and_prevents_clearance():
    guard = ClearanceGuard()

    assert not guard.update(
        time_s=0.0,
        obstacle_present=True,
        observation_valid=True,
    )
    assert guard.blocked_latch
    assert guard.clear_count == 0


def test_invalid_observation_cannot_clear_blocked_route():
    guard = ClearanceGuard(
        ClearanceGuardConfig(
            clear_frames=2,
            clear_duration_s=0.1,
        )
    )

    guard.update(
        time_s=0.0,
        obstacle_present=True,
        observation_valid=True,
    )

    assert not guard.update(
        time_s=0.1,
        obstacle_present=False,
        observation_valid=False,
    )
    assert guard.blocked_latch
    assert guard.clear_count == 0


def test_missing_track_does_not_clear_without_fresh_valid_evidence():
    guard = ClearanceGuard(
        ClearanceGuardConfig(
            clear_frames=2,
            clear_duration_s=0.1,
        )
    )

    guard.update(
        time_s=0.0,
        obstacle_present=True,
        observation_valid=True,
    )

    assert not guard.update(
        time_s=0.1,
        obstacle_present=False,
        observation_valid=False,
    )
    assert guard.blocked_latch


def test_clearance_requires_both_frame_count_and_duration():
    guard = ClearanceGuard(
        ClearanceGuardConfig(
            clear_frames=3,
            clear_duration_s=0.5,
        )
    )

    guard.update(
        time_s=0.0,
        obstacle_present=True,
        observation_valid=True,
    )

    assert not guard.update(
        time_s=0.1,
        obstacle_present=False,
        observation_valid=True,
    )
    assert not guard.update(
        time_s=0.3,
        obstacle_present=False,
        observation_valid=True,
    )

    # Frame count is now satisfied, but duration is still too short.
    assert not guard.update(
        time_s=0.5,
        obstacle_present=False,
        observation_valid=True,
    )

    # Both requirements are now satisfied.
    assert guard.update(
        time_s=0.6,
        obstacle_present=False,
        observation_valid=True,
    )

    assert not guard.blocked_latch
    assert guard.clear_count == 0


def test_obstacle_reappearing_resets_accumulated_clearance():
    guard = ClearanceGuard(
        ClearanceGuardConfig(
            clear_frames=2,
            clear_duration_s=0.2,
        )
    )

    guard.update(
        time_s=0.0,
        obstacle_present=True,
        observation_valid=True,
    )

    assert not guard.update(
        time_s=0.1,
        obstacle_present=False,
        observation_valid=True,
    )
    assert guard.clear_count == 1

    assert not guard.update(
        time_s=0.2,
        obstacle_present=True,
        observation_valid=True,
    )
    assert guard.clear_count == 0
    assert guard.blocked_latch


def test_invalid_observation_resets_accumulated_clearance():
    guard = ClearanceGuard(
        ClearanceGuardConfig(
            clear_frames=2,
            clear_duration_s=0.2,
        )
    )

    guard.update(
        time_s=0.0,
        obstacle_present=True,
        observation_valid=True,
    )

    assert not guard.update(
        time_s=0.1,
        obstacle_present=False,
        observation_valid=True,
    )
    assert guard.clear_count == 1

    assert not guard.update(
        time_s=0.2,
        obstacle_present=False,
        observation_valid=False,
    )
    assert guard.clear_count == 0
    assert guard.blocked_latch


def test_time_must_advance_monotonically():
    guard = ClearanceGuard()

    guard.update(
        time_s=1.0,
        obstacle_present=False,
        observation_valid=True,
    )

    with pytest.raises(ValueError, match="advance monotonically"):
        guard.update(
            time_s=1.0,
            obstacle_present=False,
            observation_valid=True,
        )


@pytest.mark.parametrize(
    "config",
    [
        ClearanceGuardConfig(clear_frames=1, clear_duration_s=0.1),
        ClearanceGuardConfig(clear_frames=5, clear_duration_s=0.4),
    ],
)
def test_valid_configs_construct(config):
    guard = ClearanceGuard(config)

    assert guard.config == config


@pytest.mark.parametrize(
    ("clear_frames", "clear_duration_s"),
    [
        (0, 0.4),
        (-1, 0.4),
        (1.5, 0.4),
        (True, 0.4),
        (5, 0.0),
        (5, -0.1),
        (5, float("inf")),
        (5, float("nan")),
    ],
)
def test_invalid_configs_are_rejected(clear_frames, clear_duration_s):
    with pytest.raises(ValueError):
        ClearanceGuardConfig(
            clear_frames=clear_frames,
            clear_duration_s=clear_duration_s,
        )