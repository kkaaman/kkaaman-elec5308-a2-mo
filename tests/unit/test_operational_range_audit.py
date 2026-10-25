"""Admission diagnostics count overlapping causes without changing the policy."""

from types import SimpleNamespace

import numpy as np

from obstacle_perception.box_depth import DepthPolicy
from obstacle_perception.types import PixelBox
from scripts.review_operational_range import (
    admission_failures,
    roi_diagnostic,
    summarize_rejections,
)


def test_every_failed_policy_conjunct_is_retained():
    tracks = SimpleNamespace(
        untracked=(object(),),
        expired=(object(),),
        tracks=(SimpleNamespace(current=False, depth_valid=False),),
    )
    observation = SimpleNamespace(
        valid=False, visible_distance_m=None, obstacle_gap_m=None, clearance_valid=False
    )
    assert admission_failures(tracks, observation, ready=False) == sorted(
        [
            "untracked_missing_id",
            "expired_tracks",
            "noncurrent_tracks",
            "invalid_track_depth_or_confirmation",
            "provider_invalid",
            "missing_visible_distance",
            "missing_obstacle_and_clearance",
            "warmup_not_ready",
        ]
    )


def test_valid_observation_has_no_reported_failure():
    tracks = SimpleNamespace(
        untracked=(), expired=(), tracks=(SimpleNamespace(current=True, depth_valid=True),)
    )
    observation = SimpleNamespace(
        valid=True, visible_distance_m=20, obstacle_gap_m=20, clearance_valid=False
    )
    assert admission_failures(tracks, observation, ready=True) == []


def test_frame_counts_do_not_multiply_track_instances_and_keep_combinations():
    row = dict(
        admission_failures=["provider_invalid", "invalid_track_depth_or_confirmation"],
        source_delay_exceeds_ttl=True,
        processing_lower_bound_over_budget=False,
        current_invalid_tracks=[
            dict(
                reason="insufficient_valid_depth",
                depth_diagnostic_cause="depth_support_outside_25m_scope",
                evaluation_labelled_target=False,
            )
        ]
        * 2,
    )
    result = summarize_rejections([row])
    assert result["frame_predicate_failures"]["provider_invalid"] == 1
    assert result["current_invalid_reason_frames"]["insufficient_valid_depth"] == 1
    assert result["current_invalid_reason_instances"]["insufficient_valid_depth"] == 2
    assert sum(result["predicate_combinations"].values()) == 1
    assert result["frames_with_invalid_not_labelled_target"] == 1
    assert list(result["joint_known_failure_combinations"]) == [
        "invalid_track_depth_or_confirmation & provider_invalid & source_command_expired"
    ]


def test_out_of_scope_raw_depth_is_diagnostic_not_accepted_depth():
    from geometry.camera import CameraIntrinsics
    from obstacle_perception.box_depth import estimate_box_depth

    depth = np.full((40, 40), 31.6)
    box = PixelBox(0, 0, 40, 40)
    policy = DepthPolicy(max_depth_m=25)
    raw = roi_diagnostic(depth, box, policy)
    assert raw["raw_median_m"] == 31.6
    assert raw["below_depth_limit_pixels"] == 0
    estimate = estimate_box_depth(
        depth, box, CameraIntrinsics(40, 40, 20, 20, 20, 20).calibration(np.eye(4)), policy
    )
    assert not estimate.valid and estimate.reason == "insufficient_valid_depth"
