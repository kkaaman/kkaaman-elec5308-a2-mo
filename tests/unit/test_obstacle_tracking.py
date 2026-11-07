"""Temporal flicker, identity isolation and aligned depth evidence."""

from dataclasses import replace

import numpy as np
import pytest

from autonomy.interfaces import EgoPose, FrameBundle, VehicleState
from geometry.camera import CameraIntrinsics
from obstacle_perception.tracking import TrackDepthFilter, TrackPolicy
from obstacle_perception.types import (
    CallTiming,
    DetectionBatch,
    DetectionObservation,
    ObservationMetadata,
    PixelBox,
)


def sample(i, ids=(0,), depth=10.0, **metadata):
    frame = FrameBundle(
        "e",
        i,
        i / 10,
        0,
        EgoPose(VehicleState(0, 0, 0, 0)),
        np.zeros((40, 40, 3), np.uint8),
        CameraIntrinsics(40, 40, 20, 20, 20, 20).calibration(np.eye(4)),
        None if depth is None else np.full((40, 40), depth),
        None if depth is None else i,
    )
    m = replace(ObservationMetadata("e", i, i / 10, "rgb", "v1"), **metadata)
    batch = DetectionBatch(
        m,
        tuple(DetectionObservation("car", 0.9, PixelBox(0, 0, 40, 40), x) for x in ids),
        40,
        40,
        CallTiming(0),
    )
    return frame, batch


def tracker(**policy):
    t = TrackDepthFilter(source_id="rgb", model_version="v1", policy=TrackPolicy(**policy))
    t.reset(episode_id="e")
    return t


def test_flicker_and_robust_history():
    t = tracker()
    assert not t.update(*sample(0)).tracks[0].confirmed
    assert not t.update(*sample(1, ids=())).tracks[0].depth_valid
    row = t.update(*sample(2, depth=11.0)).tracks[0]
    assert row.confirmed and row.depth_valid and row.robust_distance_m == 10.5
    row = t.update(*sample(3, depth=90.0)).tracks[0]
    assert row.robust_distance_m == 11
    result = t.update(*sample(4, ids=()))
    assert result.tracks[0].confirmed and not result.tracks[0].depth_valid
    assert result.tracks[0].distance_m is None
    assert not result.usable_for_clearance


def test_id_switch_none_and_expiry():
    t = tracker()
    t.update(*sample(0))
    assert t.update(*sample(1)).tracks[0].confirmed
    result = t.update(*sample(2, ids=(1, None, None)))
    assert not next(x for x in result.tracks if x.identity.track_id == 1).confirmed
    assert len(result.untracked) == 2
    result = t.update(*sample(9, ids=()))
    assert len(result.expired) == 2 and not result.tracks and not result.usable_for_clearance
    assert not t.update(*sample(10)).tracks[0].confirmed


@pytest.mark.parametrize(
    "changes",
    [
        {"episode_id": "other"},
        {"source_id": "other"},
        {"model_version": "v2"},
        {"frame_id": 8},
        {"sim_time_s": 9.0},
    ],
)
def test_association_rejected_without_advancing_state(changes):
    t = tracker()
    f, b = sample(0)
    with pytest.raises(ValueError):
        t.update(f, replace(b, metadata=replace(b.metadata, **changes)))
    assert not t.update(f, b).tracks[0].confirmed


@pytest.mark.parametrize("depth", [None, float("nan"), 1000.0])
def test_invalid_depth_cannot_join_confirmed_track(depth):
    t = tracker()
    t.update(*sample(0))
    row = t.update(*sample(1, depth=depth)).tracks[0]
    assert row.confirmed and not row.depth_valid
    assert row.distance_m is None and row.robust_distance_m is None


def test_invalid_visibility_and_order_reset():
    t = tracker()
    t.update(*sample(0))
    t.update(*sample(1))
    f, b = sample(2, ids=())
    result = t.update(f, replace(b, valid=False, invalid_reason="visibility_unknown"))
    assert not result.tracks[0].depth_valid
    with pytest.raises(ValueError):
        t.update(f, b)
    t.reset(episode_id="e")
    assert not t.update(*sample(0)).tracks[0].confirmed


def test_class_change_resets_and_image_mismatch():
    t = tracker()
    t.update(*sample(0))
    f, b = sample(1)
    with pytest.raises(ValueError):
        t.update(f, replace(b, image_width=41))
    changed = replace(b.detections[0], class_name="person")
    assert not t.update(f, replace(b, detections=(changed,))).tracks[0].confirmed


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hits": 0},
        {"hits": 4},
        {"window": True},
        {"expiry_s": float("nan")},
        {"distance_samples": 0},
    ],
)
def test_policy_validation(kwargs):
    with pytest.raises(ValueError):
        TrackPolicy(**kwargs)
