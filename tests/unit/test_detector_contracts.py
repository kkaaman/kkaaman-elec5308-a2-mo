"""Detector and obstacle contracts (BL-C-01): one protocol, namespaced
identities, explicit validity, and obstacles with source/frame/extent/confidence."""

import math

import numpy as np
import pytest

from autonomy.interfaces import Detection2D, DetectionBatch, Detector, ImageBox
from obstacle_perception.types import (
    Footprint,
    Obstacle,
    ObstacleClass,
    ObstacleSourceResult,
    RangeMethod,
    SourceHealth,
)
from sim_harness.fake_detectors import FakeLocalDetector, FakeRemoteDetector, ScriptedObject

RGB = np.zeros((6, 8, 3), np.uint8)
CAR = ScriptedObject("car-1", "car", 0.9, ImageBox(1, 1, 4, 3))
PED = ScriptedObject("ped-1", "person", 0.7, ImageBox(5, 1, 6, 4))


def scene(frame_id):
    return [CAR] if frame_id < 3 else [PED, CAR]


@pytest.fixture(params=[FakeLocalDetector, FakeRemoteDetector])
def detector(request):
    det = request.param(scene)
    det.reset("episode-1")
    return det


# ------------------------------------------------------------------ the protocol
def test_local_and_remote_fakes_satisfy_one_protocol(detector):
    assert isinstance(detector, Detector)
    batch = detector.infer(RGB, frame_id=0)
    assert batch.valid and batch.frame_id == 0 and batch.source == detector.source
    assert batch.session_id == "episode-1" and batch.model_version == detector.model_version
    assert batch.latency_s >= 0


def test_track_ids_persist_within_a_session(detector):
    first = detector.infer(RGB, 0).detections
    later = detector.infer(RGB, 4).detections
    car_id = first[0].track_id
    assert {d.class_name: d.track_id for d in later}["car"] == car_id
    assert {d.class_name: d.track_id for d in later}["person"] != car_id


def test_reset_drops_track_history(detector):
    detector.infer(RGB, 4)                       # build up some track history
    detector.reset("episode-2")
    batch = detector.infer(RGB, 0)
    assert batch.session_id == "episode-2"
    assert batch.detections[0].track_id == 1     # numbering restarted
    assert batch.track_key(batch.detections[0]) == f"{detector.source}/episode-2/1"


def test_detect_without_tracking_has_no_ids(detector):
    batch = detector.infer(RGB, 0, track=False)
    assert not batch.tracked and all(d.track_id is None for d in batch.detections)
    assert batch.track_key(batch.detections[0]) is None


def test_identities_are_namespaced_by_source():
    local, remote = FakeLocalDetector(scene), FakeRemoteDetector(scene)
    local.reset("s")
    remote.reset("s")
    lb, rb = local.infer(RGB, 0), remote.infer(RGB, 0)
    assert lb.detections[0].track_id == rb.detections[0].track_id == 1
    assert lb.track_key(lb.detections[0]) != rb.track_key(rb.detections[0])


def test_remote_outage_is_an_invalid_batch_not_an_empty_one():
    remote = FakeRemoteDetector(scene, outage_frames=[2])
    remote.reset("s")
    batch = remote.infer(RGB, 2)
    assert not batch.valid and batch.reason == "remote timeout" and batch.detections == ()


def test_infer_before_reset_is_an_error():
    with pytest.raises(RuntimeError, match="reset"):
        FakeLocalDetector(scene).infer(RGB, 0)


# ------------------------------------------------------------- batch validation
def _batch(**kw):
    base = dict(frame_id=1, source="local", session_id="s", model_version="m",
                detections=(), tracked=True, host_start_s=1.0, host_end_s=1.01)
    base.update(kw)
    return DetectionBatch(**base)


def test_batch_rules():
    det = Detection2D("car", 0.9, ImageBox(0, 0, 2, 2), track_id=4)
    with pytest.raises(ValueError, match="reason"):
        _batch(valid=False)
    with pytest.raises(ValueError, match="cannot carry"):
        _batch(valid=False, reason="x", detections=(det,))
    with pytest.raises(ValueError, match="untracked"):
        _batch(tracked=False, detections=(det,))
    with pytest.raises(ValueError, match="precede"):
        _batch(host_end_s=0.5)


def test_inverted_or_empty_boxes_are_rejected():
    with pytest.raises(ValueError):
        ImageBox(5, 1, 4, 3)
    with pytest.raises(ValueError):
        ImageBox(1, 1, 1, 3)
    with pytest.raises(ValueError, match="confidence"):
        Detection2D("car", 1.2, ImageBox(0, 0, 1, 1))


# ---------------------------------------------------------------------- obstacles
def obstacle(**kw):
    base = dict(
        source="rgbd", frame_id=7, sim_time_s=0.7,
        obstacle_class=ObstacleClass.from_label("car"), label="car", confidence=0.8,
        footprint_vehicle=Footprint("vehicle", 15.0, 0.2, 0.0, 4.5, 1.9),
        range_m=12.3, range_sigma_m=0.3, range_method=RangeMethod.DEPTH_MEDIAN,
        track_key="local/episode-1/3",
    )
    base.update(kw)
    return Obstacle(**base)


def test_obstacle_carries_source_frame_extent_and_confidence():
    o = obstacle()
    assert (o.source, o.frame_id, o.confidence) == ("rgbd", 7, 0.8)
    assert o.obstacle_class is ObstacleClass.VEHICLE
    xs = [x for x, _ in o.footprint_vehicle.corners()]
    assert min(xs) == pytest.approx(12.75) and max(xs) == pytest.approx(17.25)


def test_footprint_corners_rotate_counter_clockwise():
    fp = Footprint("world", 0.0, 0.0, math.pi / 2, 4.0, 2.0)
    front_left = fp.corners()[0]
    assert front_left == pytest.approx((-1.0, 2.0))


@pytest.mark.parametrize(
    "kw, match",
    [
        (dict(track_key="3"), "namespaced"),
        (dict(range_m=0.0), "range_m"),
        (dict(footprint_vehicle=Footprint("world", 1, 0, 0, 1, 1)), "vehicle frame"),
        (dict(route_s_m=(5.0, 2.0)), "route_s_m"),
        (dict(confidence=1.5), "confidence"),
    ],
)
def test_obstacle_validation(kw, match):
    with pytest.raises(ValueError, match=match):
        obstacle(**kw)


def test_label_mapping_and_fallback_flags():
    assert ObstacleClass.from_label("person") is ObstacleClass.PEDESTRIAN
    assert ObstacleClass.from_label("traffic cone") is ObstacleClass.UNKNOWN
    assert RangeMethod.KNOWN_WIDTH.is_fallback and not RangeMethod.DEPTH_MEDIAN.is_fallback


# ----------------------------------------------------------- source result logic
def test_no_obstacles_is_clear_only_when_healthy_and_seeing_far_enough():
    ok = ObstacleSourceResult("rgbd", 7, SourceHealth.OK, (), coverage_range_m=30.0)
    short = ObstacleSourceResult("rgbd", 7, SourceHealth.OK, (), coverage_range_m=10.0)
    degraded = ObstacleSourceResult("rgbd", 7, SourceHealth.DEGRADED, (), 30.0, reason="depth NaN")
    assert ok.reports_clear(25.0)
    assert not short.reports_clear(25.0)
    assert not degraded.reports_clear(25.0)


def test_failed_source_cannot_report_obstacles_and_needs_reason():
    with pytest.raises(ValueError, match="reason"):
        ObstacleSourceResult("rgbd", 7, SourceHealth.FAILED, (), 0.0)
    with pytest.raises(ValueError, match="cannot report"):
        ObstacleSourceResult("rgbd", 7, SourceHealth.FAILED, (obstacle(),), 0.0, reason="x")


def test_result_rejects_obstacles_from_another_frame_or_source():
    with pytest.raises(ValueError, match="frame"):
        ObstacleSourceResult("rgbd", 8, SourceHealth.OK, (obstacle(),), 30.0)
    with pytest.raises(ValueError, match="source"):
        ObstacleSourceResult("lidar", 7, SourceHealth.OK, (obstacle(),), 30.0)
