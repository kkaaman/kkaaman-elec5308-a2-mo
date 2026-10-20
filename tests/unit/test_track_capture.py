"""Capture schema, evidence compatibility and evaluation-only target association."""

import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

from autonomy.interfaces import EgoPose, FrameBundle, VehicleState
from carla_adapter.sensors import CameraRig
from geometry.camera import CameraIntrinsics
from obstacle_perception.tracking import TrackDepthFilter
from obstacle_perception.types import CallTiming, DetectionBatch, ObservationMetadata, PixelBox
from scripts import benchmark_obstacle_tracks as replay
from scripts.capture_obstacle_tracks import (
    capture_observation,
    projected_target,
    read_evidence,
    serialise_frame,
    target_indices,
    validate_environment,
)


def evidence(tmp_path):
    rig = CameraRig()
    depth = dict(
        status="MEASURED_PASS",
        convention="axial",
        rig=asdict(rig),
        server="0.9.16",
        map="Town10HD_Opt",
        opendrive_sha256="map-hash",
    )
    braking = dict(
        status="MEASURED_PASS",
        error=None,
        server="0.9.16",
        map="Carla/Maps/Town10HD_Opt",
        deceleration_mps2=1.5,
        response_budget_s=0.35,
        settings_restored=True,
        no_actor_leaks=True,
    )
    paths = []
    for name, doc in (
        ("depth", depth),
        ("braking", braking),
        ("baseline", {"cruise_speed_mps": 4}),
        ("acceptance", {"margin_max_m": 6}),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(doc))
        paths.append(path)
    return paths


def test_evidence_hashes_and_measured_values(tmp_path):
    docs, rig, provenance = read_evidence(*evidence(tmp_path))
    assert docs["braking"]["deceleration_mps2"] == 1.5
    assert docs["acceptance"]["margin_max_m"] == 6
    assert rig == CameraRig()
    assert all(len(row["sha256"]) == 64 for row in provenance.values())


@pytest.mark.parametrize("kind", ["client", "server", "map", "hash", "rig", "braking"])
def test_changed_environment_rejects_reused_calibration(tmp_path, kind):
    docs, rig, _ = read_evidence(*evidence(tmp_path))
    args = [
        "0.9.16",
        "0.9.16",
        "Carla/Maps/Town10HD_Opt",
        "map-hash",
        rig,
        docs["depth"],
        docs["braking"],
    ]
    if kind in ("client", "server", "map", "hash"):
        args[["client", "server", "map", "hash"].index(kind)] = "changed"
    elif kind == "rig":
        args[4] = CameraRig(width=320)
    else:
        args[6]["server"] = "changed"
    with pytest.raises(ValueError):
        validate_environment(*args)


def test_failed_measurement_cannot_supply_budgets(tmp_path):
    paths = evidence(tmp_path)
    doc = json.loads(paths[1].read_text())
    doc["status"] = "INCOMPLETE"
    paths[1].write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        read_evidence(*paths)


def detection(track_id):
    return SimpleNamespace(
        class_name="car", confidence=0.8, bounding_box=PixelBox(0, 0, 40, 40), track_id=track_id
    )


def test_schema_preserves_miss_nullable_id_and_invalid_observation():
    options = dict(
        gap=20.0, depth_file="depth.npy", rgb_file="rgb.npy", filter_wall_s=0.01, sensor_delay_s=0.2
    )
    miss = serialise_frame(
        1, 0.05, (), CallTiming(0.1), valid=True, reason=None, labels=[], **options
    )
    assert miss["detections"] == [] and miss["valid"] and miss["total_wall_s"] == 0.1
    found = serialise_frame(
        2, 0.1, (detection(None),), CallTiming(0.2), valid=True, reason=None, labels=[0], **options
    )
    assert found["detections"][0]["track_id"] is None
    invalid = serialise_frame(
        3, 0.15, (), None, valid=False, reason="sensor_timeout", labels=[], **options
    )
    assert invalid["total_wall_s"] is None and not invalid["valid"]
    json.dumps([miss, found, invalid], allow_nan=False)
    with pytest.raises(ValueError):
        serialise_frame(4, 0.2, (), None, valid=True, reason=None, labels=[], **options)


def test_labels_reject_occluder_invalid_depth_and_duplicate_matches():
    calibration = CameraIntrinsics(40, 40, 20, 20, 20, 20).calibration(np.eye(4))
    target = dict(xyxy=[0, 0, 40, 40], axial_min_m=9.0, axial_max_m=12.0)
    d = detection(1)
    assert target_indices((d,), np.full((40, 40), 10.0), calibration, target) == [0]
    assert target_indices((d,), np.full((40, 40), 5.0), calibration, target) == []
    assert target_indices((d,), np.full((40, 40), np.nan), calibration, target) == []
    assert target_indices((d, d), np.full((40, 40), 10.0), calibration, target) == []


def test_projected_bbox_uses_carla_forward_axis_and_image_clipping():
    intrinsics = CameraIntrinsics(40, 40, 20, 20, 20, 20)
    vertices = [[10, -1, -1], [10, 1, 1], [12, -1, -1], [12, 1, 1]]
    result = projected_target(vertices, np.eye(4), intrinsics)
    assert result["xyxy"] == [18, 18, 22, 22]
    assert result["axial_min_m"] == 10
    assert projected_target([[-1, 0, 0]], np.eye(4), intrinsics) is None


class SensorStream:
    def __init__(self, frame):
        self.frame = frame

    def read(self, frame_id, timeout_s):
        assert frame_id == self.frame.frame_id
        if frame_id >= 2:
            raise TimeoutError("RGB-D source frame unavailable")
        return SimpleNamespace(to_bundle=lambda *args, **kwargs: self.frame)


def test_live_sensor_failures_match_serialised_replay_at_each_frame(tmp_path, monkeypatch):
    calibration = CameraIntrinsics(40, 40, 20, 20, 20, 20).calibration(np.eye(4))
    pose = EgoPose(VehicleState(0, 0, 0, 0))
    rig = CameraRig(width=40, height=40)
    temporal = TrackDepthFilter(source_id="rgb", model_version="v")
    temporal.reset(episode_id="e")
    detector = SimpleNamespace(track=lambda rgb: (detection(1),), last_call_timing=CallTiming(0.01))
    live, rows = [], []
    for frame_id, sim_time in enumerate((0.0, 0.1, 0.2, 0.3, 0.8)):
        frame = FrameBundle(
            "e",
            frame_id,
            sim_time,
            0,
            pose,
            np.zeros((40, 40, 3), np.uint8),
            calibration,
            np.full((40, 40), 10.0),
            frame_id,
        )
        metadata = ObservationMetadata("e", frame_id, sim_time, "rgb", "v")
        ds, timing, bundle, reason, rgb, depth, wall, result = capture_observation(
            SensorStream(frame), detector, temporal, metadata, pose, calibration, rig, "axial"
        )
        live.append(result)
        np.save(tmp_path / f"{frame_id}.npy", depth)
        rows.append(
            serialise_frame(
                frame_id,
                sim_time,
                ds,
                timing,
                valid=reason is None,
                reason=reason,
                gap=20,
                labels=[0] if ds else [],
                depth_file=f"{frame_id}.npy",
                rgb_file=None,
                filter_wall_s=wall,
                sensor_delay_s=0,
            )
        )
        if frame_id >= 2:
            assert bundle is None and rgb is None and np.isnan(depth).all()
            assert timing is None and "TimeoutError" in reason
            assert not result.usable_for_clearance
            assert all(not t.current and not t.depth_valid for t in result.tracks)
    assert live[1].tracks[0].confirmed
    assert live[2].tracks[0].confirmed and not live[2].tracks[0].current
    assert not live[3].tracks[0].confirmed
    assert not live[4].tracks and len(live[4].expired) == 1
    doc = dict(
        origin="synthetic",
        settings=dict(
            confidence=0.25,
            image_size=640,
            tracker="botsort",
            tracker_settings={},
            model_version="v",
            model_sha256="fixture",
            device="cpu",
            versions={},
        ),
        track_policy={},
        proposed_cruise_mps=4,
        braking_deceleration_mps2=1.5,
        additional_response_s=0.35,
        stopping_margin_m=6,
        sequences=[
            dict(
                name="approach",
                episode_id="e",
                source_id="rgb",
                calibration=asdict(calibration),
                start_sim_time_s=0,
                frames=rows,
            )
        ],
    )
    path = tmp_path / "sequences.json"
    path.write_text(json.dumps(doc, allow_nan=False))
    replayed = []

    class RecordingFilter(TrackDepthFilter):
        def update(self, frame, batch):
            result = super().update(frame, batch)
            replayed.append(result)
            return result

    monkeypatch.setattr(replay, "TrackDepthFilter", RecordingFilter)
    replay.benchmark(json.loads(path.read_text()), tmp_path)
    assert len(replayed) == len(live)
    for recorded, actual in zip(live, replayed):
        assert actual == recorded


@pytest.mark.parametrize(
    "field,value",
    [
        ("episode_id", "other"),
        ("source_id", "other"),
        ("model_version", "other"),
        ("frame_id", 7),
        ("sim_time_s", 0.7),
    ],
)
def test_invalid_failure_requires_matching_metadata(field, value):
    calibration = CameraIntrinsics(40, 40, 20, 20, 20, 20).calibration(np.eye(4))
    frame = FrameBundle(
        "e",
        2,
        0.2,
        0,
        EgoPose(VehicleState(0, 0, 0, 0)),
        np.zeros((40, 40, 3), np.uint8),
        calibration,
    )
    metadata = ObservationMetadata("e", 2, 0.2, "rgb", "v")
    temporal = TrackDepthFilter(source_id="rgb", model_version="v")
    temporal.reset(episode_id="e")
    invalid = DetectionBatch(
        replace(metadata, **{field: value}), (), 40, 40, CallTiming(0), False, "sensor_timeout"
    )
    with pytest.raises(ValueError):
        temporal.update(frame, invalid)
    # Rejection must not consume the correctly associated source observation.
    result = temporal.update(frame, replace(invalid, metadata=metadata))
    assert not result.usable_for_clearance and not result.tracks
