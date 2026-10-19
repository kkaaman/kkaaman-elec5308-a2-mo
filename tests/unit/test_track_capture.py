"""Capture schema, evidence compatibility and evaluation-only target association."""

import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from carla_adapter.sensors import CameraRig
from geometry.camera import CameraIntrinsics
from obstacle_perception.types import CallTiming, PixelBox
from scripts.capture_obstacle_tracks import (
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
