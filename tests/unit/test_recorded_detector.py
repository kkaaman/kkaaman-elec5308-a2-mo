"""Offline fixture identity, reset, missing frames, timing and box compatibility."""

import json
from pathlib import Path

import numpy as np
import pytest

from autonomy.interfaces import CameraCalibration, EgoPose, FrameBundle
from navigation_control.types import VehicleState
from obstacle_perception.boxes import canonical_box
from obstacle_perception.types import PixelBox
from perception.recorded_detector import RecordedDetector, benchmark_calls, verify_local_asset
from perception.types import BoundingBox as ObjectBox
from traffic_perception.detector import ModelPrediction
from traffic_perception.types import BoundingBox as TrafficBox
from traffic_perception.types import TrafficCategory, TrafficDetection

ROOT = Path(__file__).resolve().parents[2]


def detector():
    f = json.loads((ROOT / "configs/models.yaml").read_text())["fixtures"][0]
    return RecordedDetector(ROOT / f["path"], f["sha256"])


def frame(i=0, episode="e"):
    return FrameBundle(
        episode,
        i,
        i * 0.05,
        0,
        EgoPose(VehicleState(0, 0, 0, 0)),
        np.zeros((24, 32, 3), np.uint8),
        CameraCalibration(20, 20, 16, 12, tuple(map(tuple, np.eye(4)))),
    )


def test_reset_replay_empty_and_invalid():
    d = detector()
    with pytest.raises(RuntimeError):
        d.detect(frame())
    d.reset(episode_id="e")
    assert d.detect(frame()).detections == d.detect(frame()).detections
    assert d.track(frame(1)).detections[0].track_id == 1
    assert d.detect(frame(2)).valid and not d.detect(frame(2)).detections
    assert not d.detect(frame(3)).valid
    with pytest.raises(ValueError):
        d.detect(frame(4))
    d.reset(episode_id="new")
    with pytest.raises(RuntimeError):
        d.detect(frame())
    assert d.detect(frame(0, "new")).metadata.episode_id == "new"
    d.close()
    with pytest.raises(RuntimeError):
        d.reset(episode_id="e")


def test_fixture_checksum_and_first_warm_timing(tmp_path):
    p = tmp_path / "asset"
    p.write_bytes(b"bad")
    with pytest.raises(ValueError):
        verify_local_asset(p, "0" * 64)
    result = benchmark_calls(detector, frame())
    assert len(result["warm_call_wall_s"]) == 3
    assert result["first_call_wall_s"] >= 0 and result["construction_wall_s"] >= 0


@pytest.mark.parametrize("cls", [ObjectBox, TrafficBox, PixelBox])
@pytest.mark.parametrize(
    "coordinates", [(0, 0, 0, 1), (2, 0, 1, 1), (-1, 0, 1, 1), (0, 0, float("nan"), 1)]
)
def test_all_box_constructors_reject_invalid(cls, coordinates):
    with pytest.raises(ValueError):
        cls(*coordinates)


def test_legacy_traffic_constructors_keep_shape():
    box = TrafficBox(0, 0, 2, 3)
    assert box.width_px == 2 and box.height_px == 3
    p = ModelPrediction(TrafficCategory.LIGHT, "red", 0.9, box)
    d = TrafficDetection(p.category, p.label, p.confidence, p.bounding_box)
    assert isinstance(d.bounding_box, PixelBox)
    assert canonical_box(d.bounding_box) is d.bounding_box


def test_canonical_box_traffic_backend_runs_once():
    from traffic_perception.detector import TrafficPerception

    calls = []
    prediction = ModelPrediction(TrafficCategory.SIGN, "stop", 0.9, PixelBox(0, 0, 4, 8))

    def backend(image):
        calls.append(image.shape)
        return [prediction]

    pipeline = TrafficPerception(backend, focal_length_px=20, object_widths_m={"stop": 2})
    result = pipeline.detect(np.zeros((24, 32, 3), dtype=np.uint8))
    assert len(calls) == 1 and result[0].distance_m == 10


def test_replay_to_depth_association_preserves_source_identity():
    from dataclasses import replace

    from obstacle_perception.box_depth import estimate_batch_depth

    f = replace(frame(), depth_m=np.full((24, 32), 10.0, np.float32), depth_frame_id=0)
    d = detector()
    d.reset(episode_id="e")
    batch = d.detect(f)
    assert estimate_batch_depth(f, batch)[0].valid
    with pytest.raises(ValueError, match="mismatch"):
        estimate_batch_depth(replace(f, episode_id="another"), batch)
    with pytest.raises(ValueError, match="aligned depth"):
        estimate_batch_depth(frame(), batch)
