"""Canonical local model factory for the existing E4 latency sweep."""

from pathlib import Path

from autonomy.admission import read_json
from perception.frame_detector import FrameDetector
from perception.object_detection import ObjectDetector
from perception.recorded_detector import verify_local_asset
from perception.types import PerceptionConfig


def make_detector():
    root = Path(__file__).resolve().parents[1]
    freeze = read_json(root / "configs/mission_baseline.json")["baseline_threshold_freeze"]
    digest = freeze["detector"]["model_sha256"]
    weights = verify_local_asset(root / "yolo26n.pt", digest)
    backend = ObjectDetector(PerceptionConfig(model_name=str(weights)))
    return FrameDetector(backend, source_id="replay", model_version="yolo26n:sha256:" + digest)
