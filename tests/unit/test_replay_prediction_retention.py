"""New evidence must preserve every measured frame for later independent scoring."""

import json
import sys
from types import ModuleType

import numpy as np

from obstacle_perception.types import CallTiming, DetectionBatch, ObservationMetadata
from scripts.benchmark_remote_detector import main


def test_rgb_selection_and_predictions_survive_without_accuracy_claim(tmp_path, monkeypatch):
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(200):
        np.save(frames / f"{i:03d}-rgb.npy", np.zeros((4, 4, 3), np.uint8))
    np.save(frames / "000-depth.npy", np.zeros((4, 4), np.float32))

    class Detector:
        def reset(self, **kw):
            pass

        def close(self):
            pass

        def detect(self, f):
            return DetectionBatch(
                ObservationMetadata(f.episode_id, f.frame_id, f.sim_time_s, "replay", "m"),
                (),
                4,
                4,
                CallTiming(0),
            )

    factory = ModuleType("test_replay_factory")
    factory.create = Detector
    monkeypatch.setitem(sys.modules, "test_replay_factory", factory)
    output = tmp_path / "out"
    assert (
        main(
            [
                "--frames",
                str(frames),
                "--frame-pattern",
                "*-rgb.npy",
                "--output",
                str(output),
                "--local-factory",
                "test_replay_factory:create",
                "--model-version",
                "m",
                "--hardware-notes",
                "fixture, not performance",
                "--origin",
                "synthetic-smoke",
                "--sizes",
                "4x4",
                "--warmup",
                "1",
            ]
        )
        == 0
    )
    manifest = json.loads((output / "summary.json").read_text())
    predictions = [json.loads(x) for x in (output / "predictions.jsonl").read_text().splitlines()]
    assert len(predictions) == 200
    assert {r["frame"] for r in predictions} == {r["file"] for r in manifest["replay"]}
    assert manifest["accuracy"]["status"] == "NOT_MEASURED"
    assert manifest["modes"]["LAN"]["status"] == "NOT RUN"
    assert all(row["status"] == "OK" and row["detections"] == [] for row in predictions)
