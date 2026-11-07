"""Adapted PR69 replay exercises current tracking, depth and failure accounting."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from evaluation.perception_replay import axial_image, range_summary, replay
from obstacle_perception.types import PixelBox
from perception.frame_detector import FrameDetector
from perception.types import Detection


class Backend:
    generation = 0

    def reset(self, *, episode_id):
        self.episode_id = episode_id
        self.generation += 1

    def track(self, rgb):
        return (Detection("car", 0.9, PixelBox(5, 5, 35, 35), 1),)

    def close(self):
        pass


def test_replay_uses_actual_current_track_filter_and_reports_first_range(tmp_path):
    Image.fromarray(np.zeros((40, 40, 3), np.uint8)).save(tmp_path / "rgb.png")
    instance = np.zeros((40, 40, 3), np.uint8)
    instance[5:35, 5:35, 0] = 14
    instance[5:35, 5:35, 2] = 1
    Image.fromarray(instance).save(tmp_path / "instance.png")
    np.save(tmp_path / "depth.npy", np.full((40, 40), 10.0, np.float32))
    rows = [
        dict(
            sample_id=str(i),
            sequence_id="test-scene",
            sequence_index=i,
            frame_id=i,
            sim_time_s=i * 0.05,
            rgb="rgb.png",
            instance="instance.png",
            depth="depth.npy",
            visible_boxes=[dict(class_name="car", instance_code=1, xyxy=[5, 5, 35, 35])],
        )
        for i in range(3)
    ]
    detector = FrameDetector(Backend(), source_id="recording", model_version="fixture")
    manifest = dict(image_size_wh=[40, 40], camera=dict(fov_deg=90, x_actor_m=1.5, z_actor_m=2))
    records = replay(Path(tmp_path), manifest, rows, detector, "axial")
    assert records[0]["truths"][0]["confirmed"] is False
    assert records[1]["truths"][0]["confirmed"] is True
    summary = range_summary(records)
    assert summary["class_metrics"]["car"] == dict(tp=3, fp=0, fn=0)
    assert summary["sequences"][0]["first_confirmed_distance_m"] == 10
    assert summary["sequences"][0]["confirmation_s"] == pytest.approx(0.05)
    assert summary["usable_range_lower_bound_m"]["car"] == 0
    assert summary["status"] == "MEASURED_NOT_OPERATIONAL_ACCEPTANCE"
    from obstacle_perception.box_depth import DepthPolicy

    np.save(tmp_path / "depth.npy", np.full((40, 40), 30.0, np.float32))
    limited = replay(
        Path(tmp_path), manifest, rows, detector, "axial", depth_policy=DepthPolicy(max_depth_m=25)
    )
    assert all(
        not t["valid_depth"] and not t["confirmed"] for row in limited for t in row["truths"]
    )
    assert range_summary(limited)["class_metrics"]["car"] == dict(tp=3, fp=0, fn=0)
    assert sum(row["invalid_depth_count"] for row in limited) == 3
    np.save(tmp_path / "depth.npy", np.full((40, 40), 10.0, np.float32))
    # An injected ID restart is not confirmation continuity.
    records = replay(Path(tmp_path), manifest, rows, detector, "axial", fault="flicker")
    assert not records[0]["truths"][0]["detected"]
    assert records[0]["class_counts"]["car"][2] == 1


def test_depth_convention_must_be_explicit():
    with pytest.raises(ValueError, match="convention"):
        axial_image(np.ones((2, 2)), None, "guess")


def test_frozen_policy_rescoring_reuses_predictions_without_model(tmp_path, monkeypatch):
    import json

    from scripts import benchmark_perception as script

    Image.fromarray(np.zeros((40, 40, 3), np.uint8)).save(tmp_path / "rgb.png")
    mask = np.zeros((40, 40, 3), np.uint8)
    mask[5:35, 5:35, 0] = 14
    mask[5:35, 5:35, 2] = 1
    Image.fromarray(mask).save(tmp_path / "instance.png")
    np.save(tmp_path / "depth.npy", np.full((40, 40), 30.0, np.float32))
    (tmp_path / "manifest.json").write_text("{}")
    manifest = dict(
        image_size_wh=[40, 40],
        camera=dict(fov_deg=90, x_actor_m=1.5, z_actor_m=2),
        opendrive_sha256="map",
    )
    rows = [
        dict(
            sample_id=str(i),
            split="test",
            sequence_id="scene",
            sequence_index=i,
            frame_id=i,
            sim_time_s=i * 0.05,
            rgb="rgb.png",
            instance="instance.png",
            depth="depth.npy",
            visible_boxes=[dict(class_name="car", instance_code=1, xyxy=[5, 5, 35, 35])],
        )
        for i in range(2)
    ]
    old = tmp_path / "original"
    old.mkdir()
    model = "yolo26n:sha256:" + "a" * 64
    (old / "summary.json").write_text(json.dumps(dict(model_version=model)))
    records = [
        dict(
            sample_id=str(i),
            frame_id=i,
            sim_time_s=i * 0.05,
            valid=True,
            predictions=[dict(class_name="car", confidence=0.9, xyxy=[5, 5, 35, 35], track_id=1)],
        )
        for i in range(2)
    ]
    original = old / "records.jsonl"
    original.write_text("\n".join(map(json.dumps, records)) + "\n")
    before = original.read_bytes()
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            dict(
                status="MEASURED_PASS",
                opendrive_sha256="map",
                convention="axial",
                rig=dict(manifest["camera"], width=40, height=40),
            )
        )
    )
    monkeypatch.setattr(script, "read_dataset", lambda folder: (manifest, rows))
    monkeypatch.setattr(
        script,
        "load_admitted_baseline",
        lambda path: dict(
            baseline_threshold_freeze=dict(
                detector=dict(model_sha256="a" * 64),
                track_policy={},
                depth_policy=dict(max_depth_m=25),
            )
        ),
    )
    monkeypatch.setattr(script, "make_detector", lambda: pytest.fail("Inference must not repeat"))
    output = tmp_path / "rescored"
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark",
            "--dataset",
            str(tmp_path),
            "--depth-calibration",
            str(calibration),
            "--predictions",
            str(original),
            "--output",
            str(output),
        ],
    )
    assert script.main() == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["depth_policy"]["max_depth_m"] == 25
    assert summary["class_metrics"]["car"]["tp"] == 2
    assert summary["sequences"][0]["confirmation_s"] is None
    assert original.read_bytes() == before
