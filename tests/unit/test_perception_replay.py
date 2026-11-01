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
    # An injected ID restart is not confirmation continuity.
    records = replay(Path(tmp_path), manifest, rows, detector, "axial", fault="flicker")
    assert not records[0]["truths"][0]["detected"]
    assert records[0]["class_counts"]["car"][2] == 1


def test_depth_convention_must_be_explicit():
    with pytest.raises(ValueError, match="convention"):
        axial_image(np.ones((2, 2)), None, "guess")
