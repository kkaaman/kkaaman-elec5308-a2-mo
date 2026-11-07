"""Pristine observer composition retains invalidity and explicit evaluation scope."""

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from scripts import evaluate_lane_dataset as module


def test_pose_conversion_and_missing_prediction_are_not_accuracy_pass(tmp_path, monkeypatch):
    Image.fromarray(np.zeros((40, 40, 3), np.uint8)).save(tmp_path / "rgb.png")
    Image.fromarray(np.zeros((40, 40), np.uint8)).save(tmp_path / "lane.png")
    (tmp_path / "manifest.json").write_text("{}")
    manifest = dict(
        camera=dict(fov_deg=90, x_actor_m=1.5, z_actor_m=2),
        image_size_wh=[40, 40],
        opendrive_sha256="map",
    )
    row = dict(
        split="test",
        rgb="rgb.png",
        lane="lane.png",
        sample_id="s",
        run_id="r",
        sequence_id="scene",
        sequence_index=0,
        frame_id=1,
        sim_time_s=0.05,
        weather="ClearNoon",
        ego_world=dict(location_m=[10, 3, 0], rotation_deg_pitch_yaw_roll=[0, 90, 0]),
    )
    monkeypatch.setattr(module, "read_dataset", lambda folder: (manifest, [row]))
    seen = []

    class Lanes:
        def __init__(self, topology):
            pass

        def signed_offset(self, x, y):
            seen.append((x, y))
            return 0.2, "lane", False

    monkeypatch.setattr("evaluation.a1_comparison.LaneCentrelines", Lanes)
    segmenter = SimpleNamespace(predict=lambda rgb: np.zeros((40, 40), np.uint8))
    result = module.evaluate(
        tmp_path,
        segmenter,
        topology=SimpleNamespace(opendrive_sha256="map"),
        overlay_output=tmp_path / "overlays",
    )
    assert seen[0][0] == pytest.approx(10)
    assert seen[0][1] == pytest.approx(-3 + 1.38641)
    record = result["records"][0]
    assert not record["lane_monitor_valid"]
    assert record["control_authority"] is False and record["evaluation_truth_only"] is True
    assert result["lane_observer_summary"]["valid_frames"] == 0
    assert (tmp_path / "overlays/s.png").is_file()
    with pytest.raises(ValueError, match="map"):
        module.evaluate(tmp_path, segmenter, topology=SimpleNamespace(opendrive_sha256="wrong"))
