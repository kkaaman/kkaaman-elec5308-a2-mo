"""A held-out audit cannot accept leaked, partial or relabelled replay."""

import json

import pytest

from scripts.score_detector_replay import score


def fixture(tmp_path):
    manifest = dict(
        model_version="m",
        origin="synthetic-smoke",
        deadline_s=0.1,
        replay=[dict(file="a.npy", sha256="a" * 64)],
        modes={"local": {"configurations": [dict(width=20, height=20, jpeg_quality=None)]}},
    )
    (tmp_path / "summary.json").write_text(json.dumps(manifest))
    row = dict(
        mode="local",
        width=20,
        height=20,
        jpeg_quality=None,
        frame="a.npy",
        original_width=10,
        original_height=10,
        status="OK",
        total_s=0.01,
        detections=[dict(class_name="car", confidence=0.9, xyxy=[0, 0, 20, 20])],
    )
    (tmp_path / "predictions.jsonl").write_text(json.dumps(row) + "\n")
    annotations = dict(
        model_version="m",
        tuning_scene_ids=["train"],
        tuning_image_sha256=[],
        frames={
            "a.npy": dict(
                scene_id="test",
                split="test",
                sha256="a" * 64,
                labels=[dict(class_name="car", xyxy=[0, 0, 10, 10])],
            )
        },
    )
    return annotations, row


def test_scaled_matching_and_per_class_denominators(tmp_path):
    annotations, _ = fixture(tmp_path)
    result = score(tmp_path, annotations)
    assert result["configurations"][0]["accuracy"]["recall"] == 1
    assert result["per_class"][0]["tp"] == 1
    assert result["per_class"][1]["recall"] is None
    assert result["status"] == "SCORED_NOT_ACCEPTANCE"


@pytest.mark.parametrize("mutation", ["scene", "image", "hash", "split", "missing", "model"])
def test_leakage_and_identity_rejected(tmp_path, mutation):
    annotations, _ = fixture(tmp_path)
    row = annotations["frames"]["a.npy"]
    if mutation == "scene":
        annotations["tuning_scene_ids"].append("test")
    if mutation == "image":
        annotations["tuning_image_sha256"].append("a" * 64)
    if mutation == "hash":
        row["sha256"] = "b" * 64
    if mutation == "split":
        row["split"] = "val"
    if mutation == "missing":
        annotations["frames"] = {}
    if mutation == "model":
        annotations["model_version"] = "other"
    with pytest.raises(ValueError):
        score(tmp_path, annotations)


def test_failed_inference_counts_false_negative_without_retiming(tmp_path):
    annotations, row = fixture(tmp_path)
    row.update(status="ERROR", detections=[], total_s=0.2)
    (tmp_path / "predictions.jsonl").write_text(json.dumps(row))
    report = score(tmp_path, annotations)["configurations"][0]
    assert report["accuracy"]["fn"] == 1
    assert report["deadline_miss_rate"] == 1


def test_duplicate_and_missing_predictions_rejected(tmp_path):
    annotations, row = fixture(tmp_path)
    (tmp_path / "predictions.jsonl").write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        score(tmp_path, annotations)
    (tmp_path / "predictions.jsonl").write_text("")
    with pytest.raises(ValueError, match="Incomplete"):
        score(tmp_path, annotations)
