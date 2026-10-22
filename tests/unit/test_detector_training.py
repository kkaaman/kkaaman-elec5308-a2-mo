"""Detector fine-tuning labels and the baseline-vs-candidate comparison (no training run)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from autonomy.paths import REPO_ROOT


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train = load("training/batch35/train_detector.py", "train_detector")
compare = load("scripts/compare_detectors.py", "compare_detectors")
COCO = {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7}


def paint(image, tag, identity, x0, y0, x1, y1):
    image[y0:y1, x0:x1] = (tag, identity // 256, identity % 256)


def test_every_obstacle_class_is_labelled_and_tiny_blobs_are_not():
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    paint(image, 12, 300, 10, 10, 16, 40)  # child-sized pedestrian, 6x30 px
    paint(image, 13, 301, 100, 100, 110, 130)  # rider -> person
    paint(image, 14, 302, 200, 200, 260, 240)
    paint(image, 15, 303, 300, 100, 380, 160)
    paint(image, 16, 304, 400, 100, 500, 170)
    paint(image, 18, 305, 520, 200, 540, 220)
    paint(image, 19, 306, 560, 200, 575, 220)
    paint(image, 14, 307, 600, 300, 603, 305)  # 15 px: below the capture's 16-pixel rule
    paint(image, 7, 308, 50, 300, 60, 330)  # traffic light: not an obstacle class
    boxes = sorted(train.road_user_boxes(image))
    assert boxes == sorted(
        [
            ("person", 10, 10, 16, 40),
            ("person", 100, 100, 110, 130),
            ("car", 200, 200, 260, 240),
            ("truck", 300, 100, 380, 160),
            ("bus", 400, 100, 500, 170),
            ("motorcycle", 520, 200, 540, 220),
            ("bicycle", 560, 200, 575, 220),
        ]
    )


def test_two_actors_of_one_class_stay_separate():
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    paint(image, 12, 1, 0, 0, 5, 10)
    paint(image, 12, 2, 20, 0, 25, 10)
    assert len(train.road_user_boxes(image)) == 2


def test_yolo_lines_are_normalised_centre_size():
    (line,) = train.yolo_lines([("car", 160, 90, 480, 270)], COCO, 640, 360)
    assert line == "2 0.500000 0.500000 0.500000 0.500000"


def test_export_writes_matching_images_and_labels(tmp_path):
    root = tmp_path / "dataset"
    (root / "rgb").mkdir(parents=True)
    (root / "instance").mkdir()
    rows = []
    for index, split in enumerate(("train", "val", "test")):
        rgb, instance = f"rgb/{index}.png", f"instance/{index}.png"
        Image.fromarray(np.zeros((36, 64, 3), dtype=np.uint8)).save(root / rgb)
        mask = np.zeros((36, 64, 3), dtype=np.uint8)
        paint(mask, 12, 9, 10, 5, 15, 30)
        Image.fromarray(mask).save(root / instance)
        rows.append(dict(sample_id=f"s{index}", split=split, rgb=rgb, instance=instance))
    stats = train.export_yolo(root, rows, tmp_path / "yolo", COCO)
    assert stats["images"] == {"train": 1, "val": 1}
    assert stats["boxes"]["person"] == 2
    assert stats["person_height_px"]["20-40"] == 2
    assert (tmp_path / "yolo/images/train/s0.png").is_file()
    assert (tmp_path / "yolo/labels/val/s1.txt").read_text().startswith("0 ")
    assert not (tmp_path / "yolo/images/test").exists()


def report(passes, **extra):
    sequences = []
    for (sequence, name, distance), ok in passes.items():
        sequences.append(
            dict(
                sequence_id=sequence,
                class_name=name,
                scenario_target=True,
                requested_lateral_m=0.0,
                distance_m=distance,
                frames=20,
                confirmation_s=0.1 if ok else None,
                recall=0.95 if ok else 0.0,
                valid_depth_fraction=0.95 if ok else 0.0,
                max_abs_depth_error_m=0.1 if ok else None,
            )
        )
    return dict(
        dict(
            dataset_index_sha256="d",
            split="val",
            depth_calibration_sha256="c",
            detector=dict(image_size=1280, confidence=0.25),
            model_sha256="m",
            usable_range_lower_bound_m=dict(car=17.5, person=17.5),
            warm_p95_s=0.022,
            false_positives=10,
            sequences=sequences,
        ),
        **extra,
    )


def test_compare_reports_fixed_broken_and_missed():
    poses = {("a", "person", 5.3): False, ("b", "person", 20.2): False, ("c", "car", 10.0): True}
    baseline = report(poses)
    candidate = report(
        {("a", "person", 5.3): True, ("b", "person", 20.2): False, ("c", "car", 10.0): False},
        model_sha256="n",
    )
    result = compare.compare(baseline, candidate)
    assert [r["sequence_id"] for r in result["fixed"]] == ["a"]
    assert [r["sequence_id"] for r in result["broken"]] == ["c"]
    assert len(result["never_detected"]["baseline"]) == 2
    assert len(result["never_detected"]["candidate"]) == 2


def test_compare_refuses_reports_that_differ_beyond_the_weights():
    baseline = report({("a", "car", 5.0): True})
    with pytest.raises(ValueError, match="detector"):
        compare.compare(baseline, report({("a", "car", 5.0): True}, detector=dict(image_size=640)))


def test_cli_writes_a_new_file_only(tmp_path):
    for name in ("b.json", "c.json"):
        (tmp_path / name).write_text(json.dumps(report({("a", "car", 5.0): True})))
    out = tmp_path / "out.json"
    out.write_text("{}")
    with pytest.raises(FileExistsError):
        import sys

        argv = sys.argv
        sys.argv = [
            "x",
            "--baseline",
            str(tmp_path / "b.json"),
            "--candidate",
            str(tmp_path / "c.json"),
            "--output",
            str(out),
        ]
        try:
            compare.main()
        finally:
            sys.argv = argv


def test_training_refuses_the_demo_map_by_default():
    source = Path(train.__file__).read_text(encoding="utf-8")
    assert "recorded_on_demo_map(manifest)" in source
    assert "--allow-demo-map" in source
