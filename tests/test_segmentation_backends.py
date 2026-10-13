"""Tests for segmentation labels, metrics and backends (no model weights or network)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from segmentation.backends import (
    BACKENDS,
    YoloSegSegmenter,
    create_segmenter,
    overlay_labelled,
)
from segmentation.evaluation import confusion_matrix, per_class_iou, summarize
from segmentation.labels import (
    CARLA_ROADLINE_RGB,
    IGNORE_ID,
    carla_palette_to_trainid,
    carla_tags_to_trainid,
)
from segmentation.postprocessing import CITYSCAPES_PALETTE

REPO_ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- labels


def test_carla_tags_map_to_cityscapes_trainids():
    tags = np.array([[1, 19, 24], [0, 20, 28]], dtype=np.uint8)
    np.testing.assert_array_equal(
        carla_tags_to_trainid(tags), [[0, 18, 0], [IGNORE_ID, IGNORE_ID, IGNORE_ID]]
    )
    # car (CARLA 14) -> trainId 13, pedestrian (CARLA 12) -> trainId 11
    assert carla_tags_to_trainid(np.array([[14, 12]]))[0].tolist() == [13, 11]


def test_carla_tags_reject_colour_images():
    with pytest.raises(ValueError):
        carla_tags_to_trainid(np.zeros((2, 2, 3), dtype=np.uint8))


def test_palette_round_trip_and_unknown_colours():
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[0, 0] = CITYSCAPES_PALETTE[13]  # car
    rgb[0, 1] = CITYSCAPES_PALETTE[11]  # person
    rgb[0, 2] = CARLA_ROADLINE_RGB  # road line -> road
    rgb[1, 0] = CITYSCAPES_PALETTE[10]  # sky
    rgb[1, 1] = (1, 2, 3)  # not a class colour
    rgb[1, 2] = (0, 0, 0)  # CARLA unlabeled
    mask, matched = carla_palette_to_trainid(rgb)
    assert mask.tolist() == [[13, 11, 0], [10, IGNORE_ID, IGNORE_ID]]
    assert matched == pytest.approx(4 / 6)


# ----------------------------------------------------------------------------- metrics


def test_confusion_counts_unlabelled_predictions_as_misses():
    gt = np.array([[13, 13, 11, IGNORE_ID]])
    pred = np.array([[13, IGNORE_ID, 13, 13]])
    cm = confusion_matrix(gt, pred)
    assert cm[13, 13] == 1 and cm[13, 19] == 1 and cm[11, 13] == 1
    assert cm.sum() == 3  # ignored ground truth is excluded
    iou = per_class_iou(cm)
    assert iou[13] == pytest.approx(1 / 3)  # tp 1, fn 1 (unlabelled), fp 1 (person pixel)
    assert iou[11] == 0.0
    assert np.isnan(iou[0])  # road absent from ground truth


def test_summarize_hides_19_class_scores_for_objects_only_models():
    gt = np.array([[0, 0, 13, 13]])
    pred = np.array([[IGNORE_ID, IGNORE_ID, 13, 13]])
    cm = confusion_matrix(gt, pred)
    objects_only = summarize(cm, labels_all_classes=False)
    assert objects_only["miou_19"] is None and objects_only["pixel_accuracy"] is None
    assert objects_only["obstacle_miou"] == 1.0
    full = summarize(cm, labels_all_classes=True)
    assert full["miou_19"] == pytest.approx(0.5)  # road 0.0, car 1.0
    assert full["per_class_iou"]["car"] == 1.0 and full["per_class_iou"]["sky"] is None


def test_confusion_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        confusion_matrix(np.zeros((2, 2)), np.zeros((2, 3)))


# ----------------------------------------------------------------------------- backends


class _Tensor:
    """Mimics the .cpu().numpy() chain of a torch tensor."""

    def __init__(self, values):
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values

    def __len__(self):
        return len(self.values)


class _Boxes:
    def __init__(self, cls, conf, xyxy):
        self.cls, self.conf, self.xyxy = _Tensor(cls), _Tensor(conf), _Tensor(xyxy)

    def __len__(self):
        return len(self.cls)


class _Result:
    def __init__(self, masks, cls, conf, xyxy):
        self.masks = type("Masks", (), {"data": _Tensor(masks)})()
        self.boxes = _Boxes(cls, conf, xyxy)


class _Model:
    def __init__(self, result):
        self.result, self.calls = result, []

    def predict(self, image, **kwargs):
        self.calls.append((image, kwargs))
        return [self.result]


def _yolo_with(result):
    segmenter = YoloSegSegmenter.__new__(YoloSegSegmenter)  # skip loading real weights
    segmenter.model = _Model(result)
    segmenter.names = {0: "person", 2: "car", 5: "bus"}
    segmenter.class_map = {0: 11, 2: 13, 5: 15}
    segmenter.confidence, segmenter.image_size, segmenter.device = 0.25, 640, "cuda"
    return segmenter


def test_yolo_masks_map_to_trainids_and_most_confident_wins():
    car = np.zeros((4, 6), dtype=np.uint8)
    car[1:3, 1:4] = 1
    person = np.zeros((4, 6), dtype=np.uint8)
    person[1:3, 3:5] = 1  # overlaps the car at column 3
    result = _Result([car, person], cls=[2, 0], conf=[0.9, 0.5], xyxy=[[1, 1, 4, 3], [3, 1, 5, 3]])
    segmenter = _yolo_with(result)
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[..., 0] = 200  # red in RGB

    mask = segmenter.predict(image)
    assert mask[1, 1] == 13 and mask[1, 4] == 11
    assert mask[1, 3] == 13  # overlap goes to the more confident car
    assert mask[0, 0] == IGNORE_ID

    sent_image, kwargs = segmenter.model.calls[0]
    assert sent_image[0, 0].tolist() == [0, 0, 200]  # converted to BGR for Ultralytics
    assert kwargs["retina_masks"] is True and kwargs["classes"] == [0, 2, 5]
    assert kwargs["device"] == "0"


def test_yolo_resizes_low_resolution_masks():
    small = np.ones((2, 3), dtype=np.uint8)
    result = _Result([small], cls=[5], conf=[0.8], xyxy=[[0, 0, 6, 4]])
    instances = _yolo_with(result).predict_instances(np.zeros((4, 6, 3), dtype=np.uint8))
    assert instances[0]["mask"].shape == (4, 6) and instances[0]["mask"].all()
    assert instances[0]["label"] == "bus" and instances[0]["train_id"] == 15


def test_yolo_with_no_detections_returns_all_unlabelled():
    segmenter = _yolo_with(_Result(np.zeros((0, 4, 6)), cls=[], conf=[], xyxy=np.zeros((0, 4))))
    mask = segmenter.predict(np.zeros((4, 6, 3), dtype=np.uint8))
    assert (mask == IGNORE_ID).all()


def test_overlay_leaves_unlabelled_pixels_unchanged():
    image = np.full((2, 2, 3), 50, dtype=np.uint8)
    mask = np.array([[13, IGNORE_ID], [IGNORE_ID, IGNORE_ID]], dtype=np.uint8)
    out = overlay_labelled(image, mask, alpha=0.5)
    assert out[0, 1].tolist() == [50, 50, 50]
    assert out[0, 0].tolist() != [50, 50, 50]


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="unknown segmentation backend"):
        create_segmenter("deeplab")
    assert BACKENDS == ("bisenetv2", "segformer_b0", "yolo_seg")


# ----------------------------------------------------------------------------- script


def _load_script():
    path = REPO_ROOT / "scripts" / "evaluate_segmentation.py"
    spec = importlib.util.spec_from_file_location("evaluate_segmentation", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_script_end_to_end_with_fake_backend(tmp_path, monkeypatch):
    import json

    import cv2

    script = _load_script()
    gt = np.zeros((8, 12), dtype=np.uint8)  # road
    gt[2:6, 3:9] = 13  # car
    label_rgb = CITYSCAPES_PALETTE[gt]
    cv2.imwrite(str(tmp_path / "semantic.png"), cv2.cvtColor(label_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(tmp_path / "rgb.png"), np.full((8, 12, 3), 90, dtype=np.uint8))

    class FakeSegmenter:
        name, device, labels_all_classes, weights = "segformer_b0", "cpu", True, "fake"

        def predict(self, image):
            return gt.copy()  # perfect prediction

        def visualize(self, image, mask, alpha=0.5):
            return image

    monkeypatch.setattr(script, "create_segmenter", lambda *a, **k: FakeSegmenter())
    code = script.main(
        [
            "--images", str(tmp_path / "rgb.png"),
            "--labels", str(tmp_path / "semantic.png"),
            "--label-format", "carla-palette",
            "--backend", "segformer_b0",
            "--min-timed", "2", "--warmup", "1",
            "--out", str(tmp_path / "out"),
        ]
    )  # fmt: skip
    assert code == 0
    report = json.loads(next((tmp_path / "out").glob("*/results.json")).read_text())
    metrics = report["results"][0]["metrics"]
    assert metrics["miou_19"] == 1.0 and metrics["obstacle_miou"] == 1.0
    assert report["results"][0]["label_colour_match"] == 1.0
    assert list((tmp_path / "out").glob("*/overlays/*.png"))


def test_backend_spec_parsing():
    script = _load_script()
    assert script.parse_backend("segformer_b0") == ("segformer_b0", None)
    assert script.parse_backend("yolo_seg=yolov8n-seg.pt") == ("yolo_seg", "yolov8n-seg.pt")
    with pytest.raises(Exception, match="unknown backend"):
        script.parse_backend("maskrcnn")
