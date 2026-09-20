"""Automated tests for object detection and tracking."""

from types import SimpleNamespace

import numpy as np
import pytest

from perception import BoundingBox, Detection, ObjectDetector, PerceptionConfig, object_detection


class FakeModel:
    """Record YOLO calls and return a configured result."""

    names = {0: "person", 2: "car", 5: "bus"}

    def __init__(self, result):
        self.result = result
        self.predict_calls = []
        self.track_calls = []

    def predict(self, image, **kwargs):
        self.predict_calls.append((image, kwargs))
        return [self.result]

    def track(self, frame, **kwargs):
        self.track_calls.append((frame, kwargs))
        return [self.result]


def make_result(*boxes):
    """Create the small portion of an Ultralytics result used by the adapter."""

    return SimpleNamespace(boxes=boxes, names=FakeModel.names)


def make_box(xyxy, class_id, confidence, track_id=None):
    """Create the small portion of an Ultralytics box used by the adapter."""

    return SimpleNamespace(
        xyxy=np.array([xyxy], dtype=float),
        cls=np.array([class_id]),
        conf=np.array([confidence]),
        id=None if track_id is None else np.array([track_id]),
    )


def install_fake_model(monkeypatch, result, config=None):
    """Construct an ObjectDetector backed by a deterministic fake model."""

    model = FakeModel(result)
    monkeypatch.setattr(object_detection, "YOLO", lambda model_name: model)
    return ObjectDetector(config), model


def test_detect_converts_rgb_and_returns_expected_detection(monkeypatch):
    result = make_result(make_box((1, 2, 30, 40), 2, 0.875))
    config = PerceptionConfig(
        model_name="test-model.pt",
        confidence_threshold=0.4,
        image_size=320,
        allowed_classes=("car",),
    )
    detector, model = install_fake_model(monkeypatch, result, config)
    image_rgb = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)

    detections = detector.detect(image_rgb)

    assert detections == (
        Detection(
            class_name="car",
            confidence=0.875,
            bounding_box=BoundingBox(1.0, 2.0, 30.0, 40.0),
        ),
    )
    image_bgr, options = model.predict_calls[0]
    np.testing.assert_array_equal(image_bgr, image_rgb[:, :, ::-1])
    assert options == {
        "conf": 0.4,
        "imgsz": 320,
        "classes": [2],
        "verbose": False,
    }


def test_track_returns_ids_and_persists_tracker_state(monkeypatch):
    result = make_result(make_box((4, 5, 60, 70), 5, 0.75, track_id=12))
    detector, model = install_fake_model(monkeypatch, result)
    frame_rgb = np.array([[[1, 2, 3]]], dtype=np.uint8)

    detections = detector.track(frame_rgb)

    assert detections == (
        Detection(
            class_name="bus",
            confidence=0.75,
            bounding_box=BoundingBox(4.0, 5.0, 60.0, 70.0),
            track_id=12,
        ),
    )
    frame_bgr, options = model.track_calls[0]
    np.testing.assert_array_equal(frame_bgr, frame_rgb[:, :, ::-1])
    assert options["persist"] is True
    assert options["tracker"] == "botsort.yaml"


def test_detect_returns_empty_tuple_when_result_has_no_boxes(monkeypatch):
    detector, _ = install_fake_model(
        monkeypatch,
        SimpleNamespace(boxes=None, names=FakeModel.names),
    )

    assert detector.detect(np.zeros((2, 2, 3), dtype=np.uint8)) == ()


@pytest.mark.parametrize("method_name", ["detect", "track"])
def test_rejects_non_array_input(monkeypatch, method_name):
    detector, model = install_fake_model(monkeypatch, make_result())

    with pytest.raises(TypeError, match="must be a NumPy array"):
        getattr(detector, method_name)([[[0, 0, 0]]])

    assert model.predict_calls == []
    assert model.track_calls == []


@pytest.mark.parametrize(
    "invalid_shape",
    [
        (10, 10),
        (10, 10, 1),
        (10, 10, 4),
        (1, 10, 10, 3),
    ],
)
@pytest.mark.parametrize("method_name", ["detect", "track"])
def test_rejects_arrays_that_are_not_rgb(monkeypatch, method_name, invalid_shape):
    detector, model = install_fake_model(monkeypatch, make_result())

    with pytest.raises(ValueError, match=r"must have shape \(H, W, 3\)"):
        getattr(detector, method_name)(np.zeros(invalid_shape, dtype=np.uint8))

    assert model.predict_calls == []
    assert model.track_calls == []
