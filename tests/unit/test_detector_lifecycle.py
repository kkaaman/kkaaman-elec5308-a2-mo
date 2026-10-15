"""Offline assets, tracker selection, episode isolation and per-call timing."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from perception import ObjectDetector, PerceptionConfig, object_detection


class FakeBackend:
    names = {2: "car"}

    def __init__(self, path):
        self.history = []
        self.options = None
        self.fail = False

    def predict(self, image, **options):
        self.options = options
        if self.fail:
            raise RuntimeError("backend failure")
        return [SimpleNamespace(boxes=None, names=self.names)]

    def track(self, image, **options):
        self.history.append(image.copy())
        return self.predict(image, **options)


def detector(tmp_path, monkeypatch, tracker="botsort.yaml"):
    path = tmp_path / "local.pt"
    path.write_bytes(b"stub weights; never read by a real learned model")
    monkeypatch.setattr(object_detection, "YOLO", FakeBackend)
    return ObjectDetector(
        PerceptionConfig(model_name=str(path), tracker_config=tracker)
    )


def test_reset_discards_history_and_selects_tracker(tmp_path, monkeypatch):
    model = detector(tmp_path, monkeypatch, "bytetrack.yaml")
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    assert model.track(image) == ()
    old = model.model
    assert len(old.history) == 1
    assert Path(old.options["tracker"]).name == "bytetrack.yaml"
    model.reset(episode_id="second")
    assert model.model is not old
    assert model.model.history == []
    assert model.episode_id == "second"
    assert model.last_call_timing is None
    model.track(image)
    assert len(model.model.history) == 1
    assert len(old.history) == 1


@pytest.mark.parametrize("kind", ["missing", "empty", "pointer", "url"])
def test_missing_assets_never_call_yolo(tmp_path, monkeypatch, kind):
    calls = []
    monkeypatch.setattr(object_detection, "YOLO", lambda name: calls.append(name))
    path = tmp_path / "weights.pt"
    if kind == "empty":
        path.write_bytes(b"")
    elif kind == "pointer":
        path.write_text(
            "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n"
        )
    name = "https://example.com/model.pt" if kind == "url" else str(path)
    with pytest.raises((ValueError, FileNotFoundError)):
        ObjectDetector(PerceptionConfig(model_name=name))
    assert calls == []


def test_timing_on_success_validation_and_backend_failure(tmp_path, monkeypatch):
    model = detector(tmp_path, monkeypatch)
    model.detect(np.zeros((4, 6, 3), dtype=np.uint8))
    timing = model.last_call_timing
    assert 0 <= timing.inference_wall_s <= timing.total_wall_s
    with pytest.raises(ValueError):
        model.detect(np.zeros((4, 6, 3), dtype=float))
    assert model.last_call_timing.inference_wall_s is None
    model.model.fail = True
    with pytest.raises(RuntimeError, match="backend failure"):
        model.detect(np.zeros((4, 6, 3), dtype=np.uint8))
    assert model.last_call_timing.inference_wall_s >= 0


def test_custom_tracker_is_local_and_reid_disabled(tmp_path, monkeypatch):
    config = tmp_path / "tracker.yaml"
    config.write_text("tracker_type: bytetrack\nwith_reid: false\n")
    model = detector(tmp_path, monkeypatch, str(config))
    model.track(np.zeros((4, 6, 3), dtype=np.uint8))
    assert model.model.options["tracker"] == str(config.resolve())
    config.write_text("tracker_type: botsort\nwith_reid: true\n")
    with pytest.raises(ValueError, match="ReID"):
        detector(tmp_path, monkeypatch, str(config))


def test_failed_reset_cannot_reuse_old_tracker(tmp_path, monkeypatch):
    model = detector(tmp_path, monkeypatch)
    old = model.model
    model._factory = lambda path: old
    with pytest.raises(RuntimeError, match="history isolation"):
        model.reset(episode_id="new")
    with pytest.raises(RuntimeError, match="unavailable"):
        model.track(np.zeros((4, 6, 3), dtype=np.uint8))


@pytest.mark.parametrize(
    "config",
    [
        {"confidence_threshold": float("nan")},
        {"confidence_threshold": 2},
        {"image_size": True},
        {"image_size": 0},
        {"tracker_config": ""},
        {"allowed_classes": ()},
    ],
)
def test_invalid_configuration_rejected(config):
    with pytest.raises(ValueError):
        PerceptionConfig(**config)


def test_close_releases_backend_and_empty_image_rejected(tmp_path, monkeypatch):
    model = detector(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        model.detect(np.zeros((0, 6, 3), dtype=np.uint8))
    model.close()
    model.close()
    assert model.model is None
    with pytest.raises(RuntimeError):
        model.detect(np.zeros((4, 6, 3), dtype=np.uint8))


def test_relative_weights_resolve_absolute_before_factory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "weights.pt").write_bytes(b"local stub")
    calls = []

    def factory(path):
        calls.append(path)
        return FakeBackend(path)

    monkeypatch.setattr(object_detection, "YOLO", factory)
    model = ObjectDetector(PerceptionConfig(model_name="weights.pt"))
    assert calls == [str((tmp_path / "weights.pt").resolve())]
    model.reset(episode_id="episode-2")
    assert calls == [str((tmp_path / "weights.pt").resolve())] * 2


def test_missing_weights_on_reset_fail_closed(tmp_path, monkeypatch):
    model = detector(tmp_path, monkeypatch)
    (tmp_path / "local.pt").unlink()
    with pytest.raises(FileNotFoundError):
        model.reset(episode_id="episode-2")
    assert model.model is None
    with pytest.raises(RuntimeError):
        model.track(np.zeros((4, 6, 3), dtype=np.uint8))
