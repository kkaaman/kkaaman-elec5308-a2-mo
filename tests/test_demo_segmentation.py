"""Tests for the demos' shared segmentation options (no CARLA, weights or network)."""

import argparse
import time

import numpy as np
import pytest

import demo_segmentation as ds


def _args(*argv):
    parser = argparse.ArgumentParser()
    ds.add_segmentation_args(parser)
    return parser.parse_args(list(argv))


def test_defaults_keep_the_a1_model():
    args = _args()
    assert args.seg_backend == "bisenetv2"
    assert args.seg_weights is None
    assert tuple(args.segformer_size) == (512, 256)


def test_unknown_backend_is_rejected():
    with pytest.raises(SystemExit):
        _args("--seg-backend", "deeplab")


def test_required_weight_files_only_checks_local_files():
    assert ds.required_weight_files(_args(), "../lane.pth") == ["../lane.pth", "../segmentation_model.pth"]
    assert ds.required_weight_files(_args("--seg-backend", "segformer_b0"), "../lane.pth") == ["../lane.pth"]


def test_require_real_weights_rejects_lfs_pointers_and_missing_files(tmp_path):
    real = tmp_path / "real.pth"
    real.write_bytes(b"PK\x03\x04weights")
    ds.require_real_weights(str(real))  # no exit

    pointer = tmp_path / "pointer.pth"
    pointer.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:ab\nsize 9\n")
    with pytest.raises(SystemExit, match="Git LFS pointer"):
        ds.require_real_weights(str(pointer))
    with pytest.raises(SystemExit, match="not found"):
        ds.require_real_weights(str(tmp_path / "missing.pth"))


def test_build_segmenter_passes_the_right_options(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ds, "create_segmenter", lambda *a, **k: calls.append((a, k)) or object())
    ds.build_segmenter("bisenetv2", None, (512, 256))
    ds.build_segmenter("segformer_b0", None, (1024, 512))
    (tmp_path / "src").mkdir()
    monkeypatch.chdir(tmp_path / "src")
    ds.build_segmenter("yolo_seg", None, (512, 256))  # nothing in ../ -> default download
    (tmp_path / "yolo26n-seg.pt").write_bytes(b"x")
    ds.build_segmenter("yolo_seg", None, (512, 256))  # reuses ../yolo26n-seg.pt
    assert calls == [
        (("bisenetv2", "../segmentation_model.pth"), {"input_size": (512, 256)}),
        (("segformer_b0", None), {"input_size": (1024, 512)}),
        (("yolo_seg", None), {}),
        (("yolo_seg", "../yolo26n-seg.pt"), {}),
    ]


def test_timed_segmenter_reports_time_and_delegates():
    class Slow:
        name = "segformer_b0"

        def predict(self, frame):
            time.sleep(0.002)
            return np.zeros(frame.shape[:2], dtype=np.uint8)

        def visualize(self, frame, mask, alpha=0.5):
            return frame * 0 + int(alpha * 100)

    timed = ds.TimedSegmenter(Slow())
    assert timed.timing_text() == ""
    frame = np.ones((4, 4, 3), dtype=np.uint8)
    for _ in range(3):
        assert timed.predict(frame).shape == (4, 4)
    assert timed.mean_ms >= 1.5
    assert timed.timing_text().startswith("segformer_b0 ") and timed.timing_text().endswith(" ms/frame")
    assert timed.visualize(frame, None, alpha=0.35)[0, 0, 0] == 35
