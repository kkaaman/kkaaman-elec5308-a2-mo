"""Tests for scripts/detector_seg_experiment.py (no CARLA, models or GPU)."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "detector_seg_experiment", REPO_ROOT / "scripts" / "detector_seg_experiment.py"
)
exp = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = exp
_spec.loader.exec_module(exp)


def _depth_to_bgra(depth_m: np.ndarray) -> np.ndarray:
    v = np.round(depth_m / 1000.0 * 16777215).astype(np.int64)
    out = np.zeros(depth_m.shape + (4,), dtype=np.uint8)
    out[..., 2] = v & 255  # R
    out[..., 1] = (v >> 8) & 255  # G
    out[..., 0] = (v >> 16) & 255  # B
    return out


def _instance_bgr(tag: np.ndarray, inst: np.ndarray) -> np.ndarray:
    return np.dstack([inst & 255, (inst >> 8) & 255, tag]).astype(np.uint8)


def _pedestrian_scene():
    """A walking pedestrian at 10 m in front of a 40 m background.

    Head, narrow torso and two spread legs: the tight box around the silhouette
    is about two-thirds background, as in a mid-stride pose.
    """
    depth = np.full((60, 80), 40.0, dtype=np.float32)
    mask = np.zeros((60, 80), dtype=bool)
    mask[10:16, 37:43] = True  # head
    mask[16:32, 37:43] = True  # torso
    mask[32:50, 30:34] = True  # left leg
    mask[32:50, 46:50] = True  # right leg
    depth[mask] = 10.0
    return depth, mask


# ----------------------------------------------------------------------------- decoding


def test_depth_and_instance_decoding_round_trip():
    depth = np.array([[2.5, 30.0], [999.9, 0.5]], dtype=np.float32)
    np.testing.assert_allclose(exp.decode_depth(_depth_to_bgra(depth)), depth, atol=1e-3)
    tag = np.array([[14, 12]])
    inst = np.array([[300, 7]])
    t, i = exp.decode_instance(_instance_bgr(tag, inst))
    assert t.tolist() == [[14, 12]] and i.tolist() == [[300, 7]]


def test_ground_truth_mask_finds_the_new_obstacle_not_parked_cars():
    tag = np.zeros((10, 10), dtype=np.int32)
    inst = np.zeros((10, 10), dtype=np.int32)
    tag[0:3, 0:3], inst[0:3, 0:3] = 14, 5  # parked car, present in baseline too
    base_tag = tag.copy()
    tag[5:9, 4:8], inst[5:9, 4:8] = 14, 77  # newly placed car
    tag[4, 4:8], inst[4, 4:8] = 14, 77  # its top row overlaps a baseline car pixel region
    base_tag[4, 4:8] = 14
    mask = exp.ground_truth_mask(tag, inst, base_tag, 14)
    assert mask[5:9, 4:8].all() and mask[4, 4:8].all()  # whole instance, incl. overlap
    assert not mask[0:3, 0:3].any()
    assert exp.ground_truth_mask(tag, inst, tag, 14) is None  # nothing new -> not visible


def test_box_helpers():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 3:9] = True
    assert exp.mask_box(mask) == (3.0, 2.0, 9.0, 5.0)
    assert exp.box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)
    assert exp.box_iou((0, 0, 1, 1), (5, 5, 6, 6)) == 0.0
    inner = exp.box_region((10, 10), (0, 0, 8, 8), 0.5)
    assert inner.sum() == 16 and inner[2:6, 2:6].all()


def test_median_depth_ignores_sky():
    depth = np.array([[5.0, 1000.0, 7.0]])
    assert exp.median_depth(depth, np.ones_like(depth, dtype=bool)) == 6.0
    assert exp.median_depth(np.array([[1000.0]]), np.array([[True]])) is None


# ----------------------------------------------------------------------------- scoring


def _frame(depth, mask, cls="pedestrian"):
    return {"class": cls, "depth": depth, "gt_box": exp.mask_box(mask),
            "gt_distance_m": exp.median_depth(depth, mask), "gt_near_m": exp.near_depth(depth, mask)}  # fmt: skip


def test_mask_beats_box_for_a_thin_pedestrian():
    depth, mask = _pedestrian_scene()
    det = {"label": "person", "confidence": 0.8, "box": exp.mask_box(mask), "mask": mask}
    row = exp.evaluate_detection(_frame(depth, mask), [det])
    assert row["detected"] and row["iou"] == 1.0
    assert row["err_box_full_m"] == pytest.approx(30.0, abs=0.01)  # background wins the box median
    assert row["err_mask_m"] == pytest.approx(0.0, abs=0.01)
    assert abs(row["err_mask_m"]) < abs(row["err_box_inner_m"])


def test_nearest_surface_box_reading_is_pulled_in_by_the_road_but_mask_is_not():
    depth, mask = _pedestrian_scene()
    depth[44:50, 30:50][~mask[44:50, 30:50]] = 9.0  # road surface around the feet, nearer than the person
    det = {"label": "person", "confidence": 0.8, "box": exp.mask_box(mask), "mask": mask}
    row = exp.evaluate_detection(_frame(depth, mask), [det])
    assert row["err_near_box_m"] == pytest.approx(-1.0, abs=0.01)  # picks up the road, 1 m too close
    assert row["err_near_mask_m"] == pytest.approx(0.0, abs=0.01)
    assert exp.near_depth(np.array([[5.0, 6.0, 1000.0]]), np.ones((1, 3), dtype=bool), 0) == 5.0


def test_wrong_class_or_poor_overlap_is_not_a_detection():
    depth, mask = _pedestrian_scene()
    car = {"label": "car", "confidence": 0.9, "box": (37.0, 10.0, 43.0, 50.0), "mask": None}
    far = {"label": "person", "confidence": 0.9, "box": (0.0, 0.0, 10.0, 10.0), "mask": None}
    row = exp.evaluate_detection(_frame(depth, mask), [car, far])
    assert not row["detected"] and row["err_box_full_m"] is None
    box_only = {"label": "person", "confidence": 0.5, "box": exp.mask_box(mask), "mask": None}
    row = exp.evaluate_detection(_frame(depth, mask), [box_only])
    assert row["detected"] and row["err_mask_m"] is None  # detection-only model has no mask


def test_summary_rates_and_median_errors():
    rows = [
        {"model": "m", "class": "car", "detected": True, "err_box_full_m": 1.0, "err_box_inner_m": -0.5, "err_mask_m": None},
        {"model": "m", "class": "car", "detected": True, "err_box_full_m": -3.0, "err_box_inner_m": 0.5, "err_mask_m": None},
        {"model": "m", "class": "car", "detected": False, "err_box_full_m": None, "err_box_inner_m": None, "err_mask_m": None},
    ]
    s = exp.summarize(rows)["m"]["car"]
    assert s["detected"] == 2 and s["frames"] == 3 and s["detection_rate"] == pytest.approx(0.667)
    assert s["median_abs_err_box_full_m"] == 2.0 and s["median_abs_err_box_inner_m"] == 0.5
    assert "median_abs_err_mask_m" not in s


# ----------------------------------------------------------------------------- analysis end to end


def _write_run(run_dir: Path):
    depth, mask = _pedestrian_scene()
    tag = np.zeros(depth.shape, dtype=np.int64)
    inst = np.zeros(depth.shape, dtype=np.int64)
    (run_dir / "baseline").mkdir(parents=True)
    cv2.imwrite(str(run_dir / "baseline" / "instance.png"), _instance_bgr(tag, inst))
    tag[mask], inst[mask] = 12, 42
    d = run_dir / "frames" / "pedestrian_10m"
    d.mkdir(parents=True)
    cv2.imwrite(str(d / "instance.png"), _instance_bgr(tag, inst))
    cv2.imwrite(str(d / "rgb.png"), np.zeros(depth.shape + (3,), dtype=np.uint8))
    np.save(d / "depth.npy", depth)
    meta = {"frames": [{"name": "pedestrian_10m", "class": "pedestrian", "gap_m": 10, "frame": 1}]}
    (run_dir / "capture.json").write_text(json.dumps(meta))
    return mask


class _T:
    def __init__(self, v):
        self.v = np.asarray(v)

    def cpu(self):
        return self

    def numpy(self):
        return self.v


def _fake_ultralytics(mask):
    class _Boxes:
        cls, conf, xyxy = _T([0]), _T([0.7]), _T([list(exp.mask_box(mask))])

        def __len__(self):
            return 1

    class Result:
        def __init__(self, with_mask):
            self.names = {0: "person"}
            self.boxes = _Boxes()
            self.masks = types.SimpleNamespace(data=_T(mask[None].astype(np.uint8))) if with_mask else None

    class YOLO:
        def __init__(self, weights):
            self.with_mask = "seg" in weights

        def predict(self, img, **kw):
            assert kw["retina_masks"] is True
            return [Result(self.with_mask)]

    return types.SimpleNamespace(YOLO=YOLO)


def test_analyze_end_to_end_with_fake_models(tmp_path, monkeypatch, capsys):
    mask = _write_run(tmp_path / "run")
    monkeypatch.setitem(sys.modules, "ultralytics", _fake_ultralytics(mask))
    exp.main(["analyze", str(tmp_path / "run"), "--models", "yolo26n.pt", "yolo26n-seg.pt"])
    report = json.loads((tmp_path / "run" / "analysis.json").read_text())
    det, seg = report["summary"]["yolo26n.pt"]["pedestrian"], report["summary"]["yolo26n-seg.pt"]["pedestrian"]
    assert det["detected"] == seg["detected"] == 1
    assert det["median_abs_err_box_full_m"] == pytest.approx(30.0, abs=0.01)
    assert "median_abs_err_mask_m" not in det
    assert seg["median_abs_err_mask_m"] == pytest.approx(0.0, abs=0.01)
    assert seg["median_abs_err_near_mask_m"] == pytest.approx(0.0, abs=0.01)
    row = next(r for r in report["rows"] if r["model"] == "yolo26n-seg.pt")
    assert row["gt_gap_m"] == pytest.approx(10.0 - 0.95, abs=0.01)  # nearest depth minus camera-to-bumper
    out = capsys.readouterr().out
    assert "Nearest: box" in out and "| 1/1 | 30.0" in out
