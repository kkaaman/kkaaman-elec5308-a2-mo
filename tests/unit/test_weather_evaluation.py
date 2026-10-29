"""Weather scoring must retain invalid frames and reject dishonest denominators."""

import copy
from pathlib import Path

import pytest

from evaluation.weather import evaluate_weather
from scripts.prepare_weather_matrix import prepare


def inputs():
    root = Path(__file__).resolve().parents[2]
    matrix = prepare(root / "configs/weather/robustness_v1.json")
    labels = [
        dict(class_name=k, xyxy=[0, 0, 10, 10], visible=True, depth_valid=True, distance_m=20)
        for k in ("car", "person")
    ]
    detections = [
        dict(class_name=k, xyxy=[0, 0, 10, 10], confirmed=True) for k in ("car", "person")
    ]
    frames = [
        dict(
            cell_id=c["cell_id"],
            frame_id=1,
            sim_time_s=1,
            valid=True,
            labels=copy.deepcopy(labels),
            detections=copy.deepcopy(detections),
        )
        for c in matrix["cells"]
    ]
    return dict(
        origin="synthetic", specification_sha256=matrix["specification_sha256"], frames=frames
    ), matrix


def test_complete_synthetic_counts_never_claim_operational_acceptance():
    obs, matrix = inputs()
    report = evaluate_weather(obs, matrix)
    assert report["status"] == "COMPLETE" and report["origin"] == "synthetic"
    assert not report["operational_acceptance"]
    assert all(r["tp"] == 9 and r["fn"] == 0 and r["recall"] == 1 for r in report["rows"])


def test_invalid_frame_cannot_earn_tp_or_confirmation():
    obs, matrix = inputs()
    obs["frames"][0]["valid"] = False
    report = evaluate_weather(obs, matrix)
    car = next(
        r for r in report["rows"] if r["class_name"] == "car" and r["weather"] == "ClearNoon"
    )
    assert car["tp"] == 8 and car["fn"] == 1 and car["invalid_frames"] == 1
    assert report["status"] == "INCOMPLETE" and report["invalid_cells"]


@pytest.mark.parametrize(
    "mutation",
    ["duplicate", "unknown", "bool", "nan_box", "zero_box", "iou", "duplicate_cell", "unpaired"],
)
def test_bad_evidence_rejected(mutation):
    obs, matrix = inputs()
    if mutation == "duplicate":
        obs["frames"].append(copy.deepcopy(obs["frames"][0]))
    if mutation == "unknown":
        obs["frames"][0]["cell_id"] = "fake"
    if mutation == "bool":
        obs["frames"][0]["valid"] = "false"
    if mutation == "nan_box":
        obs["frames"][0]["labels"][0]["xyxy"][0] = float("nan")
    if mutation == "zero_box":
        obs["frames"][0]["labels"][0]["xyxy"] = [0, 0, 0, 0]
    if mutation == "iou":
        obs["match_iou"] = 0
    if mutation == "duplicate_cell":
        matrix["cells"][1] = copy.deepcopy(matrix["cells"][0])
    if mutation == "unpaired":
        matrix["cells"][1]["seed"] += 1
    with pytest.raises(ValueError):
        evaluate_weather(obs, matrix)


def test_confirmation_uses_frame_order_not_json_order():
    obs, matrix = inputs()
    first = obs["frames"][0]
    second = copy.deepcopy(first)
    second.update(frame_id=2, sim_time_s=2)
    for label in second["labels"]:
        label["distance_m"] = 10
    obs["frames"].insert(0, second)
    report = evaluate_weather(obs, matrix)
    row = next(
        r for r in report["rows"] if r["class_name"] == "car" and r["weather"] == "ClearNoon"
    )
    assert row["first_confirmed_range_m"]["minimum"] == 20
    second["sim_time_s"] = 0.5
    with pytest.raises(ValueError, match="strictly agree"):
        evaluate_weather(obs, matrix)


def test_unmatched_detection_counts_false_positive():
    obs, matrix = inputs()
    obs["frames"][0]["detections"].append(
        dict(class_name="car", xyxy=[20, 20, 30, 30], confirmed=True)
    )
    report = evaluate_weather(obs, matrix)
    assert sum(r["fp"] for r in report["rows"]) == 1
