"""Same-sequence first events and conservative speed recommendations."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "track_benchmark", Path(__file__).resolve().parents[2] / "scripts/benchmark_obstacle_tracks.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def document(tmp_path):
    np.save(tmp_path / "depth.npy", np.full((40, 40), 10.0))
    return dict(
        origin="synthetic",
        settings=dict(
            confidence=0.25,
            image_size=640,
            tracker="botsort",
            tracker_settings={},
            model_version="v",
            model_sha256="fixture",
            device="cpu",
            versions={},
        ),
        track_policy=dict(hits=2, window=3),
        proposed_cruise_mps=4.0,
        braking_deceleration_mps2=2.0,
        additional_response_s=0.2,
        stopping_margin_m=2.0,
        sequences=[
            dict(
                name="approach",
                episode_id="e",
                source_id="rgb",
                calibration=dict(
                    fx_px=20,
                    fy_px=20,
                    cx_px=20,
                    cy_px=20,
                    camera_to_ego=((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
                ),
                start_sim_time_s=0.0,
                frames=[
                    dict(
                        frame_id=i,
                        sim_time_s=i / 10,
                        total_wall_s=0.05,
                        sensor_read_wall_s=0.0,
                        depth_npy="depth.npy",
                        target_gap_m=4.0 - i,
                        target_detection_indices=[0],
                        detections=[
                            dict(class_name="car", confidence=0.9, xyxy=[0, 0, 40, 40], track_id=0)
                        ],
                    )
                    for i in range(2)
                ],
            )
        ],
    )


def test_events_and_lower_speed(tmp_path):
    doc = document(tmp_path)
    report = module.benchmark(doc, tmp_path)
    row = report["sequences"][0]
    assert row["first_raw"] == dict(frame_id=0, range_m=4.0, latency_s=0.05)
    assert row["first_confirmed"]["frame_id"] == 1
    assert row["first_confirmed"]["range_m"] == 3.0
    assert row["first_confirmed"]["latency_s"] == pytest.approx(0.15)
    assert row["source_resolutions_wh"] == [(40, 40)]
    assert report["recommended_speed_mps"] == 1.5
    assert report["status"] == "SYNTHETIC_ONLY"


def test_missed_confirmation_recommends_hold(tmp_path):
    doc = document(tmp_path)
    for row in doc["sequences"][0]["frames"]:
        row["detections"][0]["track_id"] = None
    report = module.benchmark(doc, tmp_path)
    assert report["sequences"][0]["first_confirmed"] is None
    assert report["recommended_speed_mps"] == 0


def test_missing_settings_rejected(tmp_path):
    doc = document(tmp_path)
    del doc["settings"]["confidence"]
    with pytest.raises(ValueError):
        module.benchmark(doc, tmp_path)


def test_duplicate_sequence_names_rejected(tmp_path):
    doc = document(tmp_path)
    doc["sequences"].append(doc["sequences"][0])
    with pytest.raises(ValueError, match="unique"):
        module.benchmark(doc, tmp_path)


def test_serial_backlog_and_sensor_delay_are_not_lost(tmp_path):
    doc = document(tmp_path)
    rows = doc["sequences"][0]["frames"]
    rows[0]["total_wall_s"] = 2.0
    for row in rows:
        row["sensor_delay_s"] = 0.2
        row["filter_wall_s"] = 0.01
    report = module.benchmark(doc, tmp_path)
    row = report["sequences"][0]
    assert row["first_raw"]["latency_s"] == pytest.approx(2.21)
    assert row["first_confirmed"]["latency_s"] == pytest.approx(2.27)
    assert row["first_confirmed_response_s"] == pytest.approx(2.17)
    assert report["recommended_speed_mps"] < 1.5


def test_invalid_sensor_frame_has_no_invented_inference_timing(tmp_path):
    doc = document(tmp_path)
    row = doc["sequences"][0]["frames"][0]
    row.update(
        valid=False,
        invalid_reason="sensor_timeout",
        total_wall_s=None,
        detections=[],
        target_detection_indices=[],
        sensor_delay_s=0.2,
    )
    report = module.benchmark(doc, tmp_path)
    assert report["sequences"][0]["first_raw"]["frame_id"] == 1
    assert report["sequences"][0]["first_confirmed"] is None


def test_incomplete_capture_rejected(tmp_path):
    doc = document(tmp_path)
    doc["capture_status"] = "INCOMPLETE"
    with pytest.raises(ValueError, match="incomplete"):
        module.benchmark(doc, tmp_path)


def test_depth_artifact_hash_mismatch_rejected(tmp_path):
    doc = document(tmp_path)
    doc["sequences"][0]["frames"][0]["depth_sha256"] = "wrong"
    with pytest.raises(ValueError, match="hash"):
        module.benchmark(doc, tmp_path)


def waited_sequence(tmp_path):
    from copy import deepcopy

    doc = document(tmp_path)
    template = doc["sequences"][0]["frames"][0]
    rows = []
    for i, wait in enumerate((0.0, 0.5, 1.0, 0.0, 0.0)):
        row = deepcopy(template)
        row.update(
            frame_id=i,
            sim_time_s=i * 0.25,
            target_gap_m=8.0,
            total_wall_s=0.125,
            sensor_read_wall_s=wait,
        )
        if i in (1, 2):
            row.update(
                valid=False,
                invalid_reason="TimeoutError: missing RGB-D",
                total_wall_s=None,
                detections=[],
                target_detection_indices=[],
            )
        rows.append(row)
    doc["sequences"][0]["frames"] = rows
    return doc


def test_failure_waits_accumulate_before_confirmation_and_speed_screen(tmp_path):
    import math

    doc = waited_sequence(tmp_path)
    result = module.benchmark(doc, tmp_path)
    row = result["sequences"][0]
    # Serial live timeline: .125, .75, 1.75, 1.875, 2.0 seconds.
    assert row["first_confirmed"]["frame_id"] == 4
    assert row["first_confirmed"]["latency_s"] == 2.0
    assert row["first_confirmed_response_s"] == 1.0
    expected = math.floor((math.sqrt((2 * 1.2) ** 2 + 2 * 2 * 6) - 2 * 1.2) * 10) / 10
    assert result["measured_speed_cap_mps"] == expected == 3.0
    assert result["recommended_speed_mps"] == 3.0
    assert result["screening_decision"] == "FAILED"
    assert result["timing"]["proposed_cruise_supported"] is False
    for frame in doc["sequences"][0]["frames"]:
        frame["sensor_read_wall_s"] = 0.0
    optimistic = module.benchmark(doc, tmp_path)
    assert optimistic["sequences"][0]["first_confirmed"]["latency_s"] == 1.125
    assert optimistic["measured_speed_cap_mps"] > result["measured_speed_cap_mps"]


def test_legacy_missing_read_time_blocks_measured_screen(tmp_path):
    doc = waited_sequence(tmp_path)
    del doc["sequences"][0]["frames"][1]["sensor_read_wall_s"]
    result = module.benchmark(doc, tmp_path)
    assert result["status"] == "BLOCKED_INCOMPLETE_TIMING"
    assert result["measured_speed_cap_mps"] is None
    assert result["screening_decision"] == "BLOCKED"
    assert result["recommended_speed_mps"] == 0.0
    assert result["timing"]["proposed_cruise_supported"] is None
    doc.update(schema_version=3, settings_restored=True, no_actor_leaks=True)
    with pytest.raises(ValueError, match="Missing measured sensor read timing"):
        module.benchmark(doc, tmp_path)


@pytest.mark.parametrize("wait", [-0.1, float("nan"), float("inf"), True])
def test_invalid_read_timing_rejected(tmp_path, wait):
    doc = document(tmp_path)
    doc["sequences"][0]["frames"][0]["sensor_read_wall_s"] = wait
    with pytest.raises(ValueError, match="Invalid sensor read timing"):
        module.benchmark(doc, tmp_path)
