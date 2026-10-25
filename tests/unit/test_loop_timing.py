"""TEST-22 / TEST-17: loop-timing profiles from synthetic diagnostics and run files."""

import csv
import json

import pytest

from autonomy.interfaces import Diagnostics
from evaluation.loop_timing import (
    compare_profiles,
    load_timing_rows,
    profile_rows,
    rows_from_diagnostics,
)
from scripts.profile_loop_timing import main as profile_main


def synthetic_diagnostics(count, detector_ms, policy_ms):
    diagnostics, ends = [], []
    for index in range(count):
        detector = (detector_ms + (index % 5)) / 1000.0
        policy = policy_ms / 1000.0
        diagnostics.append(
            Diagnostics(
                (("detector", detector), ("policy", policy), ("total", detector + policy + 0.001)),
                True,
                ("policy_command",),
            )
        )
        ends.append(detector + policy + 0.004)
    return diagnostics, ends


def test_profile_from_synthetic_diagnostics_reports_stage_and_end_to_end_percentiles():
    diagnostics, ends = synthetic_diagnostics(40, 30.0, 2.0)
    rows = rows_from_diagnostics(diagnostics, ends)
    profile = profile_rows(rows, label="headless")
    assert profile["sample_count"] == 40
    assert set(profile["stages"]) == {"detector", "policy", "total"}
    detector = profile["stages"]["detector"]
    assert detector["n"] == 40
    assert detector["p50_ms"] == pytest.approx(32.0)
    assert detector["p95_ms"] == pytest.approx(34.0)
    assert detector["max_ms"] == pytest.approx(34.0)
    end = profile["end_to_end"]
    assert end["source_key"] == "capture_to_apply_ms"
    assert end["p95_ms"] == pytest.approx(40.0)
    assert end["status"] == "PASS" and end["budget_ms"] == 100.0
    assert profile["observation_valid_fraction"] == 1.0
    assert profile["real_time_factor"]["n"] == 0


def test_gate_fails_over_budget_and_is_insufficient_with_few_samples():
    diagnostics, ends = synthetic_diagnostics(40, 95.0, 10.0)
    over = profile_rows(rows_from_diagnostics(diagnostics, ends), label="hud")
    assert over["end_to_end"]["status"] == "FAIL"
    few, few_ends = synthetic_diagnostics(5, 10.0, 1.0)
    sparse = profile_rows(rows_from_diagnostics(few, few_ends), label="smoke")
    assert sparse["end_to_end"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert "5 end-to-end samples" in sparse["end_to_end"]["evidence_issues"][0]
    none = profile_rows(rows_from_diagnostics(few), label="no-latency")
    assert none["end_to_end"]["p95_ms"] is None
    assert none["end_to_end"]["status"] == "INSUFFICIENT_EVIDENCE"
    with pytest.raises(ValueError):
        rows_from_diagnostics(few, [0.1])
    with pytest.raises(ValueError):
        profile_rows([], budget_ms=0)


def test_compare_profiles_reports_deltas_and_blocks_failed_final():
    base_diag, base_ends = synthetic_diagnostics(30, 20.0, 2.0)
    hud_diag, hud_ends = synthetic_diagnostics(30, 20.0, 2.0)
    hud_rows = rows_from_diagnostics(hud_diag, [end + 0.070 for end in hud_ends])
    for row in hud_rows:
        row["hud_s"] = 0.070
    profiles = {
        "headless": profile_rows(rows_from_diagnostics(base_diag, base_ends), label="headless"),
        "hud": profile_rows(hud_rows, label="hud"),
    }
    comparison = compare_profiles(profiles, baseline="headless", final="hud")
    assert comparison["deltas"]["hud"]["end_to_end_p95_delta_ms"] == pytest.approx(70.0)
    assert comparison["deltas"]["hud"]["stage_p95_delta_ms"]["detector"] == pytest.approx(0.0)
    assert profiles["hud"]["stages"]["hud"]["p95_ms"] == pytest.approx(70.0)
    assert comparison["gate"]["status"] == "PASS" and comparison["gate"]["decision"] == "ADMIT"
    heavy_rows = rows_from_diagnostics(hud_diag, [end + 0.090 for end in hud_ends])
    profiles["hud"] = profile_rows(heavy_rows, label="hud")
    blocked = compare_profiles(profiles, baseline="headless", final="hud")
    assert blocked["gate"] == {
        "configuration": "hud",
        "status": "FAIL",
        "decision": "BLOCKED",
        "reason": blocked["gate"]["reason"],
    }
    assert "exceeds 100 ms" in blocked["gate"]["reason"]
    missing = compare_profiles(profiles, baseline="headless", final="absent")
    assert missing["gate"]["decision"] == "BLOCKED"
    with pytest.raises(KeyError):
        compare_profiles(profiles, baseline="absent")


def write_s0_run(folder, latency_ms, count=30):
    folder.mkdir(parents=True)
    fields = [
        "simulation_time_s", "capture_receipt_monotonic_s", "apply_monotonic_s",
        "capture_to_apply_ms", "frame_age", "real_time_factor", "phase", "status",
    ]
    with (folder / "samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(count):
            receipt = 100.0 + index * 0.06
            writer.writerow({
                "simulation_time_s": index * 0.05,
                "capture_receipt_monotonic_s": receipt,
                "apply_monotonic_s": receipt + latency_ms / 1000.0,
                "capture_to_apply_ms": latency_ms,
                "frame_age": 1,
                "real_time_factor": 0.8,
                "phase": "cruise",
                "status": "RUNNING",
            })


def write_timing_jsonl(path, count=30):
    rows = []
    for index in range(count):
        rows.append({
            "episode": "ep-1", "index": index, "sim_time_s": index * 0.05,
            "capture_started_monotonic_s": 50.0 + index * 0.06,
            "decision_monotonic_s": 50.0 + index * 0.06 + 0.045,
            "sensor_availability_s": 0.010, "bundle_s": 0.002, "stack_s": 0.033,
            "capture_to_decision_s": 0.045, "receipt_to_decision_s": 0.035,
            "detector_s": 0.025, "filter_s": 0.003, "controller_s": 0.001,
            "decision_and_overhead_s": 0.004, "warmed": True,
        })
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_load_run_folder_and_timing_jsonl(tmp_path):
    run = tmp_path / "run-01"
    write_s0_run(run, 16.0)
    rows, kind = load_timing_rows(run)
    assert kind == "s0_samples" and len(rows) == 30
    profile = profile_rows(rows, label="s0")
    assert profile["end_to_end"]["p95_ms"] == pytest.approx(16.0)
    assert profile["stages"] == {}
    assert profile["real_time_factor"]["median"] == pytest.approx(0.8)
    assert profile["frame_age"]["max"] == 1
    timing = tmp_path / "timing.jsonl"
    write_timing_jsonl(timing)
    rows, kind = load_timing_rows(timing)
    assert kind == "timing_jsonl"
    warm = profile_rows(rows, label="warm")
    assert warm["end_to_end"]["source_key"] == "capture_to_decision_s"
    assert warm["end_to_end"]["p95_ms"] == pytest.approx(45.0)
    assert warm["receipt_to_decision"]["p95_ms"] == pytest.approx(35.0)
    assert list(warm["stages"]) == [
        "sensor_availability", "bundle", "detector", "filter", "controller",
        "decision_and_overhead", "stack",
    ]
    assert warm["stages"]["detector"]["p95_ms"] == pytest.approx(25.0)
    assert warm["real_time_factor"]["median"] == pytest.approx(29 * 0.05 / (29 * 0.06))
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        load_timing_rows(tmp_path / "empty")
    with pytest.raises(FileNotFoundError):
        load_timing_rows(tmp_path / "notes.txt")


def test_script_writes_profile_and_plot(tmp_path):
    write_s0_run(tmp_path / "headless", 16.0)
    write_s0_run(tmp_path / "hud", 24.0)
    output = tmp_path / "profile"
    code = profile_main([
        "--run", f"headless={tmp_path / 'headless'}", "--run", f"hud={tmp_path / 'hud'}",
        "--baseline", "headless", "--final", "hud", "--output", str(output),
    ])
    assert code == 0
    report = json.loads((output / "latency_profile.json").read_text())
    assert report["comparison"]["gate"]["decision"] == "ADMIT"
    assert report["comparison"]["deltas"]["hud"]["end_to_end_p95_delta_ms"] == pytest.approx(8.0)
    assert (output / "latency_profile.png").stat().st_size > 0
    with pytest.raises(SystemExit):
        profile_main(["--run", "a=x", "--run", "a=y", "--output", str(tmp_path / "dup")])
