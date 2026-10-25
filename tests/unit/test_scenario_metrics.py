"""TEST-05 / TEST-07 reporting: S3 false stops and slowdown, S4 hold, S5 resume/envelope."""

import csv
import json

import pytest

from evaluation.scenario_metrics import (
    aggregate_scenario_runs,
    evaluate_scenario_run,
    load_scenario_limits,
    stable_cruise_mean_speed,
)
from scripts.evaluate_scenario_runs import main as evaluate_main

DT = 0.1


def cruise_sample(index, speed=4.0, target=4.0, **extra):
    sample = {
        "simulation_time_s": index * DT,
        "signed_cte_m": 0.05 if index % 2 else -0.05,
        "speed_mps": speed,
        "target_speed_mps": target,
        "phase": "cruise",
        "stable_cruise": True,
        "in_goal_window": False,
        "capture_to_apply_ms": 20.0,
        "frame_age": 1,
        "real_time_factor": 0.9,
    }
    sample.update(extra)
    return sample


def streams():
    return [{"kind": "event_stream_status", "collision_complete": True, "lane_invasion_complete": True}]


def s0_samples(count=60, speed=4.0):
    return [cruise_sample(i, speed=speed) for i in range(count)]


def test_limits_come_from_the_frozen_fixture():
    s3 = load_scenario_limits("S3")
    assert s3["false_stops"] == {
        "rule": "=", "lower": 0.0, "upper": 0.0, "unit": "count", "applicable": True,
        "required_runs": 5, "applicability_reason": None,
    }
    assert s3["cruise_speed_drop_mps"]["upper"] == 0.5
    s4 = load_scenario_limits("S4")
    assert s4["route_completion"]["applicable"] is False and s4["route_completion"]["rule"] == "N/A"
    assert s4["hold_duration_s"]["lower"] == 60.0
    assert s4["stop_margin_min_m"]["lower"] == 2.5 and s4["stop_margin_max_m"]["upper"] == 6.0
    assert load_scenario_limits("S5")["resume_delay_median_s"]["rule"] == "report"
    with pytest.raises(ValueError):
        load_scenario_limits("S9")


def test_s3_reports_false_stops_and_slowdown_against_matched_s0():
    clean = evaluate_scenario_run("S3", s0_samples(), streams(), completed=True, matched_s0_samples=s0_samples())
    assert clean["scenario"] == "S3" and clean["status"] == "PASS"
    assert clean["metrics"]["false_stops"]["value"] == 0
    assert clean["metrics"]["cruise_speed_drop_mps"]["value"] == pytest.approx(0.0)
    assert clean["metrics"]["cruise_speed_drop_mps"]["status"] == "PASS"
    # A 0.8 m/s slowdown against the matched S0 fails the frozen 0.5 m/s limit.
    slow = evaluate_scenario_run("S3", s0_samples(speed=3.2), streams(), completed=True, matched_s0_samples=s0_samples())
    assert slow["metrics"]["cruise_speed_drop_mps"]["value"] == pytest.approx(0.8)
    assert slow["status"] == "FAIL" and "cruise_speed_drop_mps" in slow["failed_metrics"]
    # A 1.5 s unexpected stop outside the goal window is one false stop.
    samples = s0_samples()
    for index in range(20, 36):
        samples[index].update(target_speed_mps=0.0, speed_mps=0.0, phase="approach", stable_cruise=False)
    stopped = evaluate_scenario_run("S3", samples, streams(), completed=True, matched_s0_samples=s0_samples())
    assert stopped["metrics"]["false_stops"]["value"] == 1
    assert stopped["metrics"]["false_stops"]["status"] == "FAIL"
    # No matched S0: the slowdown row is insufficient evidence, never PASS.
    unmatched = evaluate_scenario_run("S3", s0_samples(), streams(), completed=True)
    assert unmatched["metrics"]["cruise_speed_drop_mps"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert unmatched["status"] == "INSUFFICIENT_EVIDENCE"
    assert stable_cruise_mean_speed([]) == (None, 0)


def hold_run(hold_seconds, gap_m=4.0, gap_present=True, approach_from=20):
    samples = [cruise_sample(i) for i in range(approach_from)]
    index = approach_from
    for step in range(10):
        samples.append(cruise_sample(index, speed=4.0 - 0.4 * step, target=0.0, phase="approach", stable_cruise=False))
        index += 1
    hold_count = int(round(hold_seconds / DT))
    for _ in range(hold_count):
        extra = {"bumper_gap_m": gap_m} if gap_present else {}
        samples.append(cruise_sample(index, speed=0.0, target=0.0, phase="hold", stable_cruise=False, **extra))
        index += 1
    return samples, index


def test_s4_hold_duration_with_route_completion_not_applicable():
    samples, _ = hold_run(61.0)
    summary = evaluate_scenario_run("S4", samples, streams(), completed=False)
    metrics = summary["metrics"]
    assert metrics["route_completion"]["status"] == "NOT_APPLICABLE"
    assert "route_completion" not in summary["failed_metrics"]
    assert metrics["hold_duration_s"]["value"] == pytest.approx(61.0 - DT)
    assert metrics["hold_duration_s"]["status"] == "PASS"
    assert metrics["hold_max_speed_mps"]["value"] == 0.0 and metrics["hold_max_speed_mps"]["status"] == "PASS"
    assert metrics["stop_margin_min_m"]["value"] == 4.0 and metrics["stop_margin_max_m"]["status"] == "PASS"
    assert "false_stops" not in metrics
    assert summary["report_only"]["false_stops_including_planned_hold"]["value"] == 1
    assert summary["report_only"]["hold_windows"][0]["samples"] == 610
    assert summary["status"] == "PASS"
    short, _ = hold_run(30.0)
    failed = evaluate_scenario_run("S4", short, streams(), completed=False)
    assert failed["metrics"]["hold_duration_s"]["status"] == "FAIL" and failed["status"] == "FAIL"
    no_gap, _ = hold_run(61.0, gap_present=False)
    missing = evaluate_scenario_run("S4", no_gap, streams(), completed=False)
    assert missing["metrics"]["stop_margin_min_m"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert "bumper_gap_m" in missing["metrics"]["stop_margin_min_m"]["evidence_issues"][0]
    close, _ = hold_run(61.0, gap_m=1.0)
    assert evaluate_scenario_run("S4", close, streams(), completed=False)["metrics"]["stop_margin_min_m"]["status"] == "FAIL"


def test_s5_resume_delay_and_speed_envelope():
    samples, index = hold_run(5.0, gap_m=5.0)
    removed_at = samples[-1]["simulation_time_s"] + DT
    # Two more stopped ticks after removal, then motion resumes.
    for _ in range(2):
        samples.append(cruise_sample(index, speed=0.0, target=4.0, phase="approach", stable_cruise=False))
        index += 1
    for _ in range(40):
        samples.append(cruise_sample(index))
        index += 1
    events = streams() + [
        {"kind": "obstacle_spawned", "simulation_time_s": 0.0},
        {"kind": "obstacle_confirmed", "simulation_time_s": 2.0, "gap_m": 24.0},
        {"kind": "obstacle_removed", "simulation_time_s": removed_at},
    ]
    summary = evaluate_scenario_run(
        "S5", samples, events, completed=True, deceleration_mps2=1.497566177392398, stop_margin_m=6.0
    )
    resume = summary["metrics"]["resume_delay_median_s"]
    assert resume["status"] == "REPORTED" and resume["value"] == pytest.approx(2 * DT)
    envelope = summary["report_only"]["speed_envelope"]
    assert envelope["status"] == "REPORTED"
    assert envelope["speed_at_confirmation_mps"] == 4.0
    assert envelope["required_stopping_distance_m"] == pytest.approx(16 / (2 * 1.497566177392398) + 6.0)
    assert envelope["envelope_ok"] is True
    assert summary["metrics"]["route_completion"]["status"] == "PASS"
    assert summary["status"] == "PASS"
    without_events = evaluate_scenario_run("S5", samples, streams(), completed=True)
    assert without_events["metrics"]["resume_delay_median_s"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert without_events["report_only"]["speed_envelope"]["status"] == "INSUFFICIENT_EVIDENCE"
    with pytest.raises(ValueError):
        evaluate_scenario_run("S0", samples, events, completed=True)


def test_aggregate_requires_five_trials_and_keeps_every_run():
    samples, _ = hold_run(61.0)
    runs = [evaluate_scenario_run("S4", samples, streams(), completed=False) for _ in range(5)]
    result = aggregate_scenario_runs("S4", runs)
    assert result["status"] == "PASS" and result["passed_runs"] == 5
    assert result["worst_run_metrics"]["route_completion"]["status"] == "NOT_APPLICABLE"
    assert result["worst_run_metrics"]["hold_duration_s"]["value"] == pytest.approx(61.0 - DT)
    assert aggregate_scenario_runs("S4", runs[:3])["status"] == "INSUFFICIENT_EVIDENCE"
    short, _ = hold_run(30.0)
    runs[2] = evaluate_scenario_run("S4", short, streams(), completed=False)
    failed = aggregate_scenario_runs("S4", runs)
    assert failed["status"] == "FAIL" and failed["failed_runs"] == 1 and len(failed["runs"]) == 5
    with pytest.raises(ValueError):
        aggregate_scenario_runs("S1", runs)


def write_run(folder, samples, events, completed):
    folder.mkdir(parents=True)
    fields = sorted({key for sample in samples for key in sample})
    with (folder / "samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            writer.writerow({key: sample.get(key, "") for key in fields})
    (folder / "events.json").write_text(json.dumps(events))
    (folder / "manifest.json").write_text(json.dumps({
        "failure_reason": None, "route_install_count": 1, "cleanup_errors": [],
        "route_sha256": "a" * 64, "config_sha256": "b" * 64,
    }))
    (folder / "summary.json").write_text(json.dumps({"completed": completed}))


def test_script_recomputes_from_persisted_folders(tmp_path, capsys):
    s0 = tmp_path / "S0" / "run-01"
    write_run(s0, s0_samples(), streams(), True)
    folders = []
    for index in range(5):
        folder = tmp_path / "S3" / f"run-{index + 1:02d}"
        write_run(folder, s0_samples(speed=3.9), streams(), True)
        folders.append(str(folder))
    output = tmp_path / "S3-summary.json"
    code = evaluate_main(["S3", *folders, "--matched-s0", str(s0), "--output", str(output)])
    result = json.loads(output.read_text())
    assert code == 0 and result["status"] == "PASS"
    assert result["worst_run_metrics"]["cruise_speed_drop_mps"]["value"] == pytest.approx(0.1)
    assert result["runs"][0]["run_id"] == "run-01"
    capsys.readouterr()
