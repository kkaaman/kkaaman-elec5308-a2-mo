"""Specification checks for S0 formulas, missing evidence and frozen trials."""

import copy
import json
import math
from pathlib import Path

import pytest

from evaluation.s0_metrics import (
    S0_LIMITS,
    S0_REQUIRED_RUNS,
    aggregate_s0_runs,
    evaluate_s0_run,
)


def samples():
    return [
        {
            "simulation_time_s": index * 0.1,
            "signed_cte_m": 0.2 if index % 2 else -0.2,
            "speed_mps": 8.2,
            "target_speed_mps": 8.0,
            "phase": "cruise",
            "stable_cruise": True,
            "capture_to_apply_ms": 20.0,
            "frame_age": 1,
            "real_time_factor": 0.8,
        }
        for index in range(31)
    ]


def streams_complete():
    return [
        {
            "kind": "event_stream_status",
            "collision_complete": True,
            "lane_invasion_complete": True,
        }
    ]


def passing_run(run_id=1):
    summary = evaluate_s0_run(samples(), streams_complete(), completed=True)
    summary.update(run_id=run_id, route_sha256="a" * 64, config_sha256="b" * 64)
    return summary


def test_limits_and_required_trials_match_authoritative_fixture():
    definitions = json.loads(
        (Path(__file__).parents[1] / "fixtures" / "acceptance_definitions.json").read_text(
            encoding="utf-8"
        )
    )["metrics"]
    names = {
        "Route completion (minimum per run)": "route_completion",
        "Collisions (maximum per run)": "collisions",
        "Unplanned lane invasions (max per run)": "unplanned_lane_invasions",
        "Cross-track mean (worst run)": "mean_abs_cte_m",
        "Cruise speed RMS (worst run)": "cruise_speed_rms_mps",
        "Capture-to-apply p95 (worst run)": "capture_to_apply_p95_ms",
        "False stops (maximum per run)": "false_stops",
    }
    s0 = [metric for metric in definitions if metric["scenario"] == "S0"]
    assert len(s0) == len(S0_LIMITS)
    result = passing_run()
    for definition in s0:
        name = names[definition["metric"]]
        bound = definition["lower"] if definition["rule"] == ">=" else definition["upper"]
        assert S0_LIMITS[name] == bound
        assert result["metrics"][name]["rule"] == definition["rule"]
        assert definition["required_runs"] == S0_REQUIRED_RUNS
        assert definition["applicable"] is True


def test_goal_window_is_geometric_before_finished_status():
    trace = samples()
    for row in trace:
        row.update(phase="approach", target_speed_mps=0.05, in_goal_window=True)
    metric = evaluate_s0_run(trace, streams_complete(), completed=True)["metrics"]["false_stops"]
    assert metric["value"] == 0
    for row in trace:
        row["in_goal_window"] = False
    metric = evaluate_s0_run(trace, streams_complete(), completed=True)["metrics"]["false_stops"]
    assert metric["value"] == 1


def test_invalid_goal_window_does_not_silently_exempt_a_stop():
    trace = samples()
    trace[0]["in_goal_window"] = "true"
    metric = evaluate_s0_run(trace, streams_complete(), completed=True)["metrics"]["false_stops"]
    assert metric["status"] == "INSUFFICIENT_EVIDENCE"


def test_signed_cte_cannot_cancel_and_speed_rms_is_not_mean_error():
    trace = samples()
    for index, row in enumerate(trace):
        row["signed_cte_m"] = 0.25 if index % 2 else -0.25
        row["speed_mps"] = 8.4 if index % 2 else 7.6
    result = evaluate_s0_run(trace, streams_complete(), completed=True)
    assert result["status"] == "PASS"
    assert result["metrics"]["mean_abs_cte_m"]["value"] == pytest.approx(0.25)
    assert result["metrics"]["cruise_speed_rms_mps"]["value"] == pytest.approx(0.4)
    assert result["metrics"]["mean_abs_cte_m"]["sample_count"] == len(trace)


def test_motion_cte_includes_braking_but_excludes_stationary_hold_and_goal():
    trace = samples()
    trace[0].update(phase="approach", signed_cte_m=0.4)
    trace[1].update(phase="hold", speed_mps=0, signed_cte_m=5)
    trace[2].update(phase="goal", speed_mps=0, signed_cte_m=5)
    result = evaluate_s0_run(trace, streams_complete(), completed=True)
    assert result["metrics"]["mean_abs_cte_m"]["sample_count"] == len(trace) - 2
    assert result["metrics"]["mean_abs_cte_m"]["value"] == pytest.approx((28 * 0.2 + 0.4) / 29)


def test_cruise_selection_excludes_transients_without_masking_speed_error():
    trace = samples()
    for row in trace[1:]:
        row.update(phase="approach", speed_mps=2)
    trace[0]["speed_mps"] = 6.0
    result = evaluate_s0_run(trace, streams_complete(), completed=True)
    assert result["metrics"]["cruise_speed_rms_mps"]["value"] == 2.0
    assert result["metrics"]["cruise_speed_rms_mps"]["sample_count"] == 1
    assert result["status"] == "FAIL"


def test_latency_uses_fixed_linear_percentile_and_reports_performance_separately():
    trace = samples()[:4]
    for row, latency in zip(trace, [0.0, 10.0, 20.0, 100.0]):
        row["capture_to_apply_ms"] = latency
    trace[0]["frame_age"] = 5
    result = evaluate_s0_run(trace, streams_complete(), completed=True)
    assert result["metrics"]["capture_to_apply_p95_ms"]["value"] == pytest.approx(88.0)
    assert result["metrics"]["capture_to_apply_p95_ms"]["sample_count"] == 4
    assert result["report_only"]["frame_age_max"] == 5
    assert result["report_only"]["real_time_factor_min"] == 0.8
    assert result["report_only"]["status"] == "REPORTED"


def test_monotonic_audit_times_must_match_supplied_latency():
    trace = samples()
    trace[0].update(capture_receipt_monotonic_s=100.0, apply_monotonic_s=100.02)
    result = evaluate_s0_run(trace, streams_complete(), completed=True)
    assert result["status"] == "PASS"
    trace[0]["apply_monotonic_s"] = 99.9
    result = evaluate_s0_run(trace, streams_complete(), completed=True)
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["metrics"]["capture_to_apply_p95_ms"]["value"] is None


@pytest.mark.parametrize("completed", [False, True])
def test_empty_trace_is_retained_and_never_passes(completed):
    result = evaluate_s0_run([], [], completed=completed)
    assert result["status"] == ("INSUFFICIENT_EVIDENCE" if completed else "FAIL")
    for name in S0_LIMITS.keys() - {"route_completion"}:
        assert result["metrics"][name]["value"] is None
    assert result["sample_count"] == 0


@pytest.mark.parametrize(
    "field",
    [
        "signed_cte_m",
        "speed_mps",
        "target_speed_mps",
        "capture_to_apply_ms",
        "frame_age",
        "real_time_factor",
        "simulation_time_s",
        "phase",
        "stable_cruise",
    ],
)
def test_missing_applicable_sample_fields_are_not_silently_dropped(field):
    trace = samples()
    del trace[4][field]
    result = evaluate_s0_run(trace, streams_complete(), completed=True)
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["sample_count"] == len(trace)
    assert result["evidence_issues"]


@pytest.mark.parametrize("timestamp", [float("nan"), 0.0, -1.0])
def test_bad_or_unordered_simulation_timestamps_retain_insufficient_trial(timestamp):
    trace = samples()
    trace[5]["simulation_time_s"] = timestamp
    assert (
        evaluate_s0_run(trace, streams_complete(), completed=True)["status"]
        == "INSUFFICIENT_EVIDENCE"
    )


def test_missing_cruise_window_is_insufficient_not_zero_error():
    trace = samples()
    for row in trace:
        row["stable_cruise"] = False
    metric = evaluate_s0_run(trace, streams_complete(), completed=True)["metrics"][
        "cruise_speed_rms_mps"
    ]
    assert metric["status"] == "INSUFFICIENT_EVIDENCE"
    assert metric["value"] is None
    assert metric["sample_count"] == 0


def test_zero_incidents_requires_complete_sensor_streams():
    result = evaluate_s0_run(samples(), [], completed=True)
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["metrics"]["collisions"]["value"] is None
    assert result["metrics"]["unplanned_lane_invasions"]["value"] is None


def test_collision_dedup_preserves_raw_count_and_later_contact():
    events = [
        {"kind": "collision", "contact_id": "42:100"},
        {"kind": "collision", "contact_id": "42:100"},
        {"kind": "collision", "contact_id": "42:500"},
        *streams_complete(),
    ]
    metric = evaluate_s0_run(samples(), events, completed=True)["metrics"]["collisions"]
    assert metric["value"] == 2
    assert metric["raw_event_count"] == 3
    assert metric["status"] == "FAIL"


def test_actual_collision_fails_even_if_cleanup_or_id_evidence_is_missing():
    result = evaluate_s0_run(samples(), [{"kind": "collision"}], completed=True)
    assert result["status"] == "FAIL"
    assert result["metrics"]["collisions"]["value"] == 1
    assert result["evidence_issues"]


def test_junctions_are_not_blanket_excluded_and_legal_exclusion_needs_declaration():
    events = [
        {"kind": "lane_invasion", "event_id": 1, "is_junction": True},
        {"kind": "lane_invasion", "event_id": 2, "predeclared_legal": True},
        {
            "kind": "lane_invasion",
            "event_id": 3,
            "predeclared_legal": True,
            "legal_manoeuvre_id": "mapped-turn-1",
        },
        *streams_complete(),
    ]
    metric = evaluate_s0_run(samples(), events, completed=True)["metrics"][
        "unplanned_lane_invasions"
    ]
    assert metric["value"] == 2
    assert metric["raw_event_count"] == 3
    assert metric["excluded_legal_event_count"] == 1


def test_last_stream_marker_controls_completeness():
    events = [
        *streams_complete(),
        {
            "kind": "event_stream_status",
            "collision_complete": False,
            "lane_invasion_complete": True,
        },
    ]
    assert (
        evaluate_s0_run(samples(), events, completed=True)["metrics"]["collisions"]["status"]
        == "INSUFFICIENT_EVIDENCE"
    )


@pytest.mark.parametrize("stop_duration,expected", [(0.9, 0), (1.0, 1), (1.1, 1)])
def test_false_stop_duration_uses_simulation_time_including_resume_boundary(
    stop_duration, expected
):
    trace = samples()[:3]
    trace[0].update(simulation_time_s=0.0, target_speed_mps=0.0, stable_cruise=False)
    trace[1].update(simulation_time_s=stop_duration, target_speed_mps=8.0)
    trace[2]["simulation_time_s"] = stop_duration + 0.1
    metric = evaluate_s0_run(trace, streams_complete(), completed=True)["metrics"]["false_stops"]
    assert metric["value"] == expected


def test_continuous_long_stop_counts_once_goal_braking_is_excluded():
    trace = samples()
    for row in trace:
        row.update(target_speed_mps=0.0, stable_cruise=False)
    result = evaluate_s0_run(trace, streams_complete(), completed=True)
    assert result["metrics"]["false_stops"]["value"] == 1
    for row in trace:
        row["phase"] = "goal"
    result = evaluate_s0_run(trace, streams_complete(), completed=True)
    assert result["metrics"]["false_stops"]["value"] == 0


def test_false_stop_at_exact_decimal_second_is_not_hidden_by_roundoff():
    trace = samples()[:3]
    trace[0].update(simulation_time_s=0.4, target_speed_mps=0.0, stable_cruise=False)
    trace[1].update(simulation_time_s=1.4, target_speed_mps=8.0)
    trace[2]["simulation_time_s"] = 1.5
    metric = evaluate_s0_run(trace, streams_complete(), completed=True)["metrics"]["false_stops"]
    assert metric["value"] == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"completed": False},
        {"failure_reason": "timeout"},
        {"route_install_count": 2},
        {"cleanup_ok": False},
    ],
)
def test_endpoint_failure_timeout_route_reset_and_cleanup_failure_are_retained(kwargs):
    arguments = {"completed": True, **kwargs}
    result = evaluate_s0_run(samples(), streams_complete(), **arguments)
    assert result["status"] == "FAIL"
    assert result["sample_count"] == 31


def test_non_boolean_endpoint_is_not_treated_as_progress_completion():
    result = evaluate_s0_run(samples(), streams_complete(), completed=1.0)
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["metrics"]["route_completion"]["value"] is None


def test_five_runs_report_worst_trial_without_pooling_or_dropping_failures():
    runs = [passing_run(index) for index in range(5)]
    trace = samples()
    for row in trace:
        row["signed_cte_m"] = 0.31
    runs[4] = evaluate_s0_run(
        trace, streams_complete(), completed=True, failure_reason="CTE trial failed"
    )
    runs[4].update(run_id=4, route_sha256="a" * 64, config_sha256="b" * 64)
    result = aggregate_s0_runs(runs)
    assert result["status"] == "FAIL"
    assert result["passed_runs"] == 4
    assert result["failed_runs"] == 1
    assert result["worst_run_metrics"]["mean_abs_cte_m"] == pytest.approx(0.31)
    assert result["runs"] == runs
    assert result["runs"][4]["failure_reason"] == "CTE trial failed"


def test_exact_five_consistent_trials_pass():
    result = aggregate_s0_runs([passing_run(index) for index in range(5)])
    assert result["status"] == "PASS"
    assert result["recorded_runs"] == 5


def test_matching_smoke_hashes_do_not_turn_nonformal_trials_into_acceptance():
    runs = [passing_run(index) for index in range(5)]
    for run in runs:
        run["formal_acceptance"] = False
    result = aggregate_s0_runs(runs)
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["recorded_runs"] == 5
    assert result["runs"] == runs
    assert any("formal S0" in issue for issue in result["evidence_issues"])


@pytest.mark.parametrize("count", [0, 1, 4, 6])
def test_wrong_trial_count_cannot_pass(count):
    result = aggregate_s0_runs([passing_run(index) for index in range(count)])
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["recorded_runs"] == count


@pytest.mark.parametrize("field", ["route_sha256", "config_sha256"])
def test_changed_route_or_configuration_invalidates_frozen_suite(field):
    runs = [passing_run(index) for index in range(5)]
    runs[3][field] = "c" * 64
    assert aggregate_s0_runs(runs)["status"] == "FAIL"


def test_missing_provenance_or_duplicate_run_id_cannot_pass():
    runs = [passing_run(index) for index in range(5)]
    del runs[2]["config_sha256"]
    assert aggregate_s0_runs(runs)["status"] == "INSUFFICIENT_EVIDENCE"
    runs[2] = copy.deepcopy(runs[0])
    assert aggregate_s0_runs(runs)["status"] == "FAIL"


def test_inputs_and_raw_evidence_are_not_mutated():
    trace, events = samples(), streams_complete()
    original = copy.deepcopy((trace, events))
    evaluate_s0_run(trace, events, completed=True)
    assert (trace, events) == original


def test_nonfinite_and_malformed_samples_are_retained_without_crashing():
    trace = samples()
    trace[4]["signed_cte_m"] = math.nan
    trace.append(None)
    result = evaluate_s0_run(trace, [None, *streams_complete()], completed=True)
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["sample_count"] == 32


def test_malformed_phase_and_untyped_events_cannot_prove_zero_incidents():
    trace = samples()
    trace[0]["phase"] = []
    result = evaluate_s0_run(trace, [{}, *streams_complete()], completed=True)
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["metrics"]["collisions"]["value"] is None
