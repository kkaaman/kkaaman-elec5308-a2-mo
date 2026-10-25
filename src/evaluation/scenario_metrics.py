"""S3/S4/S5 reporting on top of the frozen S0 evaluation (TEST-05, TEST-07).

Limits come from ``tests/fixtures/acceptance_definitions.json`` (the frozen
task-board copy) and are never restated here. The shared S0 metrics are
reused unchanged through :func:`evaluation.s0_metrics.evaluate_s0_run`; this
module adds only the scenario-specific rows and applies the fixture's
applicability (S4 route completion is NOT APPLICABLE, never a failure).

Extra evidence the scenario runner must retain, all optional (absence gives
INSUFFICIENT_EVIDENCE, never zero or PASS):

* samples: ``phase`` may be ``hold`` while the behaviour layer holds; a
  ``bumper_gap_m`` column carries the evaluation-truth front-bumper to
  obstacle-surface gap in metres (NaN/blank when no obstacle is scripted);
* events: ``obstacle_spawned`` / ``obstacle_removed`` / ``obstacle_confirmed``
  with ``simulation_time_s`` (and ``gap_m`` on ``obstacle_confirmed``).

Stopped margin is measured on hold samples with speed <= 0.1 m/s. Resume delay
is ``obstacle_removed`` to the first later sample with speed > 0.1 m/s on
simulation timestamps. The S5 visibility/speed envelope is report-only: at
first confirmation it records ego speed, gap, and the stopping distance
``v^2 / (2 a) + stop_margin`` with the frozen deceleration.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping
from pathlib import Path

from autonomy.paths import FIXTURES_DIR
from evaluation.s0_metrics import evaluate_s0_run

__all__ = [
    "SCENARIO_LIMIT_NAMES",
    "aggregate_scenario_runs",
    "evaluate_scenario_run",
    "load_scenario_limits",
    "stable_cruise_mean_speed",
]

_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
SCENARIO_LIMIT_NAMES = {
    "False stops (maximum per run)": "false_stops",
    "Cruise speed drop versus matched S0": "cruise_speed_drop_mps",
    "Minimum stopped bumper margin": "stop_margin_min_m",
    "Maximum stopped bumper margin": "stop_margin_max_m",
    "HOLD maximum speed": "hold_max_speed_mps",
    "Continuous HOLD duration (minimum)": "hold_duration_s",
    "Resume delay (median)": "resume_delay_median_s",
    "Route completion (minimum per run)": "route_completion",
}


def _finite(value, *, minimum=None):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and (minimum is None or value >= minimum)
    )


def load_scenario_limits(scenario, path=None) -> dict:
    """Return ``{metric_name: {rule, lower, upper, applicable, required_runs}}`` for a scenario."""
    path = Path(path) if path else FIXTURES_DIR / "acceptance_definitions.json"
    definitions = json.loads(path.read_text(encoding="utf-8"))
    limits = {}
    for row in definitions["metrics"]:
        if row["scenario"] != scenario:
            continue
        name = SCENARIO_LIMIT_NAMES.get(row["metric"])
        if name is None:
            continue
        limits[name] = {
            "rule": row["rule"],
            "lower": row["lower"],
            "upper": row["upper"],
            "unit": row["unit"],
            "applicable": row.get("applicable", True),
            "required_runs": row["required_runs"],
            "applicability_reason": row.get("applicability_reason"),
        }
    if not limits:
        raise ValueError(f"No scenario limits for {scenario!r}")
    return limits


def _judge(name, value, limits, sample_count, issues):
    limit = limits.get(name)
    if limit is None:
        return {
            "value": value,
            "status": "REPORTED",
            "rule": "report",
            "sample_count": sample_count,
            "evidence_issues": list(issues),
        }
    if limit["applicable"] is False or limit["rule"] == "N/A":
        return {
            "value": value,
            "status": "NOT_APPLICABLE",
            "rule": "N/A",
            "reason": limit.get("applicability_reason"),
            "sample_count": sample_count,
            "evidence_issues": list(issues),
        }
    rule = limit["rule"]
    if rule == "report":
        status = "REPORTED" if value is not None and not issues else _INSUFFICIENT
    elif value is None:
        status = _INSUFFICIENT
    else:
        if rule == ">=":
            passes = value >= limit["lower"]
        elif rule == "<=":
            passes = value <= limit["upper"]
        elif rule == "=":
            passes = value == limit["lower"]
        else:
            raise ValueError(f"Unknown rule {rule!r}")
        status = "FAIL" if not passes else (_INSUFFICIENT if issues else "PASS")
    return {
        "value": value,
        "unit": limit["unit"],
        "status": status,
        "rule": rule,
        "lower": limit["lower"],
        "upper": limit["upper"],
        "sample_count": sample_count,
        "evidence_issues": list(issues),
    }


def stable_cruise_mean_speed(samples):
    """Mean ``speed_mps`` over ``phase == cruise`` and ``stable_cruise is True`` samples."""
    speeds = [
        s["speed_mps"]
        for s in samples
        if isinstance(s, Mapping)
        and s.get("phase") == "cruise"
        and s.get("stable_cruise") is True
        and _finite(s.get("speed_mps"), minimum=0)
    ]
    return (statistics.mean(speeds), len(speeds)) if speeds else (None, 0)


def _hold_windows(samples):
    """Continuous ``phase == hold`` intervals as (start_s, end_s, [samples])."""
    windows = []
    current = None
    for sample in samples:
        timestamp = sample.get("simulation_time_s")
        if not _finite(timestamp, minimum=0):
            continue
        if sample.get("phase") == "hold":
            if current is None:
                current = [timestamp, timestamp, []]
            current[1] = timestamp
            current[2].append(sample)
        elif current is not None:
            windows.append(tuple(current))
            current = None
    if current is not None:
        windows.append(tuple(current))
    return windows


def _hold_metrics(samples, limits):
    windows = _hold_windows(samples)
    issues = [] if windows else ["No hold-phase samples were recorded"]
    longest = max((end - start for start, end, _ in windows), default=None)
    hold_samples = [s for _, _, group in windows for s in group]
    speeds = [s.get("speed_mps") for s in hold_samples]
    speed_issues = list(issues)
    if any(not _finite(v, minimum=0) for v in speeds):
        speed_issues.append("A hold sample has no valid speed_mps")
    hold_max_speed = max((v for v in speeds if _finite(v, minimum=0)), default=None)
    gaps = []
    gap_issues = list(issues)
    stopped = [s for s in hold_samples if _finite(s.get("speed_mps"), minimum=0) and s["speed_mps"] <= 0.1]
    if hold_samples and not stopped:
        gap_issues.append("No hold sample reached <= 0.1 m/s")
    for sample in stopped:
        gap = sample.get("bumper_gap_m")
        if _finite(gap):
            gaps.append(gap)
    if stopped and not gaps:
        gap_issues.append("Stopped hold samples carry no evaluation-truth bumper_gap_m")
    return {
        "hold_duration_s": _judge("hold_duration_s", longest, limits, len(hold_samples), issues),
        "hold_max_speed_mps": _judge(
            "hold_max_speed_mps", hold_max_speed, limits, len(hold_samples), speed_issues
        ),
        "stop_margin_min_m": _judge(
            "stop_margin_min_m", min(gaps) if gaps else None, limits, len(gaps), gap_issues
        ),
        "stop_margin_max_m": _judge(
            "stop_margin_max_m", max(gaps) if gaps else None, limits, len(gaps), gap_issues
        ),
        "hold_windows": [
            {"start_s": start, "end_s": end, "duration_s": end - start, "samples": len(group)}
            for start, end, group in windows
        ],
    }


def _resume_delay(samples, events, limits):
    removals = [
        e["simulation_time_s"]
        for e in events
        if e.get("kind") == "obstacle_removed" and _finite(e.get("simulation_time_s"), minimum=0)
    ]
    issues = [] if removals else ["No obstacle_removed event was recorded"]
    delays = []
    for removed in removals:
        moving = [
            s["simulation_time_s"]
            for s in samples
            if _finite(s.get("simulation_time_s"), minimum=0)
            and s["simulation_time_s"] >= removed
            and _finite(s.get("speed_mps"), minimum=0)
            and s["speed_mps"] > 0.1
        ]
        if moving:
            delays.append(min(moving) - removed)
        else:
            issues.append(f"No resumed motion after removal at {removed:g} s")
    return _judge(
        "resume_delay_median_s",
        statistics.median(delays) if delays else None,
        limits,
        len(delays),
        issues,
    )


def _speed_envelope(samples, events, *, deceleration_mps2, stop_margin_m):
    """Report-only visibility/speed envelope at first obstacle confirmation."""
    confirmations = [
        e
        for e in events
        if e.get("kind") == "obstacle_confirmed" and _finite(e.get("simulation_time_s"), minimum=0)
    ]
    if not confirmations:
        return {"status": _INSUFFICIENT, "evidence_issues": ["No obstacle_confirmed event"]}
    first = min(confirmations, key=lambda e: e["simulation_time_s"])
    at = first["simulation_time_s"]
    nearest = min(
        (s for s in samples if _finite(s.get("simulation_time_s"), minimum=0)),
        key=lambda s: abs(s["simulation_time_s"] - at),
        default=None,
    )
    speed = nearest.get("speed_mps") if nearest else None
    gap = first.get("gap_m")
    if gap is None and nearest is not None:
        gap = nearest.get("bumper_gap_m")
    issues = []
    if not _finite(speed, minimum=0):
        issues.append("No ego speed sample near confirmation")
    if not _finite(gap):
        issues.append("No evaluation-truth gap at confirmation")
    required = (
        speed**2 / (2 * deceleration_mps2) + stop_margin_m if _finite(speed, minimum=0) else None
    )
    return {
        "status": _INSUFFICIENT if issues else "REPORTED",
        "confirmed_at_s": at,
        "speed_at_confirmation_mps": speed if _finite(speed) else None,
        "gap_at_confirmation_m": gap if _finite(gap) else None,
        "required_stopping_distance_m": required,
        "deceleration_mps2": deceleration_mps2,
        "stop_margin_m": stop_margin_m,
        "envelope_ok": (gap >= required) if not issues else None,
        "evidence_issues": issues,
        "definition": "gap_at_confirmation >= v^2/(2a) + stop_margin; report-only, not a frozen threshold",
    }


def evaluate_scenario_run(
    scenario,
    samples,
    events,
    *,
    completed,
    matched_s0_summary=None,
    matched_s0_samples=None,
    deceleration_mps2=None,
    stop_margin_m=None,
    limits_path=None,
    **s0_kwargs,
) -> dict:
    """Evaluate one S3, S4 or S5 trial from retained samples/events.

    The shared S0 metrics are computed first; then scenario rows are added and
    the fixture's applicability is applied. ``matched_s0_summary`` or
    ``matched_s0_samples`` (S3) supply the matched S0 stable-cruise mean speed
    for the slowdown row; ``deceleration_mps2`` / ``stop_margin_m`` (S5) feed the
    report-only envelope. ``completed`` must be the endpoint predicate result.
    """
    if scenario not in ("S3", "S4", "S5"):
        raise ValueError("scenario must be S3, S4 or S5")
    limits = load_scenario_limits(scenario, limits_path)
    base = evaluate_s0_run(samples, events, completed=completed, **s0_kwargs)
    valid_samples = [s for s in samples if isinstance(s, Mapping)]
    valid_events = [e for e in events if isinstance(e, Mapping)]
    metrics = dict(base["metrics"])
    report = dict(base.get("report_only", {}))
    completion_limit = limits.get("route_completion", {})
    if completion_limit.get("applicable") is False:
        metrics["route_completion"] = _judge("route_completion", int(bool(completed)), limits, 1, [])
    if scenario == "S3":
        own_mean, own_n = stable_cruise_mean_speed(valid_samples)
        issues = [] if own_mean is not None else ["No stable cruise samples in this run"]
        matched = None
        if matched_s0_samples is not None:
            matched, _ = stable_cruise_mean_speed(matched_s0_samples)
        elif isinstance(matched_s0_summary, Mapping):
            matched = matched_s0_summary.get("stable_cruise_mean_speed_mps")
        if not _finite(matched):
            issues.append("No matched S0 stable-cruise mean speed supplied")
        drop = matched - own_mean if own_mean is not None and _finite(matched) else None
        metrics["cruise_speed_drop_mps"] = _judge("cruise_speed_drop_mps", drop, limits, own_n, issues)
        metrics["cruise_speed_drop_mps"]["matched_s0_mean_speed_mps"] = matched
        metrics["cruise_speed_drop_mps"]["stable_cruise_mean_speed_mps"] = own_mean
        report["intentional_stop_note"] = "S3: intentional obstacle stop not applicable; false stops judged"
    if scenario in ("S4", "S5"):
        hold = _hold_metrics(valid_samples, limits)
        report["hold_windows"] = hold.pop("hold_windows")
        metrics.update(hold)
        # Expected hold in S4/S5 is a planned stop; false stops are not applied
        # by the fixture to these scenarios, so report rather than judge.
        false_stops = metrics.pop("false_stops", None)
        if false_stops is not None:
            report["false_stops_including_planned_hold"] = false_stops
    if scenario == "S5":
        metrics["resume_delay_median_s"] = _resume_delay(valid_samples, valid_events, limits)
        if deceleration_mps2 is not None and stop_margin_m is not None:
            report["speed_envelope"] = _speed_envelope(
                valid_samples,
                valid_events,
                deceleration_mps2=deceleration_mps2,
                stop_margin_m=stop_margin_m,
            )
        else:
            report["speed_envelope"] = {
                "status": _INSUFFICIENT,
                "evidence_issues": ["deceleration_mps2 and stop_margin_m not supplied"],
            }
    issues = list(
        dict.fromkeys(
            issue
            for metric in metrics.values()
            for issue in metric.get("evidence_issues", [])
            if metric.get("status") not in ("NOT_APPLICABLE",)
        )
    )
    failed = [name for name, metric in metrics.items() if metric["status"] == "FAIL"]
    failures = list(base["failures"])
    status = "FAIL" if failures or failed else _INSUFFICIENT if issues else "PASS"
    summary = dict(base)
    summary.update(
        scenario=scenario,
        status=status,
        failed_metrics=failed,
        metrics=metrics,
        report_only=report,
        evidence_issues=issues,
        stable_cruise_mean_speed_mps=stable_cruise_mean_speed(valid_samples)[0],
        limits_source=str(limits_path or FIXTURES_DIR / "acceptance_definitions.json"),
    )
    summary["definitions"] = dict(
        base["definitions"],
        hold_rule="continuous phase == hold interval on simulation timestamps; longest reported",
        stop_margin_rule="evaluation-truth bumper_gap_m on hold samples with speed <= 0.1 m/s",
        resume_rule="obstacle_removed event to first later sample with speed > 0.1 m/s",
        slowdown_rule="matched S0 stable-cruise mean speed minus this run's stable-cruise mean speed",
    )
    return summary


def aggregate_scenario_runs(scenario, summaries, *, required_runs=5) -> dict:
    """Retain every trial; worst value per metric; any FAIL fails the suite."""
    if scenario not in ("S3", "S4", "S5"):
        raise ValueError("scenario must be S3, S4 or S5")
    issues, failures = [], []
    if len(summaries) != required_runs:
        issues.append(f"Expected exactly {required_runs} runs, observed {len(summaries)}")
    if any(run.get("scenario") != scenario for run in summaries):
        failures.append("A retained run belongs to a different scenario")
    if any(run.get("status") == "FAIL" for run in summaries):
        failures.append("At least one retained run failed")
    if any(run.get("status") not in {"PASS", "FAIL"} for run in summaries):
        issues.append("At least one retained run has insufficient evidence")
    names = sorted({name for run in summaries for name in run.get("metrics", {})})
    worst = {}
    for name in names:
        rows = [run.get("metrics", {}).get(name, {}) for run in summaries]
        if any(row.get("status") == "NOT_APPLICABLE" for row in rows):
            worst[name] = {"value": None, "status": "NOT_APPLICABLE"}
            continue
        values = [row.get("value") for row in rows]
        rule = next((row.get("rule") for row in rows if row.get("rule")), None)
        if values and all(_finite(v) for v in values):
            if rule == ">=":
                value = min(values)
            elif rule == "report":
                value = statistics.median(values)
            else:
                value = max(values)
            worst[name] = {"value": value, "rule": rule}
        else:
            worst[name] = {"value": None, "rule": rule}
            if rule != "report":
                issues.append(f"At least one run lacks a valid {name} value")
    status = "FAIL" if failures else _INSUFFICIENT if issues else "PASS"
    return {
        "scenario": scenario,
        "status": status,
        "required_runs": required_runs,
        "recorded_runs": len(summaries),
        "passed_runs": sum(run.get("status") == "PASS" for run in summaries),
        "failed_runs": sum(run.get("status") == "FAIL" for run in summaries),
        "worst_run_metrics": worst,
        "failures": failures,
        "evidence_issues": issues,
        "runs": list(summaries),
    }
