"""Frozen S0 acceptance metrics with explicit evidence and failure retention.

The limits mirror ``tests/fixtures/acceptance_definitions.json``; the fixture
remains authoritative and a specification test prevents silent divergence.
Input samples are unmodified observations, not an already filtered trace.
``stable_cruise`` must be declared from phase/target history, never speed error.

CTE uses moving samples (>0.1 m/s) outside the goal phase. Cruise RMS uses only
``phase == 'cruise'`` and ``stable_cruise is True``. Percentiles use linear
interpolation at index ``(n - 1) * p``. False-stop duration is measured between
simulation timestamps, including the interval up to the first non-stop sample.
S0 has no planned stops outside the goal window.

Collision callbacks share a ``contact_id`` for one continuous contact; a later
contact must receive a new ID. Lane events use ``event_id``. Only lane events
with ``predeclared_legal is True`` and a nonempty ``legal_manoeuvre_id`` are
excluded. The final ``event_stream_status`` event must confirm both complete
sensor streams, so an empty event list alone cannot prove zero incidents.
Keep raw events and the signed CTE trace alongside the returned summary.
"""

import math
import statistics
from collections.abc import Mapping

S0_REQUIRED_RUNS = 5
S0_LIMITS = {
    "route_completion": 1.0,
    "collisions": 0,
    "unplanned_lane_invasions": 0,
    "mean_abs_cte_m": 0.3,
    "cruise_speed_rms_mps": 0.5,
    "capture_to_apply_p95_ms": 100.0,
    "false_stops": 0,
}
_PHASES = {"cruise", "approach", "hold", "goal"}
_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"


def _finite(value, *, minimum=None):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and (minimum is None or value >= minimum)
    )


def _nonempty_id(value):
    return isinstance(value, (int, str)) and not isinstance(value, bool) and str(value) != ""


def _metric(name, value, unit, sample_count, issues=()):
    limit = S0_LIMITS[name]
    rule = ">=" if name == "route_completion" else "=" if unit == "count" else "<="
    passes = value is not None and (
        value >= limit if rule == ">=" else value == limit if rule == "=" else value <= limit
    )
    status = (
        "FAIL"
        if value is not None and not passes
        else (_INSUFFICIENT if issues or value is None else "PASS")
    )
    return {
        "value": value,
        "unit": unit,
        "status": status,
        "sample_count": sample_count,
        "rule": rule,
        "limit": limit,
        "evidence_issues": list(issues),
    }


def _percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    left = math.floor(index)
    right = math.ceil(index)
    return ordered[left] + (ordered[right] - ordered[left]) * (index - left)


def _timeline_issues(samples):
    issues = []
    if not samples:
        return ["No control samples were recorded"]
    previous = None
    for index, sample in enumerate(samples):
        timestamp = sample.get("simulation_time_s")
        if not _finite(timestamp, minimum=0):
            issues.append(f"Sample {index} has no valid simulation_time_s")
        elif previous is not None and timestamp <= previous:
            issues.append(f"Sample {index} simulation timestamp is not strictly increasing")
        if _finite(timestamp, minimum=0):
            previous = timestamp
        if not isinstance(sample.get("phase"), str) or sample["phase"] not in _PHASES:
            issues.append(f"Sample {index} has no valid phase")
    return issues


def _tracking_metrics(samples, timeline_issues):
    cte_issues = list(timeline_issues)
    cruise_issues = list(timeline_issues)
    cte = []
    cruise_error = []
    for index, sample in enumerate(samples):
        speed = sample.get("speed_mps")
        if not _finite(speed, minimum=0):
            cte_issues.append(f"Sample {index} has no valid speed_mps")
        elif speed > 0.1 and sample.get("phase") != "goal":
            value = sample.get("signed_cte_m")
            if _finite(value):
                cte.append(abs(value))
            else:
                cte_issues.append(f"Moving sample {index} has no valid signed_cte_m")
        if sample.get("phase") != "cruise":
            continue
        stable = sample.get("stable_cruise")
        if not isinstance(stable, bool):
            cruise_issues.append(f"Cruise sample {index} has no stable_cruise declaration")
        elif stable:
            target = sample.get("target_speed_mps")
            if _finite(speed, minimum=0) and _finite(target, minimum=0):
                cruise_error.append((speed - target) ** 2)
            else:
                cruise_issues.append(f"Stable cruise sample {index} has invalid speed data")
    if not cte:
        cte_issues.append("No applicable motion samples were recorded")
    if not cruise_error:
        cruise_issues.append("No stable cruise samples were recorded")
    return {
        "mean_abs_cte_m": _metric(
            "mean_abs_cte_m",
            statistics.mean(cte) if cte and not cte_issues else None,
            "m",
            len(cte),
            cte_issues,
        ),
        "cruise_speed_rms_mps": _metric(
            "cruise_speed_rms_mps",
            math.sqrt(statistics.mean(cruise_error))
            if cruise_error and not cruise_issues
            else None,
            "m/s",
            len(cruise_error),
            cruise_issues,
        ),
    }


def _latency_metrics(samples):
    values = []
    ages = []
    factors = []
    latency_issues = []
    report_issues = []
    for index, sample in enumerate(samples):
        value = sample.get("capture_to_apply_ms")
        if _finite(value, minimum=0):
            values.append(value)
        else:
            latency_issues.append(f"Sample {index} has no valid capture_to_apply_ms")
        receipt = sample.get("capture_receipt_monotonic_s")
        applied = sample.get("apply_monotonic_s")
        if receipt is not None or applied is not None:
            if not (_finite(receipt, minimum=0) and _finite(applied, minimum=0)):
                latency_issues.append(f"Sample {index} has incomplete monotonic audit times")
            elif (
                applied < receipt
                or not _finite(value, minimum=0)
                or not math.isclose(value, (applied - receipt) * 1000, rel_tol=1e-9, abs_tol=1e-6)
            ):
                latency_issues.append(
                    f"Sample {index} latency disagrees with monotonic audit times"
                )
        age = sample.get("frame_age")
        if isinstance(age, int) and not isinstance(age, bool) and age >= 0:
            ages.append(age)
        else:
            report_issues.append(f"Sample {index} has no valid frame_age")
        factor = sample.get("real_time_factor")
        if _finite(factor) and factor > 0:
            factors.append(factor)
        else:
            report_issues.append(f"Sample {index} has no positive real_time_factor")
    if not values:
        latency_issues.append("No capture-to-apply samples were recorded")
    if not samples:
        report_issues.append("No frame-age or real-time-factor samples were recorded")
    return (
        _metric(
            "capture_to_apply_p95_ms",
            _percentile(values, 0.95) if values and not latency_issues else None,
            "ms",
            len(values),
            latency_issues,
        ),
        {
            "status": _INSUFFICIENT if report_issues else "REPORTED",
            "frame_age_max": max(ages) if ages else None,
            "frame_age_sample_count": len(ages),
            "real_time_factor_min": min(factors) if factors else None,
            "real_time_factor_median": statistics.median(factors) if factors else None,
            "real_time_factor_sample_count": len(factors),
            "evidence_issues": report_issues,
        },
    )


def _incident_metrics(events, invalid_event_count):
    markers = [event for event in events if event.get("kind") == "event_stream_status"]
    marker = markers[-1] if markers else {}
    metrics = {}
    for kind, name, id_field, stream_flag in (
        ("collision", "collisions", "contact_id", "collision_complete"),
        ("lane_invasion", "unplanned_lane_invasions", "event_id", "lane_invasion_complete"),
    ):
        issues = []
        if marker.get(stream_flag) is not True:
            issues.append(f"The {kind} event stream is not confirmed complete")
        if invalid_event_count:
            issues.append(f"{invalid_event_count} event records are not mappings")
        identifiers = set()
        raw_count = 0
        excluded = set()
        for index, event in enumerate(events):
            if event.get("kind") != kind:
                continue
            raw_count += 1
            identifier = event.get(id_field)
            if not _nonempty_id(identifier):
                issues.append(f"{kind} event {index} has no valid {id_field}")
                identifier = ("missing", index)
            else:
                identifier = (type(identifier).__name__, identifier)
            if (
                kind == "lane_invasion"
                and event.get("predeclared_legal") is True
                and _nonempty_id(event.get("legal_manoeuvre_id"))
            ):
                excluded.add(identifier)
            else:
                identifiers.add(identifier)
        count = len(identifiers)
        # Observed incidents prove failure even when stream completeness is unknown.
        metric = _metric(name, count if count or not issues else None, "count", raw_count, issues)
        metric["raw_event_count"] = raw_count
        metric["excluded_legal_event_count"] = len(excluded - identifiers)
        metrics[name] = metric
    return metrics


def _false_stops(samples, timeline_issues):
    issues = list(timeline_issues)
    count = 0
    start = None
    counted = False
    for index, sample in enumerate(samples):
        target = sample.get("target_speed_mps")
        timestamp = sample.get("simulation_time_s")
        if not _finite(target, minimum=0):
            issues.append(f"Sample {index} has no valid target_speed_mps")
        if not _finite(timestamp, minimum=0):
            continue
        goal_window = sample.get("in_goal_window", sample.get("phase") == "goal")
        if type(goal_window) is not bool:
            issues.append(f"Sample {index} has no valid geometric goal-window declaration")
        stopped = _finite(target, minimum=0) and target < 0.1 and goal_window is False
        if start is not None and not counted and timestamp - start >= 1.0 - 1e-9:
            count += 1
            counted = True
        if stopped and start is None:
            start = timestamp
        elif not stopped:
            start = None
            counted = False
    return _metric("false_stops", count if not issues else None, "count", len(samples), issues)


def evaluate_s0_run(
    samples: list[dict],
    events: list[dict],
    *,
    completed: bool,
    failure_reason: str | None = None,
    route_install_count: int = 1,
    cleanup_ok: bool = True,
) -> dict:
    """Evaluate one retained trial; incomplete evidence never becomes zero/PASS.

    ``completed`` must come from the endpoint completion predicate, not projected
    progress alone. The caller adds run identity, route/config hashes and paths
    to the returned summary before aggregating the frozen five-trial suite.
    A known failure takes precedence over an evidence gap in the overall status.
    """
    invalid_samples = sum(not isinstance(sample, Mapping) for sample in samples)
    valid_samples = [sample for sample in samples if isinstance(sample, Mapping)]
    invalid_events = sum(
        not isinstance(event, Mapping) or not isinstance(event.get("kind"), str) for event in events
    )
    valid_events = [
        event
        for event in events
        if isinstance(event, Mapping) and isinstance(event.get("kind"), str)
    ]
    timeline_issues = _timeline_issues(valid_samples)
    if invalid_samples:
        timeline_issues.append(f"{invalid_samples} sample records are not mappings")
    metrics = {
        "route_completion": _metric(
            "route_completion",
            int(completed) if isinstance(completed, bool) else None,
            "fraction",
            1 if isinstance(completed, bool) else 0,
            [] if isinstance(completed, bool) else ["Endpoint completion evidence is not boolean"],
        )
    }
    metrics.update(_tracking_metrics(valid_samples, timeline_issues))
    metrics.update(_incident_metrics(valid_events, invalid_events))
    latency, report_only = _latency_metrics(valid_samples)
    if invalid_samples:
        latency["value"] = None
        latency["status"] = _INSUFFICIENT
        latency["evidence_issues"].append(f"{invalid_samples} sample records are not mappings")
    metrics["capture_to_apply_p95_ms"] = latency
    metrics["false_stops"] = _false_stops(valid_samples, timeline_issues)
    issues = list(
        dict.fromkeys(
            issue
            for metric in [*metrics.values(), report_only]
            for issue in metric["evidence_issues"]
        )
    )
    failures = []
    if failure_reason:
        failures.append(failure_reason)
    if not isinstance(route_install_count, int) or isinstance(route_install_count, bool):
        issues.append("Route-install count is not an integer")
    elif route_install_count != 1:
        failures.append(f"Expected one route installation, observed {route_install_count}")
    if cleanup_ok is False:
        failures.append("Episode cleanup failed")
    elif cleanup_ok is not True:
        issues.append("Episode cleanup evidence is not boolean")
    failed_metrics = [name for name, metric in metrics.items() if metric["status"] == "FAIL"]
    status = "FAIL" if failures or failed_metrics else _INSUFFICIENT if issues else "PASS"
    return {
        "scenario": "S0",
        "status": status,
        "completed": completed,
        "failure_reason": failure_reason,
        "failures": failures,
        "failed_metrics": failed_metrics,
        "sample_count": len(samples),
        "route_install_count": route_install_count,
        "cleanup_ok": cleanup_ok,
        "metrics": metrics,
        "report_only": report_only,
        "evidence_issues": issues,
        "definitions": {
            "cte_motion_rule": "speed_mps > 0.1 and phase != goal; mean absolute signed rear-axle CTE",
            "cruise_rule": "phase == cruise and stable_cruise is True; no speed-error selection",
            "percentile_rule": "linear interpolation at (n - 1) * 0.95",
            "false_stop_rule": "target_speed_mps < 0.1 for >= 1 simulation second outside goal",
            "collision_deduplication": "one contact_id per continuous contact; retain raw callbacks",
            "lane_exclusion_rule": "predeclared_legal is True and nonempty legal_manoeuvre_id only",
            "latency_scope": "host-monotonic capture receipt to command application only",
        },
    }


def aggregate_s0_runs(summaries: list[dict]) -> dict:
    """Retain every trial and require exactly five runs with frozen provenance."""
    issues = []
    failures = []
    if len(summaries) != S0_REQUIRED_RUNS:
        issues.append(f"Expected exactly {S0_REQUIRED_RUNS} runs, observed {len(summaries)}")
    if any(run.get("status") == "FAIL" for run in summaries):
        failures.append("At least one retained run failed")
    if any(run.get("status") not in {"PASS", "FAIL"} for run in summaries):
        issues.append("At least one retained run has insufficient evidence or invalid status")
    if any(
        "formal_acceptance" in run and run["formal_acceptance"] is not True for run in summaries
    ):
        issues.append("Smoke or temporary-HOLD trials cannot establish formal S0 acceptance")
    provenance = {}
    for field in ("route_sha256", "config_sha256"):
        values = [run.get(field) for run in summaries]
        hashes = {value for value in values if isinstance(value, str) and len(value) == 64}
        if len(hashes) > 1:
            failures.append(f"Frozen {field} differs between runs")
        if len(values) == 0 or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in values
        ):
            issues.append(f"Every run must record a valid lowercase {field}")
        provenance[field] = next(iter(hashes)) if len(hashes) == 1 else None
    identifiers = [run.get("run_id") for run in summaries]
    if all(_nonempty_id(value) for value in identifiers):
        if len({(type(value).__name__, value) for value in identifiers}) != len(identifiers):
            failures.append("Run IDs are not unique")
    else:
        issues.append("Every run must have an explicit run_id")
    worst = {}
    for name in S0_LIMITS:
        values = [run.get("metrics", {}).get(name, {}).get("value") for run in summaries]
        if values and all(_finite(value) for value in values):
            worst[name] = min(values) if name == "route_completion" else max(values)
            if (
                worst[name] < S0_LIMITS[name]
                if name == "route_completion"
                else worst[name] > S0_LIMITS[name]
            ):
                failures.append(f"At least one run exceeds the {name} acceptance limit")
        else:
            worst[name] = None
            issues.append(f"At least one run lacks a valid {name} value")
        if any(run.get("metrics", {}).get(name, {}).get("status") != "PASS" for run in summaries):
            issues.append(f"At least one run lacks passing {name} evidence")
    status = "FAIL" if failures else _INSUFFICIENT if issues else "PASS"
    return {
        "scenario": "S0",
        "status": status,
        "required_runs": S0_REQUIRED_RUNS,
        "recorded_runs": len(summaries),
        "passed_runs": sum(run.get("status") == "PASS" for run in summaries),
        "failed_runs": sum(run.get("status") == "FAIL" for run in summaries),
        "worst_run_metrics": worst,
        "failures": failures,
        "evidence_issues": issues,
        "frozen_provenance": provenance,
        "runs": list(summaries),
    }
