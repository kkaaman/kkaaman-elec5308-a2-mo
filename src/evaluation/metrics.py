"""Scenario applicability and per-run metrics; absent data is never a pass."""

import json
import math
from pathlib import Path

import numpy as np

from evaluation.event_checks import count_contacts, count_unplanned_invasions
from evaluation.run_logger import file_sha256

METRIC_KEYS = {
    "Route completion (minimum per run)": "route_completion",
    "Collisions (maximum per run)": "collisions",
    "Unplanned lane invasions (max per run)": "unplanned_lane_invasions",
    "Cross-track mean (worst run)": "cte_mean_m",
    "Cruise speed RMS (worst run)": "cruise_speed_rms_mps",
    "Capture-to-apply p95 (worst run)": "capture_apply_p95_ms",
    "False stops (maximum per run)": "false_stops",
    "Minimum stopped bumper margin": "hold_margin_min_m",
    "Maximum stopped bumper margin": "hold_margin_max_m",
    "HOLD maximum speed": "hold_max_speed_mps",
    "Resume delay (median)": "resume_delay_s",
    "Continuous HOLD duration (minimum)": "continuous_hold_s",
    "Cruise speed drop versus matched S0": "cruise_drop_mps",
}


def _max_duration(rows, predicate, dt):
    start, duration = None, 0.0
    previous = None
    for row in rows:
        active = predicate(row)
        if active:
            if start is None or (previous is not None and row["sim_time_s"] - previous > dt * 1.5):
                start = row["sim_time_s"]
            duration = max(duration, row["sim_time_s"] - start + dt)
        else:
            start = None
        previous = row["sim_time_s"]
    return duration


def _false_stops(rows, dt, waiting=None):
    """Stops >= 1 s outside the goal window with no obstacle in the corridor.

    ``waiting`` is the (removal, motion) interval after a scripted removal:
    holding until fresh depth confirms clearance is the expected resume
    behaviour, reported as the resume delay, not a false stop. A run that
    never moves again after a removal gets no exemption.
    """
    count, elapsed, counted = 0, 0.0, False
    for row in rows:
        exempt = waiting is not None and waiting[0] <= row["sim_time_s"] < waiting[1]
        qualifies = (
            not exempt
            and row["target_mps"] < 0.1
            and row["distance_to_goal_m"] > 2.0
            and not row.get("truth_in_corridor", False)
        )
        if qualifies:
            elapsed += dt
            if elapsed >= 1.0 - 1e-9 and not counted:
                count, counted = count + 1, True
        else:
            elapsed, counted = 0.0, False
    return count


def _timing(rows, manifest):
    """Wall-clock budget overruns; report-only next to the scored p95."""
    budget = manifest.get("step_wall_budget_s", 0.1)
    capture = [r["capture_apply_s"] for r in rows]
    steps = [r.get("stages", {}).get("total") for r in rows]
    steps = [s for s in steps if s is not None]
    return dict(
        step_budget_s=budget,
        step_over_budget=sum(s > budget for s in steps),
        step_max_ms=max(steps) * 1000 if steps else None,
        capture_apply_over_100ms=sum(c > 0.1 for c in capture),
        capture_apply_max_ms=max(capture) * 1000,
        tick_to_apply_max_ms=max(r.get("tick_to_apply_s", 0.0) for r in rows) * 1000,
        stalls_over_500ms=sum(r.get("tick_to_apply_s", 0.0) > 0.5 for r in rows),
        guard=manifest.get("wall_guard"),
    )


def _localisation(rows):
    """Shadow pose-estimate error while moving (report-only; never scored)."""
    moving = [r["localisation"] for r in rows if "localisation" in r and r["speed_mps"] > 0.5]
    if not moving:
        return None

    def stats(values):
        values = np.abs(np.asarray(values, dtype=float))
        return dict(
            mean=float(values.mean()),
            p95=float(np.percentile(values, 95)),
            max=float(values.max()),
        )

    measured = np.array([r["yaw_rate_rps"] for r in moving])
    truth = np.array([r["true_yaw_rate_rps"] for r in moving])
    agreement = (
        float(np.corrcoef(measured, truth)[0, 1])
        if measured.std() > 1e-9 and truth.std() > 1e-9
        else None
    )
    return dict(
        moving_samples=len(moving),
        lateral_error_m=stats([r["lateral_error_m"] for r in moving]),
        along_error_m=stats([r["along_error_m"] for r in moving]),
        position_error_m=stats([r["error_m"] for r in moving]),
        yaw_error_deg=stats([math.degrees(r["yaw_error_rad"]) for r in moving]),
        final_position_error_m=float(moving[-1]["error_m"]),
        # Sign/scale sanity check of the gyro input against the true yaw rate.
        gyro_truth_correlation=agreement,
    )


def summarize(rows, events, manifest):
    dt = manifest["dt_s"]
    if not rows:
        return {"samples": 0, "metrics": {key: None for key in METRIC_KEYS.values()}}
    moving = [r for r in rows if r["speed_mps"] > 0.1]
    cruise = [r for r in rows if r.get("cruise_window", False)]
    hold = [r for r in rows if r["state"] == "HOLD"]
    margins = [
        r["truth_gap_m"] for r in hold if r["speed_mps"] <= 0.1 and r.get("truth_gap_m") is not None
    ]
    last = rows[-1]
    completed = bool(
        last.get("finished")
        and last["speed_mps"] <= 0.1
        and last["distance_to_goal_m"] <= manifest.get("goal_tolerance_m", 1.0)
    )
    removals = [e["sim_time_s"] for e in events if e["type"] == "obstacle_removed"]
    resume = waiting = None
    if removals:
        resumed = next(
            (r for r in rows if r["sim_time_s"] >= removals[0] and r["speed_mps"] > 0.1), None
        )
        if resumed:
            resume = resumed["sim_time_s"] - removals[0]
            waiting = (removals[0], resumed["sim_time_s"])
    sensor_events = manifest.get("event_sensors_verified", False)
    # Collision contacts are keyed by actor, separated by >=1 s of no callback.
    # The same functions are exercised live by scripts/verify_event_sensors.py.
    contacts = count_contacts(events)
    invasions = count_unplanned_invasions(events)
    metrics = dict(
        route_completion=1.0 if completed else 0.0,
        collisions=contacts if sensor_events else None,
        unplanned_lane_invasions=invasions if sensor_events else None,
        cte_mean_m=float(np.mean([abs(r["cte_m"]) for r in moving])) if moving else None,
        cruise_speed_rms_mps=float(
            np.sqrt(np.mean([(r["speed_mps"] - manifest["cruise_mps"]) ** 2 for r in cruise]))
        )
        if cruise
        else None,
        cruise_mean_mps=float(np.mean([r["speed_mps"] for r in cruise])) if cruise else None,
        capture_apply_p95_ms=float(
            np.percentile([r["capture_apply_s"] * 1000 for r in rows], 95, method="linear")
        ),
        tick_to_apply_p95_ms=float(
            np.percentile(
                [r.get("tick_to_apply_s", r["capture_apply_s"]) * 1000 for r in rows],
                95,
                method="linear",
            )
        ),
        false_stops=(
            _false_stops(rows, dt, waiting) if manifest.get("truth_channel_complete") else None
        ),
        hold_margin_min_m=min(margins) if margins else None,
        hold_margin_max_m=max(margins) if margins else None,
        hold_max_speed_mps=max((r["speed_mps"] for r in hold), default=None),
        continuous_hold_s=_max_duration(
            rows,
            lambda r: (
                r["state"] == "HOLD" and r["speed_mps"] <= 0.1 and r.get("truth_in_corridor", False)
            ),
            dt,
        ),
        resume_delay_s=resume,
        cruise_drop_mps=None,
    )
    if manifest["scenario"] == "S4":
        metrics["route_completion"] = None
    return dict(
        samples=len(rows),
        timing=_timing(rows, manifest),
        localisation=_localisation(rows),
        cruise_samples=len(cruise),
        moving_samples=len(moving),
        metrics=metrics,
        max_simulation_age_s=max(r.get("simulation_age_s", 0) for r in rows),
        real_time_factor=(rows[-1]["sim_time_s"] - rows[0]["sim_time_s"])
        / max(1e-9, manifest.get("loop_wall_s", 0)),
        scope=manifest["engine"],
        percentile_method="numpy linear",
    )


def evaluate(summary, criteria, scenario, *, collected=True):
    outcomes = []
    for definition in criteria["metrics"]:
        if definition["scenario"] != scenario:
            continue
        key = METRIC_KEYS[definition["metric"]]
        value = summary["metrics"].get(key)
        rule = definition["rule"]
        if not definition["applicable"]:
            status = "NOT_APPLICABLE"
        elif value is None or not math.isfinite(value):
            status = "MISSING"
        elif rule == "report":
            status = "REPORTED"
        else:
            passed = (
                (rule == "<=" and value <= definition["upper"])
                or (rule == ">=" and value >= definition["lower"])
                or (rule == "=" and value == definition["lower"])
            )
            status = "PASS" if passed else "FAIL"
        outcomes.append(dict(metric=key, value=value, status=status))
    passed = (
        collected
        and bool(outcomes)
        and all(x["status"] in {"PASS", "REPORTED", "NOT_APPLICABLE"} for x in outcomes)
    )
    return dict(status="PASS" if passed else "INCOMPLETE_OR_FAIL", metrics=outcomes)


def load_run(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    if set(manifest.get("files_sha256", {})) != {"samples.jsonl", "events.jsonl"}:
        raise ValueError("Run evidence hashes are missing")
    for name, digest in manifest["files_sha256"].items():
        if name not in {"samples.jsonl", "events.jsonl"} or file_sha256(folder / name) != digest:
            raise ValueError("Run evidence hash mismatch")
    rows = [json.loads(line) for line in (folder / "samples.jsonl").read_text().splitlines()]
    events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
    return manifest, rows, events
