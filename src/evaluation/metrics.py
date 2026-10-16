"""Offline S0 metrics. Missing evidence is blocked, never interpreted as zero."""
from __future__ import annotations

import json
import math
from pathlib import Path


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return None
    p = (len(ordered) - 1) * q
    lo, hi = math.floor(p), math.ceil(p)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (p - lo)


def first_sustained(rows, predicate, duration, dt):
    start = previous = None
    for row in rows:
        t = row["sim_time_s"]
        if previous is not None and t - previous > dt * 1.01:
            start = None
        if predicate(row):
            if start is None:
                start = t
            if t - start >= duration - 1e-9:
                return row
        else:
            start = None
        previous = t
    return None


def stopped_margin(rows, config):
    """GT front-to-surface route gap when the sustained stop first holds."""
    row = first_sustained(rows, lambda r: abs(r["ego"]["speed_mps"]) <= config["stopped_speed_mps"],
                          config["stopped_duration_s"], config["dt_s"])
    return None if row is None else row.get("truth_front_gap_m")


def read_json(text):
    def reject(value):
        raise ValueError(f"nonfinite JSON value: {value}")
    return json.loads(text, parse_constant=reject)


def evaluate_run(directory):
    directory = Path(directory)
    manifest = read_json((directory / "manifest.json").read_text())
    if manifest["schema_version"] != 1:
        raise ValueError("unsupported log schema")
    c = manifest["config"]["acceptance"]
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in c.values()):
        raise ValueError("acceptance settings must be finite and positive")
    if not isinstance(c["trials"], int):
        raise ValueError("trials must be an integer")
    rows = [read_json(s) for s in (directory / "ticks.jsonl").read_text().splitlines()]
    events = [read_json(s) for s in (directory / "evaluation.jsonl").read_text().splitlines()]
    if not rows or len(rows) != len(events):
        raise ValueError("empty or incomplete run")
    for i, (r, e) in enumerate(zip(rows, events)):
        if r["frame_id"] != e["frame_id"] or (i and (r["frame_id"] <= rows[i-1]["frame_id"] or r["sim_time_s"] <= rows[i-1]["sim_time_s"])):
            raise ValueError("misaligned or unordered log")
        if i and (r["frame_id"] != rows[i-1]["frame_id"] + 1 or not math.isclose(r["sim_time_s"] - rows[i-1]["sim_time_s"], c["dt_s"], abs_tol=1e-8)):
            raise ValueError("missing frame or incorrect simulation step")
        if not r["host_tick_s"] <= r["host_receive_s"] <= r["host_command_s"]:
            raise ValueError("invalid host clock order")
    complete = first_sustained(rows, lambda r: r["goal_distance_m"] <= c["goal_tolerance_m"] and abs(r["ego"]["speed_mps"]) <= c["stopped_speed_mps"], c["stopped_duration_s"], c["dt_s"]) is not None
    false_stops = 0
    start = previous = stable_since = last_target = None
    counted = False
    errors, latencies, ages, ctes = [], [], [], []
    collisions, lane_events = 0, 0
    contacts = {}
    for r, e in zip(rows, events):
        t, d = r["sim_time_s"], r["diagnostics"]
        continuous = previous is None or t - previous <= c["dt_s"] * 1.01
        low = d["target_speed_mps"] < 0.1 and not r["planned_final_stop"]
        if low:
            if start is None or not continuous:
                start, counted = t, False
            if not counted and t - start >= c["stopped_duration_s"] - 1e-9:
                false_stops += 1
                counted = True
        else:
            start, counted = None, False
        target, caps = d["target_speed_mps"], d["speed_caps_mps"]
        cruise = d["behaviour"] == "CRUISE" and all(caps.get(k, caps.get("cruise", target)) >= caps.get("cruise", target) for k in ("curve", "endpoint"))
        if not cruise or target != last_target or not continuous:
            stable_since = t if cruise else None
        if stable_since is not None and t - stable_since >= c["stable_cruise_s"] - 1e-9:
            errors.append((r["ego"]["speed_mps"] - target) ** 2)
        latencies.append(1000 * (r["host_command_s"] - r["host_tick_s"]))
        ages.append(1000 * (r["host_receive_s"] - r["host_tick_s"]))
        if d["signed_cte_m"] is not None:
            ctes.append(d["signed_cte_m"])
        for event in e["collisions"]:
            key = event["other_actor_id"]
            if key is None or key not in contacts or t - contacts[key] > c["collision_dedup_s"] + 1e-9:
                collisions += 1
            contacts[key] = t
        lane_events += len(e["lane_invasions"])
        last_target, previous = target, t
    rms = math.sqrt(sum(errors) / len(errors)) if errors else None
    mean_cte = sum(map(abs, ctes)) / len(ctes) if len(ctes) == len(rows) else None
    metrics = {"route_complete": complete, "collisions": collisions, "lane_invasions": lane_events,
               "false_stops": false_stops, "cruise_rms_mps": rms, "cruise_samples": len(errors),
               "mean_cte_m": mean_cte, "max_cte_m": max(map(abs, ctes)) if ctes else None,
               "latency_p50_ms": percentile(latencies, .5), "latency_p95_ms": percentile(latencies, .95),
               "latency_max_ms": max(latencies), "frame_age_max_ms": max(ages),
               "missed_deadlines": sum(v > c["dt_s"] * 1000 for v in latencies)}
    wall = rows[-1]["host_command_s"] - rows[0]["host_tick_s"]
    metrics["real_time_factor"] = (rows[-1]["sim_time_s"] - rows[0]["sim_time_s"]) / wall if wall > 0 else None
    checks = {"route_complete": complete, "collisions": collisions == 0, "lane_invasions": lane_events == 0,
              "false_stops": false_stops == 0, "cruise_rms_mps": None if rms is None else rms <= c["cruise_rms_mps"],
              "mean_cte_m": None if mean_cte is None else mean_cte <= c["mean_cte_m"],
              "latency_p95_ms": metrics["latency_p95_ms"] <= c["latency_p95_ms"]}
    supported = manifest["config"]["scenario"] == "S0"
    status = "FAIL" if False in checks.values() else "BLOCKED" if not supported or None in checks.values() else "PASS"
    return {"schema_version": 1, "status": status, "evidence_kind": manifest["config"].get("evidence_kind", "unspecified"),
            "metrics": metrics, "checks": checks, "scope": "S0 only; S1-S5 and dataset depth evaluation remain blocked"}


def aggregate(directories):
    directories = list(directories)
    runs = [evaluate_run(d) for d in directories]
    manifests = [json.loads((Path(d) / "manifest.json").read_text()) for d in directories]
    if not manifests:
        return {"status": "BLOCKED", "runs": []}
    baseline = manifests[0]["config"]
    compatible = all(m["config"] == baseline for m in manifests)
    unique = len({str(Path(d).resolve()) for d in directories}) == len(directories)
    seeds = len({m["seed"] for m in manifests}) == len(manifests)
    status = "FAIL" if any(r["status"] == "FAIL" for r in runs) else "BLOCKED"
    if compatible and unique and seeds and len(runs) >= baseline["acceptance"]["trials"] and all(r["status"] == "PASS" for r in runs):
        status = "PASS"
    return {"status": status, "runs": runs, "compatible_config": compatible, "unique_runs_and_seeds": unique and seeds}
