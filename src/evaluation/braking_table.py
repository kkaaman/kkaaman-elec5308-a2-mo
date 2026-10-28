"""Achieved braking against what the safety rules assume (third review, section 4).

The mapless stack's speed rules each assume a deceleration and a response time: the
planner's ramp (``PlannerConfig.decel_mps2``, 1 m/s^2), a predicted conflict
(``StackConfig.conflict_decel_mps2``, 2 m/s^2) and the following rule's worst case
(``FollowingConfig.ego_brake_mps2``, 4 m/s^2 after ``response_s``, 0.5 s). This module
measures what the car achieved, from the evaluator's truth speed in drive logs, or from the
open-loop brake steps of BL-E-03 (``scripts/calibrate_longitudinal.py``), and sets it
against those assumptions.

From a drive log, a braking episode is a run of ticks with a brake command of at least
``min_brake`` from a speed of at least ``min_speed_mps``. For each: the starting speed,
the median command, the lag to the first tick decelerating at 0.5 m/s^2 or more, the
mean deceleration over the episode, the peak (over 3 ticks, above 0.5 m/s: the last ticks
before a standstill read as a jump to zero), the stopping distance when it ended at a
standstill, and the effective deceleration v0^2 / (2 x stopping distance), lag included.
Closed-loop commands vary within an episode, so the table bins by the median command; the
open-loop steps are the cleaner measure. An effective deceleration above 1 g is reported
as implausible and left out of the comparison (BL-E-03 on Town05, 28 Oct: 0.85 m from
5 m/s, twice).
"""

import csv
import json
from pathlib import Path

import numpy as np

ASSUMED = {
    "planner ramp (PlannerConfig.decel_mps2)": 1.0,
    "predicted conflict (StackConfig.conflict_decel_mps2)": 2.0,
    "following worst case (FollowingConfig.ego_brake_mps2)": 4.0,
}
ASSUMED_RESPONSE_S = 0.5  # FollowingConfig.response_s


def read_log(run):
    path = Path(run) / "drive_log.jsonl" if Path(run).is_dir() else Path(run)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def episodes(log, min_brake=0.05, min_speed_mps=0.3, min_ticks=4):
    """Braking episodes in a drive log (rows with ``time_s``, ``truth.speed_mps`` and
    ``command`` = [steer, throttle, brake], applied after the row's tick)."""
    out, k = [], 0
    while k < len(log) - 1:
        row = log[k]
        brake = (row.get("command") or [0, 0, 0])[2]
        if brake < min_brake or row["truth"]["speed_mps"] < min_speed_mps:
            k += 1
            continue
        j = k
        while j + 1 < len(log) and (log[j].get("command") or [0, 0, 0])[2] >= min_brake and log[j]["truth"]["speed_mps"] > 0.05:
            j += 1
        span = log[k : j + 1]
        if len(span) > min_ticks:
            out.append(_episode(span))
        k = j + 1
    return out


def _episode(span):
    t = np.array([r["time_s"] for r in span])
    v = np.array([r["truth"]["speed_mps"] for r in span])
    brakes = np.array([(r.get("command") or [0, 0, 0])[2] for r in span[:-1]])
    decel = -np.diff(v) / np.maximum(np.diff(t), 1e-3)
    lag = next((float(t[i + 1] - t[0]) for i, d in enumerate(decel) if d >= 0.5), None)
    moving = decel[v[1:] > 0.5]
    if len(moving) >= 3:
        peak = float(np.convolve(moving, np.ones(3) / 3, mode="valid").max())
    else:
        peak = float(moving.max()) if len(moving) else float(decel.max())
    stopped = v[-1] <= 0.05
    distance = float(np.sum(0.5 * (v[1:] + v[:-1]) * np.diff(t)))
    effective = v[0] ** 2 / (2 * distance) if stopped and distance > 0 else None
    d = span[0].get("driver") or {}
    return dict(
        start_s=round(float(t[0]), 2),
        duration_s=round(float(t[-1] - t[0]), 2),
        speed_mps=round(float(v[0]), 2),
        brake_median=round(float(np.median(brakes)), 2),
        brake_max=round(float(brakes.max()), 2),
        lag_s=None if lag is None else round(lag, 2),
        mean_decel_mps2=round(float((v[0] - v[-1]) / max(t[-1] - t[0], 1e-3)), 2),
        peak_decel_mps2=round(peak, 2),
        stopping_distance_m=round(distance, 2) if stopped else None,
        effective_decel_mps2=None if effective is None else round(float(effective), 2),
        why=d.get("stop_reason") or d.get("speed_cap_reason") or "",
    )


def steps_from_calibration(csv_path):
    """BL-E-03's open-loop brake steps (``longitudinal.csv``) as episodes."""
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    trials = {}
    for r in rows:
        if r["phase"] == "brake":
            trials.setdefault(r["trial"], []).append(r)
    out = []
    for trace in trials.values():
        span = [dict(time_s=float(r["sim_time_s"]), truth=dict(speed_mps=float(r["speed_mps"])), command=[0.0, 0.0, float(r["brake"])]) for r in trace]
        if len(span) > 3:
            out.append(_episode(span))
    return out


def table(eps, brake_bands=((0.05, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.01)), speed_bands=((0.3, 2.0), (2.0, 4.0), (4.0, 9.0))):
    """Mean, peak and effective deceleration and lag by command and starting speed."""
    rows = []
    for blo, bhi in brake_bands:
        for vlo, vhi in speed_bands:
            sel = [e for e in eps if blo <= e["brake_median"] < bhi and vlo <= e["speed_mps"] < vhi]
            if not sel:
                continue
            mean = np.array([e["mean_decel_mps2"] for e in sel])
            peak = np.array([e["peak_decel_mps2"] for e in sel])
            lags = [e["lag_s"] for e in sel if e["lag_s"] is not None]
            eff = [e["effective_decel_mps2"] for e in sel if e.get("effective_decel_mps2") is not None]
            rows.append(
                dict(
                    brake=f"{blo:.2f}-{bhi:.2f}",
                    speed_mps=f"{vlo:g}-{vhi:g}",
                    episodes=len(sel),
                    mean_decel_median_mps2=round(float(np.median(mean)), 2),
                    peak_decel_median_mps2=round(float(np.median(peak)), 2),
                    peak_decel_p10_mps2=round(float(np.percentile(peak, 10)), 2),
                    effective_decel_median_mps2=round(float(np.median(eff)), 2) if eff else None,
                    lag_median_s=round(float(np.median(lags)), 2) if lags else None,
                    lag_max_s=round(float(max(lags)), 2) if lags else None,
                )
            )
    return rows


def against_assumptions(eps, full_brake=0.9, g_mps2=9.81):
    """Each rule's assumed stop against the stops full brake achieved, from 2 m/s or more
    with a brake command of at least ``full_brake``: the rule's stopping distance
    v x response + v^2 / (2 a) (the following rule's ``response_s``; none for the planner's
    ramp and the conflict rule, which carry no reaction allowance) against the measured
    one, lag included. ``margin_m`` is the least of (assumed - measured) over the stops; a
    rule holds when it is not negative. Comparing distances, not a peak or an average
    deceleration, keeps the lag and the end of the stop in. An effective deceleration
    above 1 g is implausible and left out (counted)."""
    strong = [e for e in eps if e["brake_max"] >= full_brake and e["speed_mps"] >= 2.0]
    stops = [e for e in strong if e.get("effective_decel_mps2") is not None]
    plausible = [e for e in stops if e["effective_decel_mps2"] <= g_mps2]
    if not plausible:
        return dict(note=f"no stop from 2 m/s or more with a brake command of {full_brake:g} or more: run BL-E-03 (scripts/calibrate_longitudinal.py)")
    lags = [e["lag_s"] for e in strong if e["lag_s"] is not None]
    rules = {}
    for name, a in ASSUMED.items():
        response = ASSUMED_RESPONSE_S if "following" in name else 0.0
        margin = min(e["speed_mps"] * response + e["speed_mps"] ** 2 / (2 * a) - e["stopping_distance_m"] for e in plausible)
        rules[name] = dict(assumed_mps2=a, response_s=response, margin_m=round(float(margin), 2), achieved=margin >= 0.0)
    return dict(
        stops=len(plausible),
        implausible_excluded=len(stops) - len(plausible),
        worst_effective_decel_mps2=round(float(min(e["effective_decel_mps2"] for e in plausible)), 2),
        by_speed=[dict(speed_mps=e["speed_mps"], stopping_distance_m=e["stopping_distance_m"], effective_decel_mps2=e["effective_decel_mps2"]) for e in sorted(plausible, key=lambda e: e["speed_mps"])],
        measured_lag_max_s=round(float(max(lags)), 2) if lags else None,
        assumed_response_s=ASSUMED_RESPONSE_S,
        response_ok=None if not lags else max(lags) <= ASSUMED_RESPONSE_S,
        rules=rules,
    )
