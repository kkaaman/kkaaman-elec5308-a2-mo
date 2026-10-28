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
mean deceleration over the episode, the peak (over 3 ticks), and the stopping distance
when it ended at a standstill. Closed-loop commands vary within an episode, so the table
bins by the median command; the open-loop steps are the cleaner measure.
"""

import csv
import json
import math
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
    peak = float(np.convolve(decel, np.ones(3) / 3, mode="valid").max()) if len(decel) >= 3 else float(decel.max())
    stopped = v[-1] <= 0.05
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
        stopping_distance_m=round(float(np.sum(0.5 * (v[1:] + v[:-1]) * np.diff(t))), 2) if stopped else None,
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
    """Mean and peak deceleration and lag by command and starting speed."""
    rows = []
    for blo, bhi in brake_bands:
        for vlo, vhi in speed_bands:
            sel = [e for e in eps if blo <= e["brake_median"] < bhi and vlo <= e["speed_mps"] < vhi]
            if not sel:
                continue
            mean = np.array([e["mean_decel_mps2"] for e in sel])
            peak = np.array([e["peak_decel_mps2"] for e in sel])
            lags = [e["lag_s"] for e in sel if e["lag_s"] is not None]
            rows.append(
                dict(
                    brake=f"{blo:.2f}-{bhi:.2f}",
                    speed_mps=f"{vlo:g}-{vhi:g}",
                    episodes=len(sel),
                    mean_decel_median_mps2=round(float(np.median(mean)), 2),
                    peak_decel_median_mps2=round(float(np.median(peak)), 2),
                    peak_decel_p10_mps2=round(float(np.percentile(peak, 10)), 2),
                    lag_median_s=round(float(np.median(lags)), 2) if lags else None,
                    lag_max_s=round(float(max(lags)), 2) if lags else None,
                )
            )
    return rows


def against_assumptions(eps, full_brake=0.9):
    """Each assumed deceleration against the peak achieved at a brake command of at least
    ``full_brake`` from 2 m/s or more (the 10th percentile, so a rule is checked against a
    bad case, not the median), and the measured lag against ``response_s``."""
    strong = [e for e in eps if e["brake_max"] >= full_brake and e["speed_mps"] >= 2.0]
    if not strong:
        return dict(note=f"no episode with a brake command of {full_brake:g} or more from 2 m/s: run BL-E-03 (scripts/calibrate_longitudinal.py)")
    peak = float(np.percentile([e["peak_decel_mps2"] for e in strong], 10))
    lags = [e["lag_s"] for e in strong if e["lag_s"] is not None]
    lag = max(lags) if lags else math.nan
    return dict(
        episodes=len(strong),
        achieved_peak_decel_p10_mps2=round(peak, 2),
        measured_lag_max_s=None if not lags else round(lag, 2),
        assumed_response_s=ASSUMED_RESPONSE_S,
        response_ok=None if not lags else lag <= ASSUMED_RESPONSE_S,
        rules={name: dict(assumed_mps2=a, achieved=peak >= a) for name, a in ASSUMED.items()},
    )
