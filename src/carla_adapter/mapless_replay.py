"""Replay a recorded mapless drive through the driver offline (``frames/`` from
``run_mapless_drive.py --record-frames``).

Open loop: the recorded frames are fed to a fresh ``MaplessDriver`` built as the harness
built it (same suite, overrides, body, speed, keep-right), and the driver's commands are
dropped; the car moves as it did live. So what comes out is the driver's perception and
estimation on real data — terrain grid, corridor, junction context, exit-line
measurements — tick for tick, with configuration changes (``--junction key=value``)
tried against the same frames. Control questions need the simulator.

Truth: the drive log's rear-axle pose (evaluator core frame) and ``turn_truth.json``
(approach/junction/exit lane centrelines, kerb distances). Odometry-frame estimates (the
exit line and its kerb stations) are taken to the core frame through each tick's own
pair of logged poses (odometry and truth), so they are scored against the true exit kerb
as the car saw them, free of odometry drift. Replay errors are signed positive to the
RIGHT of travel; the drive log's ``lane_offset_m`` is positive to the LEFT.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path

import numpy as np

from carla_adapter.frame_record import iter_frames
from carla_adapter.mapless_drive import driver_config
from mapless.driver import MaplessDriver
from mapless.junction import JunctionConfig
from mapless.satnav import SatNavRoute
from mapless.sensor_suite import load_sensor_suite, suite_from_dict
from mapless.terrain import VehicleBody


def rigid_fit(source, target):
    """(R, t) with target ≈ source @ R.T + t, least squares (Kabsch, 2-D)."""
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    cs, ct = source.mean(axis=0), target.mean(axis=0)
    h = (source - cs).T @ (target - ct)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, d]) @ u.T
    return r, ct - cs @ r.T


def tick_transform(row):
    """(R, t) taking odometry points to the core frame at this tick, from the logged
    odometry pose and the truth rear-axle pose: T_core<-odo = T_truth o T_odo^-1. Per tick,
    so odometry drift (0.3-0.5 m and up to a degree over a run) does not smear into the
    kerb and lane errors as a single global fit did (fresh-eyes review of 58adaf3); the
    question is where the car believes the kerb is relative to itself."""
    ox, oy, oyaw = row["driver"]["odometry"]
    tx, ty = row["truth"]["rear_axle_xy"]
    tyaw = row["truth"]["yaw_rad"]
    d = tyaw - oyaw
    r = np.array([[math.cos(d), -math.sin(d)], [math.sin(d), math.cos(d)]])
    t = np.array([tx, ty]) - r @ np.array([ox, oy])
    return r, t


def odometry_to_core(log_rows):
    """Rigid transform from the driver's odometry frame to the truth (core) frame, fitted
    over the ticks with trusted odometry; returns (R, t, rms_m). A sanity figure for the
    recording (the per-tick transform does the scoring)."""
    pairs = [(r["driver"]["odometry"][:2], r["truth"]["rear_axle_xy"]) for r in log_rows if r["driver"].get("odometry_trusted") and r["driver"].get("odometry")]
    if len(pairs) < 10:
        raise ValueError("Not enough trusted-odometry ticks to tie the frames together")
    src = np.array([p[0] for p in pairs])
    dst = np.array([p[1] for p in pairs])
    r, t = rigid_fit(src, dst)
    rms = float(np.sqrt(np.mean(np.sum((src @ r.T + t - dst) ** 2, axis=1))))
    return r, t, rms


def exit_kerb_truth(truth, command):
    """The true exit kerb as a line (point, unit direction) in the core frame: the exit
    lane's centreline offset ``exit_kerb_distance_m`` to its right (right turn) — the kerb
    the car places the lane from."""
    exit_line = np.asarray(truth["exit"], dtype=float)
    d = exit_line[-1] - exit_line[0]
    d = d / max(np.linalg.norm(d), 1e-9)
    right = np.array([d[1], -d[0]])  # core frame is right-handed (CARLA x, -y): right of travel
    # The exit lane's right kerb for both commands: the site search accepts only exit lanes
    # with a kerb on their right, and the car places the lane from that kerb either way.
    return exit_line[0] + right * truth["exit_kerb_distance_m"], d, right


def signed_offset(points, origin, direction, right):
    """Signed lateral distance of points from the line (positive to the right of travel)."""
    rel = np.asarray(points, dtype=float).reshape(-1, 2) - origin
    return rel @ right


def build_driver(run, junction_overrides, suite_path=None):
    report = json.loads((run / "drive_report.json").read_text(encoding="utf-8"))
    meta = json.loads((run / "frames" / "meta.json").read_text(encoding="utf-8"))
    if suite_path:
        # A suite file given by hand: it must be the recorded one (the report's overrides
        # are rounded to 3 digits, so they are not re-applied; the meta carries the sha).
        suite = load_sensor_suite(suite_path)
        if suite.sha256 != meta.get("suite_sha256", report["suite_sha256"]):
            raise ValueError(f"Suite {suite_path} does not match the run's (sha {report['suite_sha256'][:12]})")
    elif "suite_spec" in meta:
        suite = suite_from_dict(meta["suite_spec"])  # the recording is self-contained
    else:
        suite = suite_from_dict(report["rig"]["suite"])  # recordings before the spec went into meta.json
    dc = report["driver_config"]
    body = VehicleBody(**dc["body"])
    ctrl = dc["control"]["controller"]
    config = driver_config(body, ctrl["wheelbase_m"], ctrl["max_steering_angle_rad"], dc["planner"]["cruise_mps"], dc["corridor"]["keep_right_m"])
    junction = JunctionConfig(**{**dataclasses.asdict(config.junction), **junction_overrides})
    config = dataclasses.replace(config, junction=junction)
    route = SatNavRoute.load(run / "route.json")
    return MaplessDriver(suite, route, config, 0.05), report, config


def parse_values(items):
    out = {}
    for item in items or ():
        key, _, value = item.partition("=")
        if not _:
            raise ValueError(f"--junction expects key=value, got {item!r}")
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def replay(run, junction_overrides, out=None, suite=None, console_every=20, until_tick=None):
    run = Path(run)
    driver, report, config = build_driver(run, junction_overrides, suite)
    log_rows = [json.loads(line) for line in (run / "drive_log.jsonl").read_text(encoding="utf-8").splitlines()]
    by_tick = {r["tick"]: r for r in log_rows}
    _, _, rms = odometry_to_core(log_rows)
    truth = json.loads((run / "turn_truth.json").read_text(encoding="utf-8")) if (run / "turn_truth.json").exists() else None
    kerb_line = exit_kerb_truth(truth, truth["command"]) if truth else None
    rows = []
    out_file = open(out, "w", encoding="utf-8") if out else None
    print(f"odometry->core rigid fit over trusted ticks: rms {rms:.2f} m (sanity; errors below use the per-tick pose pair)")
    for tick, frame in iter_frames(run / "frames"):
        if until_tick is not None and tick > until_tick:
            break
        step = driver.step(frame)
        d = step.diagnostics
        live = by_tick.get(tick, {})
        row = dict(tick=tick, time_s=live.get("time_s"), mode=step.mode.value, reason=step.reason, truth=live.get("truth"), driver=d)
        if kerb_line is not None and live.get("driver", {}).get("odometry") and live.get("truth"):
            origin, direction, right = kerb_line
            r_o2c, t_o2c = tick_transform(live)
            if d.get("turn_exit_stations_odo"):
                stations = np.asarray(d["turn_exit_stations_odo"], dtype=float) @ r_o2c.T + t_o2c
                err = signed_offset(stations, origin, direction, right)
                row["exit_kerb_error_m"] = dict(median=round(float(np.median(err)), 2), p10=round(float(np.percentile(err, 10)), 2), p90=round(float(np.percentile(err, 90)), 2), n=int(len(err)))
            if d.get("turn_exit_measured_odo"):
                point = np.asarray(d["turn_exit_measured_odo"], dtype=float) @ r_o2c.T + t_o2c
                row["exit_lane_error_m"] = round(float(signed_offset(point, origin, direction, right)[0] + truth["exit_kerb_distance_m"]), 2)
            if d.get("turn_exit_odo"):
                point = np.asarray(d["turn_exit_odo"], dtype=float) @ r_o2c.T + t_o2c
                row["latched_lane_error_m"] = round(float(signed_offset(point, origin, direction, right)[0] + truth["exit_kerb_distance_m"]), 2)
        rows.append(row)
        if out_file is not None:
            out_file.write(json.dumps(row, default=_json_default) + "\n")
        if console_every and tick % console_every == 0 and (d.get("junction_command") or d.get("turn_radius_m")):
            fit = d.get("turn_exit_fit") or {}
            print(
                f"t={row['time_s'] or 0:5.1f} {step.mode.value:8s} R={d.get('turn_radius_m')} exit={d.get('turn_exit')} kerb@{fit.get('kerb_from_probe_m')} "
                f"kerb_err={row.get('exit_kerb_error_m')} lane_err={row.get('exit_lane_error_m')} latched_err={row.get('latched_lane_error_m')} {d.get('turn') or ''} "
                + (f"[{d['turn_refine']}]" if d.get("turn_refine") else "")
            )
    if out_file is not None:
        out_file.close()
    return rows


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    return str(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="A run folder with frames/, drive_log.jsonl, drive_report.json, route.json")
    parser.add_argument("--junction", action="append", default=[], help="JunctionConfig override key=value (JSON value), repeatable")
    parser.add_argument("--suite", type=Path, default=None, help="Suite file (default: configs/sensor_suites/<run's suite>.json)")
    parser.add_argument("--out", type=Path, default=None, help="Write per-tick replay rows (jsonl)")
    parser.add_argument("--console-every", type=int, default=20)
    parser.add_argument("--until-tick", type=int, default=None)
    args = parser.parse_args(argv)
    replay(args.run, parse_values(args.junction), args.out, args.suite, args.console_every, args.until_tick)


if __name__ == "__main__":
    main()
