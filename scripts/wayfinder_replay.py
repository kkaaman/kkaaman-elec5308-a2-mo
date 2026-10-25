"""Replay a Wayfinder fixture offline: guidance log plus mini-map PNG frames (E3-03).

No simulator is used. Poses are generated along the planned route at the policy
cruise speed, with one scripted stop so the paused ETA is exercised. The output
folder receives ``route.json`` (route-manager document), ``guidance_log.json``
and ``frame_NN.png`` mini-map frames showing route, current pose and next turn.

Example::

    python scripts/wayfinder_replay.py --origin baseline_start --destination baseline_goal \\
        --output runs/wayfinder/baseline --frames 4
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from planning.lane_graph import LaneTopology
from planning.wayfinder import PoiRegistry, RouteTracker, SpeedPolicy, plan_named_route
from planning.wayfinder.minimap import MiniMapStyle, render_minimap, save_png

REPO_ROOT = Path(__file__).resolve().parents[1]


def _pose_at(points, s_m):
    """Position and heading at arc length ``s_m`` along an XY polyline."""
    arc = 0.0
    for a, b in zip(points, points[1:]):
        step = math.dist(a, b)
        if arc + step >= s_m or (a, b) == (points[-2], points[-1]):
            t = min(1.0, max(0.0, (s_m - arc) / step))
            return a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), math.atan2(b[1] - a[1], b[0] - a[0])
        arc += step
    raise AssertionError("unreachable")


def replay(route, topology, policy, dt_s, pause_fraction, pause_s, lateral_offset_m, frames, output, map_context=True):
    points = [p[:2] for p in route.points_xyz_m]
    tracker = RouteTracker(route, policy)
    length = tracker.length_m
    pause_at_m = pause_fraction * length
    log, poses = [], []
    time_s, s_m, paused_for = 0.0, 0.0, 0.0
    while True:
        x, y, yaw = _pose_at(points, s_m)
        x += -lateral_offset_m * math.sin(yaw)
        y += lateral_offset_m * math.cos(yaw)
        pausing = s_m >= pause_at_m and paused_for < pause_s
        speed = 0.0 if pausing else policy.cruise_speed_mps
        state = tracker.update(time_s, x, y, speed)
        log.append({**state.to_dict(), "yaw_rad": yaw, "scripted_pause": pausing})
        poses.append((state, yaw))
        if state.eta_state == "arrived" and not pausing:
            break
        if pausing:
            paused_for += dt_s
        else:
            s_m = min(length, s_m + speed * dt_s)
        time_s += dt_s
        if time_s > 10 * length / policy.paused_threshold_mps:
            raise RuntimeError("replay did not reach the goal")
    paused_index = next((i for i, row in enumerate(log) if row["eta_state"] == "paused"), None)
    chosen = sorted({round(i * (len(log) - 1) / max(1, frames - 1)) for i in range(frames)} | ({paused_index} if paused_index is not None else set()))
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for number, index in enumerate(chosen):
        state, yaw = poses[index]
        image = render_minimap(route, state, yaw, topology if map_context else None, MiniMapStyle())
        path = output / f"frame_{number:02d}.png"
        save_png(image, path)
        written.append({"file": path.name, "log_index": index, "time_s": state.time_s, "eta_state": state.eta_state,
                        "next_maneuver": state.next_maneuver, "next_maneuver_m": round(state.next_maneuver_m, 1)})
    (output / "route.json").write_text(json.dumps(route.to_document(), indent=2) + "\n", encoding="utf-8")
    summary = {
        "origin": route.origin_name, "destination": route.destination_name, "length_m": round(length, 3),
        "lanes": len(route.segment_ids), "maneuvers": [m.kind for m in route.maneuvers],
        "policy": {"cruise_speed_mps": policy.cruise_speed_mps, "paused_threshold_mps": policy.paused_threshold_mps,
                   "arrival_radius_m": policy.arrival_radius_m, "reference": policy.reference},
        "dt_s": dt_s, "steps": len(log), "paused_steps": sum(row["eta_state"] == "paused" for row in log),
        "max_off_route_m": round(max(row["off_route_m"] for row in log), 3),
        "progress_monotonic": all(a["progress_m"] <= b["progress_m"] + 1e-9 for a, b in zip(log, log[1:])),
        "frames": written, "units": "m, m/s, s",
    }
    (output / "guidance_log.json").write_text(json.dumps({"summary": summary, "steps": log}, indent=1) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topology", type=Path, default=REPO_ROOT / "tests/fixtures/Town10HD.lane_graph.json")
    parser.add_argument("--pois", type=Path, default=REPO_ROOT / "configs/wayfinder/pois_town10hd.json")
    parser.add_argument("--origin", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speed", type=float, default=4.0, help="cruise speed for poses and the ETA policy, m/s")
    parser.add_argument("--dt", type=float, default=0.5, help="replay step, s")
    parser.add_argument("--pause-fraction", type=float, default=0.45, help="route fraction at which the scripted stop occurs")
    parser.add_argument("--pause-s", type=float, default=4.0, help="scripted stop duration, s")
    parser.add_argument("--lateral-offset", type=float, default=0.3, help="constant lateral pose offset, m")
    parser.add_argument("--frames", type=int, default=4, help="number of evenly spaced PNG frames (plus the paused frame)")
    parser.add_argument("--no-map-context", action="store_true", help="draw only the route, not the whole lane map")
    args = parser.parse_args(argv)
    if args.dt <= 0 or args.frames < 1 or not 0 <= args.pause_fraction < 1 or args.pause_s < 0:
        parser.error("dt and frames must be positive, pause-fraction in [0,1), pause-s nonnegative")
    topology = LaneTopology.load(args.topology)
    registry = PoiRegistry.load(args.pois)
    policy = SpeedPolicy(cruise_speed_mps=args.speed)
    route = plan_named_route(topology, registry, args.origin, args.destination)
    summary = replay(route, topology, policy, args.dt, args.pause_fraction, args.pause_s, args.lateral_offset,
                     args.frames, args.output, not args.no_map_context)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
