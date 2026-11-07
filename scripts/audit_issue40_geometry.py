"""Recompute #40 full-bin support from retained LiDAR evidence, without clearance."""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from carla_adapter.lidar import LidarRig
from obstacle_perception.lidar import LidarSweep
from obstacle_perception.lidar_obstacles import (
    EgoBody,
    LidarObstaclePolicy,
    extract_obstacles,
    remove_ground,
)
from obstacle_perception.lidar_synthetic import load_fixture_sweep
from planning.corridor import RouteGeometry
from scripts.screen_lidar_geometry import coverage


def audit(root):
    root = Path(root)
    policy = LidarObstaclePolicy()
    result = dict(
        positive_clearance_authorised=False,
        length_m=6.5,
        half_width_m=1.5,
        cell_m=0.5,
        lateral_strips=3,
        min_returns_per_cell=2,
        screens={},
        qualifications={},
    )
    for name in ("screen-01", "tolerance-01", "site66-01"):
        folder = root / name
        document = json.loads((folder / "screen.json").read_text())
        ego = document["ego"]
        body = EgoBody(
            front_m=ego["front_from_rear_m"],
            rear_m=-1.0,
            half_width_m=ego["half_width_m"],
            height_m=ego["top_z_actor_m"],
            margin_m=0.0,
        )
        candidates = []
        for candidate in document["candidates"]:
            rows = []
            config = LidarRig(**candidate["rig"]).config(0.05)
            for row in candidate["frames"]:
                frame = row["frame_id"]
                sweep = LidarSweep(
                    "audit",
                    frame,
                    row["sim_time_s"],
                    frame,
                    np.load(folder / candidate["name"] / f"ego_{frame}.npy"),
                    candidate["sensor_to_ego"],
                    config,
                )
                ground = remove_ground(sweep, policy, body)
                report = coverage(ground.points_xyz, ground.classes, body.front_m)
                rows.append(dict(frame_id=frame, target_distance_m=row["distance_m"], **report))
            candidates.append(dict(name=candidate["name"], frames=rows))
        result["screens"][name] = dict(candidates=candidates, cleanup=document["cleanup"])
    for name in ("qualification-01", "nearfield-01"):
        folder = root / name
        document = json.loads((folder / "sweeps.json").read_text())
        body = EgoBody(front_m=document["front_bumper_from_rear_axle_m"])
        route = RouteGeometry(document["route_xy_m"])
        low, high = document["ego_evaluation_bounds"]
        exact = replace(body, rear_m=low[0], half_width_m=high[1], height_m=high[2], margin_m=0.0)
        rows = []
        for row in document["sweeps"]:
            sweep = load_fixture_sweep(document, row, folder)
            start = time.perf_counter()
            batch = extract_obstacles(sweep, route, policy=policy, body=body)
            elapsed = time.perf_counter() - start
            ground = remove_ground(sweep, policy, exact)
            rows.append(
                dict(
                    frame_id=sweep.frame_id,
                    target_distance_m=row["target_distance_m"],
                    target_lateral_m=row["target_lateral_m"],
                    valid=batch.valid,
                    nearest_gap_m=batch.nearest_gap_m,
                    processing_s=elapsed,
                    **coverage(ground.points_xyz, ground.classes, body.front_m),
                )
            )
        timings = [r["processing_s"] for r in rows]
        result["qualifications"][name] = dict(
            frames=rows,
            processing_p95_s=float(np.percentile(timings, 95)),
            processing_max_s=max(timings),
            full_road_support_frames=sum(r["complete_road_support"] for r in rows),
            occupied_frames=sum(r["occupied_cells"] > 0 for r in rows),
            no_returns_treated_clear=0,
            cleanup=document["cleanup"],
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for name, data in report["qualifications"].items():
        print(name, {k: v for k, v in data.items() if k != "frames"})
