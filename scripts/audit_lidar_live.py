"""Read-only recorded LiDAR geometry/coverage audit; never authorizes clearance."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from autonomy.lidar_fusion import LidarCorridorSource
from carla_adapter.lidar import lidar_points_to_ego
from obstacle_perception.lidar_obstacles import (
    CLASS_OBSTACLE,
    CLASS_RAISED,
    CLASS_ROAD,
    EgoBody,
    LidarObstaclePolicy,
    corridor_mask,
    remove_ground,
)
from obstacle_perception.lidar_synthetic import load_fixture_sweep
from planning.corridor import RouteGeometry


def audit(manifest_path, output):
    manifest_path = Path(manifest_path)
    base = manifest_path.parent
    doc = json.loads(manifest_path.read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    route = RouteGeometry(doc["route_xy_m"])
    body = EgoBody(front_m=doc["front_bumper_from_rear_axle_m"])
    policy = LidarObstaclePolicy()
    source = LidarCorridorSource(route, body=body, policy=policy)
    source.reset(episode_id=doc["episode_id"])
    records = {r["frame_id"]: r for r in doc["capture_records"]}
    rows = []
    for row in doc["sweeps"]:
        sweep = load_fixture_sweep(doc, row, base)
        record = records[sweep.frame_id]
        raw_path = base / "raw-sensor" / record["raw_sensor_file"]
        payload = raw_path.read_bytes()
        assert hashlib.sha256(payload).hexdigest() == record["raw_sensor_sha256"]
        transformed, dropped = lidar_points_to_ego(
            payload, sweep.sensor_to_ego, range_m=sweep.config.range_m
        )
        assert np.array_equal(transformed, sweep.points_xyzi)
        ground = remove_ground(sweep, policy, body)
        inside = corridor_mask(
            route,
            ground.points_xyz[:, :2],
            half_width_m=policy.corridor_half_width_m,
            ahead_m=policy.corridor_ahead_m,
        )
        # Straight fixture: route and ego X align. Bins begin at the measured bumper.
        gap = ground.points_xyz[:, 0] - body.front_m
        corridor = inside & (gap >= 0) & (gap < policy.corridor_ahead_m)
        bins = np.floor(gap[corridor] / policy.clear_bin_m).astype(int)
        lateral = ground.points_xyz[corridor, 1]
        strips = np.clip(
            np.floor(
                (lateral + policy.corridor_half_width_m)
                / (2 * policy.corridor_half_width_m)
                * policy.clear_strips
            ).astype(int),
            0,
            policy.clear_strips - 1,
        )
        shape = (int(np.ceil(policy.corridor_ahead_m / policy.clear_bin_m)), policy.clear_strips)
        road = np.zeros(shape, int)
        occupied = np.zeros(shape, int)
        classes = ground.classes[corridor]
        np.add.at(road, (bins[classes == CLASS_ROAD], strips[classes == CLASS_ROAD]), 1)
        obstacle = np.isin(classes, [CLASS_OBSTACLE, CLASS_RAISED])
        np.add.at(occupied, (bins[obstacle], strips[obstacle]), 1)
        covered = road >= policy.clear_min_points
        broad = covered[:, policy.clear_strips // 2] & (covered.sum(1) >= policy.clear_min_strips)
        # 0 UNKNOWN,1 OCCUPIED; no positive clear inference from ground absence.
        state = np.where(occupied > 0, 1, 0).astype(np.uint8)
        np.savez_compressed(
            output / f"processed_{sweep.frame_id:06d}.npz",
            transformed_xyzi=sweep.points_xyzi,
            ground_points_xyz=ground.points_xyz,
            height_above_road_m=ground.height_m,
            ground_class=ground.classes,
            corridor_mask=inside,
            road_bin_counts=road,
            occupied_bin_counts=occupied,
            bin_state=state,
            road_supported_bins=broad,
        )
        source.observe(sweep)
        exact = source.report(
            SimpleNamespace(
                episode_id=sweep.episode_id, frame_id=sweep.frame_id, sim_time_s=sweep.sim_time_s
            )
        )
        stale = source.report(
            SimpleNamespace(
                episode_id=sweep.episode_id,
                frame_id=sweep.frame_id + 1,
                sim_time_s=sweep.sim_time_s + 0.05,
            )
        )
        future = source.report(
            SimpleNamespace(
                episode_id=sweep.episode_id,
                frame_id=sweep.frame_id - 1,
                sim_time_s=sweep.sim_time_s - 0.05,
            )
        )
        road_gaps = gap[corridor & (ground.classes == CLASS_ROAD)]
        rows.append(
            dict(
                frame_id=sweep.frame_id,
                target_distance_m=row["target_distance_m"],
                target_lateral_m=row["target_lateral_m"],
                raw_transform_max_error_m=0.0,
                first_road_return_gap_m=float(road_gaps.min()) if len(road_gaps) else None,
                first_broad_road_bin_gap_m=float(np.flatnonzero(broad)[0] * policy.clear_bin_m)
                if broad.any()
                else None,
                occupied_bins=int((state == 1).sum()),
                unknown_bins=int((state == 0).sum()),
                clear_bins=0,
                near_0_1m_road_returns=int(((road_gaps >= 0) & (road_gaps < 1)).sum()),
                near_0_2m_road_returns=int(((road_gaps >= 0) & (road_gaps < 2)).sum()),
                clearance_valid=exact.clearance_valid,
                stale_rejected=not stale.valid,
                future_rejected=not future.valid,
                dropped_raw_rows=dropped,
                received_monotonic_s=record["received_monotonic_s"],
                read_completed_monotonic_s=record["read_completed_monotonic_s"],
                receipt_to_read_s=record["read_completed_monotonic_s"]
                - record["received_monotonic_s"],
            )
        )
    first = [r["first_road_return_gap_m"] for r in rows if r["first_road_return_gap_m"] is not None]
    broad = [
        r["first_broad_road_bin_gap_m"] for r in rows if r["first_broad_road_bin_gap_m"] is not None
    ]
    summary = dict(
        frames=len(rows),
        raw_transform_checks_passed=len(rows),
        first_road_return_gap_m_min=min(first) if first else None,
        first_road_return_gap_m_max=max(first) if first else None,
        first_broad_road_bin_gap_m_min=min(broad) if broad else None,
        first_broad_road_bin_gap_m_max=max(broad) if broad else None,
        any_positive_clearance=any(r["clearance_valid"] for r in rows),
        near_0_1m_road_return_total=sum(r["near_0_1m_road_returns"] for r in rows),
        stale_rejections=sum(r["stale_rejected"] for r in rows),
        future_rejections=sum(r["future_rejected"] for r in rows),
        max_receipt_to_read_s=max(r["receipt_to_read_s"] for r in rows),
        bin_state_meaning={
            "0": "UNKNOWN including observed road without a traversability certificate",
            "1": "OCCUPIED return evidence",
            "clear": "none asserted",
        },
        clear_start_m=policy.clear_start_m,
        clear_start_is_coverage_certificate=False,
        conclusion="No positive from-bumper clearance certificate for issue40; no full-height rig requirement inferred",
    )
    (output / "coverage.json").write_text(
        json.dumps(dict(summary=summary, frames=rows), indent=2) + "\n"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.input, args.output), indent=2))
