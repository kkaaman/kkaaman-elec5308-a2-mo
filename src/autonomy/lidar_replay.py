"""Replay a recorded or synthetic LiDAR sweep set through the obstacle path (E2 evidence).

Each manifest row is rebuilt into a :class:`LidarSweep`, run through ground
removal, corridor crop, clustering and tracking via :class:`LidarCorridorSource`,
then fused with the row's RGB-D channel (``depth_gap_m``/``depth_valid``; absent or
``null`` means the depth observation was dropped). The summary records per-frame
validity, gaps, nearest source, disagreement and the HOLD reasons, plus aggregate
error against ``expected_gap_m`` where the fixture supplies it. Nothing here drives
a vehicle; it measures the pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from autonomy.lidar_fusion import LidarCorridorSource
from obstacle_perception.fusion import SourceCorridorReport, fuse_corridor_reports
from obstacle_perception.lidar_obstacles import EgoBody, LidarObstaclePolicy
from obstacle_perception.lidar_synthetic import load_fixture_sweep
from planning.corridor import RouteGeometry

DEPTH_SOURCE_ID = "front-rgbd"


@dataclass(frozen=True)
class _Frame:
    episode_id: str
    frame_id: int
    sim_time_s: float


def depth_report(row, source_id=DEPTH_SOURCE_ID):
    """Source report for the manifest's simulated/recorded RGB-D channel."""
    gap = row.get("depth_gap_m")
    if not row.get("depth_valid", False) or gap is None:
        return SourceCorridorReport(source_id, False, invalid_reason="depth_unavailable")
    return SourceCorridorReport(source_id, True, float(gap), float(gap), False)


def replay_manifest(
    manifest, directory, *, policy=None, body=None, half_width_m=None, ahead_m=None
):
    """Run every sweep in ``manifest`` (dict) found under ``directory`` and summarise."""
    directory = Path(directory)
    route = RouteGeometry(manifest["route_xy_m"])
    front = float(manifest.get("front_bumper_from_rear_axle_m", EgoBody().front_m))
    body = body or EgoBody(front_m=front)
    policy = policy or LidarObstaclePolicy()
    if half_width_m is not None or ahead_m is not None:
        policy = replace(
            policy,
            **{
                k: v
                for k, v in dict(
                    corridor_half_width_m=half_width_m, corridor_ahead_m=ahead_m
                ).items()
                if v is not None
            },
        )
    source = LidarCorridorSource(route, policy=policy, body=body)
    source.reset(episode_id=manifest["episode_id"])
    frames = []
    errors = []
    identities = set()
    reasons = {}
    nearest_counts = {}
    for row in manifest["sweeps"]:
        sweep = load_fixture_sweep(manifest, row, directory)
        tracks = source.observe(sweep)
        frame = _Frame(sweep.episode_id, sweep.frame_id, sweep.sim_time_s)
        lidar = source.report(frame)
        depth = depth_report(row)
        fused = fuse_corridor_reports(
            (depth, lidar), required_sources=(depth.source_id, lidar.source_id)
        )
        identities.update(lidar.track_identities)
        for reason in fused.invalid_reasons:
            reasons[reason] = reasons.get(reason, 0) + 1
        if fused.nearest_source_id is not None:
            nearest_counts[fused.nearest_source_id] = (
                nearest_counts.get(fused.nearest_source_id, 0) + 1
            )
        expected = row.get("expected_gap_m")
        batch = source.last_batch
        if lidar.valid and lidar.nearest_gap_m is not None and expected is not None:
            errors.append(abs(lidar.nearest_gap_m - float(expected)))
        frames.append(
            dict(
                frame_id=sweep.frame_id,
                sim_time_s=sweep.sim_time_s,
                note=row.get("note"),
                point_count=sweep.point_count,
                density_fraction=sweep.density_fraction,
                sweep_valid=batch.valid,
                sweep_invalid_reason=batch.invalid_reason,
                clusters=len(batch.clusters),
                in_corridor_clusters=sum(c.in_corridor for c in batch.clusters),
                road_points=batch.road_points,
                raised_points=batch.raised_points,
                obstacle_points=batch.obstacle_points,
                seen_clear_m=batch.seen_clear_m,
                lidar_report=dict(
                    valid=lidar.valid,
                    nearest_gap_m=lidar.nearest_gap_m,
                    visible_distance_m=lidar.visible_distance_m,
                    clearance_valid=lidar.clearance_valid,
                    invalid_reason=lidar.invalid_reason,
                    track_ids=[t.track_id for t in lidar.track_identities],
                ),
                depth_report=dict(
                    valid=depth.valid,
                    nearest_gap_m=depth.nearest_gap_m,
                    invalid_reason=depth.invalid_reason,
                ),
                expected_gap_m=expected,
                lidar_gap_error_m=(
                    None
                    if not (
                        lidar.valid and lidar.nearest_gap_m is not None and expected is not None
                    )
                    else lidar.nearest_gap_m - float(expected)
                ),
                fused=fused.to_dict(),
                governor_input=dict(
                    valid=fused.valid,
                    obstacle_gap_m=fused.obstacle_gap_m,
                    visible_distance_m=fused.visible_distance_m,
                    clearance_valid=fused.clearance_valid,
                    hold=not fused.valid,
                ),
                tracks=tracks.to_dict()["tracks"],
            )
        )
    summary = dict(
        input_origin=manifest.get("origin"),
        episode_id=manifest["episode_id"],
        frames=len(frames),
        sweep_valid_frames=sum(f["sweep_valid"] for f in frames),
        fused_valid_frames=sum(f["fused"]["valid"] for f in frames),
        hold_frames=sum(not f["fused"]["valid"] for f in frames),
        hold_reasons=reasons,
        nearest_source_counts=nearest_counts,
        max_disagreement_m=max(
            (
                f["fused"]["disagreement_m"]
                for f in frames
                if f["fused"]["disagreement_m"] is not None
            ),
            default=None,
        ),
        depth_loss_frames=sum(not f["depth_report"]["valid"] for f in frames),
        depth_loss_frames_treated_clear=sum(
            (not f["depth_report"]["valid"]) and f["fused"]["valid"] for f in frames
        ),
        lidar_loss_frames=sum(not f["lidar_report"]["valid"] for f in frames),
        lidar_loss_frames_treated_clear=sum(
            (not f["lidar_report"]["valid"]) and f["fused"]["valid"] for f in frames
        ),
        max_abs_lidar_gap_error_m=max(errors, default=None),
        lidar_track_ids=sorted(i.track_id for i in identities),
        lidar_track_sources=sorted({i.source_id for i in identities}),
        policy=policy.__dict__,
        body=body.__dict__,
    )
    return dict(summary=summary, frames=frames)


def replay_path(manifest_path, **kwargs):
    """Load ``manifest_path`` (JSON) and replay the sweeps stored beside it."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    return replay_manifest(manifest, manifest_path.parent, **kwargs)


__all__ = ["DEPTH_SOURCE_ID", "depth_report", "replay_manifest", "replay_path"]
