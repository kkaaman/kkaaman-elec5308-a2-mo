"""Offline operational-policy audit of fresh BL-C captures; never drives CARLA."""

import argparse
import ast
import hashlib
import json
import math
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from autonomy.interfaces import CameraCalibration, EgoPose, FrameBundle, VehicleState
from autonomy.obstacle_policy import CorridorObservation, TrackedGovernancePolicy
from navigation_control.types import ControlCommand
from obstacle_perception.box_depth import DepthPolicy, estimate_box_depth
from obstacle_perception.tracking import TrackDepthFilter, TrackPolicy
from obstacle_perception.types import (
    CallTiming,
    DetectionBatch,
    DetectionObservation,
    ObservationMetadata,
    PixelBox,
)
from planning.speed_governor import GovernorConfig, SpeedGovernor


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def admission_failures(tracks, observation, *, ready):
    """Expose every conjunct in TrackedGovernancePolicy, without changing it."""
    conditions = {
        "warmup_not_ready": not ready,
        "untracked_missing_id": bool(tracks.untracked),
        "expired_tracks": bool(tracks.expired),
        "noncurrent_tracks": any(not t.current for t in tracks.tracks),
        "invalid_track_depth_or_confirmation": any(not t.depth_valid for t in tracks.tracks),
        "provider_invalid": not observation.valid,
        "missing_visible_distance": observation.visible_distance_m is None,
        "missing_obstacle_and_clearance": observation.obstacle_gap_m is None
        and not observation.clearance_valid,
    }
    return sorted(key for key, failed in conditions.items() if failed)


def roi_diagnostic(depth, box, policy):
    """Describe raw central ROI support; out-of-scope depths are NOT accepted."""
    h, w = depth.shape
    x0, x1 = max(0, box.x_min), min(w, box.x_max)
    y0, y1 = max(0, box.y_min), min(h, box.y_max)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = (x1 - x0) * policy.inner_fraction / 2, (y1 - y0) * policy.inner_fraction / 2
    roi = depth[math.ceil(cy - dy) : math.floor(cy + dy), math.ceil(cx - dx) : math.floor(cx + dx)]
    good = roi[np.isfinite(roi) & (roi > 0)]
    return dict(
        pixels=int(roi.size),
        finite_positive_pixels=len(good),
        below_depth_limit_pixels=int(np.sum(good < policy.max_depth_m)),
        at_or_above_depth_limit_pixels=int(np.sum(good >= policy.max_depth_m)),
        raw_median_m=float(np.median(good)) if len(good) else None,
    )


def summarize_rejections(rows):
    counts, combinations, track_reason_frames, track_reason_instances = (
        Counter(),
        Counter(),
        Counter(),
        Counter(),
    )
    raw_causes = Counter()
    track_combinations = Counter()
    joint_known_combinations = Counter()
    for row in rows:
        counts.update(row["admission_failures"])
        combinations[" & ".join(row["admission_failures"]) or "ADMISSIBLE"] += 1
        reasons = [t["reason"] for t in row["current_invalid_tracks"]]
        track_reason_frames.update(set(reasons))
        track_combinations[" & ".join(sorted(set(reasons))) or "NONE"] += 1
        known = list(row["admission_failures"])
        if row["source_delay_exceeds_ttl"]:
            known.append("source_command_expired")
        if row["processing_lower_bound_over_budget"]:
            known.append("processing_budget_lower_bound_exceeded")
        joint_known_combinations[" & ".join(sorted(known)) or "NONE"] += 1
        track_reason_instances.update(reasons)
        raw_causes.update(set(t["depth_diagnostic_cause"] for t in row["current_invalid_tracks"]))
    for name in (
        "warmup_not_ready",
        "untracked_missing_id",
        "expired_tracks",
        "noncurrent_tracks",
        "invalid_track_depth_or_confirmation",
        "provider_invalid",
        "missing_visible_distance",
        "missing_obstacle_and_clearance",
    ):
        counts.setdefault(name, 0)
    return dict(
        frame_predicate_failures=dict(sorted(counts.items())),
        predicate_combinations=dict(sorted(combinations.items())),
        track_reason_combinations=dict(sorted(track_combinations.items())),
        joint_known_failure_combinations=dict(sorted(joint_known_combinations.items())),
        frames_with_invalid_not_labelled_target=sum(
            any(not t["evaluation_labelled_target"] for t in r["current_invalid_tracks"])
            for r in rows
        ),
        current_invalid_reason_frames=dict(sorted(track_reason_frames.items())),
        current_invalid_reason_instances=dict(sorted(track_reason_instances.items())),
        raw_depth_diagnostic_frames=dict(sorted(raw_causes.items())),
    )


def audit(capture_path, root):
    root, capture_path = Path(root), Path(capture_path)
    doc = json.loads(capture_path.read_text())
    if doc.get("kind") == "operational_same_frame_approach":
        from scripts.review_same_frame_capture import audit_same_frame

        return audit_same_frame(capture_path, root)
    baseline = json.loads((root / "configs/mission_baseline.json").read_text())
    profile = baseline["baseline_threshold_freeze"]
    warm_path = root / "docs/evidence/BL-D-06/warm-operational.json"
    warm = json.loads(warm_path.read_text())
    if (
        doc["capture_status"] != "COMPLETE"
        or not doc["settings_restored"]
        or not doc["no_actor_leaks"]
    ):
        raise ValueError("Complete captured evidence required")
    for key in ("model_sha256", "confidence", "image_size", "tracker", "tracker_settings"):
        if doc["settings"][key] != warm["settings"][key]:
            raise ValueError("Capture/operational setting mismatch")
    if doc["track_policy"] != profile["track_policy"]:
        raise ValueError("Track policy mismatch")
    summaries, records = [], []
    for sequence in doc["sequences"]:
        temporal = TrackDepthFilter(
            source_id=sequence["source_id"],
            model_version=doc["settings"]["model_version"],
            policy=TrackPolicy(**profile["track_policy"]),
            depth_policy=DepthPolicy(**profile["depth_policy"]),
        )

        provider_state = {}

        def provider(frame, batch, tracks):
            gaps = []
            for track in tracks.tracks:
                if track.current and track.depth_valid:
                    estimate = estimate_box_depth(
                        frame.depth_m, track.detection.box, frame.calibration, temporal.depth_policy
                    )
                    if estimate.valid:
                        gaps.append(
                            float(estimate.point_ego_m[0] - warm["ego_front_from_rear_axle_m"])
                        )
            gap = min(gaps) if gaps else None
            observation = CorridorObservation(batch.metadata, bool(gaps), gap, gap, False)
            provider_state.update(tracks=tracks, observation=observation)
            return observation

        cfg = profile["candidate_values"]
        policy = TrackedGovernancePolicy(
            temporal,
            SpeedGovernor(
                GovernorConfig(
                    cfg["cruise_speed_mps"],
                    cfg["deceleration_mps2"],
                    1.5,
                    cfg["response_budget_s"],
                    cfg["stop_margin_m"],
                    1.0,
                    2.0,
                )
            ),
            provider,
            lambda frame, target, dt: ControlCommand(0, float(target > 0), 0),
            dt_s=0.05,
        )
        from autonomy.interfaces import LaneGraph

        policy.reset(episode_id=sequence["episode_id"], graph=LaneGraph((), ()))
        counts, reasons, ready = Counter(), Counter(), 0.0
        sequence_records = []
        for index, row in enumerate(sequence["frames"]):
            depth_path = capture_path.parent / row["depth_npy"]
            if digest(depth_path) != row["depth_sha256"]:
                raise ValueError("Depth hash mismatch")
            depth = np.load(depth_path, allow_pickle=False)
            h, w = depth.shape
            m = ObservationMetadata(
                sequence["episode_id"],
                row["frame_id"],
                row["sim_time_s"],
                sequence["source_id"],
                doc["settings"]["model_version"],
            )
            frame = FrameBundle(
                m.episode_id,
                m.frame_id,
                m.sim_time_s,
                0,
                EgoPose(VehicleState(0, 0, 0, 0)),
                np.zeros((h, w, 3), np.uint8),
                CameraCalibration(**sequence["calibration"]),
                depth,
                m.frame_id,
            )
            detections = tuple(
                DetectionObservation(
                    d["class_name"], d["confidence"], PixelBox(*d["xyxy"]), d["track_id"]
                )
                for d in row["detections"]
            )
            batch = DetectionBatch(
                m,
                detections,
                w,
                h,
                CallTiming(row["total_wall_s"] or 0),
                row["valid"],
                row["invalid_reason"],
            )
            # Deliberately bypass deadlines/warm-up: an optimistic upper-bound
            # diagnostic, not a claim that the captured schedule is operational.
            command = policy.evaluate(frame, batch, fresh=True, ready=True)
            elapsed = m.sim_time_s - sequence["start_sim_time_s"]
            ready = (
                max(elapsed + row["sensor_delay_s"], ready)
                + row["sensor_read_wall_s"]
                + (row["total_wall_s"] or 0)
                + row["filter_wall_s"]
            )
            diagnostics = policy.track_diagnostics
            current_invalid = [t for t in diagnostics if t["current"] and not t["depth_valid"]]
            for t in current_invalid:
                reasons[t["reason"]] += 1
            target_ids = {detections[i].track_id for i in row["target_detection_indices"]}
            target_valid = any(
                t["track_id"] in target_ids and t["current"] and t["depth_valid"]
                for t in diagnostics
            )
            failures = admission_failures(
                provider_state["tracks"], provider_state["observation"], ready=True
            )
            # Enrich diagnostics only AFTER the operational policy has run.
            for diagnostic in current_invalid:
                match = next((d for d in detections if d.track_id == diagnostic["track_id"]), None)
                diagnostic["evaluation_labelled_target"] = diagnostic["track_id"] in target_ids
                if match is not None:
                    estimate = estimate_box_depth(
                        depth, match.box, frame.calibration, temporal.depth_policy
                    )
                    raw = roi_diagnostic(depth, match.box, temporal.depth_policy)
                    diagnostic.update(
                        box=list(match.box.__dict__.values()),
                        confidence=match.confidence,
                        measured_depth_reason=estimate.reason,
                        raw_roi=raw,
                    )
                    if raw["pixels"] < temporal.depth_policy.min_pixels:
                        cause = "roi_too_small"
                    elif (
                        estimate.reason == "insufficient_valid_depth"
                        and raw["at_or_above_depth_limit_pixels"]
                        >= raw["pixels"] * temporal.depth_policy.min_valid_fraction
                    ):
                        cause = "depth_support_outside_25m_scope"
                    else:
                        cause = estimate.reason or "confirmation_only"
                    diagnostic["depth_diagnostic_cause"] = cause
                else:
                    diagnostic["depth_diagnostic_cause"] = "missing_current_detection"
            counts["frames"] += 1
            counts["target_depth_valid_frames"] += target_valid
            counts["current_invalid_track_frames"] += bool(current_invalid)
            counts["observation_valid_frames"] += policy.observation_valid
            counts["positive_commands"] += command.throttle > 0
            counts["age_over_budget_frames"] += (
                ready - elapsed > profile["operational_gate"]["max_observation_age_s"]
            )
            counts["source_delay_exceeds_ttl_frames"] += (
                row["sensor_delay_s"] >= profile["operational_gate"]["command_ttl_s"]
            )
            record = dict(
                sequence=sequence["name"],
                index=index,
                frame_id=m.frame_id,
                target_depth_valid=target_valid,
                observation_valid=policy.observation_valid,
                throttle=command.throttle,
                brake=command.brake,
                current_invalid_tracks=current_invalid,
                admission_failures=failures,
                source_delay_exceeds_ttl=row["sensor_delay_s"]
                >= profile["operational_gate"]["command_ttl_s"],
                reconstructed_age_over_budget=ready - elapsed
                > profile["operational_gate"]["max_observation_age_s"],
                processing_lower_bound_s=(row["total_wall_s"] or 0) + row["filter_wall_s"],
                processing_lower_bound_over_budget=(row["total_wall_s"] or 0) + row["filter_wall_s"]
                > profile["operational_gate"]["max_step_wall_s"],
                source_batch_invalid=not row["valid"],
                evidence_missing_capture_origin="capture_started_monotonic_s" not in row,
                evidence_missing_full_processing="controller_wall_s" not in row
                or "governor_wall_s" not in row,
            )
            records.append(record)
            sequence_records.append(record)
        summaries.append(
            dict(
                sequence=sequence["name"],
                counts=dict(counts),
                current_invalid_reasons=dict(reasons),
                rejection_breakdown=summarize_rejections(sequence_records),
            )
        )
    with zipfile.ZipFile(root / "docs/evidence/BL-D-06/warm-source.zip") as archive:
        comparison = {}
        for name in warm["source_hashes"]:
            old, new = archive.read(name), (root / name).read_bytes()
            comparison[name] = dict(
                identical=old == new,
                measured_sha256=hashlib.sha256(old).hexdigest(),
                current_sha256=hashlib.sha256(new).hexdigest(),
            )
            if old != new:
                comparison[name]["difference"] = (
                    "Vehicle adapter apply docstring and acknowledgement error text/detail only; expiry checks and decision path unchanged."
                )
                if name != "src/carla_adapter/vehicle.py":
                    raise ValueError("Operational runtime differs from measured snapshot")

                # The known adapter difference is after the measured decision.
                def method(payload, name):
                    module = ast.parse(payload)
                    cls = next(n for n in module.body if isinstance(n, ast.ClassDef))
                    return ast.dump(
                        next(
                            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
                        ),
                        include_attributes=False,
                    )

                if method(old, "apply_packet") != method(new, "apply_packet"):
                    raise ValueError("Packet expiry semantics changed")
    return dict(
        status="BLOCKED_OPERATIONAL_RANGE_ADMISSION",
        capture_sha256=digest(capture_path),
        capture_schema=doc["schema_version"],
        operational_profile={
            key: profile[key]
            for key in (
                "operational_gate",
                "track_policy",
                "depth_policy",
                "camera_rig",
                "detector",
                "fixed_delta_seconds",
                "candidate_values",
                "implementation_policy",
            )
        },
        runtime_source_comparison=comparison,
        rejection_breakdown=summarize_rejections(records),
        timing_evidence_counts={
            key: sum(bool(r[key]) for r in records)
            for key in (
                "source_delay_exceeds_ttl",
                "reconstructed_age_over_budget",
                "processing_lower_bound_over_budget",
                "source_batch_invalid",
                "evidence_missing_capture_origin",
                "evidence_missing_full_processing",
            )
        },
        timing_evidence_note="0.2 s source delay is recorded and exceeds TTL. Host age is a serial replay reconstruction, not a measured capture-age gate trace. Full controller/governor duration and owner capture clock were not recorded; full runtime admission/warm-up cannot be reconstructed. Forced fresh/ready policy predicates are counted separately.",
        method="Replay all source rows through current TrackedGovernancePolicy and configured 25 m depth policy, optimistic fresh=True/ready=True, same perceived-point provider as historical warm collector. Evaluation labels identify target diagnostics only after policy evaluation. No CARLA command applied.",
        sequences=summaries,
        rows=records,
        blocker="No full operational obstacle/visibility observation is valid in the approach dataset, even with deadline/warm-up checks optimistically bypassed. Target-only confirmed depth range cannot establish the usable range of the all-track governor. Historical warm positives are at another site and do not measure approach first-admission range.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input, Path(__file__).resolve().parents[1])
    result["audit_script_sha256"] = digest(__file__)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
