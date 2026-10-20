"""Replay captured RGB-D/model outputs; evaluation labels stay in reporting."""

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from autonomy.interfaces import CameraCalibration, EgoPose, FrameBundle, VehicleState
from obstacle_perception.tracking import TrackDepthFilter, TrackPolicy
from obstacle_perception.types import (
    CallTiming,
    DetectionBatch,
    DetectionObservation,
    ObservationMetadata,
    PixelBox,
)


def benchmark(document, root):
    """Schema documented in docs/track_stabilisation.md; one target per sequence."""
    if document.get("capture_status", "COMPLETE") != "COMPLETE":
        raise ValueError("Capture incomplete; retained frames are diagnostic only")
    if document.get("schema_version", 0) >= 2 and (
        not document.get("settings_restored") or not document.get("no_actor_leaks")
    ):
        raise ValueError("Capture cleanup/provenance incomplete")
    settings = document["settings"]
    for key in (
        "confidence",
        "image_size",
        "tracker",
        "tracker_settings",
        "model_version",
        "model_sha256",
        "device",
        "versions",
    ):
        if key not in settings:
            raise ValueError(f"Missing benchmark setting: {key}")
    if document["origin"] not in ("recorded", "synthetic"):
        raise ValueError("Explicit recorded/synthetic origin required")
    names = [s["name"] for s in document["sequences"]]
    if len(names) != len(set(names)):
        raise ValueError("Sequence names must be unique")
    policy = TrackPolicy(**document["track_policy"])
    rows = []
    missing_read_timing = 0
    for sequence in document["sequences"]:
        t = TrackDepthFilter(
            source_id=sequence["source_id"], model_version=settings["model_version"], policy=policy
        )
        t.reset(episode_id=sequence["episode_id"])
        first_raw = first_confirmed = None
        resolutions = set()
        ready_s = 0.0
        confirmed_response_s = None
        for row in sequence["frames"]:
            metadata = ObservationMetadata(
                sequence["episode_id"],
                row["frame_id"],
                row["sim_time_s"],
                sequence["source_id"],
                settings["model_version"],
            )
            # Labels select the evaluation target only after the operational filter.
            depth_path = root / row["depth_npy"]
            if (
                "depth_sha256" in row
                and hashlib.sha256(depth_path.read_bytes()).hexdigest() != row["depth_sha256"]
            ):
                raise ValueError("Depth artifact hash mismatch")
            depth = np.load(depth_path, allow_pickle=False)
            h, w = depth.shape
            resolutions.add((w, h))
            frame = FrameBundle(
                metadata.episode_id,
                metadata.frame_id,
                metadata.sim_time_s,
                0,
                EgoPose(VehicleState(0, 0, 0, 0)),
                np.zeros((h, w, 3), np.uint8),
                CameraCalibration(**sequence["calibration"]),
                depth,
                metadata.frame_id,
            )
            detections = tuple(
                DetectionObservation(
                    d["class_name"], d["confidence"], PixelBox(*d["xyxy"]), d.get("track_id")
                )
                for d in row["detections"]
            )
            wall = row["total_wall_s"]
            if wall is None and row.get("valid", True):
                raise ValueError("Valid observation missing actual detector timing")
            sensor_delay = row.get("sensor_delay_s", 0.0)
            filter_wall = row.get("filter_wall_s") or 0.0
            read_wall = row.get("sensor_read_wall_s")
            if read_wall is None:
                if document.get("schema_version", 0) >= 3:
                    raise ValueError("Missing measured sensor read timing")
                missing_read_timing += 1
                read_wall = 0.0  # Legacy diagnostic lower bound, never a measured cap.
            if isinstance(read_wall, bool) or not math.isfinite(read_wall) or read_wall < 0:
                raise ValueError("Invalid sensor read timing")
            if any(not math.isfinite(x) or x < 0 for x in (sensor_delay, filter_wall)):
                raise ValueError("Invalid sensor/filter delay")
            batch = DetectionBatch(
                metadata,
                detections,
                w,
                h,
                CallTiming(0 if wall is None else wall),
                row.get("valid", True),
                row.get("invalid_reason"),
            )
            result = t.update(frame, batch)
            labels = row["target_detection_indices"]
            if any(type(i) is not int or i < 0 or i >= len(detections) for i in labels):
                raise ValueError("Invalid target label")
            target = [detections[i] for i in labels]
            capture_elapsed = metadata.sim_time_s - sequence["start_sim_time_s"]
            if not math.isfinite(capture_elapsed) or capture_elapsed < 0:
                raise ValueError("Invalid sequence start time")
            # Serial inference must account for backlog: slow earlier calls cannot
            # disappear when a later frame confirms the target.
            ready_s = (
                max(capture_elapsed + sensor_delay, ready_s)
                + read_wall
                + (wall or 0.0)
                + filter_wall
            )
            response_s = ready_s - capture_elapsed
            if not batch.valid:
                if labels:
                    raise ValueError("Invalid observations cannot have target labels")
                continue
            event = dict(
                frame_id=metadata.frame_id,
                range_m=row["target_gap_m"],
                latency_s=ready_s,
            )
            if (
                not math.isfinite(sequence["start_sim_time_s"])
                or sequence["start_sim_time_s"] < 0
                or metadata.sim_time_s < sequence["start_sim_time_s"]
            ):
                raise ValueError("Target event precedes sequence start")
            if first_raw is None and target:
                first_raw = event
            ids = {d.track_id for d in target if d.track_id is not None}
            if first_confirmed is None and any(
                x.current and x.confirmed and x.depth_valid and x.identity.track_id in ids
                for x in result.tracks
            ):
                first_confirmed = event
                confirmed_response_s = response_s
        rows.append(
            dict(
                sequence=sequence["name"],
                first_raw=first_raw,
                first_confirmed=first_confirmed,
                source_resolutions_wh=sorted(resolutions),
                first_confirmed_response_s=confirmed_response_s,
            )
        )
    # Worst sequence, including misses. Axial depth never masquerades as bumper gap.
    speed = document["proposed_cruise_mps"]
    decel = document["braking_deceleration_mps2"]
    delay = document["additional_response_s"]
    margin = document["stopping_margin_m"]
    if (
        not rows
        or any(not math.isfinite(x) or x < 0 for x in (speed, delay, margin))
        or not math.isfinite(decel)
        or decel <= 0
    ):
        raise ValueError("Finite nonnegative speed/delay/margin and positive deceleration required")
    caps = []
    for row in rows:
        event = row["first_confirmed"]
        if event is None:
            caps.append(0.0)
            continue
        gap = event["range_m"]
        if not math.isfinite(gap) or gap < 0:
            raise ValueError("Evaluation gap must be finite and nonnegative")
        response = delay + row["first_confirmed_response_s"]
        caps.append(
            max(
                0.0,
                math.sqrt((decel * response) ** 2 + 2 * decel * max(0.0, gap - margin))
                - decel * response,
            )
        )
    cap = math.floor(min(caps) * 10) / 10
    return dict(
        origin=document["origin"],
        settings=settings,
        track_policy=asdict(policy),
        sequences=rows,
        proposed_cruise_mps=speed,
        measured_speed_cap_mps=None if missing_read_timing else cap,
        recommended_speed_mps=0.0 if missing_read_timing else min(speed, cap),
        status=(
            "BLOCKED_INCOMPLETE_TIMING"
            if missing_read_timing
            else "SYNTHETIC_ONLY"
            if document["origin"] == "synthetic"
            else "MEASURED"
        ),
        screening_decision=(
            "BLOCKED" if missing_read_timing else "SUPPORTED" if speed <= cap else "FAILED"
        ),
        timing=dict(
            missing_sensor_read_timing_frames=missing_read_timing,
            latency_basis="legacy_lower_bound"
            if missing_read_timing
            else "serial_sim_schedule_plus_measured_read_detector_filter",
            legacy_optimistic_speed_cap_mps=cap if missing_read_timing else None,
            proposed_cruise_supported=(None if missing_read_timing else speed <= cap),
        ),
        assumptions={
            k: document[k]
            for k in ("braking_deceleration_mps2", "additional_response_s", "stopping_margin_m")
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = args.input.read_bytes()
    report = benchmark(json.loads(payload), args.input.parent)
    report["input_sha256"] = hashlib.sha256(payload).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
