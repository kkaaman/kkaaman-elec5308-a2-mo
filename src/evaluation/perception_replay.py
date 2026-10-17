"""Recorded RGB-D replay; instance labels are scored after production inference."""

import math
import time
from collections import defaultdict
from dataclasses import asdict

import numpy as np
from PIL import Image

from autonomy.interfaces import EgoPose, FrameBundle
from evaluation.dataset import local_file
from geometry.camera import CameraIntrinsics
from navigation_control.types import VehicleState
from obstacle_perception.association import associate
from obstacle_perception.box_depth import estimate_batch_depth
from obstacle_perception.track_filter import TrackFilter


def axial_image(raw, calibration, convention):
    if convention not in {"axial", "radial"}:
        raise ValueError("Measured depth convention required")
    if convention == "axial":
        return raw.copy()
    vv, uu = np.indices(raw.shape)
    scale = np.sqrt(
        1
        + ((uu - calibration.cx_px) / calibration.fx_px) ** 2
        + ((vv - calibration.cy_px) / calibration.fy_px) ** 2
    )
    return (raw / scale).astype(np.float32)


def overlap(a, b):
    area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - area
    return area / union if union > 0 else 0


def replay(folder, manifest, rows, detector, convention, *, fault=None):
    width, height = manifest["image_size_wh"]
    rig = manifest["camera"]
    calibration = CameraIntrinsics.from_horizontal_fov(
        width, height, math.radians(rig["fov_deg"])
    ).calibration(
        (
            (0, 0, 1, rig["x_actor_m"] + 1.38641),
            (-1, 0, 0, 0),
            (0, -1, 0, rig["z_actor_m"]),
            (0, 0, 0, 1),
        )
    )
    current = None
    tracker = TrackFilter()
    records = []
    for row in sorted(rows, key=lambda r: (r["sequence_id"], r["frame_id"])):
        sequence = row["sequence_id"]
        if sequence != current:
            detector.reset(episode_id=sequence)
            tracker.reset(sequence)
            if hasattr(detector, "warmup"):
                detector.warmup()
            current = sequence
        with Image.open(local_file(folder, row["rgb"])) as image:
            rgb = np.asarray(image.convert("RGB")).copy()
        depth = axial_image(
            np.load(local_file(folder, row["depth"]), allow_pickle=False), calibration, convention
        )
        frame = FrameBundle(
            sequence,
            row["frame_id"],
            row["sim_time_s"],
            time.monotonic(),
            EgoPose(VehicleState(0, 0, 0, 0)),
            rgb,
            calibration,
            depth,
            row["frame_id"],
        )
        start = time.perf_counter()
        batch = detector.detect(frame)
        if fault:
            from dataclasses import replace

            if fault == "flicker" and row["sequence_index"] % 5 in (0, 1):
                batch = replace(batch, detections=())
            if fault == "id_reset" and row["sequence_index"] == 10:
                detector.reset(episode_id=sequence)
                batch = detector.detect(frame)
        associated = associate(frame, batch)
        tracked = tracker.update(associated.footprints, batch.metadata)
        estimates = estimate_batch_depth(frame, batch)
        confirmed = {o.track_id for o in tracked if o.confirmed and o.observed_this_frame}
        predictions = []
        for d, z in zip(batch.detections, estimates):
            predictions.append(
                dict(
                    class_name=d.class_name,
                    xyxy=[d.box.x_min, d.box.y_min, d.box.x_max, d.box.y_max],
                    confidence=d.confidence,
                    track_id=d.track_id,
                    confirmed=d.track_id in confirmed,
                    valid_depth=z.valid,
                    depth_m=z.axial_depth_m,
                    depth_reason=z.reason,
                )
            )
        duration = time.perf_counter() - start
        # Evaluation truth is opened after inference, association and tracking.
        with Image.open(local_file(folder, row["instance"])) as image:
            instance = np.asarray(image)
        identities = instance[:, :, 1].astype(np.uint32) * 256 + instance[:, :, 2]
        truths = []
        used = set()
        for truth in row["visible_boxes"]:
            tag = 12 if truth["class_name"] == "person" else 14
            values = depth[
                (instance[:, :, 0] == tag)
                & (identities == truth["instance_code"])
                & np.isfinite(depth)
            ]
            distance = float(np.median(values)) if len(values) else None
            candidates = [
                (overlap(p["xyxy"], truth["xyxy"]), i)
                for i, p in enumerate(predictions)
                if p["class_name"] == truth["class_name"] and i not in used
            ]
            score, index = max(candidates, default=(0, -1))
            matched = predictions[index] if score >= 0.5 else None
            if matched:
                used.add(index)
            truths.append(
                dict(
                    instance_code=truth["instance_code"],
                    class_name=truth["class_name"],
                    distance_m=distance,
                    detected=matched is not None,
                    confirmed=bool(matched and matched["confirmed"]),
                    valid_depth=bool(matched and matched["valid_depth"]),
                    track_id=matched["track_id"] if matched else None,
                    depth_error_m=matched["depth_m"] - distance
                    if matched and matched["valid_depth"] and distance
                    else None,
                )
            )
        records.append(
            dict(
                sample_id=row["sample_id"],
                sequence_id=sequence,
                sequence_index=row["sequence_index"],
                sim_time_s=row["sim_time_s"],
                frame_id=row["frame_id"],
                wall_s=duration,
                predictions=predictions,
                truths=truths,
                false_positives=len(predictions) - len(used),
                invalid_depth_count=associated.invalid_depth_count,
                tracked_count=len(tracked),
                footprints=[asdict(item) for item in tracked],
            )
        )
    return records


def range_summary(records):
    groups = defaultdict(list)
    for row in records:
        for truth in row["truths"]:
            groups[(row["sequence_id"], truth["class_name"], truth["instance_code"])].append(
                (row, truth)
            )
    sequences = []
    for (sequence, name, identity), trace in groups.items():
        distances = [t["distance_m"] for _, t in trace if t["distance_m"] is not None]
        if not distances:
            continue
        initial = trace[0][0]["sim_time_s"]
        confirmation = next((r["sim_time_s"] - initial for r, t in trace if t["confirmed"]), None)
        ids = [t["track_id"] for _, t in trace if t["track_id"] is not None]
        sequences.append(
            dict(
                sequence_id=sequence,
                class_name=name,
                instance_code=identity,
                distance_m=float(np.median(distances)),
                frames=len(trace),
                confirmation_s=confirmation,
                recall=sum(t["detected"] for _, t in trace) / len(trace),
                valid_depth_fraction=sum(t["valid_depth"] for _, t in trace) / len(trace),
                id_switches=sum(a != b for a, b in zip(ids, ids[1:])),
                max_abs_depth_error_m=max(
                    (abs(t["depth_error_m"]) for _, t in trace if t["depth_error_m"] is not None),
                    default=None,
                ),
            )
        )
    bins = []
    usable = {}
    for name in ("car", "person"):
        for centre in (5, 10, 15, 20, 25, 30):
            rows = [
                r
                for r in sequences
                if r["class_name"] == name and abs(r["distance_m"] - centre) <= 2.5
            ]
            # Each pose is one independent observation, not twenty independent tests.
            valid = [
                r
                for r in rows
                if r["frames"] >= 10
                and r["confirmation_s"] is not None
                and r["confirmation_s"] <= 0.15
                and r["recall"] >= 0.9
                and r["valid_depth_fraction"] >= 0.9
                and r["max_abs_depth_error_m"] is not None
                and r["max_abs_depth_error_m"] <= 1.0
            ]
            passed = len({r["sequence_id"] for r in rows}) >= 5 and len(valid) == len(rows)
            bins.append(
                dict(
                    class_name=name,
                    bin_centre_m=centre,
                    independent_sequences=len({r["sequence_id"] for r in rows}),
                    usable_sequences=len(valid),
                    status="PASS" if passed else "INSUFFICIENT_OR_FAIL",
                )
            )
        contiguous = []
        for b in [x for x in bins if x["class_name"] == name]:
            if b["status"] != "PASS":
                break
            contiguous.append(b["bin_centre_m"] - 2.5)
        usable[name] = max(contiguous, default=0.0)
    times = [r["wall_s"] for r in records]
    return dict(
        status="MEASURED",
        bins=bins,
        sequences=sequences,
        usable_range_lower_bound_m=usable,
        first_call_s=times[0] if times else None,
        warm_p95_s=float(np.percentile(times[1:], 95)) if len(times) > 1 else None,
        pedestrian_25m_covered=usable["person"] >= 25,
        false_positives=sum(r["false_positives"] for r in records),
        scope="Stationary held-out sequences. Range requires >=5 independent poses per bin, >=90% recall/depth, confirmation <=150 ms. Moving/occluded objects require further evidence.",
    )
