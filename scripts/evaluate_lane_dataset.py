"""Score pristine held-out lane labels using existing dataset/metric contracts.

Binary marking quality does not certify lane-centre selection or divider
agreement. Display/video recordings cannot supply pristine ground truth.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.dataset import confusion, local_file, metrics, read_dataset, sha256


def evaluate(folder, segmenter, *, topology=None, overlay_output=None):
    manifest, rows = read_dataset(folder)
    rows = [row for row in rows if row["split"] == "test"]
    if not rows:
        raise ValueError("No pristine held-out test scenes")
    observer = None
    if topology is not None:
        from evaluation.a1_comparison import LaneCentrelines
        from geometry.camera import CameraIntrinsics
        from visualisation.lane_departure import LaneDepartureMonitor

        if topology.opendrive_sha256 != manifest["opendrive_sha256"]:
            raise ValueError("Lane-reference map differs from captured map")
        lanes = LaneCentrelines(topology)
        rig = manifest["camera"]
        width, height = manifest["image_size_wh"]
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
        observer = LaneDepartureMonitor()
        label_observer = LaneDepartureMonitor()
        if overlay_output is not None:
            overlay_output.mkdir(parents=True, exist_ok=False)
    totals = np.zeros(4, dtype=np.int64)
    scenes = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    records = []
    for row in rows:
        with Image.open(local_file(folder, row["rgb"])) as image:
            rgb = np.asarray(image).copy()
            prediction = segmenter.predict(rgb)
        with Image.open(local_file(folder, row["lane"])) as image:
            truth_mask = np.asarray(image).copy()
            counts = confusion(prediction, truth_mask)
        totals += counts
        scenes[row["run_id"]] += counts
        record = dict(sample_id=row["sample_id"], run_id=row["run_id"], **metrics(counts))
        if observer is not None:
            from perception.lane_path import extract_ego_path
            from visualisation.hud import draw_departure_overlay
            from visualisation.lane_departure import lane_estimate_from_centerline

            pose = row["ego_world"]
            x, y, _ = pose["location_m"]
            yaw = -math.radians(pose["rotation_deg_pitch_yaw_roll"][1])
            rear_x, rear_y = x - 1.38641 * math.cos(yaw), -y - 1.38641 * math.sin(yaw)
            cte, section, junction = lanes.signed_offset(rear_x, rear_y)
            path = extract_ego_path(prediction)
            estimate = (
                None
                if path is None
                else lane_estimate_from_centerline(
                    path["centerline"],
                    calibration,
                    frame_id=row["frame_id"],
                    sim_time_s=row["sim_time_s"],
                )
            )
            report = observer.update(
                estimate,
                None if junction else cte,
                now_sim_time_s=row["sim_time_s"],
                frame_id=row["frame_id"],
            )
            label_path = extract_ego_path(truth_mask)
            label_estimate = (
                None
                if label_path is None
                else lane_estimate_from_centerline(
                    label_path["centerline"],
                    calibration,
                    frame_id=row["frame_id"],
                    sim_time_s=row["sim_time_s"],
                )
            )
            label_report = label_observer.update(
                label_estimate,
                None if junction else cte,
                now_sim_time_s=row["sim_time_s"],
                frame_id=row["frame_id"],
            )
            record["label_mask_observer"] = label_report.as_record()
            record.update(
                report.as_record(),
                sequence_id=row["sequence_id"],
                weather=row["weather"],
                map_section=section,
                reference_rear_axle_xy_m=[rear_x, rear_y],
                in_junction=bool(junction),
                evaluation_truth_only=True,
                control_authority=False,
            )
            if overlay_output is not None and row["sequence_index"] == 0:
                target = overlay_output / (row["sample_id"].replace("/", "_") + ".png")
                Image.fromarray(draw_departure_overlay(rgb, report)).save(target)
        records.append(record)
    return dict(
        status="SCORED_NOT_LANE_CENTRE_ACCEPTANCE",
        evaluated_split="test",
        dataset_manifest_sha256=sha256(folder / "manifest.json"),
        samples=len(rows),
        aggregate=metrics(totals),
        lane_observer_summary=dict(
            recorded_frames=len(records),
            valid_frames=sum(bool(r.get("lane_monitor_valid")) for r in records),
            disagreements=sum(r.get("lane_monitor_status") == "disagree" for r in records),
            status="RECORDED_NOT_AUTOMATIC_ACCEPTANCE" if observer else "NOT_RUN",
            reference="Measured rear-axle offset + exported map topology; matched pristine frame pose; evaluation only",
        ),
        scenes={key: metrics(value) for key, value in scenes.items()},
        records=records,
        limitations=[
            "Adjacent frames are correlated within scenes.",
            "Binary marking quality does not establish correct lane/divider selection.",
            "Recorded curves and target-induced S5 stops remain separate acceptance.",
        ],
    )


def main():
    from perception.lane_inference import LaneSegmenter
    from perception.recorded_detector import verify_local_asset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument(
        "--topology",
        type=Path,
        help="Existing exported map for evaluation-only observer comparison",
    )
    parser.add_argument("--overlays", type=Path, help="New directory; first frame per scene")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    weights = verify_local_asset(args.weights, args.weights_sha256)
    from planning.lane_graph import LaneTopology

    topology = LaneTopology.load(args.topology) if args.topology else None
    report = evaluate(
        args.dataset, LaneSegmenter(weights), topology=topology, overlay_output=args.overlays
    )
    report["weights_sha256"] = args.weights_sha256
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
