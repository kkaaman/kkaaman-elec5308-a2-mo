"""Adapt PR69 recorded RGB-D replay to the canonical TrackDepthFilter.

Evaluates only test roads/runs/images accepted by the existing dataset validator.
Never accepts a training/validation subset as held-out evidence.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from autonomy.admission import load_admitted_baseline
from evaluation.dataset import read_dataset, sha256
from evaluation.perception_replay import range_summary, replay
from obstacle_perception.box_depth import DepthPolicy
from obstacle_perception.tracking import TrackPolicy
from perception.recorded_detector import RecordedDetector
from scripts.replay_local_detector import make_detector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth-calibration", type=Path, required=True)
    parser.add_argument(
        "--predictions", type=Path, help="Rescore retained model records; never rerun inference"
    )
    args = parser.parse_args()
    manifest, rows = read_dataset(args.dataset)
    calibration = json.loads(args.depth_calibration.read_text())
    if (
        calibration.get("status") != "MEASURED_PASS"
        or calibration.get("opendrive_sha256") != manifest.get("opendrive_sha256")
        or calibration.get("rig")
        != dict(
            manifest["camera"],
            width=manifest["image_size_wh"][0],
            height=manifest["image_size_wh"][1],
        )
    ):
        raise ValueError("Matching measured map/camera depth calibration required")
    selected = [r for r in rows if r["split"] == "test"]
    if not selected:
        parser.error("No held-out test scenes")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    freeze = load_admitted_baseline(root / "configs/mission_baseline.json")[
        "baseline_threshold_freeze"
    ]
    track_policy = TrackPolicy(**freeze["track_policy"])
    depth_policy = DepthPolicy(**freeze["depth_policy"])
    if args.predictions:
        previous = json.loads((args.predictions.parent / "summary.json").read_text())
        expected = "yolo26n:sha256:" + freeze["detector"]["model_sha256"]
        if previous["model_version"] != expected:
            raise ValueError("Retained model identity differs from admitted model")
        old = [json.loads(line) for line in args.predictions.read_text().splitlines()]
        if len(old) != len(selected) or {v["sample_id"] for v in old} != {
            v["sample_id"] for v in selected
        }:
            raise ValueError("Retained predictions do not cover exactly the held-out samples")
        fixture = dict(
            schema_version=1,
            origin="recorded",
            source_id="replay",
            model_version=expected,
            frames=[
                dict(
                    frame_id=v["frame_id"],
                    sim_time_s=v["sim_time_s"],
                    width=manifest["image_size_wh"][0],
                    height=manifest["image_size_wh"][1],
                    valid=v["valid"],
                    invalid_reason=None if v["valid"] else "retained_invalid",
                    detections=[
                        {k: p[k] for k in ("class_name", "confidence", "xyxy", "track_id")}
                        for p in v["predictions"]
                    ],
                )
                for v in old
            ],
        )
        fixture_path = args.output / "retained-detector.json"
        fixture_path.write_text(json.dumps(fixture, allow_nan=False) + "\n")
        detector = RecordedDetector(fixture_path, sha256(fixture_path))
    else:
        detector = make_detector()
    try:
        records = replay(
            args.dataset,
            manifest,
            selected,
            detector,
            calibration["convention"],
            track_policy=track_policy,
            depth_policy=depth_policy,
        )
    finally:
        detector.close()
    report = range_summary(records)
    report.update(
        dataset_manifest_sha256=sha256(args.dataset / "manifest.json"),
        model_version=detector.model_version,
        evaluated_split="test",
        track_policy=asdict(track_policy),
        depth_policy=asdict(depth_policy),
        prediction_source_sha256=sha256(args.predictions) if args.predictions else None,
        timing_origin="retained fixture/filter only; not learned inference"
        if args.predictions
        else "measured model/filter replay",
        frames=len(records),
        origin="recorded",
        reference="PR69 5942fab; canonical TrackDepthFilter adaptation",
        limitations=[
            "Visible-mask truth is not amodal object truth.",
            "No tuning on test roads/runs/images is permitted.",
            "Stationary replay does not validate autonomous RESUME or #41 acceptance.",
            "Adjacent frames are correlated; range bins require independent sequences.",
        ],
    )
    for values in report["class_metrics"].values():
        tp, fp, fn = (values[k] for k in ("tp", "fp", "fn"))
        values.update(
            precision=tp / (tp + fp) if tp + fp else None,
            recall=tp / (tp + fn) if tp + fn else None,
        )
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (args.output / "records.jsonl").open("x") as stream:
        for row in records:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
