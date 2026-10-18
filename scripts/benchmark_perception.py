"""Replay the existing YOLO + depth + production track filter on saved sequences."""

import argparse
import json
from pathlib import Path

from evaluation.dataset import read_dataset, save_json, sha256
from evaluation.perception_replay import range_summary, replay
from perception import ObjectDetector, PerceptionConfig
from perception.frame_detector import FrameDetector
from perception.recorded_detector import verify_local_asset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "weights", "depth-calibration", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--fault", choices=("flicker", "id_reset"))
    parser.add_argument(
        "--image-size", type=int, default=640, help="Detector input size; frozen into the config"
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()
    manifest, rows = read_dataset(args.dataset)
    rows = [r for r in rows if r["split"] == args.split]
    if not rows or any("sequence_id" not in r for r in rows):
        raise ValueError("Capture consecutive frames with this batch collector first")
    depth = json.loads(args.depth_calibration.read_text())
    if depth["status"] != "MEASURED_PASS" or depth.get("opendrive_sha256") != manifest.get(
        "opendrive_sha256"
    ):
        raise ValueError("Matching map depth calibration required")
    asset = json.loads((Path(__file__).resolve().parents[1] / "configs/models.yaml").read_text())[
        "models"
    ][0]
    verify_local_asset(args.weights, asset["sha256"])
    args.output.mkdir(parents=True, exist_ok=False)
    detector = FrameDetector(
        ObjectDetector(
            PerceptionConfig(
                model_name=str(args.weights.resolve()),
                image_size=args.image_size,
                confidence_threshold=args.confidence,
            )
        ),
        model_version=asset["sha256"],
    )
    try:
        records = replay(
            args.dataset, manifest, rows, detector, depth["convention"], fault=args.fault
        )
    finally:
        detector.close()
    report = range_summary(records)
    report.update(
        dataset_index_sha256=manifest["samples_sha256"],
        model_sha256=asset["sha256"],
        split=args.split,
        depth_calibration_sha256=sha256(args.depth_calibration),
        fault=args.fault,
        detector=dict(image_size=args.image_size, confidence=args.confidence),
    )
    save_json(args.output / "perception.json", report)
    with (args.output / "replay.jsonl").open("x", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    print(
        f"Measured {len(records)} frames | detector {args.image_size}px conf {args.confidence}"
        f" | warm p95 {report['warm_p95_s'] * 1000:.0f} ms"
        f" | in-path usable ranges {report['usable_range_lower_bound_m']}"
        f" | all visible (report only) {report['all_visible_usable_range_m']} | {args.output}"
    )


if __name__ == "__main__":
    main()
