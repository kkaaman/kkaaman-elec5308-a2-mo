"""Adapt PR69 recorded RGB-D replay to the canonical TrackDepthFilter.

Evaluates only test roads/runs/images accepted by the existing dataset validator.
Never accepts a training/validation subset as held-out evidence.
"""

import argparse
import json
from pathlib import Path

from evaluation.dataset import read_dataset, sha256
from evaluation.perception_replay import range_summary, replay
from scripts.replay_local_detector import make_detector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth-calibration", type=Path, required=True)
    args = parser.parse_args()
    manifest, rows = read_dataset(args.dataset)
    calibration = json.loads(args.depth_calibration.read_text())
    if (calibration.get("status") != "MEASURED_PASS"
            or calibration.get("opendrive_sha256") != manifest.get("opendrive_sha256")
            or calibration.get("rig") != manifest.get("camera")):
        raise ValueError("Matching measured map/camera depth calibration required")
    selected = [r for r in rows if r["split"] == "test"]
    if not selected:
        parser.error("No held-out test scenes")
    args.output.mkdir(parents=True, exist_ok=False)
    detector = make_detector()
    try:
        records = replay(args.dataset, manifest, selected, detector, calibration["convention"])
    finally:
        detector.close()
    report = range_summary(records)
    report.update(
        dataset_manifest_sha256=sha256(args.dataset / "manifest.json"),
        model_version=detector.model_version,
        evaluated_split="test",
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
