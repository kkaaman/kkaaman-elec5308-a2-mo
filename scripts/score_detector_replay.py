"""Score retained E4 predictions without repeating inference or latency measurements.

Reuse the canonical IoU matcher. Reject scene/image leakage before scoring.
This reports detector quality only, never clearance or physical stop acceptance.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from obstacle_perception.types import DetectionObservation, PixelBox
from perception.replay_accuracy import match_counts
from scripts.benchmark_remote_detector import summarize


def score(replay, annotations):
    manifest = json.loads((replay / "summary.json").read_text())
    labels = annotations["frames"]
    tuning = set(annotations["tuning_scene_ids"])
    if annotations["model_version"] != manifest["model_version"]:
        raise ValueError("Annotation/model identity mismatch")
    expected = {x["file"]: x["sha256"] for x in manifest["replay"]}
    if set(labels) != set(expected):
        raise ValueError("Labels must explicitly cover exactly the replay inputs")
    seen_images = {}
    for name, row in labels.items():
        scene = row["scene_id"]
        if not isinstance(scene, str) or not scene.strip() or scene in tuning:
            raise ValueError("Held-out scene is missing or was used for tuning")
        if row["split"] != "test" or row["sha256"] != expected[name]:
            raise ValueError("Held-out split/image identity mismatch")
        if row["sha256"] in set(annotations["tuning_image_sha256"]):
            raise ValueError("Tuning/test image leakage")
        if row["sha256"] in seen_images and seen_images[row["sha256"]] != scene:
            raise ValueError("One image assigned to multiple independent scenes")
        seen_images[row["sha256"]] = scene
        match_counts((), row["labels"])
    groups, scene_groups = defaultdict(list), defaultdict(list)
    seen = set()
    for line in (replay / "predictions.jsonl").read_text().splitlines():
        row = json.loads(line)
        key = (row["mode"], row["width"], row["height"], row["jpeg_quality"])
        identity = (*key, row["frame"])
        if identity in seen or row["frame"] not in labels:
            raise ValueError("Duplicate or foreign prediction identity")
        seen.add(identity)
        if row["status"] not in ("OK", "ERROR"):
            raise ValueError("Unknown inference status")
        detections = tuple(
            DetectionObservation(d["class_name"], d["confidence"], PixelBox(*d["xyxy"]), None)
            for d in row["detections"]
        )
        if row["status"] != "OK" and detections:
            raise ValueError("Failed inference cannot supply detections")
        target = labels[row["frame"]]
        scale = dict(
            scale_x=row["width"] / row["original_width"],
            scale_y=row["height"] / row["original_height"],
        )
        tp, fp, fn = match_counts(detections, target["labels"], **scale)
        scored = dict(row, tp=tp, fp=fp, fn=fn)
        groups[key].append(scored)
        scene_groups[(*key, target["scene_id"])].append(scored)
    configured = {
        (mode, c["width"], c["height"], c["jpeg_quality"])
        for mode, data in manifest["modes"].items()
        for c in data.get("configurations", [])
    }
    if set(groups) != configured or any(
        {x["frame"] for x in rows} != set(expected) for rows in groups.values()
    ):
        raise ValueError("Incomplete prediction/configuration coverage")

    def result(key, rows):
        return dict(configuration=list(key), **summarize(rows, manifest["deadline_s"]))

    per_class = []
    for key, rows in groups.items():
        for name in ("car", "person"):
            counts = [0, 0, 0]
            for row in rows:
                target = labels[row["frame"]]
                detections = tuple(
                    DetectionObservation(
                        d["class_name"], d["confidence"], PixelBox(*d["xyxy"]), None
                    )
                    for d in row["detections"]
                    if d["class_name"] == name
                )
                values = match_counts(
                    detections,
                    [t for t in target["labels"] if t["class_name"] == name],
                    scale_x=row["width"] / row["original_width"],
                    scale_y=row["height"] / row["original_height"],
                )
                counts = [a + b for a, b in zip(counts, values)]
            tp, fp, fn = counts
            per_class.append(
                dict(
                    configuration=list(key),
                    class_name=name,
                    tp=tp,
                    fp=fp,
                    fn=fn,
                    precision=tp / (tp + fp) if tp + fp else None,
                    recall=tp / (tp + fn) if tp + fn else None,
                )
            )
    return dict(
        status="SCORED_NOT_ACCEPTANCE",
        origin=manifest["origin"],
        independent_scenes=len({x["scene_id"] for x in labels.values()}),
        model_version=manifest["model_version"],
        configurations=[result(k, v) for k, v in groups.items()],
        scenes=[result(k, v) for k, v in scene_groups.items()],
        per_class=per_class,
        limitations=[
            "Frame metrics are correlated within scenes; no independent-frame confidence claim.",
            "No operational first-confirmed range, invalid-depth or stop/resume acceptance inferred.",
            "Accuracy is valid only for the declared held-out scenes and annotation coverage.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = score(args.replay, json.loads(args.annotations.read_text()))
    report["annotations_sha256"] = hashlib.sha256(args.annotations.read_bytes()).hexdigest()
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
