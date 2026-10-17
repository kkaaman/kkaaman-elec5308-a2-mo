"""Measure the existing YOLO model on reviewed CARLA images; no retraining."""

import argparse
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.dataset import local_file, read_dataset, save_json, sha256


def iou(a, b):
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def match_boxes(predictions, truths, threshold=0.5):
    used = set()
    true_positive = 0
    for prediction in sorted(predictions, key=lambda p: p["confidence"], reverse=True):
        options = [
            (iou(prediction["xyxy"], box["xyxy"]), i)
            for i, box in enumerate(truths)
            if i not in used
        ]
        score, best = max(options, default=(0, -1))
        if best >= 0 and score >= threshold:
            used.add(best)
            true_positive += 1
    return true_positive, len(predictions) - true_positive, len(truths) - true_positive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=640)
    args = parser.parse_args()
    if not math.isfinite(args.confidence) or not 0 <= args.confidence <= 1 or args.image_size <= 0:
        parser.error("Invalid confidence or image size")
    weights = args.weights.resolve()
    if not weights.is_file() or weights.stat().st_size < 1024:
        raise ValueError("Provide the real local yolo26n.pt checkpoint")
    with weights.open("rb") as stream:
        if stream.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError("YOLO weights are still a Git LFS pointer")
    manifest, rows = read_dataset(args.dataset)
    rows = [r for r in rows if r["split"] == args.split]
    if not rows:
        raise ValueError("Evaluation split is empty")
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import ultralytics
    from ultralytics import YOLO

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    sync()
    started = time.perf_counter()
    model = YOLO(str(weights)).to(args.device)
    sync()
    construction = time.perf_counter() - started
    class_ids = [key for key, name in model.names.items() if name in ("person", "car")]
    if {model.names[key] for key in class_ids} != {"person", "car"}:
        raise ValueError("Model class names must include person and car")
    totals = {name: np.zeros(3, dtype=np.int64) for name in ("person", "car")}
    samples, times = [], []
    for row in rows:
        with Image.open(local_file(args.dataset, row["rgb"])) as image:
            bgr = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
        sync()
        started = time.perf_counter()
        result = model.predict(
            bgr, conf=args.confidence, imgsz=args.image_size, classes=class_ids, verbose=False
        )[0]
        sync()
        duration = time.perf_counter() - started
        times.append(duration)
        predictions = []
        if result.boxes is not None:
            for xyxy, confidence, label in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
                strict=True,
            ):
                predictions.append(
                    {
                        "class_name": result.names[int(label)],
                        "xyxy": xyxy.tolist(),
                        "confidence": float(confidence),
                    }
                )
        for name in totals:
            totals[name] += match_boxes(
                [p for p in predictions if p["class_name"] == name],
                [t for t in row["visible_boxes"] if t["class_name"] == name],
            )
        samples.append(
            {
                "sample_id": row["sample_id"],
                "frame_id": row["frame_id"],
                "inference_wall_s": duration,
                "predictions": predictions,
            }
        )
    summary = {}
    for name, values in totals.items():
        tp, fp, fn = map(int, values)
        summary[name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
        }
    report = {
        "status": "MEASURED_NOT_ACCEPTANCE",
        "weights_sha256": sha256(weights),
        "dataset_index_sha256": manifest["samples_sha256"],
        "split": args.split,
        "device": args.device,
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "confidence": args.confidence,
        "image_size": args.image_size,
        "iou_threshold": 0.5,
        "construction_wall_s": construction,
        "first_call_wall_s": times[0],
        "warm_p95_wall_s": float(np.percentile(times[1:], 95)) if len(times) > 1 else None,
        "per_class": summary,
        "frames": len(rows),
        "scope": "Static reviewed visible-mask boxes, not full/amodal boxes. No mAP, track-confirmation range or driving acceptance claim.",
        "samples": samples,
    }
    save_json(args.output / "detector.json", report)
    print({key: value for key, value in report.items() if key != "samples"})


if __name__ == "__main__":
    main()
