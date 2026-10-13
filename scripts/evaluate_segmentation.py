#!/usr/bin/env python3
"""Compare segmentation backends on the same frames (BL-B-05).

For each backend this measures inference time and, when ground-truth labels are
given, IoU against them: the 19-class mean IoU and the obstacle mean IoU
(person, rider, car, truck, bus, motorcycle, bicycle). It also saves overlay
images so the masks can be checked by eye.

Backends are given as ``name`` or ``name=weights``; repeat ``--backend`` to
compare several, including two weights files for the same backend.

Examples::

    # Timing and overlays only (no labels), on the A1 street image
    python scripts/evaluate_segmentation.py --images src/perception/street.jpg \
        --backend bisenetv2 --backend segformer_b0 \
        --backend yolo_seg=yolov8n-seg.pt --backend yolo_seg=yolo26n-seg.pt

    # Score against the smoke test's CARLA frame (colour-palette labels)
    python scripts/evaluate_segmentation.py \
        --images "runs/smoke/*/rgb.png" --labels "runs/smoke/*/semantic.png" \
        --label-format carla-palette --backend bisenetv2 --backend segformer_b0 \
        --backend yolo_seg=yolov8n-seg.pt --backend yolo_seg=yolo26n-seg.pt

Label formats: ``carla-tags`` (raw CARLA semantic image, tag in the red
channel), ``carla-palette`` (saved with CityScapesPalette) and ``trainid``
(single-channel Cityscapes trainIds).
"""

from __future__ import annotations

import argparse
import glob
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from segmentation.backends import BACKENDS, create_segmenter  # noqa: E402
from segmentation.evaluation import NUM_CLASSES, confusion_matrix, summarize  # noqa: E402
from segmentation.labels import (  # noqa: E402
    CITYSCAPES_CLASSES,
    carla_palette_to_trainid,
    carla_tags_to_trainid,
)

LABEL_FORMATS = ("carla-tags", "carla-palette", "trainid")


def parse_backend(spec: str) -> tuple[str, str | None]:
    """Split ``name`` or ``name=weights`` and validate the name."""
    name, _, weights = spec.partition("=")
    name = name.strip()
    if name not in BACKENDS:
        raise argparse.ArgumentTypeError(f"unknown backend '{name}'; choose from {', '.join(BACKENDS)}")
    return name, (weights.strip() or None)


def expand(pattern: str) -> list[Path]:
    matches = sorted(glob.glob(pattern, recursive=True))
    if not matches and Path(pattern).is_file():
        matches = [pattern]
    return [Path(m) for m in matches]


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot read image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_label(path: Path, label_format: str) -> tuple[np.ndarray, float | None]:
    """Return (trainId mask, matched fraction or None)."""
    if label_format == "trainid":
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None or mask.ndim != 2:
            raise ValueError(f"{path} is not a single-channel trainId image")
        return mask.astype(np.uint8), None
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot read label {path}")
    if label_format == "carla-tags":
        return carla_tags_to_trainid(bgr[..., 2]), None  # tag is in the red channel
    return carla_palette_to_trainid(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, int(np.ceil(pct / 100 * len(ordered))) - 1))]


def evaluate_backend(name, weights, args, images, labels, out_dir) -> dict:
    record: dict = {"backend": name, "weights": weights}
    kwargs: dict = {}
    if args.device:
        kwargs["device"] = args.device
    if name == "segformer_b0":
        kwargs["input_size"] = tuple(args.segformer_size)
    if name == "yolo_seg":
        kwargs["confidence"] = args.yolo_conf
        kwargs["image_size"] = args.yolo_imgsz
    try:
        t0 = time.perf_counter()
        segmenter = create_segmenter(name, weights, **kwargs)
        record["load_s"] = round(time.perf_counter() - t0, 2)
    except Exception as exc:  # noqa: BLE001 - one broken backend must not stop the others
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    record["weights"] = getattr(segmenter, "weights", weights)
    record["device"] = segmenter.device
    record["labels_all_classes"] = segmenter.labels_all_classes

    first = read_rgb(images[0])
    for _ in range(args.warmup):
        segmenter.predict(first)
    _sync(segmenter.device)

    times_ms: list[float] = []
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES + 1), dtype=np.int64)
    class_pixels = np.zeros(256, dtype=np.int64)
    label_match: list[float] = []
    for index, image_path in enumerate(images):
        image = read_rgb(image_path)
        start = time.perf_counter()
        mask = segmenter.predict(image)
        _sync(segmenter.device)
        times_ms.append((time.perf_counter() - start) * 1000.0)
        class_pixels += np.bincount(mask.ravel(), minlength=256)
        if labels:
            gt, matched = read_label(labels[index], args.label_format)
            if gt.shape != mask.shape:
                raise ValueError(f"label {labels[index]} shape {gt.shape} != image {mask.shape}")
            cm += confusion_matrix(gt, mask)
            if matched is not None:
                label_match.append(matched)
        if index < args.save_overlays:
            overlay = segmenter.visualize(image, mask)
            safe_weights = Path(str(record["weights"])).name.replace("/", "_")
            path = out_dir / "overlays" / f"{index:03d}_{name}_{safe_weights}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # Extra timing passes when there are only a few frames
    while len(times_ms) < args.min_timed:
        start = time.perf_counter()
        segmenter.predict(first)
        _sync(segmenter.device)
        times_ms.append((time.perf_counter() - start) * 1000.0)

    record["timing_ms"] = {
        "mean": round(statistics.fmean(times_ms), 2),
        "p50": round(_percentile(times_ms, 50), 2),
        "p95": round(_percentile(times_ms, 95), 2),
        "n": len(times_ms),
    }
    total = class_pixels.sum()
    record["predicted_class_fraction"] = {
        (CITYSCAPES_CLASSES[i] if i < NUM_CLASSES else "unlabelled"): round(float(c / total), 4)
        for i, c in enumerate(class_pixels)
        if c
    }
    if labels:
        record["metrics"] = summarize(cm, labels_all_classes=segmenter.labels_all_classes)
        if label_match:
            record["label_colour_match"] = round(float(np.mean(label_match)), 4)
    return record


def _sync(device: str) -> None:
    if device == "cuda":
        import torch

        torch.cuda.synchronize()


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare segmentation backends (BL-B-05)")
    parser.add_argument("--images", required=True, help="image file or glob (quote globs)")
    parser.add_argument("--labels", help="label file or glob, sorted to match --images")
    parser.add_argument("--label-format", choices=LABEL_FORMATS, default="carla-tags")
    parser.add_argument(
        "--backend", action="append", type=parse_backend, required=True,
        help="name or name=weights; repeat to compare (" + ", ".join(BACKENDS) + ")",
    )  # fmt: skip
    parser.add_argument("--device", help="cuda, mps or cpu (default: auto)")
    parser.add_argument("--segformer-size", type=int, nargs=2, default=(1024, 512), metavar=("W", "H"))
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--min-timed", type=int, default=20, help="minimum timed inferences")
    parser.add_argument("--save-overlays", type=int, default=3, help="overlays saved per backend")
    parser.add_argument("--out", type=Path, default=Path("runs/seg_eval"))
    args = parser.parse_args(argv)

    images = expand(args.images)
    if not images:
        parser.error(f"no images match {args.images}")
    labels = expand(args.labels) if args.labels else []
    if args.labels and len(labels) != len(images):
        parser.error(f"{len(images)} images but {len(labels)} labels; they are paired in sorted order")

    started = datetime.now(timezone.utc)
    out_dir = args.out / started.strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(images)} image(s){' with labels' if labels else ''}; first pair: {images[0]}"
          + (f" <-> {labels[0]}" if labels else ""))  # fmt: skip

    results = []
    for name, weights in args.backend:
        print(f"Evaluating {name}{'=' + weights if weights else ''} ...")
        results.append(evaluate_backend(name, weights, args, images, labels, out_dir))

    report = {
        "task": "BL-B-05 segmentation backend comparison",
        "started_utc": started.isoformat(timespec="seconds"),
        "host": platform.node(),
        "python": platform.python_version(),
        "images": [str(p) for p in images],
        "labels": [str(p) for p in labels],
        "label_format": args.label_format if labels else None,
        "results": results,
    }
    (out_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n| Backend | Weights | Device | ms mean | ms p95 | mIoU (19) | Obstacle mIoU | person | car |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        if "error" in r:
            print(f"| {r['backend']} | {r['weights']} | - | - | - | - | - | - | - |  error: {r['error']}")
            continue
        m = r.get("metrics", {})
        iou = m.get("per_class_iou", {})
        t = r["timing_ms"]
        print(
            f"| {r['backend']} | {Path(str(r['weights'])).name} | {r['device']} | {t['mean']} | {t['p95']} "
            f"| {_fmt(m.get('miou_19'))} | {_fmt(m.get('obstacle_miou'))} "
            f"| {_fmt(iou.get('person'))} | {_fmt(iou.get('car'))} |"
        )
    for r in results:
        if r.get("label_colour_match") is not None and r["label_colour_match"] < 0.9:
            print(f"\nwarning: only {r['label_colour_match']:.0%} of label pixels matched the palette; "
                  "check --label-format")  # fmt: skip
            break
    if not labels:
        print("\nNo labels given: timing and overlays only. IoU needs CARLA ground truth (BL-B-03).")
    print(f"\nResults: {out_dir / 'results.json'}  Overlays: {out_dir / 'overlays'}")
    return 0 if all("error" not in r for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
