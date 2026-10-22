"""Fine-tune the YOLO obstacle detector on CARLA road users recorded on another town.

Labels come from the capture's instance image (R = CARLA semantic tag, G/B = actor),
using the same visible-pixel rule as ``evaluation.dataset.visible_boxes``. Every
road-user class the stack treats as an obstacle is labelled (person, rider -> person,
car, truck, bus, motorcycle, bicycle), so fine-tuning does not teach the model to
ignore unlabelled vehicles. The COCO class list of the initial model is kept, so the
runtime's class handling is unchanged. Traffic lights and signs are not labelled;
keep the original model for any traffic-light work.

Training selects on the dataset's own validation split. The demo map (Town10HD) is
refused unless --allow-demo-map: its validation and test splits are the hold-out the
candidate is measured on afterwards (scripts/benchmark_perception.py --candidate).
"""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.dataset import (
    local_file,
    map_short_name,
    read_dataset,
    recorded_on_demo_map,
    save_json,
    sha256,
)

# CARLA 0.9.16 semantic tags -> COCO class names used by the stack's obstacle filter.
TAG_TO_NAME = {
    12: "person",
    13: "person",  # rider
    14: "car",
    15: "truck",
    16: "bus",
    18: "motorcycle",
    19: "bicycle",
}
MIN_PIXELS = 16  # as evaluation.dataset.visible_boxes


def road_user_boxes(instance_rgb, min_pixels=MIN_PIXELS):
    """Visible-mask boxes of road users in a saved instance image (R tag, G/B id)."""
    image = np.asarray(instance_rgb)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Instance image must be HxWx3 (R tag, G/B instance)")
    tags = image[:, :, 0]
    ids = image[:, :, 1].astype(np.uint32) * 256 + image[:, :, 2]
    boxes = []
    for tag, name in TAG_TO_NAME.items():
        selected = tags == tag
        if not selected.any():
            continue
        for identity in np.unique(ids[selected]):
            ys, xs = np.nonzero(selected & (ids == identity))
            if len(xs) < min_pixels:
                continue
            boxes.append((name, int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    return boxes


def yolo_lines(boxes, class_ids, width, height):
    lines = []
    for name, x0, y0, x1, y1 in boxes:
        cx, cy = (x0 + x1) / 2 / width, (y0 + y1) / 2 / height
        w, h = (x1 - x0) / width, (y1 - y0) / height
        lines.append(f"{class_ids[name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def _link_or_copy(source, target):
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def export_yolo(folder, rows, output, class_ids):
    """Images (hard links where possible) and YOLO labels for the train/val splits."""
    stats = {
        "images": {},
        "boxes": {name: 0 for name in sorted(set(TAG_TO_NAME.values()))},
        "person_height_px": {"<20": 0, "20-40": 0, ">=40": 0},
    }
    for split in ("train", "val"):
        chosen = [r for r in rows if r["split"] == split]
        if not chosen:
            raise ValueError(f"Dataset has no {split} rows")
        (output / "images" / split).mkdir(parents=True, exist_ok=False)
        (output / "labels" / split).mkdir(parents=True, exist_ok=False)
        for row in chosen:
            rgb = local_file(folder, row["rgb"])
            with Image.open(local_file(folder, row["instance"])) as image:
                instance = np.asarray(image.convert("RGB"))
            height, width = instance.shape[:2]
            boxes = road_user_boxes(instance)
            name = f"{row['sample_id']}{rgb.suffix}"
            _link_or_copy(rgb, output / "images" / split / name)
            (output / "labels" / split / f"{row['sample_id']}.txt").write_text(
                "\n".join(yolo_lines(boxes, class_ids, width, height)) + ("\n" if boxes else ""),
                encoding="utf-8",
            )
            for box in boxes:
                stats["boxes"][box[0]] += 1
                if box[0] == "person":
                    tall = box[4] - box[2]
                    key = "<20" if tall < 20 else "20-40" if tall < 40 else ">=40"
                    stats["person_height_px"][key] += 1
        stats["images"][split] = len(chosen)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True, help="Registered yolo26n.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1280, help="Match the runtime detector")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hours", type=float, default=3.0, help="Wall-clock training limit")
    parser.add_argument("--batch", type=int, default=-1, help="-1 lets Ultralytics size it")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=5308)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One short epoch to check the pipeline; not a candidate",
    )
    parser.add_argument(
        "--allow-demo-map",
        action="store_true",
        help="Deliberately fine-tune on images recorded on the demo map (Town10HD)",
    )
    args = parser.parse_args()
    if args.imgsz % 32 or not 320 <= args.imgsz <= 1920:
        raise ValueError("imgsz must be a multiple of 32 in [320, 1920]")
    manifest, rows = read_dataset(args.dataset)
    demo = recorded_on_demo_map(manifest)
    if demo and not args.allow_demo_map:
        raise SystemExit(
            f"Refusing to train on {map_short_name(manifest.get('map', ''))}: it is the demo map,"
            " whose validation/test splits measure the candidate. Capture another town"
            " (run_batch35.ps1 -Stage Capture -Map Town05) or pass --allow-demo-map."
        )
    from ultralytics import YOLO

    args.output.mkdir(parents=True, exist_ok=False)
    model = YOLO(str(args.initial))
    names = dict(model.names)
    class_ids = {name: index for index, name in names.items()}
    missing = sorted(set(TAG_TO_NAME.values()) - set(class_ids))
    if missing:
        raise ValueError(f"Initial model lacks classes {missing}")
    stats = export_yolo(args.dataset.resolve(), rows, args.output / "yolo", class_ids)
    data_yaml = args.output / "yolo" / "data.yaml"
    data_yaml.write_text(
        "path: " + str((args.output / "yolo").resolve()).replace("\\", "/") + "\n"
        "train: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {names[i]}\n" for i in sorted(names)),
        encoding="utf-8",
    )
    train_args = dict(
        data=str(data_yaml),
        imgsz=args.imgsz,
        epochs=1 if args.smoke else args.epochs,
        time=None if args.smoke else args.hours,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        device=args.device,
        project=str(args.output.resolve()),
        name="train",
        exist_ok=False,
        plots=True,
    )
    if args.smoke:
        train_args["fraction"] = 0.05
    model.train(**train_args)
    best = args.output / "train" / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"Training produced no best.pt in {best.parent}")
    candidate = args.output / "detector_candidate.pt"
    shutil.copyfile(best, candidate)
    report = dict(
        status="SMOKE_ONLY" if args.smoke else "CANDIDATE_REQUIRES_HOLDOUT_EVALUATION",
        candidate=str(candidate.resolve()),
        candidate_sha256=sha256(candidate),
        initial_sha256=sha256(args.initial),
        dataset_index_sha256=manifest["samples_sha256"],
        dataset_map=map_short_name(manifest.get("map", "")),
        trained_on_demo_map=demo,
        tag_to_class=TAG_TO_NAME,
        labels=stats,
        train_args={k: v for k, v in train_args.items() if k not in {"data", "project"}},
        results_csv=str((args.output / "train" / "results.csv").resolve()),
        next="scripts/benchmark_perception.py --candidate --weights <candidate> on the demo-map"
        " validation split, then the test split once",
    )
    save_json(args.output / "training.json", report)
    print(f"{report['status']} | {candidate} | sha256 {report['candidate_sha256'][:16]}")


if __name__ == "__main__":
    main()
