"""Fine-tune the existing LaneUNet; select on validation roads, never test roads."""

import argparse
import csv
import importlib.util
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.dataset import (
    KIT_VERSION,
    confusion,
    local_file,
    map_short_name,
    metrics,
    read_dataset,
    recorded_on_demo_map,
    save_json,
    sha256,
)


def model_class(repo):
    """Load the user's canonical architecture without importing package side effects."""
    source = Path(repo).resolve() / "src" / "perception" / "lane_segmentation.py"
    spec = importlib.util.spec_from_file_location("group7_training_lane_model", source)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import canonical LaneUNet: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LaneUNet, source


def make_dataset(torch, cv2, root, rows, size, augment=False):
    class LaneData(torch.utils.data.Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            row = rows[index]
            with Image.open(local_file(root, row["rgb"])) as source:
                rgb = np.asarray(source.convert("RGB"))
            with Image.open(local_file(root, row["lane"])) as source:
                mask = np.asarray(source)
            rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
            if augment:
                if random.random() < 0.5:
                    rgb, mask = rgb[:, ::-1], mask[:, ::-1]
                rgb = np.clip(rgb.astype(np.float32) * random.uniform(0.85, 1.15), 0, 255).astype(
                    np.uint8
                )
            # Match LaneSegmenter exactly: RGB float in [0,1], no ImageNet normalization.
            x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
            y = torch.from_numpy(mask.astype(np.int64).copy())
            return x, y

    return LaneData()


def evaluate_model(torch, model, loader, device):
    model.eval()
    counts = np.zeros(4, dtype=np.int64)
    with torch.inference_mode():
        for image, label in loader:
            prediction = model(image.to(device)).argmax(1).cpu().numpy()
            for guess, truth in zip(prediction, label.numpy(), strict=True):
                counts += confusion(guess, truth)
    return metrics(counts)


def checkpoint_state(torch, path, device):
    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_size < 1024:
        raise ValueError(f"Real local checkpoint missing: {path}")
    with path.open("rb") as stream:
        if stream.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError("Checkpoint is still a Git LFS pointer")
    return torch.load(path, map_location=device, weights_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hours", type=float, default=4.0)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=5308)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--smoke", action="store_true", help="One small training epoch; not a production candidate"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume last checkpoint in the SAME output directory"
    )
    parser.add_argument(
        "--allow-demo-map",
        action="store_true",
        help="Deliberately fine-tune on images recorded on the demo map (Town10HD)",
    )
    args = parser.parse_args()
    if (
        args.epochs < 1
        or args.batch < 1
        or args.patience < 1
        or not math.isfinite(args.lr)
        or args.lr <= 0
        or not math.isfinite(args.hours)
        or args.hours <= 0
    ):
        parser.error("Invalid training settings")
    import cv2
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Use scripts/setup_batch35.ps1 or explicitly request --device cpu for a smoke test"
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    manifest, rows = read_dataset(args.dataset)
    dataset_map = map_short_name(manifest.get("map", "unknown"))
    on_demo_map = recorded_on_demo_map(manifest)
    if on_demo_map and not args.allow_demo_map and not args.smoke:
        raise ValueError(
            f"Dataset was recorded on the demo map ({dataset_map}). Capture another town"
            " (run_batch35.ps1 -Stage Capture -Map Town05) or pass --allow-demo-map to"
            " train on the demo map deliberately"
        )
    groups = {name: [r for r in rows if r["split"] == name] for name in ("train", "val", "test")}
    for name, group in groups.items():
        minimum = 1 if args.smoke else 50
        if len(group) < minimum or not any(r["lane_pixels"] > 0 for r in group):
            raise ValueError(
                f"Need at least {minimum} {name} images and some visible road-line labels"
            )
    if not args.smoke and len({r["road_id"] for r in groups["train"]}) < 3:
        raise ValueError("Collect at least three distinct training roads")
    if args.smoke:
        groups["train"], groups["val"] = groups["train"][:4], groups["val"][:4]
        args.epochs = 1
    size = tuple(manifest["image_size_wh"])
    if size != (640, 360):
        raise ValueError("Canonical LaneSegmenter uses 640x360 input. Collect with the default rig")
    train_data = make_dataset(torch, cv2, args.dataset, groups["train"], size, augment=True)
    val_data = make_dataset(torch, cv2, args.dataset, groups["val"], size)
    # Single-worker loading avoids Windows multiprocessing/main-module surprises.
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch, num_workers=0)
    lane_class, source = model_class(args.repo)
    model = lane_class(num_classes=2, pretrained=False).to(device)
    model.load_state_dict(checkpoint_state(torch, args.initial, device), strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    identity = {
        "dataset_index_sha256": manifest["samples_sha256"],
        "architecture_sha256": sha256(source),
        "initial_sha256": sha256(args.initial),
        "image_size_wh": list(size),
    }
    best_path = output / "lane_candidate.pth"
    last_path = output / "last_training.pt"
    start_epoch, stale = 0, 0
    baseline = evaluate_model(torch, model, val_loader, device)
    best = baseline["foreground_iou"]
    if best is None:
        raise ValueError("Validation set has no evaluable foreground union")
    if args.resume:
        state = checkpoint_state(torch, last_path, device)
        if state["identity"] != identity or state["batch"] != args.batch or state["lr"] != args.lr:
            raise ValueError("Resume identity/settings differ; use a new output directory")
        if not best_path.exists():
            raise ValueError("Best candidate checkpoint is missing")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, best, stale = state["epoch"], state["best"], state["stale"]
    else:
        torch.save(model.state_dict(), best_path)
    pixels = len(groups["train"]) * size[0] * size[1]
    positive = sum(row["lane_pixels"] for row in groups["train"])
    foreground_weight = min(20.0, max(1.0, math.sqrt((pixels - positive) / max(positive, 1))))
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, foreground_weight], device=device)
    )
    report = {
        "status": "TRAINING",
        "schema": KIT_VERSION,
        "identity": identity,
        "device": str(device),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "baseline_validation": baseline,
        "best_validation_iou": best,
        "smoke_only": args.smoke,
        "dataset_map": dataset_map,
        "trained_on_demo_map": on_demo_map,
        "test_set_used_for_selection": False,
        "promotion": "NOT_DEPLOYED",
        "preprocessing": "640x360 RGB /255; matches canonical LaneSegmenter",
        "counts": {k: len(v) for k, v in groups.items()},
        "foreground_loss_weight": foreground_weight,
    }
    save_json(output / "training.json", report)
    started = time.monotonic()
    completed = start_epoch
    try:
        with (output / "epochs.csv").open(
            "a" if args.resume else "x", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "epoch",
                    "train_loss",
                    "validation_iou",
                    "validation_precision",
                    "validation_recall",
                    "session_wall_s",
                ),
            )
            if not args.resume:
                writer.writeheader()
            for epoch in range(start_epoch, args.epochs):
                model.train()
                total, batches = 0.0, 0
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(images)
                    loss = criterion(logits, labels)
                    foreground = logits.softmax(1)[:, 1]
                    truth = (labels == 1).float()
                    dice = (2 * (foreground * truth).sum() + 1) / (
                        foreground.sum() + truth.sum() + 1
                    )
                    loss = loss + (1 - dice)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    total += loss.item()
                    batches += 1
                current = evaluate_model(torch, model, val_loader, device)
                score = current["foreground_iou"]
                if score is None:
                    raise ValueError("Validation IoU is undefined")
                stale += 1
                if score > best:
                    best, stale = score, 0
                    temporary = output / "lane_candidate.tmp"
                    torch.save(model.state_dict(), temporary)
                    temporary.replace(best_path)
                completed = epoch + 1
                temporary = output / "last_training.tmp"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": completed,
                        "best": best,
                        "stale": stale,
                        "identity": identity,
                        "lr": args.lr,
                        "batch": args.batch,
                    },
                    temporary,
                )
                temporary.replace(last_path)
                writer.writerow(
                    {
                        "epoch": completed,
                        "train_loss": total / batches,
                        "validation_iou": score,
                        "validation_precision": current["precision"],
                        "validation_recall": current["recall"],
                        "session_wall_s": time.monotonic() - started,
                    }
                )
                stream.flush()
                print(
                    f"Epoch {completed}: loss={total / batches:.4f} val IoU={score:.4f} best={best:.4f}",
                    flush=True,
                )
                if stale >= args.patience or time.monotonic() - started >= args.hours * 3600:
                    break
        report["status"] = "SMOKE_COMPLETE" if args.smoke else "CANDIDATE_READY_FOR_EVALUATION"
    except BaseException as error:
        report["status"] = "INTERRUPTED_OR_FAILED"
        report["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report.update(
            epochs_completed=completed,
            best_validation_iou=best,
            session_wall_s=time.monotonic() - started,
            candidate_sha256=sha256(best_path),
            improved_over_initial_validation=best > baseline["foreground_iou"],
        )
        save_json(output / "training.json", report)
    print("Candidate:", best_path)
    print(
        "Use evaluate_lane.py once settings are frozen. Existing project weights were not overwritten."
    )


if __name__ == "__main__":
    main()
