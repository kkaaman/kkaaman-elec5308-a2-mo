"""Train the road segmenter on capture runs (plan v2.1, section 9). G2 framework.

``--architecture bisenetv2`` (A1 network, from scratch, auxiliary heads at 0.4) or
``segformer_b0`` (ImageNet-pretrained MiT-B0 encoder from ``transformers``, new head),
``len(ROAD_CLASSES)`` classes, cross entropy with inverse-frequency class weights, Adam
(a lower rate for the pretrained encoder), a few epochs. Selection is by validation boundary quality and class recall, not
mIoU alone (section 9): the report records per-class IoU and recall, the share of
kerb/pavement boundary pixels within 2 px of the truth boundary, and the runtime per
image on this machine.

The checkpoint ``{"state_dict", "provenance", "input_size"}`` is what
``mapless.road_segmentation.TorchRoadSegmenter`` loads; the provenance names the runs
and towns (Town07/Town10HD refused by the dataset index, unaccepted captures refused
unless ``--allow-unaccepted``), the seed, and which epoch was saved. The main checkpoint
is the epoch with the best validation mIoU (loss breaks ties); the final epoch is kept
beside it as ``*_last.pt``. Without a validation split there is nothing to select on, so
the file is named ``*_smoke.pt`` and its provenance says ``validated: false``. A
``.provenance.json`` sidecar goes beside each checkpoint.

The learning rate decays per step (cosine to 5%; ``--schedule constant`` keeps it fixed:
road_seg_b0_v2's marking class died in its last constant-rate epoch). ``--evaluate
<checkpoint>`` only scores checkpoints on the validation town with today's tag table, so an
old and a retrained checkpoint compare on the same labels.

Usage:
    python scripts/train_mapless_segmentation.py --run <capture run> [--run ...] \
        --output models/mapless_road_seg.pt --epochs 8 --batch 8 --seed 0
    python scripts/train_mapless_segmentation.py --run <capture run> [--run ...] \
        --evaluate models/road_seg_b0.pt --evaluate models/road_seg_b0_v2.pt

Status: the loop is written and exercised on a two-sample synthetic run in the unit
tests (one epoch, CPU); the first real training run on the Town03/04/05 captures is the
G2 debug pass.
"""

import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mapless.road_segmentation import (
    ARCHITECTURES,
    ROAD_CLASSES,
    TorchRoadSegmenter,
    build_model,
    label_mapping,
    logits,
    write_provenance_sidecar,
)
from mapless_training.segmentation_data import (
    TorchSegmentationDataset,
    class_weights,
    index_runs,
    provenance,
    split,
)


def confusion(pred, truth, n):
    return np.bincount(n * truth.ravel().astype(int) + pred.ravel().astype(int), minlength=n * n).reshape(n, n)


def boundary_agreement(pred, truth, class_index, tolerance_px=2):
    """Share of truth boundary pixels of ``class_index`` within ``tolerance_px`` of a predicted boundary."""
    import cv2  # noqa: PLC0415

    kernel = np.ones((3, 3), np.uint8)

    def edges(mask):
        m = mask.astype(np.uint8)
        return cv2.morphologyEx(m, cv2.MORPH_GRADIENT, kernel) > 0

    t_edge, p_edge = edges(truth == class_index), edges(pred == class_index)
    if not t_edge.any():
        return None
    dilated = cv2.dilate(p_edge.astype(np.uint8), np.ones((2 * tolerance_px + 1,) * 2, np.uint8)) > 0
    return float((t_edge & dilated).sum() / t_edge.sum())


def boundary_precision(pred, truth, class_index, tolerance_px=2):
    """Share of predicted ``class_index`` boundary pixels within ``tolerance_px`` of a truth
    boundary (None when nothing is predicted): ``boundary_agreement`` is its recall."""
    import cv2  # noqa: PLC0415

    kernel = np.ones((3, 3), np.uint8)
    t_edge = cv2.morphologyEx((truth == class_index).astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    p_edge = cv2.morphologyEx((pred == class_index).astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    if not p_edge.any():
        return None
    dilated = cv2.dilate(t_edge.astype(np.uint8), np.ones((2 * tolerance_px + 1,) * 2, np.uint8)) > 0
    return float((p_edge & dilated).sum() / p_edge.sum())


def evaluate(model, loader, device, torch, architecture="bisenetv2", warmup=2):
    """Per-class IoU, recall and precision, pavement boundary recall and precision (within
    2 px) and ms/image. CUDA kernels run
    asynchronously, so the timing synchronises before and after each forward pass and
    skips ``warmup`` untimed passes first (allocator and cuDNN autotuning); CPU needs neither."""
    n = len(ROAD_CLASSES)
    total = np.zeros((n, n), dtype=np.int64)
    boundary, boundary_p, times = [], [], []
    cuda = str(device).startswith("cuda")

    def sync():
        if cuda:
            torch.cuda.synchronize(device)

    model.eval()
    warmed = not cuda
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            if not warmed:
                for _ in range(warmup):
                    logits(model, x, architecture)
                warmed = True
            sync()
            begun = time.perf_counter()
            main, _ = logits(model, x, architecture)
            sync()
            times.append((time.perf_counter() - begun) / len(x))
            pred = main.argmax(dim=1).cpu().numpy()
            truth = y.numpy()
            total += confusion(pred, truth, n)
            for p, t in zip(pred, truth):
                b = boundary_agreement(p, t, ROAD_CLASSES.index("pavement"))
                if b is not None:
                    boundary.append(b)
                bp = boundary_precision(p, t, ROAD_CLASSES.index("pavement"))
                if bp is not None:
                    boundary_p.append(bp)
    tp = np.diag(total)
    iou = tp / np.maximum(total.sum(0) + total.sum(1) - tp, 1)
    recall = tp / np.maximum(total.sum(1), 1)
    precision = tp / np.maximum(total.sum(0), 1)
    return dict(
        iou={k: float(v) for k, v in zip(ROAD_CLASSES, iou)},
        recall={k: float(v) for k, v in zip(ROAD_CLASSES, recall)},
        precision={k: float(v) for k, v in zip(ROAD_CLASSES, precision)},
        miou=float(iou.mean()),
        pavement_boundary_within_2px=float(np.mean(boundary)) if boundary else None,  # recall
        pavement_boundary_precision_2px=float(np.mean(boundary_p)) if boundary_p else None,
        ms_per_image=float(1000 * np.mean(times)) if times else None,
        timing=dict(device=str(device), synchronised=cuda, warmup_passes=warmup if cuda else 0, batches=len(times)),
    )


def seed_everything(seed, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def better(candidate, best):
    """Selection by validation mIoU, loss as the tie-break (and the whole criterion when
    there is no validation split)."""
    if best is None:
        return True
    if candidate["miou"] is not None and best["miou"] is not None and candidate["miou"] != best["miou"]:
        return candidate["miou"] > best["miou"]
    return candidate["loss"] < best["loss"]


SCHEDULES = ("constant", "cosine")


def train(run_dirs, output, epochs=8, batch=8, size_wh=(512, 256), lr=1e-3, device=None, cameras=None, max_samples=None, workers=0, architecture="bisenetv2", pretrained=True, seed=0, allow_unaccepted=False, progress_every=200, schedule="cosine"):
    import cv2  # noqa: PLC0415
    import torch  # noqa: PLC0415

    if architecture not in ARCHITECTURES:
        raise ValueError(f"architecture must be one of {ARCHITECTURES}")
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}")
    seed_everything(seed, torch)
    samples, manifests = index_runs(run_dirs, cameras, allow_unaccepted=allow_unaccepted)
    train_samples, validation_samples, other = split(samples)
    if max_samples:
        train_samples, validation_samples = train_samples[:max_samples], validation_samples[: max(1, max_samples // 4)]
    if not train_samples:
        raise ValueError("no training samples (training towns: Town01, 03, 04, 05, 06)")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    weights = torch.tensor(class_weights(train_samples, size_wh, cv2, seed=seed), dtype=torch.float32, device=device)
    model = build_model(architecture, len(ROAD_CLASSES), pretrained=pretrained and architecture == "segformer_b0").to(device)
    if architecture == "segformer_b0":
        # The pretrained encoder learns slowly, the new head at the full rate.
        encoder = [p for n, p in model.named_parameters() if n.startswith("segformer.")]
        head = [p for n, p in model.named_parameters() if not n.startswith("segformer.")]
        optimiser = torch.optim.AdamW([dict(params=encoder, lr=lr / 10), dict(params=head, lr=lr)], weight_decay=0.01)
    else:
        optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    shuffle_rng = torch.Generator().manual_seed(seed)
    # drop_last: BiSeNetV2's context embedding batch-normalises a 1x1 map, which a final
    # batch of one image cannot (fresh-eyes review of 58adaf3; it raises in training mode).
    # Persistent workers: with Windows spawn, each epoch's fresh workers re-import the main
    # script, and a checkout of another branch in the same clone mid-run killed epoch 6 of
    # the BiSeNetV2 run (the script was gone); workers started once stay alive.
    persistent = dict(persistent_workers=workers > 0)
    loader = torch.utils.data.DataLoader(TorchSegmentationDataset(train_samples, size_wh), batch_size=batch, shuffle=True, num_workers=workers, generator=shuffle_rng, drop_last=True, **persistent)
    val_loader = (
        torch.utils.data.DataLoader(TorchSegmentationDataset(validation_samples, size_wh), batch_size=batch, num_workers=workers, **persistent)
        if validation_samples
        else None
    )
    validated = val_loader is not None
    # Cosine decay per step to 5% of each group's rate. At a constant rate (road_seg_b0_v2,
    # 27 Oct) the loss spiked in epochs 4 and 7 and in epoch 8 the marking class died
    # (IoU 0.71 at epoch 6, 0.0 at epoch 8); selection kept epoch 6.
    total_steps = max(1, epochs * len(loader))
    scheduler = None
    if schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda k: 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(k, total_steps) / total_steps)))
    history, best = [], None

    def common(history):
        return dict(
            created=datetime.now(timezone.utc).isoformat(),
            epochs=epochs,
            batch=batch,
            learning_rate=lr,
            schedule=schedule,
            seed=seed,
            device=device,
            towns_left_out=other,
            history=list(history),
            architecture=architecture,
            pretrained_encoder=bool(pretrained and architecture == "segformer_b0"),
            allow_unaccepted=allow_unaccepted,
            validated=validated,
            smoke=not validated,
        )

    for epoch in range(epochs):
        model.train()
        losses = []
        started = time.perf_counter()
        for step, (x, y) in enumerate(loader, 1):
            x, y = x.to(device), y.to(device)
            main, aux = logits(model, x, architecture)
            loss = loss_fn(main, y) + 0.4 * sum(loss_fn(a, y) for a in aux)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            if scheduler is not None:
                scheduler.step()
            losses.append(float(loss.detach()))
            if progress_every and step % progress_every == 0:
                # A full epoch over 48k images is 20 minutes or more; the console says so.
                elapsed = time.perf_counter() - started
                print(
                    json.dumps(dict(epoch=epoch + 1, step=step, of=len(loader), loss=round(float(np.mean(losses[-progress_every:])), 4), images_per_s=round(step * batch / elapsed, 1), eta_min=round((len(loader) - step) * elapsed / step / 60, 1))),
                    flush=True,
                )
        row = dict(epoch=epoch + 1, loss=float(np.mean(losses)), seconds=round(time.perf_counter() - started, 1))
        if validated:
            row["validation"] = evaluate(model, val_loader, device, torch, architecture)
        candidate = dict(epoch=epoch + 1, loss=row["loss"], miou=row["validation"]["miou"] if validated else None)
        improved = better(candidate, best)
        if improved:
            # A CPU copy, so the chosen epoch survives the ones after it.
            best = dict(candidate, state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        history.append(row)
        print(json.dumps(row), flush=True)
        if improved and epoch + 1 < epochs:
            # The best so far is on disk after every improving epoch (marked in progress), so
            # a run that dies keeps its best checkpoint (the BiSeNetV2 run lost an hour).
            save_checkpoints(output, best, None, epoch + 1, epochs, validated, manifests, train_samples, validation_samples, size_wh, common(history), torch, in_progress=True)
    output, info = save_checkpoints(output, best, model, epochs, epochs, validated, manifests, train_samples, validation_samples, size_wh, common(history), torch)
    return output, info


def save_checkpoints(output, best, model, epochs_done, epochs, validated, manifests, train_samples, validation_samples, size_wh, common, torch, in_progress=False):
    """Write the selected checkpoint (and, at the end of a validated run, the last epoch's)."""
    output = Path(output)
    if not validated and "smoke" not in output.stem:
        # Nothing was validated, so the file says so: the car should not take it for a trained model.
        output = output.with_name(f"{output.stem}_smoke{output.suffix}")
    selection = dict(
        validated=validated,
        criterion="validation mIoU, loss as tie-break" if validated else "training loss (no validation split: smoke checkpoint)",
        best_epoch=best["epoch"],
        best_miou=best["miou"],
        best_loss=best["loss"],
        last_epoch=epochs_done,
        in_progress=in_progress,
    )
    # The last epoch is kept beside the selected one (for resuming and the report) when a
    # validation split did the selecting; a smoke checkpoint is its own last epoch.
    last = output.with_name(f"{output.stem}_last{output.suffix}") if validated and model is not None else None
    info = provenance(manifests, train_samples, validation_samples, size_wh, extra=dict(common, selection=dict(selection, saved="best"), epoch=best["epoch"], last_checkpoint=str(last) if last else None))
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=best["state_dict"], provenance=info, input_size=list(size_wh)), output)
    write_provenance_sidecar(output, info)
    if last is not None:
        last_info = provenance(manifests, train_samples, validation_samples, size_wh, extra=dict(common, selection=dict(selection, saved="last"), epoch=epochs, last_checkpoint=None))
        torch.save(dict(state_dict=model.state_dict(), provenance=last_info, input_size=list(size_wh)), last)
        write_provenance_sidecar(last, last_info)
    return output, info


def evaluate_checkpoints(run_dirs, checkpoints, batch=16, workers=0, device=None, cameras=None, allow_unaccepted=False, max_samples=None):
    """Score checkpoints on the validation town of ``run_dirs`` with today's labels.

    Each checkpoint is scored against the current ``CARLA_TAG_TO_CLASS``, whatever mapping it
    was trained on, so an old and a retrained checkpoint compare on the same labels. Its
    own mapping, and whether it matches, are reported beside the scores.
    """
    import torch  # noqa: PLC0415

    samples, _ = index_runs(run_dirs, cameras, allow_unaccepted=allow_unaccepted)
    _, validation_samples, other = split(samples)
    if max_samples:
        validation_samples = validation_samples[:max_samples]
    if not validation_samples:
        raise ValueError("no validation samples (the validation town is Town02)")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    current = label_mapping()
    results = []
    for checkpoint in checkpoints:
        segmenter = TorchRoadSegmenter(checkpoint, device=device, allow_label_mismatch=True)
        size = segmenter.input_size
        loader = torch.utils.data.DataLoader(TorchSegmentationDataset(validation_samples, size), batch_size=batch, num_workers=workers)
        scores = evaluate(segmenter.model, loader, device, torch, segmenter.architecture)
        trained_on = segmenter.provenance.get("label_mapping")
        results.append(
            dict(
                checkpoint=str(checkpoint),
                architecture=segmenter.architecture,
                epoch=segmenter.provenance.get("epoch"),
                trained_label_mapping=trained_on,
                label_mapping_matches=trained_on == current,
                validation_samples=len(validation_samples),
                towns_left_out=other,
                validation=scores,
            )
        )
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", type=Path, action="append", required=True, help="Capture run folder (repeatable)")
    parser.add_argument("--output", type=Path, help="Checkpoint path (.pt); required unless --evaluate")
    parser.add_argument("--evaluate", type=Path, action="append", default=None, help="Only score this checkpoint on the validation town with today's labels (repeatable); no training")
    parser.add_argument("--schedule", choices=SCHEDULES, default="cosine", help="Learning rate: cosine decay per step to 5%% (default) or constant")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=16, help="Images per step (16 fits the RTX 4000 Ada at 512x256 with room to spare)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--camera", action="append", default=None, help="Only these cameras (default: all)")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=6, help="Data-loading processes (0: load on the main thread; the dataset pickles for Windows spawn)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="bisenetv2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-unaccepted", action="store_true", help="Smoke runs only: train on captures the inspection did not accept")
    parser.add_argument("--progress-every", type=int, default=200, help="Print a progress line every this many steps (0: none)")
    args = parser.parse_args(argv)
    if args.evaluate:
        for row in evaluate_checkpoints(args.run, args.evaluate, args.batch, args.workers, args.device, args.camera, args.allow_unaccepted, args.max_samples):
            print(json.dumps(row, default=str), flush=True)
        return
    if args.output is None:
        parser.error("--output is required to train")
    path, info = train(
        args.run, args.output, args.epochs, args.batch, (args.width, args.height), args.lr, args.device, args.camera, args.max_samples, args.workers, args.architecture,
        seed=args.seed, allow_unaccepted=args.allow_unaccepted, progress_every=args.progress_every, schedule=args.schedule,
    )
    print(json.dumps(dict(checkpoint=str(path), towns=info["training_towns"], samples=info["samples"], selection=info["selection"], last=info["history"][-1] if info["history"] else None), default=str))


if __name__ == "__main__":
    main()
