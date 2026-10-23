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
and towns (Town07/Town10HD refused by the dataset index) so the deployed loader can
check it. A ``.provenance.json`` sidecar goes beside it.

Usage:
    python scripts/train_mapless_segmentation.py --run <capture run> [--run ...] \
        --output models/mapless_road_seg.pt --epochs 8 --batch 8

Status: the loop is written and exercised on a two-sample synthetic run in the unit
tests (one epoch, CPU); the first real training run on the Town03/04/05 captures is the
G2 debug pass.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mapless.road_segmentation import (
    ARCHITECTURES,
    ROAD_CLASSES,
    build_model,
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


def evaluate(model, loader, device, torch, architecture="bisenetv2"):
    n = len(ROAD_CLASSES)
    total = np.zeros((n, n), dtype=np.int64)
    boundary, times = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            begun = time.perf_counter()
            main, _ = logits(model, x.to(device), architecture)
            times.append((time.perf_counter() - begun) / len(x))
            pred = main.argmax(dim=1).cpu().numpy()
            truth = y.numpy()
            total += confusion(pred, truth, n)
            for p, t in zip(pred, truth):
                b = boundary_agreement(p, t, ROAD_CLASSES.index("pavement"))
                if b is not None:
                    boundary.append(b)
    tp = np.diag(total)
    iou = tp / np.maximum(total.sum(0) + total.sum(1) - tp, 1)
    recall = tp / np.maximum(total.sum(1), 1)
    return dict(
        iou={k: float(v) for k, v in zip(ROAD_CLASSES, iou)},
        recall={k: float(v) for k, v in zip(ROAD_CLASSES, recall)},
        miou=float(iou.mean()),
        pavement_boundary_within_2px=float(np.mean(boundary)) if boundary else None,
        ms_per_image=float(1000 * np.mean(times)) if times else None,
    )


def train(run_dirs, output, epochs=8, batch=8, size_wh=(512, 256), lr=1e-3, device=None, cameras=None, max_samples=None, workers=0, architecture="bisenetv2", pretrained=True):
    import cv2  # noqa: PLC0415
    import torch  # noqa: PLC0415

    if architecture not in ARCHITECTURES:
        raise ValueError(f"architecture must be one of {ARCHITECTURES}")
    samples, manifests = index_runs(run_dirs, cameras)
    train_samples, validation_samples, other = split(samples)
    if max_samples:
        train_samples, validation_samples = train_samples[:max_samples], validation_samples[: max(1, max_samples // 4)]
    if not train_samples:
        raise ValueError("no training samples (training towns: Town01, 03, 04, 05, 06)")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    weights = torch.tensor(class_weights(train_samples, size_wh, cv2), dtype=torch.float32, device=device)
    model = build_model(architecture, len(ROAD_CLASSES), pretrained=pretrained and architecture == "segformer_b0").to(device)
    if architecture == "segformer_b0":
        # The pretrained encoder learns slowly, the new head at the full rate.
        encoder = [p for n, p in model.named_parameters() if n.startswith("segformer.")]
        head = [p for n, p in model.named_parameters() if not n.startswith("segformer.")]
        optimiser = torch.optim.AdamW([dict(params=encoder, lr=lr / 10), dict(params=head, lr=lr)], weight_decay=0.01)
    else:
        optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    loader = torch.utils.data.DataLoader(TorchSegmentationDataset(train_samples, size_wh), batch_size=batch, shuffle=True, num_workers=workers)
    val_loader = (
        torch.utils.data.DataLoader(TorchSegmentationDataset(validation_samples, size_wh), batch_size=batch, num_workers=workers)
        if validation_samples
        else None
    )
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            main, aux = logits(model, x, architecture)
            loss = loss_fn(main, y) + 0.4 * sum(loss_fn(a, y) for a in aux)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append(float(loss.detach()))
        row = dict(epoch=epoch + 1, loss=float(np.mean(losses)))
        if val_loader is not None:
            row["validation"] = evaluate(model, val_loader, device, torch, architecture)
        history.append(row)
        print(json.dumps(row), flush=True)
    info = provenance(
        manifests,
        train_samples,
        validation_samples,
        size_wh,
        extra=dict(
            created=datetime.now(timezone.utc).isoformat(),
            epochs=epochs,
            batch=batch,
            learning_rate=lr,
            device=device,
            towns_left_out=other,
            history=history,
            architecture=architecture,
            pretrained_encoder=bool(pretrained and architecture == "segformer_b0"),
        ),
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), provenance=info, input_size=list(size_wh)), output)
    write_provenance_sidecar(output, info)
    return output, info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", type=Path, action="append", required=True, help="Capture run folder (repeatable)")
    parser.add_argument("--output", type=Path, required=True, help="Checkpoint path (.pt)")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--camera", action="append", default=None, help="Only these cameras (default: all)")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="bisenetv2")
    args = parser.parse_args(argv)
    path, info = train(args.run, args.output, args.epochs, args.batch, (args.width, args.height), args.lr, args.device, args.camera, args.max_samples, args.workers, args.architecture)
    print(json.dumps(dict(checkpoint=str(path), towns=info["training_towns"], samples=info["samples"], last=info["history"][-1] if info["history"] else None), default=str))


if __name__ == "__main__":
    main()
