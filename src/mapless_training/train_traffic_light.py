"""Train the traffic-light state classifier (``mapless.traffic_light``) on capture runs.

Each capture sample's front image is labelled from the run's truth: ``none`` unless the
car is at a traffic light (``at_traffic_light``: inside the light's trigger zone near the
stop line), else the governing light's state (red, yellow, green; off or unknown read as
none). Train on Town01/03/04/05/06, validate on Town02, refuse Town07 and Town10HD (the
held-out test town and the development benchmark), as for the segmenter.

The autopilot that drove the captures waits at reds, so red samples come in long runs of
near-identical frames: training draws samples with weights that balance the classes.
Selection: the best validation balanced accuracy, with red recall and the share of
none/green frames read as red (false stops) reported beside it.

    python scripts/train_traffic_light.py --run <capture> [--run ...] --output models/light_v1.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from mapless.traffic_light import INPUT_WH, LIGHT_STATES, preprocess, tiny_net
from mapless_training.segmentation_data import REFUSED_TOWNS, TRAIN_TOWNS, VALIDATION_TOWNS, town_of

CARLA_STATES = {"Red": "red", "Yellow": "yellow", "Green": "green"}


def label_of(truth):
    if not truth or not truth.get("at_traffic_light"):
        return "none"
    return CARLA_STATES.get(str(truth.get("traffic_light_state")), "none")


def index(runs, camera="front"):
    """[(image path, label index, town)] and per-run counts; refuses test towns."""
    items, counts = [], {}
    for run in map(Path, runs):
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        town = town_of(manifest)
        if town in REFUSED_TOWNS:
            raise ValueError(f"{run}: {town} may not be trained or validated on")
        per = {s: 0 for s in LIGHT_STATES}
        with open(run / "samples.jsonl", encoding="utf-8") as lines:
            for line in lines:
                row = json.loads(line)
                rgb = (row.get("files") or {}).get("rgb", {}).get(camera)
                if rgb is None:
                    continue
                label = label_of(row.get("truth"))
                per[label] += 1
                items.append((run / rgb, LIGHT_STATES.index(label), town))
        counts[str(run)] = dict(town=town, **per)
    return items, counts


def scores(confusion):
    """Per-class recall and precision, balanced accuracy (over the classes present),
    red recall, and the share of none/green frames read as red (false stops)."""
    c = np.asarray(confusion, dtype=float)
    rows, cols = c.sum(axis=1), c.sum(axis=0)
    recall = np.divide(np.diag(c), rows, out=np.zeros(len(c)), where=rows > 0)
    precision = np.divide(np.diag(c), cols, out=np.zeros(len(c)), where=cols > 0)
    present = rows > 0
    red, none, green = (LIGHT_STATES.index(s) for s in ("red", "none", "green"))
    go = c[none].sum() + c[green].sum()
    return dict(
        recall={s: round(float(r), 4) for s, r in zip(LIGHT_STATES, recall)},
        precision={s: round(float(p), 4) for s, p in zip(LIGHT_STATES, precision)},
        balanced_accuracy=round(float(recall[present].mean()), 4) if present.any() else 0.0,
        red_recall=round(float(recall[red]), 4),
        false_red_share=round(float((c[none, red] + c[green, red]) / go), 4) if go else 0.0,
        confusion=c.astype(int).tolist(),
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", type=Path, action="append", required=True, help="capture run folder (repeatable)")
    p.add_argument("--output", type=Path, help="checkpoint (.pt); required unless --evaluate")
    p.add_argument("--evaluate", type=Path, default=None, help="only score this checkpoint on the given runs")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--steps-per-epoch", type=int, default=600)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    import cv2  # noqa: PLC0415
    import torch  # noqa: PLC0415

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    items, counts = index(args.run)
    print(json.dumps(dict(runs=counts), indent=1), file=sys.stderr)

    class Data(torch.utils.data.Dataset):
        def __init__(self, rows, augment):
            self.rows, self.augment = rows, augment

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            path, label, _ = self.rows[i]
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            x = preprocess(bgr[:, :, ::-1], cv2)
            if self.augment:
                g = np.random.default_rng()
                x = np.clip(x * g.uniform(0.7, 1.3) + g.uniform(-0.08, 0.08), 0.0, 1.0)  # exposure
                dx, dy = int(g.integers(-12, 13)), int(g.integers(-8, 9))  # a small shift
                x = np.roll(x, (dy, dx), axis=(1, 2))
            return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), label

    def evaluate(model, rows):
        loader = torch.utils.data.DataLoader(Data(rows, False), batch_size=args.batch, num_workers=args.workers)
        confusion = np.zeros((len(LIGHT_STATES), len(LIGHT_STATES)), dtype=int)
        model.eval()
        with torch.no_grad():
            for x, y in loader:
                pred = model(x.to(device)).argmax(dim=1).cpu().numpy()
                for t, q in zip(y.numpy(), pred):
                    confusion[t, q] += 1
        return scores(confusion)

    if args.evaluate is not None:
        payload = torch.load(args.evaluate, map_location="cpu", weights_only=False)
        model = tiny_net(torch).to(device)
        model.load_state_dict(payload["state_dict"])
        print(json.dumps(evaluate(model, items), indent=1))
        return
    if args.output is None:
        p.error("--output is required to train")
    train = [r for r in items if r[2] in TRAIN_TOWNS]
    validation = [r for r in items if r[2] in VALIDATION_TOWNS]
    if not train:
        p.error("no training-town samples among the runs")
    labels = np.array([r[1] for r in train])
    freq = np.bincount(labels, minlength=len(LIGHT_STATES)).astype(float)
    weights = 1.0 / np.maximum(freq[labels], 1.0)
    sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=args.steps_per_epoch * args.batch, replacement=True)
    loader = torch.utils.data.DataLoader(Data(train, True), batch_size=args.batch, sampler=sampler, num_workers=args.workers, drop_last=True, persistent_workers=args.workers > 0)
    model = tiny_net(torch).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=args.lr, total_steps=args.epochs * args.steps_per_epoch)
    loss_fn = torch.nn.CrossEntropyLoss()
    best, history = None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        begun, total = time.time(), 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimiser.step()
            schedule.step()
            total += loss.item()
        record = dict(epoch=epoch, loss=round(total / args.steps_per_epoch, 4), seconds=round(time.time() - begun, 1))
        if validation:
            record["validation"] = evaluate(model, validation)
        history.append(record)
        print(json.dumps({k: v for k, v in record.items() if k != "validation"} | ({"balanced": record["validation"]["balanced_accuracy"], "red_recall": record["validation"]["red_recall"], "false_red": record["validation"]["false_red_share"]} if validation else {})), file=sys.stderr, flush=True)
        key = record["validation"]["balanced_accuracy"] if validation else -record["loss"]
        if best is None or key > best[0]:
            best = (key, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    provenance = dict(
        states=list(LIGHT_STATES), input_wh=list(INPUT_WH), architecture="tiny_net", training_towns=sorted({r[2] for r in train}),
        validation_towns=sorted({r[2] for r in validation}), refused_towns=list(REFUSED_TOWNS), runs=counts,
        samples=dict(train=len(train), validation=len(validation)), epochs=args.epochs, batch=args.batch, lr=args.lr, seed=args.seed,
        best_epoch=best[1], selection="validation balanced accuracy" if validation else "training loss", history=history,
        created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=best[2], provenance=provenance), args.output)
    args.output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=1), encoding="utf-8")
    print(f"saved {args.output} (epoch {best[1]})", file=sys.stderr)


if __name__ == "__main__":
    main()
