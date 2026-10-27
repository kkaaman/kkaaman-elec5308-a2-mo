"""Capture runs as a road-segmentation dataset (plan v2.1, sections 9 and 12). G2 framework.

Reads the run folders written by ``scripts/capture_mapless_dataset.py``: ``samples.jsonl``
indexes RGB images and their CARLA semantic label images per camera. Labels are mapped
to ``mapless.road_segmentation.ROAD_CLASSES`` with ``carla_tags_to_classes``.

Splits are by whole run and by town, never by neighbouring frames (section 12):
- train: Town01, 03, 04, 05, 06; validation: Town02; Town07 is the held-out final test
  town and Town10HD the development benchmark — a run from either is refused.
A run is also refused unless its capture inspection accepted it (integrity and geometry
PASS; the manifest's COMPLETE status stands in when no inspection has run), except with
``allow_unaccepted=True`` for smoke runs. ``provenance()`` records the runs, towns, suite hashes and label mapping for the
checkpoint, which the deployed loader checks before using the weights.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from evaluation.dataset import map_short_name
from mapless.road_segmentation import ROAD_CLASSES, carla_tags_to_classes, label_mapping

TRAIN_TOWNS = ("Town01", "Town03", "Town04", "Town05", "Town06")
VALIDATION_TOWNS = ("Town02",)
REFUSED_TOWNS = ("Town07", "Town10HD")


def town_of(manifest):
    name = map_short_name(manifest.get("map", ""))
    return name.replace("_Opt", "")


@dataclass(frozen=True)
class Sample:
    run: Path
    frame: int
    camera: str
    rgb: Path
    label: Path
    town: str


def acceptance_of(run, manifest):
    """The capture inspection's verdict (``<run>/inspection/summary.json``, written by
    ``evaluation.capture_inspection``) when it ran; otherwise the manifest's status alone,
    which is all a run has before inspection. ``accepted`` needs integrity and geometry PASS."""
    summary = Path(run) / "inspection" / "summary.json"
    if summary.exists():
        verdict = json.loads(summary.read_text(encoding="utf-8")).get("acceptance") or {}
        integrity, geometry = verdict.get("integrity") or {}, verdict.get("geometry") or {}
        accepted = integrity.get("status") == "PASS" and geometry.get("status") == "PASS"
        return dict(
            source="inspection",
            status=verdict.get("status"),
            integrity=integrity.get("status"),
            geometry=geometry.get("status"),
            accepted=accepted,
            reasons=list(verdict.get("reasons") or (["inspection summary holds no verdict"] if not verdict else [])),
        )
    status = manifest.get("status")
    complete = status == "COMPLETE"
    return dict(source="manifest", status=status, integrity=None, geometry=None, accepted=complete, reasons=[] if complete else [f"run status {status}"])


def index_runs(run_dirs, cameras=None, allow_unaccepted=False):
    """Samples from each run; refuses runs from test-only towns and runs the inspection (or,
    without one, the manifest) does not accept. ``allow_unaccepted`` is for smoke runs only;
    the acceptance is recorded per run either way so the provenance shows it."""
    samples, manifests = [], {}
    for run in map(Path, run_dirs):
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        town = town_of(manifest)
        if town in REFUSED_TOWNS:
            raise ValueError(f"{run}: {town} is a test-only town and may not be trained on")
        acceptance = acceptance_of(run, manifest)
        if not acceptance["accepted"] and not allow_unaccepted:
            raise ValueError(f"{run}: not accepted for training by its {acceptance['source']} ({acceptance['status']}: {'; '.join(acceptance['reasons']) or 'integrity/geometry not PASS'}); pass allow_unaccepted=True only for a smoke run")
        manifests[str(run)] = dict(town=town, suite_sha256=manifest.get("suite_sha256"), seed=manifest.get("seed"), status=manifest.get("status"), acceptance=acceptance)
        with open(run / "samples.jsonl", encoding="utf-8") as lines:
            for line in lines:
                row = json.loads(line)
                for camera, rgb in row["files"]["rgb"].items():
                    if cameras is not None and camera not in cameras:
                        continue
                    label = row["files"]["semantic"].get(camera)
                    if label is None:
                        continue
                    samples.append(Sample(run, int(row["frame"]), camera, run / rgb, run / label, town))
    return samples, manifests


def split(samples):
    """(train, validation) by town; anything else is left out and reported."""
    train = [s for s in samples if s.town in TRAIN_TOWNS]
    validation = [s for s in samples if s.town in VALIDATION_TOWNS]
    other = sorted({s.town for s in samples} - set(TRAIN_TOWNS) - set(VALIDATION_TOWNS))
    return train, validation, other


def load_pair(sample, size_wh, cv2):
    """(image uint8 HxWx3 RGB, labels uint8 HxW over ROAD_CLASSES) resized to ``size_wh``."""
    bgr = cv2.imread(str(sample.rgb), cv2.IMREAD_COLOR)
    tags = cv2.imread(str(sample.label), cv2.IMREAD_UNCHANGED)
    if bgr is None or tags is None:
        raise FileNotFoundError(f"{sample.rgb} / {sample.label}")
    if tags.ndim == 3:
        tags = tags[:, :, 2] if tags.shape[2] >= 3 else tags[:, :, 0]  # CARLA tags in the R channel of a BGR(A) PNG
    rgb = cv2.resize(bgr[:, :, ::-1], size_wh, interpolation=cv2.INTER_LINEAR)
    labels = cv2.resize(carla_tags_to_classes(tags), size_wh, interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray(rgb), labels


def weight_samples(samples, limit=200, seed=0):
    """Up to ``limit`` samples taken evenly across (run, camera) groups, round robin over a
    seeded shuffle of each group, so one long capture or one camera does not set the
    frequencies alone (the first ``limit`` samples of the index would)."""
    groups = {}
    for sample in samples:
        groups.setdefault((str(sample.run), sample.camera), []).append(sample)
    rng = np.random.default_rng(seed)
    queues = [list(rng.permutation(len(g))) for g in groups.values()]
    pools = list(groups.values())
    chosen = []
    while len(chosen) < min(limit, len(samples)):
        for pool, queue in zip(pools, queues):
            if queue and len(chosen) < limit:
                chosen.append(pool[queue.pop()])
    return chosen


def class_weights(samples, size_wh, cv2, limit=200, seed=0):
    """Inverse-frequency class weights from up to ``limit`` label images, stratified by run and camera."""
    counts = np.zeros(len(ROAD_CLASSES))
    for sample in weight_samples(samples, limit, seed):
        _, labels = load_pair(sample, size_wh, cv2)
        counts += np.bincount(labels.ravel(), minlength=len(ROAD_CLASSES))[: len(ROAD_CLASSES)]
    freq = counts / max(counts.sum(), 1)
    weights = 1.0 / np.maximum(freq, 1e-3)
    return weights / weights.mean()


def provenance(manifests, train, validation, size_wh, extra=None):
    towns = sorted({m["town"] for m in manifests.values()})
    return dict(
        classes=list(ROAD_CLASSES),
        label_mapping=label_mapping(),
        training_runs=sorted(str(s.run) for s in {s.run: s for s in train}.values()),
        training_towns=sorted({s.town for s in train}),
        validation_towns=sorted({s.town for s in validation}),
        # The car's loader checks this flag; the towns themselves are refused in index_runs.
        test_towns_excluded=not any(t in REFUSED_TOWNS for t in {s.town for s in train} | {s.town for s in validation}),
        runs=manifests,
        towns=towns,
        input_size=list(size_wh),
        samples=dict(train=len(train), validation=len(validation)),
        **(extra or {}),
    )


class TorchSegmentationDataset:
    """A ``torch.utils.data.Dataset`` over samples (imported lazily so numpy-only use works)."""

    def __init__(self, samples, size_wh):
        # Nothing but data on the instance: DataLoader workers on Windows pickle the dataset
        # (spawn), and a module object does not pickle (fresh-eyes review of 58adaf3).
        self.samples, self.size_wh = list(samples), tuple(size_wh)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        import cv2  # noqa: PLC0415
        import torch  # noqa: PLC0415

        from segmentation.preprocessing import preprocess_image  # noqa: PLC0415

        rgb, labels = load_pair(self.samples[i], self.size_wh, cv2)
        x = preprocess_image(rgb, self.size_wh)[0]
        return x, torch.from_numpy(labels.astype(np.int64))
