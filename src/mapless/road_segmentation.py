"""Road-surface segmentation for the mapless stack (plan v2.1, section 9). G2 framework.

A ``RoadSegmenter`` turns one camera image into per-pixel class probabilities over
``ROAD_CLASSES``: road, lane marking, pavement (kerb and sidewalk), road user, other.
The painted LiDAR (``mapless.painting``) and the road model (``mapless.road_model``) read
these; nothing else in the car does.

``TorchRoadSegmenter`` runs the A1 BiSeNetV2 architecture (``segmentation.bisenetv2``)
with a head of ``len(ROAD_CLASSES)`` classes, trained by
``scripts/train_mapless_segmentation.py`` on the capture labels. The checkpoint carries
its provenance (training towns, capture runs, label mapping, suite hash) and the
training side's attestation that no test-only town was used (the car's code names no
town, so the refusal of held-out towns lives in ``mapless_training``); loading refuses
a checkpoint without that attestation or with other classes, and the loader never
opens a label folder (section 12: the deployed loader cannot see labels).

``ConstantSegmenter`` is the stand-in for tests and for driving without a model: every
pixel below the horizon is road with low confidence, so the stack runs end to end and
the LiDAR terrain alone decides.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROAD_CLASSES = ("road", "marking", "pavement", "road_user", "other")

# CARLA 0.9.16 semantic tags -> ROAD_CLASSES index (labels/semantic/<camera>/<frame>.png).
CARLA_TAG_TO_CLASS = {
    1: 0,  # Roads
    24: 1,  # RoadLine
    2: 2,  # Sidewalks
    8: 2,  # Ground (curb-adjacent ground)
    12: 3,  # Pedestrians
    13: 3,  # Rider
    14: 3,  # Car
    15: 3,  # Truck
    16: 3,  # Bus
    17: 3,  # Train
    18: 3,  # Motorcycle
    19: 3,  # Bicycle
}


@dataclass(frozen=True)
class Segmentation:
    """Class probabilities for one image: H x W x C float32, C = len(ROAD_CLASSES)."""

    camera: str
    probabilities: np.ndarray

    def __post_init__(self):
        p = np.asarray(self.probabilities, dtype=np.float32)
        if p.ndim != 3 or p.shape[2] != len(ROAD_CLASSES) or not np.isfinite(p).all():
            raise ValueError("probabilities must be H x W x len(ROAD_CLASSES), finite")
        object.__setattr__(self, "probabilities", p)

    @property
    def labels(self):
        return self.probabilities.argmax(axis=2).astype(np.uint8)

    def probability(self, name):
        return self.probabilities[:, :, ROAD_CLASSES.index(name)]


def carla_tags_to_classes(tags):
    """Capture label image (uint8 CARLA tags) -> ROAD_CLASSES indices (uint8)."""
    table = np.full(256, ROAD_CLASSES.index("other"), dtype=np.uint8)
    for tag, index in CARLA_TAG_TO_CLASS.items():
        table[tag] = index
    return table[np.asarray(tags, dtype=np.uint8)]


class RoadSegmenter:
    """Interface: ``segment(image) -> Segmentation``."""

    def segment(self, image):
        raise NotImplementedError

    @property
    def provenance(self):
        return {}


class ConstantSegmenter(RoadSegmenter):
    """Everything below ``horizon_fraction`` of the image is road at ``confidence``."""

    def __init__(self, horizon_fraction=0.55, confidence=0.4):
        self.horizon_fraction, self.confidence = horizon_fraction, confidence

    def segment(self, image):
        h, w = image.rgb.shape[:2]
        p = np.zeros((h, w, len(ROAD_CLASSES)), dtype=np.float32)
        other = ROAD_CLASSES.index("other")
        p[:, :, other] = 1.0
        row = int(h * self.horizon_fraction)
        p[row:, :, ROAD_CLASSES.index("road")] = self.confidence
        p[row:, :, other] = 1.0 - self.confidence
        return Segmentation(image.name, p)

    @property
    def provenance(self):
        return dict(model="constant", note="no learned model: road assumed below the horizon")


def check_provenance(provenance):
    """Refuse a checkpoint without the training side's attestation that no held-out town
    was used, or with other classes; return the provenance otherwise."""
    if provenance.get("test_towns_excluded") is not True:
        raise ValueError("checkpoint provenance does not attest that held-out towns were excluded: refused")
    if "classes" in provenance and tuple(provenance["classes"]) != ROAD_CLASSES:
        raise ValueError("checkpoint classes do not match ROAD_CLASSES")
    for key in ("training_runs", "training_towns", "label_mapping", "created"):
        if key not in provenance:
            raise ValueError(f"checkpoint provenance lacks {key!r}")
    return provenance


class TorchRoadSegmenter(RoadSegmenter):
    """BiSeNetV2 with a ``len(ROAD_CLASSES)``-class head, from a checkpoint with provenance.

    The checkpoint is a dict ``{"state_dict": ..., "provenance": {...}, "input_size":
    [w, h]}`` written by the training script. Inference resizes to ``input_size``,
    normalises as the A1 preprocessing does, and resizes the softmax back to the image.
    """

    def __init__(self, checkpoint_path, device=None):
        import torch  # noqa: PLC0415

        from segmentation.bisenetv2 import BiSeNetV2  # noqa: PLC0415

        path = Path(checkpoint_path)
        if path.is_dir() or "labels" in path.parts:
            raise ValueError("a checkpoint is a file outside any labels folder")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "state_dict" not in payload or "provenance" not in payload:
            raise ValueError("checkpoint must hold state_dict and provenance")
        self._provenance = check_provenance(dict(payload["provenance"]))
        self.input_size = tuple(int(v) for v in payload.get("input_size", (512, 256)))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BiSeNetV2(num_classes=len(ROAD_CLASSES))
        self.model.load_state_dict(payload["state_dict"])
        self.model.to(self.device).eval()
        self._torch = torch

    @property
    def provenance(self):
        return dict(self._provenance)

    def segment(self, image):
        import cv2  # noqa: PLC0415

        from segmentation.preprocessing import preprocess_image  # noqa: PLC0415

        torch = self._torch
        h, w = image.rgb.shape[:2]
        tensor = preprocess_image(np.ascontiguousarray(image.rgb), self.input_size).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].permute(1, 2, 0).cpu().numpy()
        probs = cv2.resize(probs, (w, h), interpolation=cv2.INTER_LINEAR)
        probs = probs / np.maximum(probs.sum(axis=2, keepdims=True), 1e-6)
        return Segmentation(image.name, probs.astype(np.float32))


def load_segmenter(spec, device=None):
    """``None`` or ``"constant"`` -> ConstantSegmenter; a path -> TorchRoadSegmenter."""
    if spec in (None, "constant"):
        return ConstantSegmenter()
    return TorchRoadSegmenter(spec, device)


def write_provenance_sidecar(checkpoint_path, provenance):
    """A JSON copy of the provenance beside the checkpoint, for the report."""
    path = Path(checkpoint_path).with_suffix(".provenance.json")
    path.write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")
    return path
