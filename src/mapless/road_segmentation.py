"""Road-surface segmentation for the mapless stack (plan v2.1, section 9). G2 framework.

A ``RoadSegmenter`` turns one camera image into per-pixel class probabilities over
``ROAD_CLASSES``: road, lane marking, pavement (kerb and sidewalk), road user, other.
The painted LiDAR (``mapless.painting``) and the road model (``mapless.road_model``) read
these; nothing else in the car does.

``TorchRoadSegmenter`` runs one of two architectures with a head of ``len(ROAD_CLASSES)``
classes, chosen by the checkpoint's ``architecture``: the A1 BiSeNetV2
(``segmentation.bisenetv2``, trained from scratch, fastest, sharp boundaries from its
detail branch) or SegFormer-B0 (``transformers``, ImageNet-pretrained encoder, 3.7 M
parameters, converges with little data; logits at 1/4 resolution, upsampled). The two
are compared on the captures by boundary quality, class recall and runtime (plan,
section 9), not mIoU alone. Both are trained by
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
# 0.9.16 numbers its tags the Cityscapes way (evaluation.capture_inspection.TAG_NAMES):
# 8 is TrafficSign and 25 is Ground. The first table mapped 8 ("Ground", the pre-0.9.13
# numbering) to pavement, so sign panels trained as pavement and kerb-side ground as
# "other" (review of 3e9083a, 27 Oct); road_seg_b0.pt was trained on that table.
CARLA_TAG_TO_CLASS = {
    1: 0,  # Roads
    24: 1,  # RoadLine
    2: 2,  # Sidewalks
    25: 2,  # Ground (kerb-side ground at pavement level)
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
    """Class probabilities for one image: H x W x C float32, C = len(ROAD_CLASSES). They
    may be at a lower resolution than the image (the model's own): each cell covers the
    same share of the image, and ``paint`` maps pixel coordinates onto it."""

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


def label_mapping():
    """Today's tag table as checkpoints record it: {"<CARLA tag>": "<class name>"}."""
    return {str(k): ROAD_CLASSES[v] for k, v in CARLA_TAG_TO_CLASS.items()}


def check_provenance(provenance, allow_label_mismatch=False):
    """Refuse a checkpoint without the training side's attestation that no held-out town
    was used, with other classes, or trained on another tag table (unless
    ``allow_label_mismatch``, which is recorded in the provenance); return the provenance."""
    if provenance.get("test_towns_excluded") is not True:
        raise ValueError("checkpoint provenance does not attest that held-out towns were excluded: refused")
    if "classes" in provenance and tuple(provenance["classes"]) != ROAD_CLASSES:
        raise ValueError("checkpoint classes do not match ROAD_CLASSES")
    for key in ("training_runs", "training_towns", "label_mapping", "created"):
        if key not in provenance:
            raise ValueError(f"checkpoint provenance lacks {key!r}")
    if provenance["label_mapping"] != label_mapping():
        # road_seg_b0.pt learnt sign panels as pavement and kerb-side ground as "other"
        # (tag 8 read as Ground); the car must not run it by accident (external review).
        if not allow_label_mismatch:
            raise ValueError("checkpoint was trained on another tag table (label_mapping differs): refused; retrain, or pass allow_label_mismatch for an experiment")
        provenance = dict(provenance, label_mismatch_override=True)
    return provenance


ARCHITECTURES = ("bisenetv2", "segformer_b0")
SEGFORMER_PRETRAINED = "nvidia/mit-b0"  # ImageNet-1k encoder; the head is new


def build_model(architecture, num_classes=len(ROAD_CLASSES), pretrained=False):
    """A fresh model of ``architecture`` with ``num_classes`` outputs (training builds it
    ``pretrained`` where an encoder is available; inference loads a state dict into it)."""
    if architecture == "bisenetv2":
        from segmentation.bisenetv2 import BiSeNetV2  # noqa: PLC0415

        return BiSeNetV2(num_classes=num_classes)
    if architecture == "segformer_b0":
        from transformers import SegformerForSemanticSegmentation  # noqa: PLC0415

        if pretrained:
            return SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_PRETRAINED, num_labels=num_classes, ignore_mismatched_sizes=True)
        return SegformerForSemanticSegmentation(_segformer_b0_config(num_classes))
    raise ValueError(f"unknown architecture {architecture!r}; one of {ARCHITECTURES}")


def _segformer_b0_config(num_classes):
    """MiT-B0 hyper-parameters, so a checkpoint loads with no network access."""
    from transformers import SegformerConfig  # noqa: PLC0415

    return SegformerConfig(
        num_labels=num_classes,
        depths=[2, 2, 2, 2],
        hidden_sizes=[32, 64, 160, 256],
        decoder_hidden_size=256,
        num_attention_heads=[1, 2, 5, 8],
    )


def logits(model, x, architecture):
    """Main logits (N x C x H x W at the input size) and auxiliary logits for training."""
    import torch  # noqa: PLC0415

    if architecture == "bisenetv2":
        if model.training:
            main, *aux = model.forward_with_aux(x)
            return main, list(aux)
        return model(x), []
    out = model(pixel_values=x).logits
    return torch.nn.functional.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False), []


class TorchRoadSegmenter(RoadSegmenter):
    """BiSeNetV2 or SegFormer-B0 with a ``len(ROAD_CLASSES)``-class head, from a checkpoint
    with provenance.

    The checkpoint is a dict ``{"state_dict": ..., "provenance": {..., "architecture":
    ...}, "input_size": [w, h]}`` written by the training script. Inference resizes to
    ``input_size``, normalises as the A1 preprocessing does, and resizes the softmax back
    to the image.
    """

    def __init__(self, checkpoint_path, device=None, allow_label_mismatch=False, full_resolution=True):
        import torch  # noqa: PLC0415

        # ``full_resolution=False``: the probabilities at the model's input size, not
        # resized to the image (fifth review, F5-05: the resize, normalisation and checks
        # at full camera resolution were most of a tick's painting on the CPU side).
        self.full_resolution = bool(full_resolution)

        path = Path(checkpoint_path)
        if path.is_dir() or "labels" in path.parts:
            raise ValueError("a checkpoint is a file outside any labels folder")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "state_dict" not in payload or "provenance" not in payload:
            raise ValueError("checkpoint must hold state_dict and provenance")
        self._provenance = check_provenance(dict(payload["provenance"]), allow_label_mismatch=allow_label_mismatch)
        self.architecture = self._provenance.get("architecture", "bisenetv2")
        self.input_size = tuple(int(v) for v in payload.get("input_size", (512, 256)))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(self.architecture, len(ROAD_CLASSES))
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
            main, _ = logits(self.model, tensor, self.architecture)
            probs = torch.softmax(main, dim=1)[0].permute(1, 2, 0).cpu().numpy()
        if not self.full_resolution:
            return Segmentation(image.name, np.ascontiguousarray(probs, dtype=np.float32))
        probs = cv2.resize(probs, (w, h), interpolation=cv2.INTER_LINEAR)
        probs = probs / np.maximum(probs.sum(axis=2, keepdims=True), 1e-6)
        return Segmentation(image.name, probs.astype(np.float32))


def load_segmenter(spec, device=None, full_resolution=True):
    """``None`` or ``"constant"`` -> ConstantSegmenter; a path -> TorchRoadSegmenter."""
    if spec in (None, "constant"):
        return ConstantSegmenter()
    return TorchRoadSegmenter(spec, device, full_resolution=full_resolution)


def write_provenance_sidecar(checkpoint_path, provenance):
    """A JSON copy of the provenance beside the checkpoint, for the report."""
    path = Path(checkpoint_path).with_suffix(".provenance.json")
    path.write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")
    return path
