"""Alternative segmentation models behind the :class:`SemanticSegmenter` interface.

Every segmenter takes an HxWx3 uint8 RGB frame and returns an HxW uint8 mask of
Cityscapes trainIds (0-18) at the input resolution, with 255 for pixels the
model does not label. That lets BL-B-05 compare them on identical frames and
lets the rest of the stack switch backend by configuration only.

Backends:

``bisenetv2``
    The A1 model (:class:`SemanticSegmenter`), 19 Cityscapes classes.
``segformer_b0``
    SegFormer-B0 fine-tuned on Cityscapes, loaded with Hugging Face
    ``transformers``. Same 19 classes in the same order.
``yolo_seg``
    Ultralytics instance segmentation trained on COCO (``yolo26n-seg.pt`` by
    default, ``yolov8n-seg.pt`` also works). Only object classes are labelled:
    person, bicycle, car, motorcycle, bus, train, truck, traffic light and stop
    sign (as traffic sign); everything else is 255. It also exposes
    :meth:`YoloSegSegmenter.predict_instances`, one mask per detected object,
    which is what depth-ROI masking needs.

Heavy libraries are imported only when a backend is constructed, so importing
this module does not require ``transformers`` or ``ultralytics``.
"""

from __future__ import annotations

from typing import Any, Protocol

import cv2
import numpy as np

from .labels import CITYSCAPES_CLASSES, IGNORE_ID
from .postprocessing import colorize_mask
from .preprocessing import preprocess_image
from .segmentation import SemanticSegmenter

BACKENDS: tuple[str, ...] = ("bisenetv2", "segformer_b0", "yolo_seg")

DEFAULT_WEIGHTS = {
    "bisenetv2": "segmentation_model.pth",
    "segformer_b0": "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    "yolo_seg": "yolo26n-seg.pt",
}

#: COCO class name -> Cityscapes trainId for the classes YOLO-seg can label.
COCO_TO_TRAINID: dict[str, int] = {
    "person": 11,
    "bicycle": 18,
    "car": 13,
    "motorcycle": 17,
    "bus": 15,
    "train": 16,
    "truck": 14,
    "traffic light": 6,
    "stop sign": 7,
}


class Segmenter(Protocol):
    """Interface shared by every backend."""

    name: str
    device: str
    labels_all_classes: bool

    def predict(self, image: np.ndarray) -> np.ndarray: ...

    def visualize(self, image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray: ...


def _auto_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def overlay_labelled(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend class colours over ``image`` only where ``mask`` is labelled.

    Unlike :func:`overlay_mask`, unlabelled (255) pixels keep their original
    colour, so objects-only masks stay readable.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    colours = colorize_mask(mask)
    blended = cv2.addWeighted(image, 1 - alpha, colours, alpha, 0)
    labelled = (mask != IGNORE_ID)[..., None]
    return np.where(labelled, blended, image).astype(np.uint8)


class SegFormerSegmenter:
    """SegFormer-B0 (Cityscapes) with the :class:`SemanticSegmenter` interface.

    Args:
        weights: Hugging Face model id or a local directory saved with
            ``save_pretrained``. Downloaded once, then read from the local cache.
        input_size: (width, height) the frame is resized to before inference.
        device: "cuda", "mps" or "cpu". Auto-detected if not given.
        local_files_only: Never touch the network (for offline runs).
    """

    name = "segformer_b0"
    labels_all_classes = True

    def __init__(
        self,
        weights: str = DEFAULT_WEIGHTS["segformer_b0"],
        input_size: tuple[int, int] = (1024, 512),
        device: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        import torch  # noqa: F401  (fail early with a clear ImportError)
        from transformers import SegformerForSemanticSegmentation

        self.weights = weights
        self.input_size = input_size
        self.device = device or _auto_device()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            weights, local_files_only=local_files_only
        )
        self.model.eval().to(self.device)
        config = self.model.config
        self.num_classes = int(config.num_labels)
        labels = tuple(str(config.id2label[i]).lower() for i in range(self.num_classes))
        if labels != CITYSCAPES_CLASSES:
            raise ValueError(
                f"{weights} is not a 19-class Cityscapes model (labels: {', '.join(labels)})"
            )

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Return an HxW uint8 trainId mask at the input resolution."""
        import torch
        import torch.nn.functional as F

        SemanticSegmenter._validate_input(image)
        height, width = image.shape[:2]
        tensor = preprocess_image(image, self.input_size).to(self.device)
        with torch.no_grad():
            logits = self.model(pixel_values=tensor).logits  # (1, 19, h/4, w/4)
            logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
            mask = logits.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()
        return mask

    def visualize(self, image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        return overlay_labelled(image, mask, alpha)


class YoloSegSegmenter:
    """Ultralytics instance segmentation mapped to Cityscapes trainIds.

    Args:
        weights: Ultralytics segmentation weights, e.g. ``yolo26n-seg.pt`` or
            ``yolov8n-seg.pt`` (downloaded on first use if not present).
        confidence: Minimum detection confidence.
        image_size: Ultralytics ``imgsz`` (letterbox size).
        device: "cuda", "mps" or "cpu". Auto-detected if not given.
    """

    name = "yolo_seg"
    labels_all_classes = False

    def __init__(
        self,
        weights: str = DEFAULT_WEIGHTS["yolo_seg"],
        confidence: float = 0.25,
        image_size: int = 640,
        device: str | None = None,
    ) -> None:
        from ultralytics import YOLO

        self.weights = weights
        self.confidence = confidence
        self.image_size = image_size
        self.device = device or _auto_device()
        self.model = YOLO(weights)
        if getattr(self.model, "task", "segment") != "segment":
            raise ValueError(f"{weights} is a '{self.model.task}' model, not a segmentation model")
        self.names: dict[int, str] = dict(self.model.names)
        self.class_map = {i: COCO_TO_TRAINID[n] for i, n in self.names.items() if n in COCO_TO_TRAINID}
        if not self.class_map:
            raise ValueError(f"{weights} has none of the expected COCO classes")

    def _ultralytics_device(self) -> str:
        return "0" if self.device == "cuda" else self.device

    def predict_instances(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Detect objects and return one full-resolution boolean mask per object.

        Returns:
            List of dicts with ``label``, ``class_id`` (COCO), ``train_id``,
            ``confidence``, ``box_xyxy`` (pixels) and ``mask`` (HxW bool).
        """
        SemanticSegmenter._validate_input(image)
        height, width = image.shape[:2]
        image_bgr = image[:, :, ::-1].copy()  # Ultralytics treats numpy input as BGR
        result = self.model.predict(
            image_bgr,
            conf=self.confidence,
            imgsz=self.image_size,
            classes=sorted(self.class_map),
            retina_masks=True,  # masks at the original image resolution
            device=self._ultralytics_device(),
            verbose=False,
        )[0]
        if result.masks is None or result.boxes is None or len(result.boxes) == 0:
            return []

        masks = result.masks.data.cpu().numpy() > 0
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        instances = []
        for mask, class_id, confidence, box in zip(masks, classes, confidences, boxes):
            if mask.shape != (height, width):
                mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
            instances.append(
                {
                    "label": self.names[int(class_id)],
                    "class_id": int(class_id),
                    "train_id": self.class_map[int(class_id)],
                    "confidence": float(confidence),
                    "box_xyxy": tuple(float(v) for v in box),
                    "mask": mask,
                }
            )
        return instances

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Return an HxW uint8 trainId mask; pixels with no object are 255."""
        height, width = image.shape[:2]
        mask = np.full((height, width), IGNORE_ID, dtype=np.uint8)
        # Paint low-confidence objects first so overlaps go to the most confident one.
        for instance in sorted(self.predict_instances(image), key=lambda d: d["confidence"]):
            mask[instance["mask"]] = instance["train_id"]
        return mask

    def visualize(self, image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        return overlay_labelled(image, mask, alpha)


def create_segmenter(backend: str, weights: str | None = None, **kwargs: Any) -> Segmenter:
    """Build a segmenter by name.

    Args:
        backend: One of :data:`BACKENDS`.
        weights: Weights path or model id; defaults to :data:`DEFAULT_WEIGHTS`.
        **kwargs: Passed to the backend constructor (for example ``device``,
            ``input_size``, ``confidence``, ``image_size``).

    Example::

        segmenter = create_segmenter("segformer_b0", device="cuda")
        mask = segmenter.predict(rgb_frame)  # HxW Cityscapes trainIds
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown segmentation backend '{backend}'; choose from {', '.join(BACKENDS)}")
    weights = weights or DEFAULT_WEIGHTS[backend]
    if backend == "bisenetv2":
        segmenter = SemanticSegmenter(weights, **kwargs)
        segmenter.name = "bisenetv2"
        segmenter.weights = weights
        segmenter.labels_all_classes = True
        return segmenter
    if backend == "segformer_b0":
        return SegFormerSegmenter(weights, **kwargs)
    return YoloSegSegmenter(weights, **kwargs)
