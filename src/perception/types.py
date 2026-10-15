"""Public types for object detection and tracking."""

import math
from dataclasses import dataclass
from typing import Optional

from obstacle_perception.types import PixelBox

ROAD_RELEVANT_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
)


@dataclass(frozen=True)
class BoundingBox(PixelBox):
    """Legacy constructor backed by the canonical validated XYXY contract."""

    @property
    def width_px(self):
        return self.x_max - self.x_min

    @property
    def height_px(self):
        return self.y_max - self.y_min


@dataclass(frozen=True)
class Detection:
    """One detected or tracked object."""

    class_name: str
    confidence: float
    bounding_box: BoundingBox
    track_id: Optional[int] = None

    def __post_init__(self):
        from obstacle_perception.boxes import canonical_box

        object.__setattr__(self, "bounding_box", canonical_box(self.bounding_box))


@dataclass(frozen=True)
class PerceptionConfig:
    """Settings for object detection and tracking."""

    model_name: str = "yolo26n.pt"
    confidence_threshold: float = 0.25
    image_size: int = 640
    allowed_classes: Optional[tuple[str, ...]] = ROAD_RELEVANT_CLASSES
    tracker_config: str = "botsort.yaml"

    def __post_init__(self):
        for name in ("model_name", "tracker_config"):

            if (
                not isinstance(getattr(self, name), str)
                or not getattr(self, name).strip()
            ):
                raise ValueError(f"{name} must be a nonempty path/name")
        if (
            isinstance(self.confidence_threshold, bool)
            or not math.isfinite(self.confidence_threshold)
            or not 0 <= self.confidence_threshold <= 1
        ):
            raise ValueError("confidence_threshold must be finite and in [0, 1]")
        if type(self.image_size) is not int or self.image_size <= 0:
            raise ValueError("image_size must be a positive integer")
        if self.allowed_classes is not None:
            if isinstance(self.allowed_classes, str):
                raise ValueError("allowed_classes must be a sequence, not a string")
            values = tuple(self.allowed_classes)

            if not values or any(
                not isinstance(v, str) or not v.strip() for v in values
            ):

                raise ValueError("allowed_classes must be nonempty class names or None")
            object.__setattr__(self, "allowed_classes", values)
