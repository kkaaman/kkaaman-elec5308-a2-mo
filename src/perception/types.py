"""Public types for object detection and tracking."""

from dataclasses import dataclass
from typing import Optional

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
class BoundingBox:
    """Object bounding box in image pixels."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True)
class Detection:
    """One detected or tracked object."""

    class_name: str
    confidence: float
    bounding_box: BoundingBox
    track_id: Optional[int] = None


@dataclass(frozen=True)
class PerceptionConfig:
    """Settings for object detection and tracking."""

    model_name: str = "yolo26n.pt"
    confidence_threshold: float = 0.25
    image_size: int = 640
    allowed_classes: Optional[tuple[str, ...]] = ROAD_RELEVANT_CLASSES