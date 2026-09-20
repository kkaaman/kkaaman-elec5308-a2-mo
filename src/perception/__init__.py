"""Object detection and tracking."""

from .object_detection import ObjectDetector
from .types import BoundingBox, Detection, PerceptionConfig

__all__ = [
    "BoundingBox",
    "Detection",
    "ObjectDetector",
    "PerceptionConfig",
]