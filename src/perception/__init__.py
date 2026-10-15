"""Object detection and tracking, with lazy optional learned-backend loading."""

from .types import BoundingBox, Detection, PerceptionConfig

__all__ = ["BoundingBox", "Detection", "ObjectDetector", "PerceptionConfig"]


def __getattr__(name):
    if name == "ObjectDetector":
        from .object_detection import ObjectDetector

        return ObjectDetector
    raise AttributeError(name)
