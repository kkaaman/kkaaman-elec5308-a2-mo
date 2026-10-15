from dataclasses import dataclass
from enum import Enum

from obstacle_perception.types import PixelBox

from ._validation import finite, probability


class TrafficCategory(str, Enum):
    """Supported traffic-object categories."""

    SIGN = "sign"
    LIGHT = "light"


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
class TrafficDetection:
    """Traffic object detected in an RGB camera frame."""

    category: TrafficCategory
    label: str
    confidence: float
    bounding_box: BoundingBox
    distance_m: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.category, TrafficCategory):
            raise TypeError("category must be a TrafficCategory")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")

        probability("confidence", self.confidence)
        from obstacle_perception.boxes import canonical_box

        object.__setattr__(self, "bounding_box", canonical_box(self.bounding_box))

        if self.distance_m is not None:
            finite("distance_m", self.distance_m)
            if self.distance_m <= 0:
                raise ValueError("distance_m must be positive")
