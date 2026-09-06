from dataclasses import dataclass
from enum import Enum

from ._validation import finite, probability


class TrafficCategory(str, Enum):
    """Supported traffic-object categories."""

    SIGN = "sign"
    LIGHT = "light"


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in image pixel coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        for name in ("x_min", "y_min", "x_max", "y_max"):
            finite(name, getattr(self, name))

        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min")
        if self.y_max <= self.y_min:
            raise ValueError("y_max must be greater than y_min")

    @property
    def width_px(self) -> float:
        """Return bounding-box width in pixels."""
        return self.x_max - self.x_min

    @property
    def height_px(self) -> float:
        """Return bounding-box height in pixels."""
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

        if self.distance_m is not None:
            finite("distance_m", self.distance_m)
            if self.distance_m <= 0:
                raise ValueError("distance_m must be positive")
