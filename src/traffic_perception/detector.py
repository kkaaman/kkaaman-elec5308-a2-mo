from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from ._validation import positive, probability, rgb_image
from .distance import estimate_distance_m
from .types import BoundingBox, TrafficCategory, TrafficDetection


@dataclass(frozen=True)
class ModelPrediction:
    """Normalised prediction returned by a detection backend."""

    category: TrafficCategory
    label: str
    confidence: float
    bounding_box: BoundingBox

    def __post_init__(self) -> None:
        probability("confidence", self.confidence)
        from obstacle_perception.boxes import canonical_box

        object.__setattr__(self, "bounding_box", canonical_box(self.bounding_box))


DetectionBackend = Callable[[np.ndarray], Iterable[ModelPrediction]]


class TrafficPerception:
    """Detect traffic-relevant objects from simulator-independent RGB images."""

    def __init__(
        self,
        backend: DetectionBackend,
        confidence_threshold: float = 0.5,
        focal_length_px: float | None = None,
        object_widths_m: dict[str, float] | None = None,
    ) -> None:
        """Create a traffic perception pipeline.

        Args:
            backend: Callable that takes an RGB image and returns an iterable of
                ModelPrediction instances.
            confidence_threshold: Minimum confidence required for a prediction to
                be retained. Defaults to 0.5.
            focal_length_px: Camera focal length in pixels used for monocular
                distance estimation. If None, distance estimation is skipped.
            object_widths_m: Known real-world object widths in metres, indexed by
                label. Labels without a known width will not receive a distance
                estimate.
        """
        if not callable(backend):
            raise TypeError("backend must be callable")

        probability("confidence_threshold", confidence_threshold)

        if focal_length_px is not None:
            positive("focal_length_px", focal_length_px)

        widths = dict(object_widths_m or {})
        for label, width_m in widths.items():
            if not isinstance(label, str) or not label.strip():
                raise ValueError("object-width labels must be non-empty strings")
            positive(f"object_widths_m[{label!r}]", width_m)

        self._backend = backend
        self._confidence_threshold = confidence_threshold
        self._focal_length_px = focal_length_px
        self._object_widths_m = widths

    def detect(self, image: np.ndarray) -> tuple[TrafficDetection, ...]:
        """Detect and classify traffic objects in an RGB image."""
        rgb_image(image)

        detections = []

        for prediction in self._backend(image):
            if not isinstance(prediction, ModelPrediction):
                raise TypeError("backend must return ModelPrediction instances")

            if prediction.confidence < self._confidence_threshold:
                continue

            detections.append(
                TrafficDetection(
                    category=prediction.category,
                    label=prediction.label,
                    confidence=prediction.confidence,
                    bounding_box=prediction.bounding_box,
                    distance_m=self._estimate_distance(prediction),
                )
            )

        return tuple(detections)

    def _estimate_distance(
        self,
        prediction: ModelPrediction,
    ) -> float | None:
        if self._focal_length_px is None:
            return None

        real_width_m = self._object_widths_m.get(prediction.label)
        if real_width_m is None:
            return None

        return estimate_distance_m(
            prediction.bounding_box.width_px,
            real_width_m,
            self._focal_length_px,
        )
