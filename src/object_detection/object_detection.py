"""Object detection and tracking for RGB images."""
# FIXED (Mohammad): detect()/track() indexed the input directly with no
# validation, so a wrong-shape array (e.g. grayscale) or non-array input
# failed with an obscure low-level indexing error instead of a clear one.
# Added _validate_rgb_image(), called at the top of both methods.
import numpy as np
from ultralytics import YOLO

from .types import BoundingBox, Detection, PerceptionConfig


def _validate_rgb_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (height, width, 3)")


class ObjectDetector:
    """Detect and track road objects using a pretrained YOLO model."""

    def __init__(self, config: PerceptionConfig | None = None):
        self.config = config or PerceptionConfig()
        self.model = YOLO(self.config.model_name)
        self.class_ids = self._get_class_ids()

    def detect(self, image: np.ndarray) -> tuple[Detection, ...]:
        """Detect objects in one RGB image."""
        _validate_rgb_image(image)  # FIXED (Mohammad)
        image_bgr = image[:, :, ::-1]
        results = self.model.predict(
            image_bgr,
            conf=self.config.confidence_threshold,
            imgsz=self.config.image_size,
            classes=self.class_ids,
            verbose=False,
        )
        return self._convert(results[0], include_track_ids=False)

    def track(self, frame: np.ndarray) -> tuple[Detection, ...]:
        """Detect and track objects across consecutive RGB frames."""
        _validate_rgb_image(frame)  # FIXED (Mohammad)
        frame_bgr = frame[:, :, ::-1]
        results = self.model.track(
            frame_bgr,
            persist=True,
            tracker="botsort.yaml",
            conf=self.config.confidence_threshold,
            imgsz=self.config.image_size,
            classes=self.class_ids,
            verbose=False,
        )
        return self._convert(results[0], include_track_ids=True)

    def _get_class_ids(self):
        """Find the model IDs for the classes we care about."""
        if self.config.allowed_classes is None:
            return None
        return [
            class_id
            for class_id, name in self.model.names.items()
            if name in self.config.allowed_classes
        ]

    def _convert(
        self,
        result,
        include_track_ids: bool,
    ) -> tuple[Detection, ...]:
        """Convert YOLO results into our Detection objects."""
        detections = []
        if result.boxes is None:
            return ()
        for box in result.boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])
            track_id = None
            if include_track_ids and box.id is not None:
                track_id = int(box.id[0])
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=confidence,
                    bounding_box=BoundingBox(
                        x_min=x_min,
                        y_min=y_min,
                        x_max=x_max,
                        y_max=y_max,
                    ),
                    track_id=track_id,
                )
            )
        return tuple(detections)
