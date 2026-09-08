"""Object detection and tracking for RGB images."""

import numpy as np
from ultralytics import YOLO

from .types import BoundingBox, Detection, PerceptionConfig


class ObjectDetector:
    """Detect and track road objects using a pretrained YOLO model."""

    def __init__(self, config: PerceptionConfig | None = None):
        self.config = config or PerceptionConfig()
        self.model = YOLO(self.config.model_name)

        self.class_ids = self._get_class_ids()

    def detect(self, image: np.ndarray) -> tuple[Detection, ...]:
        """Detect objects in one RGB image."""

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