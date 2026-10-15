"""Object detection and tracking for RGB images."""

import time
from pathlib import Path

import numpy as np
import ultralytics
from ultralytics import YOLO

from obstacle_perception.types import CallTiming

from .types import BoundingBox, Detection, PerceptionConfig


def _resolve_model_path(name):
    path = Path(name).expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Local model weights missing or empty: {path}. No download attempted."
        )
    with path.open("rb") as stream:
        if stream.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError("Model weights are a Git LFS pointer, not a downloaded asset")
    if path.suffix.lower() != ".pt":
        raise ValueError("Baseline detector requires local .pt weights")
    return str(path)


def _resolve_tracker(name):
    if name in ("botsort.yaml", "bytetrack.yaml"):
        path = Path(ultralytics.__file__).resolve().parent / "cfg" / "trackers" / name
    else:
        path = Path(name).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Local tracker config missing: {path}")
    import yaml

    values = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(values, dict) or values.get("tracker_type") not in (
        "botsort",
        "bytetrack",
    ):
        raise ValueError("Tracker config must select botsort or bytetrack")
    if values.get("with_reid", False):
        raise ValueError("ReID is disabled in the offline baseline")
    return str(path)


class ObjectDetector:
    """Detect and track road objects using a pretrained YOLO model."""

    def __init__(self, config: PerceptionConfig | None = None):
        """Create a detector.

        Args:
            config: Optional detection and tracking settings. When omitted, the
                default :class:`PerceptionConfig` is used.
        """
        self.config = config or PerceptionConfig()
        self._model_path = _resolve_model_path(self.config.model_name)
        self._tracker = _resolve_tracker(self.config.tracker_config)
        self._factory = YOLO
        self.model = self._factory(self._model_path)
        self.episode_id = None
        self._closed = False
        self.last_call_timing = None
        self.class_ids = self._get_class_ids()

    def reset(self, *, episode_id: str):
        """Reload the local backend to discard all predictor/tracker history.

        Reset is intentionally more expensive than a frame call. It avoids
        depending on version-specific private Ultralytics tracker attributes.
        """
        if self._closed:
            raise RuntimeError("Detector is closed")
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("episode_id must be nonempty")
        old_model = self.model
        self.model = None
        self.episode_id = None
        self.last_call_timing = None
        new_model = self._factory(_resolve_model_path(self._model_path))
        if new_model is old_model:
            raise RuntimeError("Model factory reused the old backend; history isolation failed")
        self.model = new_model
        self.class_ids = self._get_class_ids()
        self.episode_id = episode_id

    def close(self):
        """Release this instance's backend reference; repeated close is harmless."""
        self._closed = True
        self.model = None
        self.episode_id = None
        self.last_call_timing = None

    def detect(self, image: np.ndarray) -> tuple[Detection, ...]:
        """Detect one RGB image, preserving the A1 tuple return API."""
        return self._run(image, tracking=False)

    def track(self, frame: np.ndarray) -> tuple[Detection, ...]:
        """Track consecutive RGB images; call reset between episodes."""
        return self._run(frame, tracking=True)

    def _run(self, image, tracking):
        started = time.perf_counter()
        inference_wall_s = None
        try:
            if self.model is None:
                raise RuntimeError("Detector backend unavailable or closed")
            self._validate_rgb_image(image, "frame" if tracking else "image")
            bgr = image[:, :, ::-1].copy()
            options = dict(
                conf=self.config.confidence_threshold,
                imgsz=self.config.image_size,
                classes=self.class_ids,
                verbose=False,
            )
            inference_start = time.perf_counter()
            try:
                if tracking:
                    results = self.model.track(bgr, persist=True, tracker=self._tracker, **options)
                else:
                    results = self.model.predict(bgr, **options)
            finally:
                inference_wall_s = time.perf_counter() - inference_start
            if len(results) != 1:
                raise ValueError("Expected one inference result for one image")
            return self._convert(results[0], include_track_ids=tracking)
        finally:
            self.last_call_timing = CallTiming(time.perf_counter() - started, inference_wall_s)

    @staticmethod
    def _validate_rgb_image(value: np.ndarray, parameter_name: str) -> None:
        """Validate an RGB image before it is passed to YOLO."""

        if not isinstance(value, np.ndarray):
            raise TypeError(f"{parameter_name} must be a NumPy array")

        if value.ndim != 3 or value.shape[2] != 3:
            raise ValueError(f"{parameter_name} must have shape (H, W, 3); got {value.shape}")

        if value.dtype != np.uint8 or 0 in value.shape:
            raise ValueError(f"{parameter_name} must be nonempty uint8 RGB")

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
