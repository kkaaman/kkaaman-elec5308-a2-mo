"""Adapter from the preserved A1 RGB detector API to exact-frame contracts."""

import time

import numpy as np

from obstacle_perception.types import (
    CallTiming,
    DetectionBatch,
    DetectionObservation,
    ObservationMetadata,
)


class FrameDetector:
    def __init__(self, backend, *, model_version, source_id="front-rgb", tracking=True):
        ObservationMetadata("validate", 0, 0.0, source_id, model_version)
        self.backend = backend
        self.model_version, self.source_id, self.tracking = model_version, source_id, tracking
        self.episode_id = None

    def reset(self, *, episode_id):
        self.backend.reset(episode_id=episode_id)
        self.episode_id = episode_id

    def detect(self, frame):
        if frame.episode_id != self.episode_id:
            raise ValueError("Detector episode mismatch")
        items = self.backend.track(frame.rgb) if self.tracking else self.backend.detect(frame.rgb)
        return DetectionBatch(
            ObservationMetadata(
                frame.episode_id,
                frame.frame_id,
                frame.sim_time_s,
                self.source_id,
                self.model_version,
            ),
            tuple(
                DetectionObservation(d.class_name, d.confidence, d.bounding_box, d.track_id)
                for d in items
            ),
            frame.rgb.shape[1],
            frame.rgb.shape[0],
            self.backend.last_call_timing or CallTiming(0),
        )

    def warmup(self):
        """Empty-image tracker initialization, recorded outside loop timings."""
        start = time.perf_counter()
        for _ in range(3):
            self.backend.track(np.zeros((360, 640, 3), dtype=np.uint8))
        return time.perf_counter() - start

    def close(self):
        self.backend.close()


class EmptyDetector:
    """Explicit navigation-only diagnostic; never used for obstacle acceptance."""

    def reset(self, *, episode_id):
        self.episode_id = episode_id

    def detect(self, frame):
        return DetectionBatch(
            ObservationMetadata(
                frame.episode_id, frame.frame_id, frame.sim_time_s, "navigation-only", "no-detector"
            ),
            (),
            frame.rgb.shape[1],
            frame.rgb.shape[0],
            CallTiming(0),
        )

    def close(self):
        self.episode_id = None
