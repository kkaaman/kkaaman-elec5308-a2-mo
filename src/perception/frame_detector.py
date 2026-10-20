"""Frame contract adapter preserving image detector detect/track separation."""

import time

from obstacle_perception.types import (
    CallTiming,
    DetectionBatch,
    DetectionObservation,
    ObservationMetadata,
)


class FrameDetector:
    def __init__(self, backend, *, source_id, model_version):
        self.backend = backend
        self.source_id, self.model_version = source_id, model_version

    @property
    def generation(self):
        return self.backend.generation

    def reset(self, *, episode_id):
        self.backend.reset(episode_id=episode_id)

    def detect(self, frame):
        return self._run(frame, tracking=False)

    def track(self, frame):
        return self._run(frame, tracking=True)

    def _run(self, frame, tracking):
        if self.backend.episode_id != frame.episode_id:
            raise ValueError("Reset detector for matching episode")
        started = time.perf_counter()
        metadata = ObservationMetadata(
            frame.episode_id, frame.frame_id, frame.sim_time_s, self.source_id, self.model_version
        )
        result = self.backend.track(frame.rgb) if tracking else self.backend.detect(frame.rgb)
        observations = tuple(
            DetectionObservation(d.class_name, d.confidence, d.bounding_box, d.track_id)
            for d in result
        )
        # Includes conversion to canonical observations, not just GPU inference.
        return DetectionBatch(
            metadata,
            observations,
            frame.rgb.shape[1],
            frame.rgb.shape[0],
            CallTiming(time.perf_counter() - started),
        )

    def close(self):
        self.backend.close()
