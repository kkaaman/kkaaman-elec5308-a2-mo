"""Deterministic offline detector fixtures, bound to source frame metadata."""

import hashlib
import json
import time
from pathlib import Path

from obstacle_perception.types import (
    CallTiming,
    DetectionBatch,
    DetectionObservation,
    ObservationMetadata,
    PixelBox,
)


def verify_local_asset(path, expected_sha256):
    path = Path(path)
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(c not in "0123456789abcdef" for c in expected_sha256)
    ):
        raise ValueError("Invalid SHA256")
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        prefix = stream.read(128)
        if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError("Git LFS pointer is not an asset")
        digest.update(prefix)
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError("Asset checksum mismatch")
    return path


class RecordedDetector:
    """JSON fixture reader; never imports or creates a learned detector.

    The fixture origin is explicitly 'synthetic' or 'recorded'. Timing reported
    here measures fixture lookup/conversion, not learned-model inference.
    Missing frames raise; they cannot silently become valid empty detections.
    """

    def __init__(self, path, sha256):
        doc = json.loads(verify_local_asset(path, sha256).read_text(encoding="utf-8-sig"))
        if doc.get("schema_version") != 1 or doc.get("origin") not in ("synthetic", "recorded"):
            raise ValueError("Unsupported fixture schema/origin")
        self.model_version = doc["model_version"]
        self.source_id = doc["source_id"]
        self.frames = {}
        for row in doc["frames"]:
            frame = row["frame_id"]
            if type(frame) is not int or frame < 0 or frame in self.frames:
                raise ValueError("Duplicate/invalid fixture frame")
            observations = tuple(
                DetectionObservation(
                    d["class_name"], d["confidence"], PixelBox(*d["xyxy"]), d.get("track_id")
                )
                for d in row["detections"]
            )
            # Validate everything at load, including metadata, dimensions,
            # timestamps, bounds and valid-empty/invalid distinction.
            metadata = ObservationMetadata(
                "fixture-validation", frame, row["sim_time_s"], self.source_id, self.model_version
            )
            DetectionBatch(
                metadata,
                observations,
                row["width"],
                row["height"],
                CallTiming(0),
                row.get("valid", True),
                row.get("invalid_reason"),
            )
            self.frames[frame] = (row, observations)
        self.episode_id = None
        self.closed = False

    def reset(self, *, episode_id):
        ObservationMetadata(episode_id, 0, 0, self.source_id, self.model_version)
        if self.closed:
            raise RuntimeError("Detector is closed")
        self.episode_id = episode_id

    def detect(self, frame):
        start = time.perf_counter()
        if self.closed or frame.episode_id != self.episode_id:
            raise RuntimeError("Reset detector for this episode first")
        if frame.frame_id not in self.frames:
            raise ValueError("Frame absent from recording; observation unknown")
        row, observations = self.frames[frame.frame_id]
        if abs(frame.sim_time_s - row["sim_time_s"]) > 1e-6 or frame.rgb.shape[:2] != (
            row["height"],
            row["width"],
        ):
            raise ValueError("Recorded source frame dimensions/time mismatch")
        return DetectionBatch(
            ObservationMetadata(
                frame.episode_id,
                frame.frame_id,
                frame.sim_time_s,
                self.source_id,
                self.model_version,
            ),
            observations,
            row["width"],
            row["height"],
            CallTiming(time.perf_counter() - start),
            row.get("valid", True),
            row.get("invalid_reason"),
        )

    def track(self, frame):
        return self.detect(frame)

    def close(self):
        self.closed = True
        self.episode_id = None
        self.frames.clear()


def benchmark_calls(factory, frame, warm_calls=3):
    """Separate construction, first call and subsequent calls, all wall seconds.

    Use a learned-detector factory for actual model timings. Fixture timings
    cannot be represented as YOLO performance. First-call timing includes any
    lazy backend setup. Factory must return a resettable contract detector.
    """
    if type(warm_calls) is not int or warm_calls < 1:
        raise ValueError("At least one warm call required")
    start = time.perf_counter()
    detector = factory()
    construction = time.perf_counter() - start
    try:
        detector.reset(episode_id=frame.episode_id)
        start = time.perf_counter()
        detector.detect(frame)
        first = time.perf_counter() - start
        durations = []
        for _ in range(warm_calls):
            start = time.perf_counter()
            detector.detect(frame)
            durations.append(time.perf_counter() - start)
        return dict(
            construction_wall_s=construction, first_call_wall_s=first, warm_call_wall_s=durations
        )
    finally:
        detector.close()
