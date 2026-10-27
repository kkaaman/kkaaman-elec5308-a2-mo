"""E4-04 JPEG/JSON HTTP detector; all encoding and HTTP run in a bounded worker."""
from __future__ import annotations

import base64
import io
import json
import time
import urllib.request
from urllib.parse import urlsplit

from obstacle_perception.types import (
    CallTiming,
    DetectionBatch,
    DetectionObservation,
    ObservationMetadata,
    PixelBox,
)
from perception.async_detector import AsyncDetector, positive


class JpegHttpBackend:
    """Synchronous transport for AsyncDetector, not for the control thread.

    POST JSON v1: episode_id/frame_id/sim_time_s, dimensions, source/model,
    jpeg_base64. Response has the same identity, detections, valid/reason and
    server_duration_s. Coordinates refer to original input dimensions.
    Explicit endpoint only; no model downloads or simulator imports.
    """
    def __init__(self, endpoint, *, source_id, model_version, timeout_s=0.2,
                 jpeg_quality=85, max_response_bytes=2_000_000):
        url = urlsplit(endpoint)
        if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password:
            raise ValueError("Explicit HTTP(S) endpoint without embedded credentials required")
        ObservationMetadata("validation", 0, 0, source_id, model_version)
        if type(jpeg_quality) is not int or not 1 <= jpeg_quality <= 100:
            raise ValueError("JPEG quality must be 1..100")
        if type(max_response_bytes) is not int or max_response_bytes < 1:
            raise ValueError("Positive response size limit required")
        self.endpoint = endpoint
        self.source_id, self.model_version = source_id, model_version
        self.timeout_s = positive(timeout_s, "timeout_s")
        self.quality, self.max_bytes = jpeg_quality, max_response_bytes
        self.episode_id = None
        self.last_timing = {}

    def reset(self, *, episode_id):
        self.episode_id = episode_id

    def detect(self, frame):
        from PIL import Image

        if frame.episode_id != self.episode_id:
            raise ValueError("Episode mismatch")
        start = time.monotonic()
        buffer = io.BytesIO()
        Image.fromarray(frame.rgb).save(buffer, format="JPEG", quality=self.quality)
        identity = dict(schema_version=1, episode_id=frame.episode_id, frame_id=frame.frame_id,
                        sim_time_s=frame.sim_time_s, image_width=frame.rgb.shape[1],
                        image_height=frame.rgb.shape[0], source_id=self.source_id,
                        model_version=self.model_version)
        payload = json.dumps(dict(identity, jpeg_base64=base64.b64encode(buffer.getvalue()).decode("ascii")),
                             allow_nan=False).encode()
        encoded = time.monotonic()
        request = urllib.request.Request(self.endpoint, payload, {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            raw = response.read(self.max_bytes + 1)
        received = time.monotonic()
        if len(raw) > self.max_bytes:
            raise ValueError("Response size limit exceeded")
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate JSON key")
                result[key] = value
            return result
        doc = json.loads(raw, object_pairs_hook=unique)
        expected = set(identity) | {"detections", "valid", "invalid_reason", "server_duration_s"}
        if not isinstance(doc, dict) or set(doc) != expected:
            raise ValueError("Unexpected response schema")
        for key, value in identity.items():
            if type(doc[key]) is not type(value) or doc[key] != value:
                raise ValueError(f"Response identity mismatch: {key}")
        duration = doc["server_duration_s"]
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0:
            raise ValueError("Invalid server duration")
        if not isinstance(doc["detections"], list):
            raise ValueError("detections must be a list")
        detections = []
        for row in doc["detections"]:
            if not isinstance(row, dict) or set(row) != {"class_name", "confidence", "xyxy", "track_id"}:
                raise ValueError("Unexpected detection schema")
            detections.append(DetectionObservation(row["class_name"], row["confidence"],
                              PixelBox(*row["xyxy"]), row["track_id"]))
        end = time.monotonic()
        result = DetectionBatch(
            ObservationMetadata(frame.episode_id, frame.frame_id, frame.sim_time_s,
                                self.source_id, self.model_version), tuple(detections),
            frame.rgb.shape[1], frame.rgb.shape[0], CallTiming(end-start, duration),
            doc["valid"], doc["invalid_reason"],
        )
        self.last_timing = dict(encode_s=encoded-start, http_round_trip_s=received-encoded,
                               decode_s=end-received, server_duration_s=duration,
                               rtt_residual_s=received-encoded-duration, total_s=end-start)
        return result

    def close(self):
        self.episode_id = None


class RemoteDetector(AsyncDetector):
    """submit/poll API; deliberately does not pretend detect() is nonblocking."""
    def __init__(self, endpoint, **kwargs):
        super().__init__(JpegHttpBackend(endpoint, **kwargs))
