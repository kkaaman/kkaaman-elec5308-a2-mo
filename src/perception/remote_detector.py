"""E4-04 JPEG/JSON HTTP detector; all encoding and HTTP run in a bounded worker."""
from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
import uuid
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
                 jpeg_quality=85, max_response_bytes=2_000_000, token=None, run_id=None):
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
        self.session_id = None
        self.run_id = run_id or str(uuid.uuid4())
        self.token = token if token is not None else os.environ.get("DETECTOR_TOKEN")
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("Nonempty run_id required")
        self.last_timing = {}
        self._reset_required = False

    def _release_session(self):
        if self.session_id is None:
            return
        headers = {"Content-Type":"application/json", "X-Run-Id":self.run_id,
                   "X-Session-Id":self.session_id}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        request = urllib.request.Request(self.endpoint.rsplit("/",1)[0]+"/reset", b"{}", headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                response.read(self.max_bytes + 1)
        except (OSError, ValueError) as error:
            self.last_reset_error = type(error).__name__

    def reset(self, *, episode_id):
        self._release_session()
        self.episode_id = episode_id
        self.session_id = str(uuid.uuid4())
        self.session_started = False
        self._reset_required = False

    def detect(self, frame):
        from PIL import Image

        if frame.episode_id != self.episode_id:
            raise ValueError("Episode mismatch")
        start = time.monotonic()
        if self._reset_required:
            self.reset(episode_id=frame.episode_id)
        prepared = time.monotonic()
        buffer = io.BytesIO()
        Image.fromarray(frame.rgb).save(buffer, format="JPEG", quality=self.quality)
        identity = dict(schema_version=1, episode_id=frame.episode_id, frame_id=frame.frame_id,
                        sim_time_s=frame.sim_time_s, image_width=frame.rgb.shape[1],
                        image_height=frame.rgb.shape[0], source_id=self.source_id,
                        model_version=self.model_version)
        payload = json.dumps(dict(identity, jpeg_base64=base64.b64encode(buffer.getvalue()).decode("ascii")),
                             allow_nan=False).encode()
        encoded = time.monotonic()
        headers = {"Content-Type": "application/json", "X-Run-Id": self.run_id,
                   "X-Session-Id": self.session_id, "X-Send-Monotonic-S": str(encoded)}
        headers["X-Session-Start"] = "false" if self.session_started else "true"
        self.session_started = True
        if frame.capture_started_monotonic_s is not None:
            headers["X-Capture-Monotonic-S"] = str(frame.capture_started_monotonic_s)
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        request = urllib.request.Request(self.endpoint, payload, headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                if (response.headers.get("X-Run-Id") != self.run_id
                        or response.headers.get("X-Session-Id") != self.session_id):
                    raise ValueError("Response run/session identity mismatch")
                raw = response.read(self.max_bytes + 1)
        except urllib.error.HTTPError as error:
            if error.code in (410, 503):
                self._reset_required = True
            raise
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
        self.last_timing = dict(session_reset_s=prepared-start, encode_s=encoded-prepared, http_round_trip_s=received-encoded,
                               decode_s=end-received, server_duration_s=duration,
                               rtt_residual_s=received-encoded-duration, total_s=end-start)
        return result

    def close(self):
        self._release_session()
        self.session_id = None
        self.episode_id = None


class RemoteDetector(AsyncDetector):
    """submit/poll API; deliberately does not pretend detect() is nonblocking."""
    def __init__(self, endpoint, **kwargs):
        super().__init__(JpegHttpBackend(endpoint, **kwargs))
