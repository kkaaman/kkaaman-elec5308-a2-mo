"""Bounded session-isolated optional HTTP inference service (official issue #56).

Client absolute clocks are echoed/validated only within the client domain.
Server duration uses a local monotonic clock. No simulator API is imported.
"""

import base64
import binascii
import hashlib
import io
import json
import math
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from obstacle_perception.types import (
    CallTiming,
    DetectionBatch,
    DetectionObservation,
    ObservationMetadata,
)


class ServiceError(Exception):
    """HTTP-compatible bounded-service rejection, with no private backend details."""

    def __init__(self, status, message):
        self.status = status
        super().__init__(message)


@dataclass
class Session:
    detector: object
    identity: tuple
    last_frame: int = -1
    last_sim_s: float = -1
    last_used: float = 0
    failed: bool = False
    lock: object = field(default_factory=threading.Lock)


class InferenceService:
    """One detector/tracker per run/session; reject queued or unordered requests.

    TTL is server-local idle seconds; only idle sessions expire. Maximum sessions
    bounds model instances. Per-session contention returns 409 instead of queuing.
    """

    def __init__(self, factory, *, model_version, max_sessions=2, ttl_s=60,
                 max_pixels=1920*1080, max_jpeg_bytes=4_000_000, clock=time.monotonic):
        if (type(max_sessions) is not int or max_sessions < 1 or not math.isfinite(ttl_s)
                or isinstance(ttl_s, bool) or ttl_s <= 0 or type(max_pixels) is not int or max_pixels < 1
                or type(max_jpeg_bytes) is not int or max_jpeg_bytes < 1):
            raise ValueError("Positive finite resource limits required")
        ObservationMetadata("validation", 0, 0, "remote", model_version)
        self.factory, self.model_version = factory, model_version
        self.max_sessions, self.ttl_s = max_sessions, ttl_s
        self.max_pixels, self.max_jpeg_bytes = max_pixels, max_jpeg_bytes
        self.clock, self.sessions, self.guard = clock, {}, threading.Lock()

    def _key(self, headers):
        key = tuple(headers.get(name) for name in ("x-run-id", "x-session-id"))
        if any(not isinstance(v, str) or not v.strip() or len(v) > 128
               or not v.isascii() or any(ord(c) < 33 for c in v) for v in key):
            raise ServiceError(422, "Explicit run/session identity required")
        return key

    def _expire(self):
        for key, session in list(self.sessions.items()):
            if self.clock() - session.last_used > self.ttl_s and session.lock.acquire(False):
                try:
                    session.detector.close()
                    del self.sessions[key]
                finally:
                    session.lock.release()

    def _decode(self, doc, headers):
        fields = {"schema_version", "episode_id", "frame_id", "sim_time_s", "image_width",
                  "image_height", "source_id", "model_version", "jpeg_base64"}
        if not isinstance(doc, dict) or set(doc) != fields or type(doc["schema_version"]) is not int or doc["schema_version"] != 1:
            raise ServiceError(422, "Unexpected request schema")
        try:
            metadata = ObservationMetadata(*(doc[k] for k in
                ("episode_id", "frame_id", "sim_time_s", "source_id", "model_version")))
            sent = float(headers["x-send-monotonic-s"])
            captured = headers.get("x-capture-monotonic-s")
            if not math.isfinite(sent) or sent < 0 or (captured is not None and not
                    (math.isfinite(float(captured)) and 0 <= float(captured) <= sent)):
                raise ValueError("Client clock order")
            w, h = doc["image_width"], doc["image_height"]
            if type(w) is not int or type(h) is not int or min(w,h) < 1 or w*h > self.max_pixels:
                raise ValueError("Image dimensions")
            if not isinstance(doc["jpeg_base64"], str) or len(doc["jpeg_base64"]) > 4*((self.max_jpeg_bytes+2)//3):
                raise ValueError("JPEG size")
            raw = base64.b64decode(doc["jpeg_base64"], validate=True)
            if len(raw) > self.max_jpeg_bytes:
                raise ValueError("JPEG size")
            with Image.open(io.BytesIO(raw)) as image:
                if image.format != "JPEG" or image.size != (w,h):
                    raise ValueError("JPEG format/dimensions")
                rgb = np.array(image.convert("RGB"), dtype=np.uint8)
        except (KeyError, TypeError, ValueError, OSError, binascii.Error, Image.DecompressionBombError) as error:
            raise ServiceError(422, "Invalid metadata, client timing or JPEG") from error
        if metadata.model_version != self.model_version:
            raise ServiceError(409, "Model identity mismatch")
        return metadata, rgb

    def detect(self, doc, headers):
        """Infer one ordered frame and return exact request identity plus detections."""
        started = self.clock()
        key = self._key(headers)
        metadata, rgb = self._decode(doc, headers)
        identity = (metadata.episode_id, metadata.source_id, metadata.model_version)
        with self.guard:
            self._expire()
            session = self.sessions.get(key)
            if session is None:
                if headers.get("x-session-start") != "true":
                    raise ServiceError(410, "Session absent/expired; explicit reset required")
                if len(self.sessions) >= self.max_sessions:
                    raise ServiceError(429, "Session capacity reached")
                try:
                    detector = self.factory()
                except Exception as error:
                    raise ServiceError(503, "Detector initialization failed") from error
                if any(s.detector is detector for s in self.sessions.values()):
                    raise ServiceError(503, "Detector factory did not isolate tracker state")
                try:
                    detector.reset(episode_id=metadata.episode_id)
                except Exception as error:
                    detector.close()
                    raise ServiceError(503, "Detector reset failed") from error
                session = Session(detector, identity, last_used=self.clock())
                self.sessions[key] = session
            if not session.lock.acquire(False):
                raise ServiceError(409, "Session busy; request not queued")
        try:
            if session.identity != identity:
                raise ServiceError(409, "Session identity mismatch; use a new session")
            if session.failed:
                raise ServiceError(503, "Session backend failed; explicit reset required")
            if metadata.frame_id <= session.last_frame or metadata.sim_time_s <= session.last_sim_s:
                raise ServiceError(409, "Duplicate or out-of-order frame")
            # Consume identity even on backend failure; never repeat partially advanced tracking.
            session.last_frame, session.last_sim_s = metadata.frame_id, metadata.sim_time_s
            try:
                observations = tuple(DetectionObservation(d.class_name, d.confidence,
                    d.bounding_box, d.track_id) for d in session.detector.track(rgb))
                DetectionBatch(metadata, observations, rgb.shape[1], rgb.shape[0],
                               CallTiming(0))
            except Exception as error:
                session.failed = True
                raise ServiceError(503, "Inference unavailable") from error
            result = {k:v for k,v in doc.items() if k != "jpeg_base64"}
            result.update(detections=[dict(class_name=d.class_name, confidence=d.confidence,
                xyxy=[d.box.x_min,d.box.y_min,d.box.x_max,d.box.y_max], track_id=d.track_id)
                for d in observations], valid=True, invalid_reason=None,
                server_duration_s=self.clock()-started)
            return result
        finally:
            session.last_used = self.clock()
            session.lock.release()

    def reset(self, headers):
        """Close one idle session; concurrent in-flight reset is rejected."""
        key = self._key(headers)
        with self.guard:
            session = self.sessions.get(key)
            if session is None:
                return
            if not session.lock.acquire(False):
                raise ServiceError(409, "Session busy")
            try:
                session.detector.close()
                del self.sessions[key]
            finally:
                session.lock.release()

    def close(self):
        """Close idle sessions at service shutdown; never interrupt active tracking."""
        for run, session in list(self.sessions):
            self.reset({"x-run-id":run, "x-session-id":session})


def create_app(service, *, token, max_body_bytes=6_000_000):
    """Create optional FastAPI routes; require an explicit bearer secret."""
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from starlette.concurrency import run_in_threadpool

    if not isinstance(token, str) or not token.strip():
        raise ValueError("Explicit nonempty service token required")
    if type(max_body_bytes) is not int or max_body_bytes < 1:
        raise ValueError("Positive body limit required")
    @asynccontextmanager
    async def lifespan(app):
        yield
        service.close()

    app = FastAPI(lifespan=lifespan)

    def authenticate(request):
        if not secrets.compare_digest(request.headers.get("authorization", "").encode(), ("Bearer "+token).encode()):
            raise HTTPException(401, "Unauthorized")

    @app.get("/health")
    def health():
        return dict(status="configured", inference_verified=False, model_version=service.model_version,
                    max_sessions=service.max_sessions)

    @app.post("/detect")
    async def detect(request: Request):
        authenticate(request)
        raw = bytearray()
        async for chunk in request.stream():
            if len(raw)+len(chunk) > max_body_bytes:
                raise HTTPException(413, "Request too large")
            raw.extend(chunk)
        try:
            def unique(pairs):
                result = {}
                for k,v in pairs:
                    if k in result:
                        raise ValueError("Duplicate JSON key")
                    result[k] = v
                return result
            doc = json.loads(raw, object_pairs_hook=unique)
            headers = dict(request.headers)
            result = await run_in_threadpool(service.detect, doc, headers)
            return JSONResponse(result, headers={"X-Run-Id":headers["x-run-id"],
                                                "X-Session-Id":headers["x-session-id"]})
        except ServiceError as error:
            raise HTTPException(error.status, str(error)) from error
        except (ValueError, TypeError) as error:
            raise HTTPException(422, "Malformed JSON") from error

    @app.post("/reset")
    async def reset(request: Request):
        authenticate(request)
        try:
            await run_in_threadpool(service.reset, dict(request.headers))
        except ServiceError as error:
            raise HTTPException(error.status, str(error)) from error
        return {"status":"reset"}

    return app


def from_environment():
    """Require mounted hash-pinned weights and env secret; never download a model."""
    from perception.object_detection import ObjectDetector
    from perception.types import PerceptionConfig

    path = Path(os.environ["DETECTOR_WEIGHTS"])
    expected = os.environ["DETECTOR_MODEL_SHA256"]
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Mounted model hash mismatch")
    service = InferenceService(lambda: ObjectDetector(PerceptionConfig(model_name=str(path))),
        model_version=expected, max_sessions=int(os.environ.get("DETECTOR_MAX_SESSIONS", "2")),
        ttl_s=float(os.environ.get("DETECTOR_SESSION_TTL_S", "60")))
    return create_app(service, token=os.environ["DETECTOR_TOKEN"])
