"""Deterministic fake detectors for CI and the kinematic harness.

Both fakes implement :class:`autonomy.interfaces.Detector`, so the stack can
be tested end to end without YOLO, a GPU or a network. A *scene* function says
which objects are visible in each frame; the fake assigns track ids per
session in order of first appearance, like a real tracker would.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np

from autonomy.interfaces import Detection2D, DetectionBatch, ImageBox

__all__ = ["FakeLocalDetector", "FakeRemoteDetector", "ScriptedObject", "Scene"]


@dataclass(frozen=True)
class ScriptedObject:
    """One object a fake detector will 'see'. ``key`` is its true identity."""

    key: str
    class_name: str
    confidence: float
    box: ImageBox


Scene = Callable[[int], Iterable[ScriptedObject]]


class _FakeDetectorBase:
    source = "fake"
    model_version = "fake"

    def __init__(self, scene: Scene, latency_s: float = 0.0) -> None:
        self._scene = scene
        self._latency_s = latency_s
        self._session: Optional[str] = None
        self._tracks: dict[str, int] = {}

    def reset(self, session_id: str) -> None:
        if not session_id:
            raise ValueError("session_id must be non-empty")
        self._session = session_id
        self._tracks = {}

    def _detections(self, frame_id: int, track: bool) -> tuple[Detection2D, ...]:
        out = []
        for obj in self._scene(frame_id):
            track_id = None
            if track:
                track_id = self._tracks.setdefault(obj.key, len(self._tracks) + 1)
            out.append(Detection2D(obj.class_name, obj.confidence, obj.box, track_id))
        return tuple(out)

    def _check(self, rgb: np.ndarray) -> None:
        if self._session is None:
            raise RuntimeError("reset(session_id) must be called before infer()")
        if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb must be an (H, W, 3) array")


class FakeLocalDetector(_FakeDetectorBase):
    """Stand-in for the local YOLO detector. Always answers."""

    source = "local"
    model_version = "fake-local"

    def infer(self, rgb: np.ndarray, frame_id: int, track: bool = True) -> DetectionBatch:
        self._check(rgb)
        start = time.perf_counter()
        detections = self._detections(frame_id, track)
        return DetectionBatch(frame_id, self.source, self._session, self.model_version,
                              detections, track, start, start + self._latency_s)


class FakeRemoteDetector(_FakeDetectorBase):
    """Stand-in for the hybrid cloud backend (E4).

    Frames listed in ``outage_frames`` return an invalid batch, as a timeout
    or dropped connection would. Track ids are independent of any local
    detector's, which is why identities are namespaced by source.
    """

    source = "remote"
    model_version = "fake-remote"

    def __init__(self, scene: Scene, latency_s: float = 0.05,
                 outage_frames: Iterable[int] = ()) -> None:
        super().__init__(scene, latency_s)
        self._outages = frozenset(outage_frames)

    def infer(self, rgb: np.ndarray, frame_id: int, track: bool = True) -> DetectionBatch:
        self._check(rgb)
        start = time.perf_counter()
        if frame_id in self._outages:
            return DetectionBatch(frame_id, self.source, self._session, self.model_version, (),
                                  track, start, start + self._latency_s,
                                  valid=False, reason="remote timeout")
        detections = self._detections(frame_id, track)
        return DetectionBatch(frame_id, self.source, self._session, self.model_version,
                              detections, track, start, start + self._latency_s)
