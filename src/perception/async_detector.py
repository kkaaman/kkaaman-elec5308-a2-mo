"""Bounded detector workers. The caller thread only submits and polls.

A stalled operation retains its one slot until it exits; reset never creates
replacement workers. Daemon threads cannot interrupt a stuck backend call.
"""
from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass

from obstacle_perception.types import DetectionBatch


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def matches(batch, frame):
    return (
        isinstance(batch, DetectionBatch) and batch.valid
        and batch.metadata.episode_id == frame.episode_id
        and batch.metadata.frame_id == frame.frame_id
        and batch.metadata.sim_time_s == frame.sim_time_s
        and (batch.image_height, batch.image_width) == frame.rgb.shape[:2]
    )


@dataclass
class Work:
    frame: object
    generation: int
    deadline: float
    done: threading.Event
    result: DetectionBatch | None = None
    completed: float | None = None
    error: str | None = None


class AsyncDetector:
    """One in-flight operation, zero waiting backlog; detect/reset run in worker.

    The owner must serialize submit/poll/reset/close. Backend state is confined
    to one worker. Poll never waits; invalid/late/wrong-frame results are unknown,
    never valid empty detections. Reset invalidates results across episodes.
    """
    def __init__(self, backend, *, clock=time.monotonic):
        self.backend, self.clock = backend, clock
        self.episode_id = None
        self.generation = 0
        self.closed = False
        self.work = None
        self.last_frame = -1
        self._queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="detector-worker")
        self._thread.start()

    def reset(self, *, episode_id):
        if self.closed or not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("Open worker and nonempty episode required")
        self.episode_id = episode_id
        self.generation += 1
        self.last_frame = -1

    def submit(self, frame, *, deadline):
        if self.closed or frame.episode_id != self.episode_id:
            raise ValueError("Reset open worker for matching episode first")
        if not math.isfinite(deadline) or deadline <= self.clock():
            return False
        if frame.frame_id <= self.last_frame:
            return False
        if self.work is not None and not self.work.done.is_set():
            return False
        work = Work(frame, self.generation, deadline, threading.Event())
        self.work = work
        self.last_frame = frame.frame_id
        self._queue.put_nowait(work)
        return True

    def poll(self, frame):
        work = self.work
        if self.closed or work is None or not work.done.is_set():
            return None
        if (work.generation != self.generation or frame is not work.frame
                or self.clock() > work.deadline or work.completed > work.deadline):
            return None
        return work.result if matches(work.result, frame) else None

    def _run(self):
        generation = None
        try:
            while not self._stop.is_set():
                try:
                    work = self._queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                try:
                    if work.generation != generation:
                        self.backend.reset(episode_id=work.frame.episode_id)
                        generation = work.generation
                    if self.clock() >= work.deadline:
                        raise TimeoutError("Expired before inference")
                    work.result = self.backend.detect(work.frame)
                    if not matches(work.result, work.frame):
                        raise ValueError("Invalid or mismatched detection batch")
                except Exception as exc:
                    work.error = f"{type(exc).__name__}: {exc}"
                    work.result = None
                finally:
                    work.completed = self.clock()
                    work.done.set()
        finally:
            self.backend.close()

    def close(self):
        self.closed = True
        self.generation += 1
        self._stop.set()
