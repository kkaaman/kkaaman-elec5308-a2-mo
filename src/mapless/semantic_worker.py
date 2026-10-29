"""The stack's optional semantic work off the control path (sixth review, F6-01).

Segmentation, painting and the road model feed the semantic layers, the shadow comparison
with the planned path (``semantic_tube_support``) and the log; nothing the command is
computed from waits on them. Inline, ``MaplessStack.step`` still ran them before it
returned, so the command reached the vehicle only after them: a slow segmenter tick was a
late command. ``SemanticWorker`` runs them on a thread of their own:

- **Immutable jobs.** Each ``SemanticJob`` carries what it needs as it was on its tick: the
  sweep's points (a copy), the camera images (a frame's, never written after capture), the
  pose, the frame id and time, the stack's spatial ``epoch`` (bumped on every reset of the
  odometry frame) and the terrain grid's placement then (``TerrainGrid.view``: its origin,
  no arrays) or, for a road-model build, a copy of it (``TerrainGrid.snapshot``), so the
  worker never reads arrays the control path is writing.
- **Latest wins.** The queue holds one job: a job not yet started when the next arrives is
  dropped (``dropped``), so the worker's lag never grows; a waiting build is not dropped
  for a job that only segments (that one is, instead).
- **Stale results are rejected.** ``take`` returns a result only from the current epoch and
  no older than ``max_age_s`` in simulation time (``stale_epoch``, ``stale_age``).
- **Failure stays off the control path.** An exception in the worker is counted and kept
  (``errors``, ``last_error``); the stack drives on without semantic layers.

``stats`` reports all of it per tick for the log, with the worker's own stage times.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from mapless.painting import SegmentedFrame, paint
from mapless.road_model import RoadModelBuilder


@dataclass(frozen=True)
class SemanticJob:
    """One tick's semantic work, with everything it reads (see the module docstring)."""

    frame_id: int
    time_s: float
    epoch: int
    pose: object
    xyz: np.ndarray
    cameras: tuple
    segment: bool
    grid: object  # TerrainGrid.view() then; TerrainGrid.snapshot() when ``build``
    build: bool = False
    path_body: np.ndarray | None = None  # the planned path then, for the semantic shadow


@dataclass(frozen=True)
class SemanticResult:
    """A finished road-model build: the model, its shadow comparison and the worker's times."""

    frame_id: int
    time_s: float
    epoch: int
    model: object
    semantic: dict | None
    timing_ms: dict = field(default_factory=dict)


class SemanticWorker:
    """Runs ``SemanticJob``s on a daemon thread (see the module docstring).

    ``shadow(model, pose, path_body)`` computes the semantic shadow for a build (the stack
    passes ``semantic_tube_support`` with its config); ``max_age_s``: the oldest result, in
    simulation time behind the current tick, ``take`` still returns.
    """

    def __init__(self, segmenter, painting_config, road_model_config, grid, shadow=None, max_age_s=0.5):
        self.segmenter = segmenter
        self.painting_config = painting_config
        self.builder = RoadModelBuilder(grid.view(), road_model_config)
        self.shadow = shadow
        self.max_age_s = max_age_s
        self._jobs = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._result = None
        self._epoch = None
        self._busy = False
        self._idle = threading.Event()
        self._idle.set()
        self.counts = dict(submitted=0, dropped=0, segmented=0, built=0, taken=0, stale_epoch=0, stale_age=0, errors=0)
        self.last_error = None
        self.last_timing_ms = {}
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="semantic-worker", daemon=True)
        self._thread.start()

    # -- the control path's side ------------------------------------------------------------
    def submit(self, job):
        """Queue ``job``, dropping the one still waiting, unless that one builds the road
        model and ``job`` does not: then ``job`` is dropped. Never blocks."""
        if self._closed:
            return
        with self._lock:
            self.counts["submitted"] += 1
            self._idle.clear()
            try:
                waiting = self._jobs.get_nowait()
            except queue.Empty:
                waiting = None
            if waiting is not None:
                self.counts["dropped"] += 1
                if waiting.build and not job.build:
                    job = waiting
            self._jobs.put_nowait(job)

    def take(self, time_s, epoch):
        """The newest finished build if it is from ``epoch`` and no older than ``max_age_s``
        at ``time_s``; else None (and the stale result is dropped)."""
        with self._lock:
            result, self._result = self._result, None
            if result is None:
                return None
            if result.epoch != epoch:
                self.counts["stale_epoch"] += 1
                return None
            if time_s - result.time_s > self.max_age_s:
                self.counts["stale_age"] += 1
                return None
            self.counts["taken"] += 1
            return result

    def stats(self):
        """Counters, the worker's last stage times and its last error, for the log."""
        with self._lock:
            return dict(self.counts, busy=self._busy, last_ms={k: round(v, 2) for k, v in self.last_timing_ms.items()}, last_error=self.last_error)

    def wait_idle(self, timeout_s=5.0):
        """Block until every submitted job is done (for tests and orderly shut-down)."""
        return self._idle.wait(timeout_s)

    def close(self, timeout_s=2.0):
        """Stop the thread after the job in hand."""
        self._closed = True
        try:
            self._jobs.put_nowait(None)
        except queue.Full:
            try:
                self._jobs.get_nowait()
            except queue.Empty:
                pass
            self._jobs.put_nowait(None)
        self._thread.join(timeout_s)

    # -- the worker's side --------------------------------------------------------------------
    def _run(self):
        while True:
            job = self._jobs.get()
            if job is None:
                self._idle.set()
                return
            with self._lock:
                self._busy = True
            try:
                self._do(job)
            except Exception as error:  # noqa: BLE001 - the worker must survive any stage's failure
                with self._lock:
                    self.counts["errors"] += 1
                    self.last_error = f"{type(error).__name__}: {error}"
            finally:
                with self._lock:
                    self._busy = False
                    if self._jobs.empty():
                        self._idle.set()

    def _do(self, job):
        timing = {}
        self.builder.grid = job.grid
        if job.epoch != self._epoch:
            # The odometry frame was rebuilt (or the stack reset): the evidence gathered in
            # the old one is meaningless in the new.
            self.builder.reset()
            self._epoch = job.epoch
        if job.segment and job.cameras:
            begun = time.perf_counter()
            segmentations = SegmentedFrame(job.frame_id, job.time_s, [self.segmenter.segment(im) for im in job.cameras])
            timing["segment"] = (time.perf_counter() - begun) * 1000
            begun = time.perf_counter()
            painted = paint(job.xyz, job.cameras, segmentations, self.painting_config, frame_id=job.frame_id)
            timing["paint"] = (time.perf_counter() - begun) * 1000
            if painted is not None:
                begun = time.perf_counter()
                self.builder.add(job.xyz, painted, job.pose, job.time_s)
                timing["add"] = (time.perf_counter() - begun) * 1000
            with self._lock:
                self.counts["segmented"] += 1
        if job.build:
            begun = time.perf_counter()
            model = self.builder.build(job.pose, job.time_s)
            semantic = self.shadow(model, job.pose, job.path_body) if self.shadow is not None and job.path_body is not None else None
            timing["road_model"] = (time.perf_counter() - begun) * 1000
            result = SemanticResult(job.frame_id, job.time_s, job.epoch, model, semantic, dict(timing))
            with self._lock:
                self._result = result
                self.counts["built"] += 1
        with self._lock:
            self.last_timing_ms = timing
