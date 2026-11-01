"""Optional lane/semantic HUD composed inside the shared control deadline."""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from perception.cadence import RateLimitedWorkload
from perception.lane_path import extract_ego_path
from segmentation.postprocessing import overlay_mask
from visualisation.hud import draw_departure_overlay
from visualisation.lane_departure import LaneDepartureMonitor, lane_estimate_from_centerline


class ReleaseHudObserver:
    """Diagnostic models never supply depth masks, corrections or control targets.

    Model construction belongs to startup. Missing optional models can be recorded
    by the caller and this observer omitted. Runtime failures propagate to the
    control boundary, which disables the observer and preserves its error.
    """

    def __init__(self, lane, semantic=None, *, map_cte, sink, period_ticks=10):
        self.lane, self.semantic = lane, semantic
        self.map_cte, self.sink = map_cte, sink
        self.cadence = RateLimitedWorkload(period_ticks, max_age_ticks=period_ticks - 1)
        self.monitor = LaneDepartureMonitor()
        self.episode_id = None

    def __call__(self, frame, packet):
        if frame.episode_id != self.episode_id:
            self.cadence.reset()
            self.monitor.reset()
            self.episode_id = frame.episode_id

        def compute():
            path = extract_ego_path(self.lane.predict(frame.rgb))
            estimate = None if path is None else lane_estimate_from_centerline(
                path["centerline"], frame.calibration, frame_id=frame.frame_id,
                sim_time_s=frame.sim_time_s)
            mask = None if self.semantic is None else self.semantic.predict(frame.rgb)
            return estimate, mask

        result = self.cadence.step(frame.frame_id, compute)
        estimate, mask = result.value if result.valid else (None, None)
        report = self.monitor.update(estimate, self.map_cte(frame),
                                     now_sim_time_s=frame.sim_time_s, frame_id=frame.frame_id)
        # Semantic masks are displayed only on the original computed frame.
        image = overlay_mask(frame.rgb, mask) if mask is not None and result.age_ticks == 0 else frame.rgb
        image = draw_departure_overlay(image, report)
        record = dict(report=report.as_record(), frame_id=frame.frame_id,
                      episode_id=frame.episode_id, age_ticks=result.age_ticks,
                      capture_started_monotonic_s=getattr(frame, "capture_started_monotonic_s", None),
                      snapshot_only=True,
                      computed=result.computed, compute_cost_s=result.cost_s,
                      control_authority=False, semantic_depth_masking=False)
        self.sink(image, record)


class AsyncHudObserver:
    """One diagnostic snapshot in flight; never block command dispatch on HUD.

    Outputs retain their original frame identity. Skipped snapshots are not
    current control observations. Worker failures surface on the next call and
    are disabled by ControlCycle; close drains before the journal is closed.
    """

    def __init__(self, observer, *, period_ticks):
        self.observer, self.period_ticks = observer, period_ticks
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hud")
        self.pending = None
        self.last = None

    def __call__(self, frame, packet):
        if self.pending is not None:
            if not self.pending.done():
                return
            self.pending.result()
        key = (frame.episode_id, frame.frame_id)
        if self.last is not None and key[0] == self.last[0] and key[1] - self.last[1] < self.period_ticks:
            return
        self.last = key
        self.pending = self.worker.submit(self.observer, frame, packet)

    def close(self):
        self.worker.shutdown(wait=True, cancel_futures=True)
        self.error = None
        if self.pending is not None and not self.pending.cancelled():
            failure = self.pending.exception()
            if failure is not None:
                self.error = f"{type(failure).__name__}: {failure}"


_cpu_observer = None
_cpu_output = None


def _initialize_cpu_hud(lane_path, semantic_path, topology_path, period_ticks, shape):
    import numpy as np
    import torch

    from evaluation.a1_comparison import LaneCentrelines
    from perception.lane_inference import LaneSegmenter
    from planning.lane_graph import LaneTopology
    from segmentation.segmentation import SemanticSegmenter
    global _cpu_observer
    torch.set_num_threads(1)
    lane = LaneSegmenter(lane_path, device="cpu")
    semantic = SemanticSegmenter(semantic_path, device="cpu")
    reference = LaneCentrelines(LaneTopology.load(topology_path))
    def map_cte(frame):
        state = frame.ego.state
        cte, _, junction = reference.signed_offset(state.x_m, state.y_m)
        return None if junction else cte
    def retain(image, record):
        global _cpu_output
        _cpu_output = image, record
    for _ in range(2):
        blank = np.zeros((*shape, 3), dtype=np.uint8)
        lane.predict(blank)
        semantic.predict(blank)
    _cpu_observer = ReleaseHudObserver(lane, semantic, map_cte=map_cte,
                                       sink=retain, period_ticks=period_ticks)


def _cpu_hud_ready():
    return _cpu_observer is not None


def _compute_cpu_hud(frame, packet):
    _cpu_observer(frame, packet)
    return _cpu_output


class ProcessHudObserver:
    """Bound CPU model/Python work away from the mandatory control interpreter."""

    def __init__(self, *, lane_path, semantic_path, topology_path, period_ticks, shape, sink):
        self.process = ProcessPoolExecutor(max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_cpu_hud,
            initargs=(lane_path, semantic_path, topology_path, period_ticks, shape))
        try:
            if not self.process.submit(_cpu_hud_ready).result():
                raise RuntimeError("Diagnostic worker failed startup")
        except BaseException:
            self.process.shutdown(wait=True, cancel_futures=True)
            raise
        def compute(frame, packet):
            image, record = self.process.submit(_compute_cpu_hud, frame, packet).result()
            sink(image, record)
        self.delegate = AsyncHudObserver(compute, period_ticks=period_ticks)
        self.error = None

    def __call__(self, frame, packet):
        self.delegate(frame, packet)

    def close(self):
        self.delegate.close()
        self.error = self.delegate.error
        self.process.shutdown(wait=True, cancel_futures=True)
