"""The semantic worker (sixth review, F6-01) and the optional schedule check (F6-07)."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from mapless.ego_motion import Pose2D
from mapless.road_model import RoadModelConfig
from mapless.semantic_worker import SemanticJob, SemanticWorker
from mapless.stack import StackConfig, optional_schedule
from mapless.terrain import TerrainGrid


class Gate:
    """A segmenter-shaped stand-in for the shadow function: blocks until released."""

    def __init__(self):
        self.go = threading.Event()
        self.calls = 0

    def __call__(self, model, pose, path):
        self.calls += 1
        self.go.wait(5.0)
        return dict(cells=1)


def job(tick, epoch=0, build=True, grid=None, t=None):
    grid = grid or TerrainGrid()
    return SemanticJob(
        tick, tick * 0.05 if t is None else t, epoch, Pose2D(0, 0, 0), np.zeros((0, 3)), (), False,
        grid.snapshot() if build else grid.view(), build, np.column_stack((np.arange(0.0, 20.0), np.zeros(20))),
    )


def worker(shadow=None, max_age_s=0.5, max_wall_age_s=None):
    return SemanticWorker(None, None, RoadModelConfig(), TerrainGrid(), shadow=shadow, max_age_s=max_age_s, max_wall_age_s=max_wall_age_s)


def test_a_build_comes_back_with_its_frame_its_epoch_and_the_shadow():
    w = worker(shadow=lambda model, pose, path: dict(cells=len(path)))
    try:
        w.submit(job(4))
        assert w.wait_idle()
        result = w.take(0.2, epoch=0)
        assert result is not None and result.frame_id == 4 and result.model is not None and result.semantic == dict(cells=20)
        assert "road_model" in result.timing_ms
        assert w.take(0.2, epoch=0) is None  # taken once
        assert w.stats()["built"] == 1 and w.stats()["taken"] == 1
    finally:
        w.close()


def test_stale_results_are_rejected_by_epoch_and_by_age():
    w = worker()
    try:
        w.submit(job(1, epoch=0))
        assert w.wait_idle()
        assert w.take(0.05, epoch=1) is None  # the odometry frame was rebuilt since
        w.submit(job(2, epoch=1))
        assert w.wait_idle()
        assert w.take(0.1 + 0.6, epoch=1) is None  # 0.6 s old
        stats = w.stats()
        assert stats["stale_epoch"] == 1 and stats["stale_age"] == 1 and stats["taken"] == 0
    finally:
        w.close()


def test_the_queue_keeps_the_latest_job_and_never_drops_a_waiting_build_for_a_segment():
    gate = Gate()
    w = worker(shadow=gate)
    try:
        w.submit(job(1))  # the worker takes it and blocks in the shadow
        for _ in range(100):
            if gate.calls:
                break
            threading.Event().wait(0.01)
        w.submit(job(2))  # waits
        w.submit(job(3))  # replaces 2
        w.submit(job(4, build=False))  # does not replace the waiting build 3
        gate.go.set()
        assert w.wait_idle()
        result = w.take(0.2, epoch=0)
        assert result is not None and result.frame_id == 3
        stats = w.stats()
        assert stats["submitted"] == 4 and stats["dropped"] == 2 and stats["built"] == 2
    finally:
        gate.go.set()
        w.close()


def test_a_failing_stage_is_counted_and_the_worker_carries_on():
    def boom(model, pose, path):
        raise RuntimeError("shadow failed")

    w = worker(shadow=boom)
    try:
        w.submit(job(1))
        assert w.wait_idle()
        assert w.stats()["errors"] == 1 and "shadow failed" in w.stats()["last_error"]
        w.shadow = lambda model, pose, path: None
        w.submit(job(2))
        assert w.wait_idle()
        assert w.take(0.1, epoch=0).frame_id == 2
    finally:
        w.close()


def test_a_grid_snapshot_is_a_copy_and_a_view_shares_only_its_placement():
    grid = TerrainGrid()
    snap, view = grid.snapshot(), grid.view()
    grid.road[3, 3] = 7.0
    grid.seen_by_class[next(iter(grid.seen_by_class))][3, 3] = 1.0
    grid.origin += 5.0
    assert snap.road[3, 3] == 0.0 and np.isneginf(next(iter(snap.seen_by_class.values()))[3, 3])
    assert not np.allclose(snap.origin, grid.origin) and not np.allclose(view.origin, grid.origin)
    assert isinstance(snap, TerrainGrid) and isinstance(view, TerrainGrid)


def test_f6_07_the_schedule_keeps_the_road_model_off_segmentation_ticks():
    """Sixth review, F6-07: the two optional stages share a tick exactly when the phase is a
    multiple of gcd(segment_every, road_model_every); such a phase is replaced, and when
    every phase is (segmenting every tick) the overlap is reported as unavoidable."""
    ok = optional_schedule(StackConfig())
    assert ok["road_model_phase"] == 1 and not ok["overlap"]
    moved = optional_schedule(StackConfig(road_model_phase=2))
    assert moved["configured_phase"] == 2 and moved["road_model_phase"] == 1 and not moved["overlap"] and "phase 1 used" in moved["note"]
    every = optional_schedule(StackConfig(segment_every=1))
    assert every["overlap"] and "cannot be scheduled away" in every["note"]
    for seg, rm, phase in ((2, 4, 0), (3, 6, 3), (2, 5, 0)):
        s = optional_schedule(SimpleNamespace(segment_every=seg, road_model_every=rm, road_model_phase=phase))
        ticks = range(seg * rm * 3)
        shared = [t for t in ticks if t % seg == 0 and t % rm == s["road_model_phase"]]
        assert bool(shared) == s["overlap"], (seg, rm, phase, s)
    with pytest.raises(ValueError):
        StackConfig(semantic_mode="fast")


def test_f7_04_results_carry_their_wall_clock_age_and_late_ones_are_counted():
    """Seventh review, F7-04: simulation time alone says nothing of real time in synchronous
    CARLA. A taken result reports its queue wait, run time and wall-clock age from the job's
    submission; one older on the wall clock than max_age_s is kept (shadow information) but
    counted late, and the shares of drops, failures, stale and late results are reported."""
    gate = Gate()
    w = worker(shadow=gate, max_age_s=0.05)
    try:
        w.submit(job(1))
        threading.Event().wait(0.12)  # the job runs 0.12 s on the wall clock, no simulation time
        gate.go.set()
        assert w.wait_idle()
        result = w.take(0.05, epoch=0)
        assert result is not None
        take = w.last_take
        # The job is held 0.12 s from its submission; how much of that is queue wait and how
        # much run time depends on when the worker thread picked it up (87 ms run under load).
        assert take["wall_age_ms"] >= 100.0 and take["queue_wait_ms"] + take["run_ms"] >= 100.0 and take["queue_wait_ms"] >= 0.0 and take["sim_age_s"] == 0.0
        stats = w.stats()
        assert stats["late_wall"] == 1 and stats["late_share"] == 1.0 and stats["dropped_share"] == 0.0 and stats["error_share"] == 0.0
    finally:
        gate.go.set()
        w.close()



def test_eighth_review_a_result_late_on_the_wall_clock_is_refused_when_admission_asks():
    """Before a semantic layer may steer or set a speed, wall-clock age is an admission rule:
    with max_wall_age_s set, a result recent in simulation time but old in real time is
    refused (stale_wall), as a stale one is; unset, it is kept and counted late (F7-04)."""
    gate = Gate()
    w = worker(shadow=gate, max_age_s=0.5, max_wall_age_s=0.05)
    try:
        w.submit(job(1))
        threading.Event().wait(0.12)
        gate.go.set()
        assert w.wait_idle()
        assert w.take(0.05, epoch=0) is None
        stats = w.stats()
        assert stats["stale_wall"] == 1 and stats["taken"] == 0 and stats["stale_share"] == 1.0
    finally:
        gate.go.set()
        w.close()
    assert StackConfig().semantic_max_wall_age_s is None  # shadow only: reported, not refused
    with pytest.raises(ValueError):
        StackConfig(semantic_max_wall_age_s=0.0)
