"""Slow diagnostic work cannot build an unbounded queue or hold dispatch."""

import threading
from types import SimpleNamespace

import pytest

from visualisation.release_observer import AsyncHudObserver


def test_busy_observer_skips_frames_and_drains_before_close():
    entered, release = threading.Event(), threading.Event()
    calls = []
    def observer(frame, packet):
        calls.append(frame.frame_id)
        entered.set()
        assert release.wait(2)
    hud = AsyncHudObserver(observer, period_ticks=10)
    try:
        hud(SimpleNamespace(episode_id="e",frame_id=1),None)
        assert entered.wait(2)
        hud(SimpleNamespace(episode_id="e",frame_id=11),None)
        hud(SimpleNamespace(episode_id="e",frame_id=21),None)
        assert calls == [1]
    finally:
        release.set()
        hud.close()


def test_worker_error_surfaces_for_optional_boundary_to_disable():
    def observer(*args):
        raise RuntimeError("HUD failed")
    hud=AsyncHudObserver(observer, period_ticks=10)
    try:
        frame=SimpleNamespace(episode_id="e",frame_id=1)
        hud(frame,None)
        with pytest.raises(RuntimeError, match="HUD failed"):
            hud.pending.result(timeout=2)
        with pytest.raises(RuntimeError, match="HUD failed"):
            hud(frame,None)
    finally:
        hud.close()
