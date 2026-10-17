"""Decoder vectors, exact-frame ordering, bounded storage and timeout stop."""

import threading
import time

import numpy as np
import pytest

from autonomy.interfaces import EgoPose
from carla_adapter.conversions import decode_depth, decode_rgb
from carla_adapter.sensors import CameraRig, FrameJoiner
from navigation_control.types import VehicleState


def put(joiner, kind, frame=1, timestamp=0.05, episode="e"):
    value = (
        np.zeros((2, 3, 3), np.uint8)
        if kind == "rgb"
        else np.full((2, 3), 10.0, np.float32)
        if kind == "depth"
        else EgoPose(VehicleState(0, 0, 0, 0))
    )
    joiner.put(kind, episode, frame, timestamp, value)


def test_known_bgra_rgb_and_depth():
    assert decode_rgb(bytes([3, 2, 1, 255]), 1, 1).tolist() == [[[1, 2, 3]]]
    assert decode_depth(bytes([0, 0, 1, 255]), 1, 1)[0, 0] == pytest.approx(1000 / 16777215)
    assert decode_depth(bytes([1, 0, 0, 255]), 1, 1)[0, 0] == pytest.approx(1000 * 65536 / 16777215)
    assert np.isnan(decode_depth(bytes([0, 0, 0, 255, 255, 255, 255, 255]), 2, 1)).all()


def test_decode_rejects_wrong_size():
    with pytest.raises(ValueError):
        decode_rgb(b"123", 1, 1)
    with pytest.raises(ValueError):
        CameraRig(width=0)


def test_out_of_order_exact_join():
    j = FrameJoiner("e")
    for kind in ("depth", "ego", "rgb"):
        put(j, kind)
    result = j.wait(1, 0.1, lambda: pytest.fail("unexpected stop"))
    assert result.frame_id == 1
    assert not result.rgb.flags.writeable
    put(j, "rgb")
    assert j.events["late_or_wrong_episode"] == 1


def test_timestamp_mismatch_stops():
    j = FrameJoiner("e")
    stops = []
    put(j, "rgb")
    put(j, "depth", timestamp=0.06)
    put(j, "ego")
    with pytest.raises(ValueError):
        j.wait(1, 0.1, lambda: stops.append(True))
    assert stops == [True]


def test_missing_not_mixed_and_storage_bounded():
    j = FrameJoiner("e", capacity=2)
    stops = []
    for i in range(10):
        put(j, "rgb", i)
    assert len(j._streams["rgb"]) == 2
    put(j, "depth", 8)
    put(j, "ego", 9)
    with pytest.raises(TimeoutError):
        j.wait(9, 0.001, lambda: stops.append(True))
    assert stops == [True]
    assert j.events["overflow"] == 8
    put(j, "depth", 9)
    assert 9 not in j._streams["depth"]


def test_delayed_callback_and_wrong_episode():
    j = FrameJoiner("e")
    put(j, "rgb", episode="wrong")
    put(j, "ego")
    put(j, "rgb")
    worker = threading.Thread(target=lambda: (time.sleep(0.01), put(j, "depth")))
    worker.start()
    assert j.wait(1, 0.5, lambda: pytest.fail("stop")).frame_id == 1
    worker.join()
    assert j.events["late_or_wrong_episode"] == 1


def test_frame_dimensions_fail_closed():
    j = FrameJoiner("e")
    put(j, "rgb")
    put(j, "ego")
    j.put("depth", "e", 1, 0.05, np.ones((4, 4)))
    with pytest.raises(ValueError):
        j.wait(1, 0.1, lambda: None)


def test_explicit_depth_convention_required_and_radial_conversion():
    from autonomy.interfaces import CameraCalibration
    from carla_adapter.conversions import axial_depth

    calibration = CameraCalibration(2, 2, 0, 0, tuple(map(tuple, np.eye(4))))
    depth = np.full((2, 3), 10.0, np.float32)
    with pytest.raises(ValueError):
        axial_depth(depth, calibration, convention="unknown")
    assert axial_depth(depth, calibration, convention="radial")[0, 2] == pytest.approx(
        10 / np.sqrt(2)
    )
    assert axial_depth(depth, calibration, convention="axial")[0, 2] == 10


def test_close_and_duplicate():
    j = FrameJoiner("e")
    put(j, "rgb")
    put(j, "rgb")
    assert j.events["duplicate"] == 1
    j.close()
    stops = []
    with pytest.raises(RuntimeError):
        j.wait(1, 0.01, lambda: stops.append(1))
    assert stops == [1]


def test_rgbd_sensor_cleanup_on_partial_start_failure(monkeypatch):
    import sys
    from types import SimpleNamespace

    from carla_adapter.sensors import RGBDCapture

    class Sensor:
        is_alive = True
        id = 9

        def listen(self, callback):
            self.callback = callback

        def stop(self):
            self.stopped = True

        def destroy(self):
            self.is_alive = False
            return True

    sensor = Sensor()

    class World:
        count = 0

        def get_blueprint_library(self):
            return self

        def find(self, key):
            return SimpleNamespace(set_attribute=lambda *args: None)

        def spawn_actor(self, *args, **kwargs):
            self.count += 1
            if self.count == 2:
                raise RuntimeError("injected depth spawn failure")
            return sensor

    world = World()
    owner = SimpleNamespace(world=world, _actors=[], _require_owner=lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "carla",
        SimpleNamespace(
            Transform=lambda x: x,
            Location=lambda **kwargs: kwargs,
            AttachmentType=SimpleNamespace(Rigid=1),
        ),
    )
    with pytest.raises(RuntimeError, match="injected"):
        with RGBDCapture(owner, object(), "e"):
            pass
    assert not sensor.is_alive and sensor.stopped
    assert owner._actors == [sensor]


def test_capture_timestamp_includes_queue_wait(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("carla_adapter.sensors.time.monotonic", lambda: now[0])
    joiner = FrameJoiner("e")
    put(joiner, "rgb")
    now[0] = 100.2
    put(joiner, "depth")
    put(joiner, "ego")
    now[0] = 100.3
    frame = joiner.wait(1, 1, lambda: pytest.fail("unexpected stop"))
    assert frame.received_monotonic_s == 100.0
    assert not joiner._received
