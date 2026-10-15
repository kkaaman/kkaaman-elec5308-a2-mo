"""Bounded exact-frame RGB-D join; callbacks never tick the simulator."""

import math
import threading
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np

from autonomy.interfaces import EgoPose, FrameBundle
from carla_adapter.conversions import axial_depth, decode_depth, decode_rgb


@dataclass(frozen=True)
class CameraRig:
    width: int = 640
    height: int = 360
    fov_deg: float = 90.0
    x_actor_m: float = 1.5
    z_actor_m: float = 2.0

    def __post_init__(self):
        if (
            type(self.width) is not int
            or type(self.height) is not int
            or min(self.width, self.height) < 1
        ):
            raise ValueError("Invalid camera dimensions")
        if (
            not all(math.isfinite(x) for x in (self.fov_deg, self.x_actor_m, self.z_actor_m))
            or not 0 < self.fov_deg < 180
        ):
            raise ValueError("Invalid camera rig")


@dataclass(frozen=True)
class AlignedCapture:
    episode_id: str
    frame_id: int
    sim_time_s: float
    received_monotonic_s: float
    ego: object
    rgb: np.ndarray
    decoded_depth_m: np.ndarray

    def to_bundle(self, calibration, *, established_depth_convention):
        depth = axial_depth(
            self.decoded_depth_m, calibration, convention=established_depth_convention
        )
        return FrameBundle(
            self.episode_id,
            self.frame_id,
            self.sim_time_s,
            self.received_monotonic_s,
            self.ego,
            self.rgb,
            calibration,
            depth,
            self.frame_id,
        )


class FrameJoiner:
    """At most capacity frames per stream; old/missing observations never reused."""

    def __init__(self, episode_id, capacity=8):
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("Nonempty episode required")
        if type(capacity) is not int or capacity < 1:
            raise ValueError("Positive queue capacity required")
        self.episode_id, self.capacity = episode_id, capacity
        self._streams = {key: {} for key in ("rgb", "depth", "ego")}
        self._condition = threading.Condition()
        self._retired = -1
        self._closed = False
        self.events = Counter()

    def put(self, kind, episode, frame, timestamp, value):
        if (
            kind not in self._streams
            or type(frame) is not int
            or frame < 0
            or not math.isfinite(timestamp)
            or timestamp < 0
        ):
            raise ValueError("Invalid capture metadata")
        if kind == "ego" and not isinstance(value, EgoPose):
            raise TypeError("Expected matched EgoPose snapshot")
        with self._condition:
            if self._closed or episode != self.episode_id or frame <= self._retired:
                self.events["late_or_wrong_episode"] += 1
                return
            stream = self._streams[kind]
            if frame in stream:
                self.events["duplicate"] += 1
                return
            if kind in ("rgb", "depth"):
                value = np.array(value, copy=True)
                value.setflags(write=False)
            stream[frame] = (timestamp, value)
            while len(stream) > self.capacity:
                del stream[min(stream)]
                self.events["overflow"] += 1
            self._condition.notify_all()

    def _retire(self, frame):
        self._retired = max(self._retired, frame)
        for stream in self._streams.values():
            for key in list(stream):
                if key <= frame:
                    del stream[key]
                    self.events["discarded"] += 1

    def wait(self, frame, timeout_s, stop):
        """On timeout/invalid alignment call stop outside the callback lock."""
        if (
            type(frame) is not int
            or frame < 0
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
            or not callable(stop)
        ):
            raise ValueError("Invalid join request")
        deadline = time.monotonic() + timeout_s
        try:
            with self._condition:
                while True:
                    if self._closed or frame <= self._retired:
                        raise RuntimeError("Frame join closed or already retired")
                    if all(frame in stream for stream in self._streams.values()):
                        rgb, depth, ego = (self._streams[k][frame] for k in ("rgb", "depth", "ego"))
                        if max(rgb[0], depth[0], ego[0]) - min(rgb[0], depth[0], ego[0]) > 1e-6:
                            self.events["timestamp_mismatch"] += 1
                            self._retire(frame)
                            raise ValueError("RGB/depth/ego timestamp mismatch")
                        if (
                            rgb[1].dtype != np.uint8
                            or rgb[1].ndim != 3
                            or rgb[1].shape[2] != 3
                            or depth[1].shape != rgb[1].shape[:2]
                        ):
                            self._retire(frame)
                            raise ValueError("RGB/depth dimensions or dtype mismatch")
                        result = AlignedCapture(
                            self.episode_id,
                            frame,
                            ego[0],
                            time.monotonic(),
                            ego[1],
                            rgb[1],
                            depth[1],
                        )
                        self._retire(frame)
                        self.events["joined"] += 1
                        return result
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self.events["timeout"] += 1
                        self._retire(frame)
                        raise TimeoutError(f"Missing aligned frame {frame}")
                    self._condition.wait(remaining)
        except Exception:
            stop()
            raise

    def stats(self):
        with self._condition:
            return dict(self.events)

    def close(self):
        with self._condition:
            self._closed = True
            for stream in self._streams.values():
                stream.clear()
            self._condition.notify_all()


class RGBDCapture:
    """Own two rigid cameras within an active SynchronousWorldSession."""

    def __init__(self, owner, vehicle, episode_id, rig=None, capacity=8):
        self.owner, self.vehicle = owner, vehicle
        self.rig = rig or CameraRig()
        self.joiner = FrameJoiner(episode_id, capacity)
        self.sensors = []
        self.callback_errors = []

    def __enter__(self):
        import carla

        self.owner._require_owner()
        try:
            for kind, blueprint_id in [
                ("rgb", "sensor.camera.rgb"),
                ("depth", "sensor.camera.depth"),
            ]:
                bp = self.owner.world.get_blueprint_library().find(blueprint_id)
                for key, value in {
                    "image_size_x": self.rig.width,
                    "image_size_y": self.rig.height,
                    "fov": self.rig.fov_deg,
                    "sensor_tick": 0.0,
                    "lens_k": 0.0,
                    "lens_kcube": 0.0,
                }.items():
                    bp.set_attribute(key, str(value))
                sensor = self.owner.world.spawn_actor(
                    bp,
                    carla.Transform(carla.Location(x=self.rig.x_actor_m, z=self.rig.z_actor_m)),
                    attach_to=self.vehicle,
                    attachment_type=carla.AttachmentType.Rigid,
                )
                self.sensors.append(sensor)
                # Register immediately so owner cleanup can retry on failure.
                self.owner._actors.append(sensor)
                sensor.listen(lambda image, channel=kind: self._receive(channel, image))
            return self
        except BaseException:
            self.close()
            raise

    def _receive(self, kind, image):
        try:
            if (image.width, image.height) != (self.rig.width, self.rig.height):
                raise ValueError("Camera dimensions changed")
            decoder = decode_rgb if kind == "rgb" else decode_depth
            self.joiner.put(
                kind,
                self.joiner.episode_id,
                image.frame,
                image.timestamp,
                decoder(image.raw_data, image.width, image.height),
            )
        except Exception as error:
            self.callback_errors.append(str(error))
            del self.callback_errors[:-8]

    def record_snapshot(self, frame, ego):
        snapshot = self.owner.world.get_snapshot()
        if snapshot.frame != frame:
            self.owner.emergency_stop()
            raise ValueError("Ego snapshot does not match requested camera frame")
        self.joiner.put(
            "ego", self.joiner.episode_id, frame, snapshot.timestamp.elapsed_seconds, ego
        )

    def read(self, frame, timeout_s=5.0):
        return self.joiner.wait(frame, timeout_s, self.owner.emergency_stop)

    def capture(self, frame, ego, timeout_s=5.0):
        self.record_snapshot(frame, ego)
        return self.read(frame, timeout_s)

    def close(self):
        self.joiner.close()
        errors = []
        for sensor in reversed(self.sensors):
            try:
                if sensor.is_alive:
                    sensor.stop()
                    if not sensor.destroy():
                        errors.append(f"Sensor destroy failed: {sensor.id}")
            except Exception as error:
                errors.append(str(error))
        self.sensors.clear()
        if errors:
            raise RuntimeError("; ".join(errors))

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False
