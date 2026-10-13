"""Contracts in autonomy.interfaces: units, frames, clocks and validation."""

import json
import math

import numpy as np
import pytest

from autonomy.interfaces import (
    CameraModel,
    ControlCommand,
    ControlPacket,
    Diagnostics,
    DriveMode,
    EvaluationPacket,
    Extrinsics,
    FrameBundle,
    LaneGraph,
    LaneSegment,
    Maneuver,
    Mission,
    MotionState,
    SensorStatus,
    SimulatorInterface,
    VehicleGeometry,
)

MOUNT = Extrinsics(x_m=1.5, y_m=0.0, z_m=2.4)


def camera(width=8, height=6):
    return CameraModel.from_fov(width, height, math.radians(90.0), MOUNT)


def bundle(**overrides):
    cam = overrides.pop("camera", camera())
    fields = dict(
        run_id="run-1",
        frame_id=10,
        sim_time_s=1.0,
        host_tick_s=100.0,
        host_receive_s=100.02,
        ego=MotionState(0.0, 0.0, 0.0, 5.0),
        camera=cam,
        rgb=np.zeros((cam.height_px, cam.width_px, 3), np.uint8),
        depth_m=np.full((cam.height_px, cam.width_px), 10.0, np.float32),
        sensors=(SensorStatus("rgb", True, 10), SensorStatus("depth", True, 10)),
    )
    fields.update(overrides)
    return FrameBundle(**fields)


# ----------------------------------------------------------------------- camera
def test_camera_from_fov_matches_carla_pinhole():
    cam = CameraModel.from_fov(800, 600, math.radians(90.0), MOUNT)
    assert cam.fx_px == pytest.approx(400.0)
    assert cam.fy_px == cam.fx_px
    assert (cam.cx_px, cam.cy_px) == (400.0, 300.0)
    assert cam.depth_kind == "optical_z" and not cam.depth_kind_verified


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(cx_px=900.0),
        dict(fx_px=0.0),
        dict(width_px=0),
        dict(depth_kind="disparity"),
    ],
)
def test_camera_rejects_invalid_intrinsics(kwargs):
    base = dict(width_px=800, height_px=600, fx_px=400.0, fy_px=400.0,
                cx_px=400.0, cy_px=300.0, mount=MOUNT)
    base.update(kwargs)
    with pytest.raises(ValueError):
        CameraModel(**base)


def test_numpy_scalars_are_accepted():
    Extrinsics(np.float32(1.5), np.float64(0.0), np.float32(2.4))


# ------------------------------------------------------------------- frame bundle
def test_valid_bundle_and_frame_age():
    b = bundle()
    assert b.valid
    assert b.frame_age_s == pytest.approx(0.02)


def test_bundle_rejects_bgra_or_wrong_size_rgb():
    with pytest.raises(ValueError, match="rgb"):
        bundle(rgb=np.zeros((6, 8, 4), np.uint8))
    with pytest.raises(ValueError, match="rgb"):
        bundle(rgb=np.zeros((6, 8, 3), np.float32))


def test_bundle_rejects_depth_not_matching_camera():
    with pytest.raises(ValueError, match="depth"):
        bundle(depth_m=np.zeros((5, 8), np.float32))
    with pytest.raises(ValueError, match="depth"):
        bundle(depth_m=np.zeros((6, 8), np.uint8))


def test_bundle_rejects_reading_from_another_frame():
    with pytest.raises(ValueError, match="frame 9"):
        bundle(sensors=(SensorStatus("depth", True, 9),))


def test_bundle_rejects_receive_before_tick():
    with pytest.raises(ValueError, match="host_receive_s"):
        bundle(host_receive_s=99.0)


def test_invalid_sensor_makes_bundle_invalid_and_needs_reason():
    b = bundle(sensors=(SensorStatus("rgb", True, 10), SensorStatus("depth", False, None, "timeout")))
    assert not b.valid
    with pytest.raises(ValueError, match="reason"):
        SensorStatus("depth", False)


# ----------------------------------------------------------------------- motion
def test_motion_state_clamps_noise_but_rejects_reverse():
    assert MotionState(1.0, 2.0, 0.1, -0.01).to_vehicle_state().speed_mps == 0.0
    with pytest.raises(ValueError, match="reversing"):
        MotionState(1.0, 2.0, 0.1, -0.5).to_vehicle_state()


def test_vehicle_geometry_sanity():
    VehicleGeometry("vehicle.test", 2.9, 3.9, 1.0, 1.9, math.radians(70))
    with pytest.raises(ValueError, match="front bumper"):
        VehicleGeometry("vehicle.test", 2.9, 2.5, 1.0, 1.9, math.radians(70))


# ---------------------------------------------------------------------- control
def test_stop_packet_is_emergency_brake():
    packet = ControlPacket.stop(frame_id=3, reason="sensor timeout")
    assert packet.emergency and packet.command.brake == 1.0 and packet.command.throttle == 0.0


def test_emergency_packet_cannot_accelerate():
    with pytest.raises(ValueError, match="emergency"):
        ControlPacket(1, ControlCommand(0.0, 0.3, 0.0), "bad", emergency=True)


def test_packet_needs_reason_and_a1_command():
    with pytest.raises(ValueError, match="reason"):
        ControlPacket(1, ControlCommand(0.0, 0.0, 0.5), "")
    with pytest.raises(TypeError):
        ControlPacket(1, (0.0, 0.0, 0.5), "tuple is not a command")


# ------------------------------------------------------------------ graph/mission
def graph():
    a = LaneSegment("a", ((0.0, 0.0), (10.0, 0.0)), successors=("b",))
    b = LaneSegment("b", ((10.0, 0.0), (10.0, 10.0)), in_junction=True)
    return LaneGraph("Town10HD_Opt", {"b": b, "a": a}, source_hash="abc")


def test_graph_round_trips_deterministically():
    g = graph()
    text = json.dumps(g.to_dict())
    assert [s["segment_id"] for s in g.to_dict()["segments"]] == ["a", "b"]
    again = LaneGraph.from_dict(json.loads(text))
    assert again.to_dict() == g.to_dict()
    assert again.segments["a"].length_m == pytest.approx(10.0)


def test_graph_rejects_unknown_successor_and_is_read_only():
    with pytest.raises(ValueError, match="unknown successor"):
        LaneGraph("m", {"a": LaneSegment("a", ((0, 0), (1, 0)), successors=("zzz",))})
    with pytest.raises(TypeError):
        graph().segments["c"] = None


def test_mission_requires_maneuver_values():
    Mission("Town10HD_Opt", MotionState(0, 0, 0, 0), (Maneuver.LEFT, Maneuver.STRAIGHT), (50.0, 20.0))
    with pytest.raises(TypeError):
        Mission("Town10HD_Opt", MotionState(0, 0, 0, 0), ("left",), (50.0, 20.0))


# ------------------------------------------------------------------ diagnostics
def test_diagnostics_maps_are_read_only():
    d = Diagnostics(1, 0.1, "CRUISE", DriveMode.DRIVE, 5.0, 4.8,
                    speed_caps_mps={"cruise": 5.0, "curve": 6.2}, timings_ms={"detect": 12.0})
    assert min(d.speed_caps_mps.values()) == 5.0
    with pytest.raises(TypeError):
        d.timings_ms["detect"] = 0.0


# ------------------------------------------------------------- simulator protocol
class _TinyWorld:
    vehicle_geometry = VehicleGeometry("vehicle.test", 2.9, 3.9, 1.0, 1.9, 1.2)

    def reset(self, scenario):
        return bundle(frame_id=0, sensors=())

    def apply(self, packet):
        self.last = packet

    def tick(self, timeout_s):
        return bundle(frame_id=1, sensors=())

    def evaluation(self):
        return EvaluationPacket(frame_id=1, ego_truth=MotionState(0, 0, 0, 0))

    def close(self):
        pass


def test_simulator_protocol_is_structural():
    world = _TinyWorld()
    assert isinstance(world, SimulatorInterface)
    first = world.reset({"name": "S0"})
    world.apply(ControlPacket.stop(first.frame_id, "test"))
    assert world.tick(0.5).frame_id == first.frame_id + 1
