"""Capture map policy and the rig check's evaluator-side geometry (no CARLA server)."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from carla_adapter.mapless_capture import capture_map_allowed, main, to_ego_frame
from carla_adapter.sensor_rig_check import (
    box_corners,
    inside_box,
    project,
    rear_axle_pose,
    rect_iou,
    world_to_ego,
)
from mapless.geometry import camera_calibration
from mapless.sensor_suite import CameraSpec, Mount


def test_capture_refuses_the_demo_map_before_touching_carla(tmp_path):
    assert capture_map_allowed("Carla/Maps/Town03") and capture_map_allowed("Town05_Opt")
    for name in ("Town10HD", "Town10HD_Opt", "Town12", "Town15"):
        assert not capture_map_allowed(name)
    with pytest.raises(SystemExit):
        main(["--map", "Town10HD_Opt", "--output", str(tmp_path / "x")])
    assert not (tmp_path / "x").exists()


def test_world_to_vehicle_frame_round_trip():
    transform = SimpleNamespace(location=SimpleNamespace(x=10.0, y=-5.0, z=0.2), rotation=SimpleNamespace(yaw=-90.0))
    pose = rear_axle_pose(transform, 1.4)
    # CARLA yaw -90 = facing core +y (north); rear axle 1.4 m behind.
    assert pose == pytest.approx((10.0, 5.0 - 1.4, 0.2, math.pi / 2))
    ahead = world_to_ego([(10.0, 15.0, 0.2)], pose)[0]
    assert ahead == pytest.approx((11.4, 0.0, 0.0))
    assert to_ego_frame(pose, (10.0, 15.0), math.pi / 2) == pytest.approx((11.4, 0.0, 0.0))


def test_box_membership_tells_a_box_from_its_mirror_image():
    centre, yaw, extent = np.array([20.0, 3.5, 0.75]), 0.0, np.array([1.8, 0.9, 0.75])
    corners = box_corners(centre, yaw, extent)
    assert inside_box(corners, centre, yaw, extent, 0.01).all()
    assert not inside_box(corners, centre * (1, -1, 1), -yaw, extent, 0.15).any()


def test_projection_and_overlap():
    cal = camera_calibration(CameraSpec("c", Mount(1.5, 0, 2.0), 1280, 720, 90.0), 1.38641)
    px = project(np.array([[30.0, 0.0, 2.0], [-5.0, 0.0, 2.0]]), cal)
    assert len(px) == 1 and px[0] == pytest.approx((640, 360), abs=1e-6)
    assert rect_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert rect_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
