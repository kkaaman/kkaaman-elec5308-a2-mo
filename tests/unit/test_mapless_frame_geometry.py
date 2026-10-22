"""SensorFrame contract and mount geometry for the mapless stack."""

import math

import numpy as np
import pytest

from geometry.camera import ego_point_to_pixel
from mapless.frame import CameraImage, GnssFix, ImuSample, LidarSweep, RadarReturns, SensorFrame
from mapless.geometry import camera_calibration, rotation, sensor_to_ego, yaw_from_compass
from mapless.sensor_suite import CameraSpec, Mount

OFFSET = 1.38641


def frame(**changes):
    spec = CameraSpec("front", Mount(1.5, 0.0, 2.0), 64, 36, 90.0)
    values = dict(
        episode_id="e",
        frame_id=3,
        sim_time_s=0.15,
        received_monotonic_s=10.0,
        cameras=(CameraImage("front", np.zeros((36, 64, 3), np.uint8), camera_calibration(spec, OFFSET)),),
        lidars=(LidarSweep("roof", np.zeros((5, 4), np.float32), np.eye(4)),),
        radars=(RadarReturns("radar_front", np.array([[10.0, 0.1, 0.0, -2.0]]), np.eye(4)),),
        gnss=GnssFix(49.0, 8.0, 0.0),
        imu=ImuSample((0, 0, 9.81), (0, 0, 0.1), 1.0),
        speed_mps=4.8,
    )
    values.update(changes)
    return SensorFrame(**values)


def test_frame_is_read_only_and_validated():
    f = frame()
    with pytest.raises(ValueError):
        f.cameras[0].rgb[0, 0, 0] = 1
    with pytest.raises(ValueError):
        f.lidars[0].points[0, 0] = 1
    assert f.camera("front").name == "front" and f.imu.yaw_rate_rps == 0.1
    with pytest.raises(ValueError, match="at least one camera"):
        frame(cameras=())
    with pytest.raises(ValueError, match="unique"):
        frame(radars=(RadarReturns("front", np.zeros((0, 4)), np.eye(4)),))
    with pytest.raises(ValueError, match="finite"):
        frame(lidars=(LidarSweep("roof", np.full((1, 4), np.nan), np.eye(4)),))
    with pytest.raises(ValueError, match="sensor_to_ego"):
        frame(lidars=(LidarSweep("roof", np.zeros((1, 4)), np.diag([1, -1, 1, 1])),))


def test_mount_rotation_conventions():
    # Yaw +90 deg (left): sensor forward points along vehicle +y.
    assert rotation(90, 0) @ np.array([1, 0, 0]) == pytest.approx([0, 1, 0], abs=1e-12)
    # Nose-up pitch: sensor forward gains +z.
    assert (rotation(0, 10) @ np.array([1, 0, 0]))[2] > 0.17
    m = sensor_to_ego(Mount(1.5, 0.2, 2.0, yaw_deg=180), OFFSET)
    assert m[:3, 3] == pytest.approx([1.5 + OFFSET, 0.2, 2.0])
    assert m[:3, :3] @ np.array([1, 0, 0]) == pytest.approx([-1, 0, 0], abs=1e-12)


@pytest.mark.parametrize("yaw", [0.0, 60.0, -60.0, 180.0])
def test_camera_calibration_projects_a_point_on_the_optical_axis_to_the_centre(yaw):
    spec = CameraSpec("c", Mount(1.2, 0.5, 1.9, yaw_deg=yaw), 640, 360, 100.0)
    cal = camera_calibration(spec, OFFSET)
    origin = np.array([1.2 + OFFSET, 0.5, 1.9])
    ahead = origin + 10 * np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), 0.0])
    u, v = ego_point_to_pixel(tuple(ahead), cal)
    assert (u, v) == pytest.approx((320, 180), abs=1e-6)
    # A point to the camera's left lands on the left half of the image.
    left = ahead + 2 * np.array([-math.sin(math.radians(yaw)), math.cos(math.radians(yaw)), 0.0])
    assert ego_point_to_pixel(tuple(left), cal)[0] < 320


def test_radar_points_and_compass_heading():
    r = RadarReturns("front", np.array([[10.0, math.radians(30), 0.0, -1.0]]), sensor_to_ego(Mount(2.4, 0, 0.7), OFFSET))
    x, y, z = r.points_ego()[0]
    assert (x, y, z) == pytest.approx((2.4 + OFFSET + 10 * math.cos(math.radians(30)), 5.0, 0.7))
    # Compass is clockwise from north; east (compass 90 deg) is vehicle yaw 0.
    assert yaw_from_compass(math.radians(90)) == pytest.approx(0.0)
    assert yaw_from_compass(0.0) == pytest.approx(math.pi / 2)  # facing north
    assert yaw_from_compass(math.radians(180)) == pytest.approx(-math.pi / 2)  # facing south
