"""Evaluator-side calibration geometry (no simulator)."""

import math

import numpy as np
import pytest

from evaluation import rig_calibration as cal
from mapless.geometry import camera_calibration, sensor_to_ego
from mapless.sensor_suite import CameraSpec, Mount


def test_ray_box_distance_hits_the_near_face_and_misses_cleanly():
    centre, extent = np.array([10.0, 0.0, 0.5]), np.array([2.0, 1.0, 0.5])
    assert cal.ray_box_distance([0, 0, 0.5], [1, 0, 0], centre, 0.0, extent) == pytest.approx(8.0)
    assert cal.ray_box_distance([0, 0, 0.5], [0, 1, 0], centre, 0.0, extent) is None
    # Rotated 90 degrees: the near face is now 1 m from the centre.
    assert cal.ray_box_distance([0, 0, 0.5], [1, 0, 0], centre, math.pi / 2, extent) == pytest.approx(9.0)
    assert cal.ray_box_distance(centre, [1, 0, 0], centre, 0.0, extent) == 0.0  # inside


def test_expected_range_rate_with_and_without_lever_arm():
    radar = sensor_to_ego(Mount(2.5, 0.0, 0.7), 1.4)  # 3.9 m ahead of the rear axle
    ahead, left = np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]])
    # Driving at 5 m/s towards a static point ahead: range shrinks at 5 m/s.
    assert cal.expected_range_rate(ahead, radar, (5.0, 0.0), 0.0)[0] == pytest.approx(-5.0)
    # Turning left at 0.5 rad/s: the nose moves left at omega * r, so a point to the left closes.
    with_arm = cal.expected_range_rate(left, radar, (5.0, 0.0), 0.5, velocity_origin_x_m=1.4)[0]
    assert with_arm == pytest.approx(-0.5 * (3.9 - 1.4))
    assert cal.expected_range_rate(left, radar, (5.0, 0.0), 0.5, lever_arm=False)[0] == 0.0
    # A target receding at 3 m/s while the ego is still.
    assert cal.expected_range_rate(ahead, radar, (0.0, 0.0), 0.0, (3.0, 0.0))[0] == pytest.approx(3.0)


def test_edges_masks_and_instances():
    assert cal.edge_errors((10, 10, 50, 40), (11, 10, 52, 43), 100, 100) == {"left": 1, "top": 0, "right": 2, "bottom": 3}
    # Edges at or past the image border are not scored.
    assert cal.edge_errors((-5, 10, 50, 40), (0, 10, 50, 40), 100, 100) == {"top": 0, "right": 0, "bottom": 0}
    ids = np.zeros((20, 20), int)
    tags = np.full((20, 20), 1)  # road everywhere
    ids[:, :] = 3  # the road carries an instance code too, and covers most of the window
    ids[5:10, 5:12], tags[5:10, 5:12] = 7, 14  # the target car
    ids[0:3, 0:3], tags[0:3, 0:3] = 9, 14
    assert cal.target_instance(ids, (0, 0, 20, 20)) == (3, 0)  # without tags the road wins
    key = cal.target_instance(ids, (0, 0, 20, 20), tags)
    assert key == (7, 14)  # scenery tags never count
    assert cal.target_instance(ids, (0, 0, 20, 20), tags, allowed_tags={15}) is None
    assert cal.target_instance(np.zeros((5, 5), int), (0, 0, 5, 5)) is None
    mask = cal.instance_mask(ids, tags, key)
    assert cal.mask_rect(mask) == (5, 5, 12, 10)
    ids = np.where(mask, 7, 0)
    pixels = np.array([[6.5, 6.5], [11.5, 9.5], [15.5, 15.5]])
    assert cal.mask_consistency(pixels, mask, dilate_px=0) == pytest.approx(2 / 3)
    assert cal.mask_consistency(np.array([[13.2, 7.0]]), mask, dilate_px=2) == 1.0


def test_ground_clearance_keeps_road_points_out_of_box_counts():
    centre, extent = np.array([10.0, 0.0, 0.8]), np.array([2.0, 1.0, 0.75])
    road = np.array([[10.0, 0.5, 0.0], [9.0, -0.5, 0.02]])
    body = np.array([[8.1, 0.0, 0.6]])
    points = np.vstack((road, body))
    assert cal.inside_box(points, centre, 0.0, extent, 0.15).tolist() == [True, True, True]
    assert cal.inside_box(points, centre, 0.0, extent, 0.15, 0.2).tolist() == [False, False, True]


def test_projection_and_bearing():
    cal_front = camera_calibration(CameraSpec("c", Mount(1.5, 0, 2.0), 1280, 720, 90.0), 1.4)
    px, front = cal.project(np.array([[30.0, 0.0, 2.0], [-5.0, 0.0, 2.0]]), cal_front)
    assert front.tolist() == [True, False] and px[0] == pytest.approx((640, 360))
    bearing, rng = cal.bearing_range(sensor_to_ego(Mount(0, 0, 0, yaw_deg=90), 0.0), [0.0, 10.0, 0.0])
    assert bearing == pytest.approx(0.0, abs=1e-9) and rng == pytest.approx(10.0)


def test_placements_cover_both_sides_ahead_and_behind():
    rows = cal.placements(3.8, -1.0, 1.4)
    names = {n for n, _, _ in rows}
    assert {"ahead_5m", "ahead_50m", "left_alongside", "right_ahead_15m", "left_behind_10m", "behind_15m"} <= names
    assert all(y in (0.0, 3.5, -3.5) for _, _, y in rows)
    assert len(cal.prop_placements(3.8, -1.0, 1.4)) == 5
