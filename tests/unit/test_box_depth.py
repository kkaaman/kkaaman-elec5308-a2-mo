"""Foreground/background support, invalid depth and off-axis reconstruction."""

import numpy as np
import pytest

from geometry.camera import CameraIntrinsics
from obstacle_perception.box_depth import DepthPolicy, estimate_box_depth, pinhole_fallback
from obstacle_perception.types import PixelBox


def calibration():
    return CameraIntrinsics(40, 40, 20, 20, 20, 20).calibration(np.eye(4))


def test_inner_foreground_ignores_outer_background():
    depth = np.full((40, 40), 50.0, np.float32)
    depth[10:30, 10:30] = 10
    result = estimate_box_depth(depth, PixelBox(0, 0, 40, 40), calibration())
    assert result.valid and result.axial_depth_m == 10
    assert result.point_ego_m == pytest.approx((0, 0, 10))
    assert not result.usable_for_clearance


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0, -1, 1000])
def test_unknown_depth_never_clear(value):
    result = estimate_box_depth(
        np.full((40, 40), value, dtype=float), PixelBox(0, 0, 40, 40), calibration()
    )
    assert not result.valid and result.axial_depth_m is None and not result.usable_for_clearance


def test_two_surfaces_use_the_near_one_never_the_background():
    depth = np.full((40, 40), 10.0)
    depth[:, 20:] = 50
    result = estimate_box_depth(depth, PixelBox(0, 0, 40, 40), calibration())
    assert result.valid and result.source == "foreground_cluster"
    assert result.axial_depth_m == 10


def test_small_near_patch_over_background_stays_unknown():
    depth = np.full((40, 40), 50.0)
    depth[10:30, 10:13] = 10  # ~15% of the central ROI
    depth[20:30, 13:30] = 30
    result = estimate_box_depth(depth, PixelBox(0, 0, 40, 40), calibration())
    assert not result.valid and result.reason == "mixed_or_spread_depth"


def _rig():
    import math

    return CameraIntrinsics.from_horizontal_fov(640, 360, math.pi / 2).calibration(
        ((0, 0, 1, 2.88641), (-1, 0, 0, 0), (0, -1, 0, 2.0), (0, 0, 0, 1))
    )


def test_thin_pedestrian_box_with_background_gets_body_depth():
    # Live Perception (18 Oct): YOLO person boxes beyond ~15 m are mostly
    # background, so the old spread check made their depth invalid.
    cal = _rig()
    depth = np.full((360, 640), 45.0, np.float32)
    depth[170:210, 316:324] = 15.2  # body, ~8 px wide, inside a 20 px wide box
    result = estimate_box_depth(depth, PixelBox(310, 168, 330, 212), cal)
    assert result.valid and result.source == "foreground_cluster"
    assert result.axial_depth_m == pytest.approx(15.2)


def test_road_pixels_inside_the_roi_are_ignored():
    cal = _rig()
    rows = (np.arange(360) + 0.5 - 180) / 320
    with np.errstate(divide="ignore"):
        ground = np.where(rows > 0, 2.0 / rows, np.nan)
    depth = np.repeat(ground[:, None], 640, axis=1).astype(np.float32)
    depth[150:205, 280:360] = 12.0  # car rear face; road visible below it
    result = estimate_box_depth(depth, PixelBox(280, 150, 360, 260), cal)
    assert result.valid and result.axial_depth_m == pytest.approx(12.0)


def test_clipping_empty_and_small_support():
    depth = np.ones((40, 40)) * 10
    assert estimate_box_depth(depth, PixelBox(50, 50, 60, 60), calibration()).reason == "empty_roi"
    assert not estimate_box_depth(depth, PixelBox(0, 0, 2, 2), calibration()).valid


def test_off_axis_axial_depth():
    result = estimate_box_depth(np.ones((40, 40)) * 10, PixelBox(20, 10, 40, 30), calibration())
    assert result.point_ego_m == pytest.approx((5, 0, 10))
    assert not pinhole_fallback(PixelBox(0, 0, 10, 10), 2, calibration()).valid


def test_invalid_policy():
    with pytest.raises(ValueError):
        DepthPolicy(inner_fraction=0)
