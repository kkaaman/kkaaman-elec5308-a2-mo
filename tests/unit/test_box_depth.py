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


def test_two_surfaces_rejected():
    depth = np.full((40, 40), 10.0)
    depth[:, 20:] = 50
    assert (
        estimate_box_depth(depth, PixelBox(0, 0, 40, 40), calibration()).reason
        == "mixed_or_spread_depth"
    )


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
