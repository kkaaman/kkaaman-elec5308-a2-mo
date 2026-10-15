"""Known coordinates, round trips, units and invalid camera geometry."""

import math

import numpy as np
import pytest

import ground_projection as legacy
from geometry import ground_projection as canonical
from geometry.camera import (
    CameraIntrinsics,
    ego_point_to_pixel,
    pixel_depth_to_ego,
    pixel_to_ground,
)
from geometry.transforms import (
    ego_to_world_matrix,
    inverse_rigid,
    rigid_matrix,
    transform_point,
)


@pytest.fixture
def calibration():
    return CameraIntrinsics(960, 540, 400, 400, 480, 270).calibration(
        ((0, 0, 1, 1.5), (-1, 0, 0, 0), (0, -1, 0, 1.7), (0, 0, 0, 1))
    )


def test_bl_d01_worked_coordinate_example(calibration):
    ego = pixel_depth_to_ego(680, 310, 10, calibration)
    assert ego == pytest.approx([11.5, -5, 0.7])
    world = transform_point(ego, ego_to_world_matrix(100, 20, math.pi / 2))
    assert world == pytest.approx([105, 31.5, 0.7])


@pytest.mark.parametrize("yaw", [0, math.pi / 2, -math.pi / 2, math.pi, 0.37])
@pytest.mark.parametrize(
    "pixel,depth", [((480, 270), 10), ((680, 310), 10), ((320, 400), 3)]
)
def test_pixel_world_roundtrip_multiple_headings(yaw, pixel, depth, calibration):
    ego = pixel_depth_to_ego(*pixel, depth, calibration)
    matrix = ego_to_world_matrix(100, -20, yaw, 2)
    world = transform_point(ego, matrix)
    recovered = transform_point(world, inverse_rigid(matrix))
    assert recovered == pytest.approx(ego, abs=1e-10)
    assert ego_point_to_pixel(recovered, calibration) == pytest.approx(pixel, abs=1e-8)


def test_ground_intersection_horizon_and_behind(calibration):
    assert pixel_to_ground(480, 270, calibration) is None
    assert pixel_to_ground(480, 200, calibration) is None
    assert pixel_to_ground(680, 310, calibration) == pytest.approx([18.5, -8.5, 0])
    assert ego_point_to_pixel((0, 0, 0), calibration) is None


def test_legacy_imports_are_canonical_and_preserve_valid_results():
    assert legacy.pixel_to_vehicle_local is canonical.pixel_to_vehicle_local
    assert legacy.vehicle_local_to_world is canonical.vehicle_local_to_world
    assert (
        legacy.project_ground_point_to_pixel is canonical.project_ground_point_to_pixel
    )
    args = (960, 540, 400, 2.4, 1.5)
    pixel = legacy.project_ground_point_to_pixel(11.5, -5, *args)
    assert pixel == pytest.approx((680, 366))
    assert legacy.pixel_to_vehicle_local(*pixel, *args) == pytest.approx((11.5, -5))
    assert legacy.vehicle_local_to_world(
        11.5, -5, 100, 20, math.pi / 2
    ) == pytest.approx((105, 31.5))
    assert legacy.pixel_to_vehicle_local(480, 270, *args) is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 0},
        {"height": True},
        {"fx_px": 0},
        {"fy_px": -1},
        {"fx_px": float("nan")},
        {"cx_px": float("inf")},
    ],
)
def test_invalid_intrinsics_rejected(kwargs):
    values = dict(width=960, height=540, fx_px=400, fy_px=400, cx_px=480, cy_px=270)
    values.update(kwargs)
    with pytest.raises(ValueError):
        CameraIntrinsics(**values)


@pytest.mark.parametrize("depth", [0, -1, float("nan"), float("inf")])
def test_invalid_depth_rejected(depth, calibration):
    with pytest.raises(ValueError):
        pixel_depth_to_ego(480, 270, depth, calibration)


def test_fov_radians_and_metre_distance():
    intrinsics = CameraIntrinsics.from_horizontal_fov(800, 600, math.pi / 2)
    assert intrinsics.fx_px == pytest.approx(400)
    with pytest.raises(ValueError):
        CameraIntrinsics.from_horizontal_fov(800, 600, 90)
    matrix = ego_to_world_matrix(100, 20, math.pi / 2)
    assert np.linalg.norm(
        transform_point((3, 4, 0), matrix) - matrix[:3, 3]
    ) == pytest.approx(5)


@pytest.mark.parametrize("kind", ["reflection", "scale", "nan", "bad_shape"])
def test_invalid_rigid_transform_rejected(kind):
    matrix = np.eye(4)
    if kind == "reflection":
        matrix[1, 1] = -1
    elif kind == "scale":
        matrix[0, 0] = 2
    elif kind == "nan":
        matrix[0, 3] = np.nan
    else:
        matrix = np.eye(3)
    with pytest.raises(ValueError):
        rigid_matrix(matrix)


def test_pitched_mount_intersection_and_distinct_focal_lengths(calibration):
    pitch = math.radians(10)
    c, s = math.cos(pitch), math.sin(pitch)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    matrix = np.asarray(calibration.camera_to_ego).copy()
    matrix[:3, :3] = rotation @ matrix[:3, :3]
    pitched = CameraIntrinsics(960, 540, 400, 400, 480, 270).calibration(matrix)
    assert pixel_to_ground(480, 270, pitched) == pytest.approx(
        [1.5 + 1.7 / math.tan(pitch), 0, 0]
    )
    distinct = CameraIntrinsics(960, 540, 200, 400, 480, 270).calibration(
        calibration.camera_to_ego
    )
    assert pixel_depth_to_ego(680, 310, 10, distinct) == pytest.approx([11.5, -10, 0.7])


def test_legacy_invalid_inputs_and_behind_camera():
    with pytest.raises(ValueError):
        legacy.pixel_to_vehicle_local(480, 400, 960, 540, 0, 2.4, 1.5)
    with pytest.raises(ValueError):
        legacy.pixel_to_vehicle_local(480, 400, 960, 540, 400, -1, 1.5)
    assert legacy.project_ground_point_to_pixel(1.5, 0, 960, 540, 400, 2.4, 1.5) is None
