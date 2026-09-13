import numpy as np
import pytest

from autodrive.perception.lane_path import (
    extract_ego_path,
    draw_ego_path,
)

def make_straight_lane_mask():

    mask = np.zeros(
        (540, 960),
        dtype=np.uint8,
    )
    mask[220:490, 300:306] = 1
    mask[220:490, 654:660] = 1
    return mask

def test_extract_path_from_straight_lanes():
    mask = make_straight_lane_mask()

    path = extract_ego_path(mask)

    assert path is not None

    assert len(path["left_boundary"]) > 0
    assert len(path["right_boundary"]) > 0
    assert len(path["centerline"]) > 0


def test_centerline_is_between_boundaries():
    mask = make_straight_lane_mask()

    path = extract_ego_path(mask)

    left_x = path["left_boundary"][:, 0]
    right_x = path["right_boundary"][:, 0]
    center_x = path["centerline"][:, 0]

    assert np.all(center_x > left_x)
    assert np.all(center_x < right_x)


def test_centerline_is_near_image_center():
    mask = make_straight_lane_mask()

    path = extract_ego_path(mask)

    center_x = path["centerline"][:, 0]


    assert np.allclose(
        center_x,
        480,
        atol=10,
    )

def test_blank_mask_returns_none():
    mask = np.zeros(
        (540, 960),
        dtype=np.uint8,
    )

    path = extract_ego_path(mask)

    assert path is None

def test_none_mask_raises_error():
    with pytest.raises(ValueError):
        extract_ego_path(None)


def test_three_dimensional_mask_raises_error():
    mask = np.zeros(
        (540, 960, 3),
        dtype=np.uint8,
    )

    with pytest.raises(ValueError):
        extract_ego_path(mask)


def test_draw_path_preserves_image_shape():
    mask = make_straight_lane_mask()

    path = extract_ego_path(mask)

    image = np.zeros(
        (540, 960, 3),
        dtype=np.uint8,
    )

    output = draw_ego_path(
        image,
        path,
    )

    assert output.shape == image.shape