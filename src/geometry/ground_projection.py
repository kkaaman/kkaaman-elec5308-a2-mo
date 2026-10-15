"""A1-compatible horizontal-camera helpers, backed by canonical geometry."""

import math

from .camera import CameraIntrinsics, ego_point_to_pixel, pixel_to_ground
from .transforms import vehicle_local_to_world


def _calibration(
    image_width, image_height, focal_length_px, camera_height_m, camera_forward_offset_m
):
    if not math.isfinite(camera_height_m) or camera_height_m <= 0:
        raise ValueError("Camera height must be positive metres")
    if not math.isfinite(camera_forward_offset_m):
        raise ValueError("Camera offset must be finite metres")
    intrinsics = CameraIntrinsics(
        image_width,
        image_height,
        focal_length_px,
        focal_length_px,
        image_width / 2,
        image_height / 2,
    )
    # Optical X right, Y down, Z forward -> ego X forward, Y left, Z up.
    return intrinsics.calibration(
        (
            (0, 0, 1, camera_forward_offset_m),
            (-1, 0, 0, 0),
            (0, -1, 0, camera_height_m),
            (0, 0, 0, 1),
        )
    )


def pixel_to_vehicle_local(
    u,
    v,
    image_width,
    image_height,
    focal_length_px,
    camera_height_m,
    camera_forward_offset_m,
):
    calibration = _calibration(
        image_width,
        image_height,
        focal_length_px,
        camera_height_m,
        camera_forward_offset_m,
    )
    point = pixel_to_ground(u, v, calibration)
    return None if point is None else (float(point[0]), float(point[1]))


def project_ground_point_to_pixel(
    forward_m,
    left_m,
    image_width,
    image_height,
    focal_length_px,
    camera_height_m,
    camera_forward_offset_m,
):
    calibration = _calibration(
        image_width,
        image_height,
        focal_length_px,
        camera_height_m,
        camera_forward_offset_m,
    )
    return ego_point_to_pixel((forward_m, left_m, 0), calibration)


__all__ = [
    "pixel_to_vehicle_local",
    "project_ground_point_to_pixel",
    "vehicle_local_to_world",
]
