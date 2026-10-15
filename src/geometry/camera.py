"""Pinhole optical rays, axial-depth projection and local ground intersection."""

import math
from dataclasses import dataclass

import numpy as np

from autonomy.interfaces import CameraCalibration

from .transforms import inverse_rigid, rigid_matrix, transform_point


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float

    def __post_init__(self):
        if any(
            type(value) is not int or value <= 0 for value in (self.width, self.height)
        ):
            raise ValueError("Image dimensions must be positive integers")
        if not all(
            math.isfinite(value)
            for value in (self.fx_px, self.fy_px, self.cx_px, self.cy_px)
        ):
            raise ValueError("Intrinsics must be finite")
        if self.fx_px <= 0 or self.fy_px <= 0:
            raise ValueError("Focal lengths must be positive")

    @classmethod
    def from_horizontal_fov(cls, width, height, fov_rad):
        if not math.isfinite(fov_rad) or not 0 < fov_rad < math.pi:
            raise ValueError("Horizontal FOV must lie between zero and pi radians")
        if (
            type(width) is not int
            or width <= 0
            or type(height) is not int
            or height <= 0
        ):
            raise ValueError("Image dimensions must be positive integers")
        focal = width / (2 * math.tan(fov_rad / 2))
        return cls(width, height, focal, focal, width / 2, height / 2)

    def calibration(self, camera_to_ego):
        return CameraCalibration(
            self.fx_px,
            self.fy_px,
            self.cx_px,
            self.cy_px,
            tuple(map(tuple, rigid_matrix(camera_to_ego))),
        )


def pixel_ray(u, v, calibration):
    if not isinstance(calibration, CameraCalibration):
        raise TypeError("Expected CameraCalibration")
    if not math.isfinite(u) or not math.isfinite(v):
        raise ValueError("Pixel coordinates must be finite")
    return np.array(
        [
            (u - calibration.cx_px) / calibration.fx_px,
            (v - calibration.cy_px) / calibration.fy_px,
            1.0,
        ]
    )


def pixel_depth_to_ego(u, v, axial_depth_m, calibration):
    if not math.isfinite(axial_depth_m) or axial_depth_m <= 0:
        raise ValueError("Axial depth must be finite and positive metres")
    return transform_point(
        pixel_ray(u, v, calibration) * axial_depth_m, calibration.camera_to_ego
    )


def ego_point_to_pixel(point_xyz_m, calibration):
    optical = transform_point(point_xyz_m, inverse_rigid(calibration.camera_to_ego))
    if optical[2] <= 0:
        return None
    return (
        float(calibration.cx_px + calibration.fx_px * optical[0] / optical[2]),
        float(calibration.cy_px + calibration.fy_px * optical[1] / optical[2]),
    )


def pixel_to_ground(u, v, calibration, ground_z_ego_m=0.0):
    """Intersect an optical ray with an explicitly assumed local horizontal plane.

    Returns None for parallel rays or intersections behind the camera. This
    plane assumption is not valid for arbitrary slopes, obstacles or junctions.
    """
    if not math.isfinite(ground_z_ego_m):
        raise ValueError("Ground plane height must be finite metres")
    ray = pixel_ray(u, v, calibration)
    matrix = rigid_matrix(calibration.camera_to_ego)
    origin = matrix[:3, 3]
    direction = matrix[:3, :3] @ ray
    if abs(direction[2]) < 1e-9:
        return None
    distance = (ground_z_ego_m - origin[2]) / direction[2]
    if distance <= 0:
        return None
    return origin + distance * direction
