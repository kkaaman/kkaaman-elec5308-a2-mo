"""Right-handed SI rigid transforms. Simulator axes are converted at its boundary."""

import math

import numpy as np


def rigid_matrix(matrix):
    result = np.asarray(matrix, dtype=float)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError("Transform must be finite 4x4")
    rotation = result[:3, :3]
    if (
        not np.allclose(result[3], [0, 0, 0, 1], atol=1e-7)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6)
    ):
        raise ValueError("Transform must preserve handedness and rigid distances")
    return result.copy()


def transform_point(point_xyz_m, matrix):
    point = np.asarray(point_xyz_m, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("Point must be finite XYZ metres")
    return (rigid_matrix(matrix) @ np.append(point, 1))[:3]


def inverse_rigid(matrix):
    matrix = rigid_matrix(matrix)
    result = np.eye(4)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ matrix[:3, 3]
    return result


def ego_to_world_matrix(x_m, y_m, yaw_rad, z_m=0.0):
    if not all(math.isfinite(value) for value in (x_m, y_m, yaw_rad, z_m)):
        raise ValueError("Pose must be finite metres and radians")
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.array(
        [[c, -s, 0, x_m], [s, c, 0, y_m], [0, 0, 1, z_m], [0, 0, 0, 1]], dtype=float
    )


def vehicle_local_to_world(
    forward_m, left_m, vehicle_x_m, vehicle_y_m, vehicle_yaw_rad
):
    point = transform_point(
        (forward_m, left_m, 0),
        ego_to_world_matrix(vehicle_x_m, vehicle_y_m, vehicle_yaw_rad),
    )
    return float(point[0]), float(point[1])
