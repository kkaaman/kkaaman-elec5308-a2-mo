"""Mount transforms, camera calibration and compass heading for the mapless stack.

A mount is given relative to the CARLA actor origin. The vehicle frame's origin is the
rear axle, ``rear_axle_offset_m`` behind the actor origin. A mount at actor x therefore
sits at ``x + rear_axle_offset_m`` in the vehicle frame.
"""

import math

import numpy as np

from geometry.camera import CameraIntrinsics

# Optical frame (+x right, +y down, +z forward) to sensor body frame (+x forward, +y left, +z up).
BODY_FROM_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def rotation(yaw_deg, pitch_deg):
    """Body-to-vehicle rotation: yaw about +z (left positive), then nose-up pitch."""
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    rz = np.array(
        [[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
    )
    # Nose-up pitch is a negative rotation about +y (+y points left).
    ry = np.array(
        [
            [math.cos(-pitch), 0.0, math.sin(-pitch)],
            [0.0, 1.0, 0.0],
            [-math.sin(-pitch), 0.0, math.cos(-pitch)],
        ]
    )
    return rz @ ry


def sensor_to_ego(mount, rear_axle_offset_m):
    """4x4 transform from the sensor body frame to the rear-axle vehicle frame."""
    if not math.isfinite(rear_axle_offset_m) or rear_axle_offset_m < 0:
        raise ValueError("rear_axle_offset_m must be finite and non-negative")
    m = np.eye(4)
    m[:3, :3] = rotation(mount.yaw_deg, mount.pitch_deg)
    m[:3, 3] = (mount.x_m + rear_axle_offset_m, mount.y_m, mount.z_m)
    return m


def camera_calibration(spec, rear_axle_offset_m):
    """Pinhole calibration of an RGB camera spec, in the vehicle frame."""
    body = sensor_to_ego(spec.mount, rear_axle_offset_m)
    camera_to_ego = np.eye(4)
    camera_to_ego[:3, :3] = body[:3, :3] @ BODY_FROM_OPTICAL
    camera_to_ego[:3, 3] = body[:3, 3]
    intrinsics = CameraIntrinsics.from_horizontal_fov(
        spec.width, spec.height, math.radians(spec.fov_deg)
    )
    return intrinsics.calibration(tuple(tuple(row) for row in camera_to_ego))


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def yaw_from_compass(compass_rad):
    """Vehicle-frame heading (counter-clockwise from east) from a clockwise-from-north compass."""
    return wrap(math.pi / 2 - compass_rad)
