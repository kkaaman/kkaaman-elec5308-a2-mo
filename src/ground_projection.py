"""Flat-ground inverse perspective mapping (IPM): convert image pixels
believed to lie on the road surface into real-world coordinates, relative
to the vehicle.

Assumes:
- The road is a flat, level plane (true in CARLA).
- The camera looks exactly horizontal (zero pitch) -- matches this
  project's camera mount (carla.Transform(carla.Location(x=1.5, z=2.4)),
  no rotation specified).
- A centered principal point (u0 = width/2, v0 = height/2) -- reasonable
  absent an actual calibration.

Derivation: a camera at height h above the ground, looking horizontally,
sees a pixel at row v as a ray angled downward by
    alpha = atan((v - v0) / f)
(v > v0 = below image center = angled downward; v <= v0 never hits the
ground -- the ray points at or above the horizon). That ray intersects the
ground plane at forward distance
    Z = h / tan(alpha)
For a pixel at column u, the ray's horizontal angle from the optical axis
is
    beta = atan((u - u0) / f)
and the lateral offset at that same ground point is
    X = Z * tan(beta)
"""

import math


def pixel_to_vehicle_local(
    u: float,
    v: float,
    image_width: int,
    image_height: int,
    focal_length_px: float,
    camera_height_m: float,
    camera_forward_offset_m: float,
) -> tuple[float, float] | None:
    """Back-project one image pixel onto the ground plane, in the
    vehicle's own local frame (forward, left) -- NOT yet the world frame.

    Args:
        u, v: Pixel column, row (0,0 = top-left, standard image convention).
        image_width, image_height: Frame dimensions.
        focal_length_px: Camera focal length in pixels (pinhole model).
        camera_height_m: Camera height above the ground plane.
        camera_forward_offset_m: How far forward of the vehicle's own
            origin the camera is mounted (added to Z so the result is
            relative to the vehicle, not the camera).

    Returns:
        (forward_m, left_m) in the vehicle's local frame, or None if the
        pixel is at/above the horizon (v <= v0) and therefore never
        actually intersects the ground plane.
    """
    u0 = image_width / 2
    v0 = image_height / 2

    if v <= v0:
        return None  # at or above the horizon -- no ground intersection

    alpha = math.atan((v - v0) / focal_length_px)
    forward_from_camera_m = camera_height_m / math.tan(alpha)

    beta = math.atan((u - u0) / focal_length_px)
    lateral_m = forward_from_camera_m * math.tan(beta)

    forward_m = forward_from_camera_m + camera_forward_offset_m
    # Image u increases rightward; the controller's convention (see
    # navigation_control/types.py) is +Y = left, so a rightward pixel
    # offset is a NEGATIVE left offset.
    left_m = -lateral_m

    return forward_m, left_m


def vehicle_local_to_world(
    forward_m: float,
    left_m: float,
    vehicle_x_m: float,
    vehicle_y_m: float,
    vehicle_yaw_rad: float,
) -> tuple[float, float]:
    """Rotate+translate a vehicle-local (forward, left) offset into the
    controller's world frame, using the vehicle's current pose.

    Standard 2D rotation: at yaw=0 (facing +X), "forward" aligns with +X
    and "left" aligns with +Y (matching the world frame's own "+Y = left"
    convention when facing +X) -- consistent with how steering.py already
    treats yaw elsewhere in navigation_control.
    """
    cos_yaw, sin_yaw = math.cos(vehicle_yaw_rad), math.sin(vehicle_yaw_rad)
    world_dx = forward_m * cos_yaw - left_m * sin_yaw
    world_dy = forward_m * sin_yaw + left_m * cos_yaw
    return vehicle_x_m + world_dx, vehicle_y_m + world_dy


def project_ground_point_to_pixel(
    forward_m: float,
    left_m: float,
    image_width: int,
    image_height: int,
    focal_length_px: float,
    camera_height_m: float,
    camera_forward_offset_m: float,
) -> tuple[float, float]:
    """The FORWARD (inverse) projection: given a known ground point in the
    vehicle's local frame, compute which pixel it would appear at.

    This exists purely to test pixel_to_vehicle_local(): projecting a known
    point forward, then back-projecting the resulting pixel, should recover
    the original point -- the strongest verification available without an
    actual physical/simulated camera to test against.
    """
    u0 = image_width / 2
    v0 = image_height / 2

    forward_from_camera_m = forward_m - camera_forward_offset_m
    lateral_m = -left_m  # inverse of the sign flip in pixel_to_vehicle_local

    alpha = math.atan(camera_height_m / forward_from_camera_m)
    v = v0 + focal_length_px * math.tan(alpha)

    beta = math.atan(lateral_m / forward_from_camera_m)
    u = u0 + focal_length_px * math.tan(beta)

    return u, v
