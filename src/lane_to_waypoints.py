"""Convert a lane_path.extract_ego_path() result into a short, real-world
local route usable by navigation_control.PathFollower.

This is the actual fusion point between perception (lane detection, image
pixels) and control (navigation_control, world-frame metres) -- the gap
identified back when the two modules were first reviewed separately.
"""

import math

from ground_projection import pixel_to_vehicle_local, vehicle_local_to_world

MIN_FORWARD_M = 2.0   # closer than this: numerically unstable / too near the vehicle
MAX_FORWARD_M = 20.0  # comfortably above the ~12.5m braking-distance threshold at
                       # a 5 m/s target speed, so the momentary local path's "end"
                       # never triggers unwanted premature braking (see the
                       # accompanying derivation in the project notes).
MAX_WAYPOINTS = 12    # ReferencePath does O(n) linear scans; no need for the full
                       # ~100 raw centerline points every single control cycle.

# FIXED (fusion pipeline): PathFollower treats a route's endpoint as a real
# destination -- braking_distance shrinks and speed_cap drops as the
# vehicle approaches it (see controller.py's update()). Because this route
# is a receding horizon rebuilt fresh every cycle, its "end" is just
# wherever lane detection currently stops seeing lane -- NOT a real place
# to stop. Without this, a shortening/noisy detected horizon (e.g. near
# the plaza in failure_early.png/failure_late.png) directly produces
# throttle/brake oscillation as speed_cap swings with detected route
# length. Appending a synthetic tail, continuing the last heading, keeps
# the route's endpoint always comfortably beyond any real braking
# distance at cruise speed, so PathFollower's endpoint-braking logic
# stays inert during ordinary driving -- real stops should come from
# lowering target_speed_mps at the call site instead, which
# PathFollower.update() already accepts fresh each cycle.
# FIXED (fusion pipeline, round 2): the extrapolation above only checked
# for >=2 real points before extending 30m past them -- far too low a bar.
# Observed live: a marginal 2-3 point detection reacquired after a long
# stretch of misses (near the same low-lane-marking area that caused the
# original oscillation) produced a route extended confidently in whatever
# direction those few noisy points implied. With the route now looking
# "long enough," PathFollower's speed_cap unclamped and jumped the target
# from creeping (~0.25 throttle) to full cruise -- the PID saturated at
# full throttle chasing that error, on a heading built from almost no real
# evidence, straight into a building. Two guards fix this without
# reverting Fix A's actual benefit (removing the oscillation on genuinely
# well-supported detections):
MIN_REAL_POINTS = 5           # below this, don't trust the detection enough
                                # to build ANY route from it -- fall through
                                # to the caller's existing hold/creep miss-
                                # handling instead of committing to speed.
MIN_HEADING_SEGMENT_M = 1.0    # the last two points set the extrapolated
                                # heading; if they're closer together than
                                # this, pixel-quantization noise dominates
                                # the direction estimate -- skip extending
                                # this frame (real points only) so
                                # PathFollower's own distance-based speed
                                # cap still acts as a safety net exactly
                                # when the heading can't be trusted.
EXTRAPOLATION_EXTRA_M = 30.0


def build_local_route(
    centerline,  # Nx2 array from extract_ego_path()'s "centerline" key, (x_px, y_px) pairs
    image_width: int,
    image_height: int,
    focal_length_px: float,
    camera_height_m: float,
    camera_forward_offset_m: float,
    vehicle_x_m: float,
    vehicle_y_m: float,
    vehicle_yaw_rad: float,
):
    """Convert a lane centerline (image pixels, far-to-near order, as
    extract_ego_path produces it) into a list of navigation_control
    Waypoints (world frame, near-to-far order, as PathFollower requires).

    Returns None if fewer than MIN_REAL_POINTS usable points survive
    filtering -- not just too few for a valid ReferencePath, but too few
    to trust enough to extrapolate and commit speed to (see the FIXED
    note above MIN_REAL_POINTS). The caller should treat this the same as
    "no lane detected this cycle."
    """
    # extract_ego_path's centerline goes farthest-first -> nearest-last
    # (see its own y_values = linspace(highest_supported_y, lowest...)).
    # A route to DRIVE needs the opposite order: nearest first (closest to
    # "where the car is now"), farthest last (where it's heading).
    near_to_far = centerline[::-1]

    local_points = []
    for x_px, y_px in near_to_far:
        result = pixel_to_vehicle_local(
            x_px, y_px, image_width, image_height, focal_length_px,
            camera_height_m, camera_forward_offset_m,
        )
        if result is None:
            continue  # at/above the horizon -- doesn't hit the ground plane
        forward_m, left_m = result
        if not (MIN_FORWARD_M <= forward_m <= MAX_FORWARD_M):
            continue
        local_points.append((forward_m, left_m))

    if len(local_points) < MIN_REAL_POINTS:
        return None

    # Extend past the real detection horizon along the heading implied by
    # the last two points -- but only if they're far enough apart that the
    # heading itself is trustworthy (see MIN_HEADING_SEGMENT_M above). If
    # not, skip extrapolation for this frame: the route ends at the real
    # detection horizon, so PathFollower's own destination-braking still
    # acts as a speed cap exactly when confidence is low, rather than
    # extending a shaky heading estimate at full speed.
    (fwd_a, left_a), (fwd_b, left_b) = local_points[-2], local_points[-1]
    d_fwd, d_left = fwd_b - fwd_a, left_b - left_a
    seg_len = math.hypot(d_fwd, d_left)
    if seg_len >= MIN_HEADING_SEGMENT_M:
        d_fwd, d_left = d_fwd / seg_len, d_left / seg_len
        local_points.append((
            fwd_b + d_fwd * EXTRAPOLATION_EXTRA_M,
            left_b + d_left * EXTRAPOLATION_EXTRA_M,
        ))

    # Subsample to a manageable count, always keeping the first and last
    # (nearest and farthest usable points) so the route's actual span
    # doesn't shrink just because we're thinning out the middle.
    if len(local_points) > MAX_WAYPOINTS:
        step = (len(local_points) - 1) / (MAX_WAYPOINTS - 1)
        local_points = [
            local_points[round(i * step)] for i in range(MAX_WAYPOINTS)
        ]

    from navigation_control.types import Waypoint
    waypoints = [
        Waypoint(*vehicle_local_to_world(fwd, left, vehicle_x_m, vehicle_y_m, vehicle_yaw_rad))
        for fwd, left in local_points
    ]

    # Guard against consecutive duplicate points slipping through (would
    # make ReferencePath reject the path with "segment too small") --
    # extremely unlikely given continuous geometry, but cheap to check.
    deduped = [waypoints[0]]
    for wp in waypoints[1:]:
        if wp != deduped[-1]:
            deduped.append(wp)
    if len(deduped) < 2:
        return None

    return deduped
