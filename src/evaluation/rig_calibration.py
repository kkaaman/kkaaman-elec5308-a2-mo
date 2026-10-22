"""Evaluator-side geometry for the mapless rig's calibration gate (no simulator imports).

Everything here may use simulator truth: exact target boxes, ego velocity and instance
masks. None of it reaches the car. Only the verdicts (pass/fail, error statistics,
coverage) go into the calibration report.

Frames: the vehicle frame (rear axle, +x forward, +y left, +z up). A box is
``(centre xyz, yaw, half-extent xyz)`` in that frame.

What each check can and cannot show:
- **Mask consistency.** The share of the target's LiDAR points that project onto the
  target's own instance mask. A projection can be several pixels off and still land
  inside a large mask, so this shows agreement, not pixel accuracy.
- **Reprojection accuracy.** The pixel distance between each edge of the projected
  true silhouette and the matching edge of the instance mask, counting only edges not
  clipped by the image border. This is exact for box-shaped props, and only
  approximate for cars (rounded corners and mirrors).
- **Radar range.** Measured range against the ray-box distance along each return's own
  direction.
- **Radar range rate.** Measured range rate (positive when receding) against
  ``dot(u, v_target - v_sensor)``, with ``v_sensor = v_ego + omega x r_mount``
  (the lever arm).
"""

import math

import numpy as np


# -- placements ---------------------------------------------------------------------
def placements(front_bumper_x_m, rear_bumper_x_m, centre_x_m, lane_offset_m=3.5):
    """Target centres for the coverage sweep, in the vehicle frame.

    Ego lane ahead at 5, 10, 20, 30 and 50 m past the front bumper; each adjacent lane
    alongside and 6 and 15 m ahead of the car's centre; ego lane behind at 8 and 15 m;
    each adjacent lane 10 m behind. Each entry is (name, x, y).
    """
    rows = [(f"ahead_{d}m", front_bumper_x_m + d, 0.0) for d in (5, 10, 20, 30, 50)]
    for side, y in (("left", lane_offset_m), ("right", -lane_offset_m)):
        rows += [
            (f"{side}_alongside", centre_x_m, y),
            (f"{side}_ahead_6m", centre_x_m + 6.0, y),
            (f"{side}_ahead_15m", centre_x_m + 15.0, y),
            (f"{side}_behind_10m", rear_bumper_x_m - 10.0, y),
        ]
    rows += [(f"behind_{d}m", rear_bumper_x_m - d, 0.0) for d in (8, 15)]
    return rows


def prop_placements(front_bumper_x_m, rear_bumper_x_m, centre_x_m, lane_offset_m=3.5):
    """Box-prop centres for reprojection accuracy: one view per camera."""
    return [
        ("prop_ahead_8m", front_bumper_x_m + 8.0, 0.0),
        ("prop_ahead_15m", front_bumper_x_m + 15.0, 0.0),
        ("prop_left_3m", centre_x_m + 3.0, lane_offset_m),
        ("prop_right_3m", centre_x_m + 3.0, -lane_offset_m),
        ("prop_behind_8m", rear_bumper_x_m - 8.0, 0.0),
    ]


def bearing_range(sensor_to_ego, point):
    """Bearing (deg, +left of the sensor's axis) and range of ``point`` from a sensor."""
    m = np.asarray(sensor_to_ego, dtype=float)
    local = (np.asarray(point, dtype=float) - m[:3, 3]) @ m[:3, :3]
    return math.degrees(math.atan2(local[1], local[0])), float(np.linalg.norm(local))


# -- boxes ----------------------------------------------------------------------------
def _box_local(points, centre, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    p = np.asarray(points, dtype=float)[..., :3] - centre
    return np.stack((c * p[..., 0] + s * p[..., 1], -s * p[..., 0] + c * p[..., 1], p[..., 2]), axis=-1)


def inside_box(points, centre, yaw, extent, margin, min_above_bottom_m=None):
    """Points within the box grown by ``margin``.

    With ``min_above_bottom_m``, points lower than that height above the box's bottom
    face are excluded. This keeps road returns under and beside a target out of
    LiDAR and radar box counts: the first live run counted them, and that made a
    mirror-image box over empty road look occupied.
    """
    if len(points) == 0:
        return np.zeros(0, bool)
    local = _box_local(points, centre, yaw)
    extent = np.asarray(extent, dtype=float)
    inside = np.all(np.abs(local) <= extent + margin, axis=-1)
    if min_above_bottom_m is not None:
        inside &= local[..., 2] >= -extent[2] + min_above_bottom_m
    return inside


def box_corners(centre, yaw, extent):
    c, s = math.cos(yaw), math.sin(yaw)
    signs = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], dtype=float)
    local = signs * np.asarray(extent, dtype=float)
    return np.column_stack((c * local[:, 0] - s * local[:, 1], s * local[:, 0] + c * local[:, 1], local[:, 2])) + centre


def ray_box_distance(origin, direction, centre, yaw, extent):
    """Distance along a unit ray to the box surface (slab method), or None if it misses."""
    o = _box_local(np.asarray(origin, dtype=float), centre, yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    d = np.asarray(direction, dtype=float)
    d = np.array((c * d[0] + s * d[1], -s * d[0] + c * d[1], d[2]))
    t_near, t_far = -math.inf, math.inf
    for axis in range(3):
        if abs(d[axis]) < 1e-12:
            if abs(o[axis]) > extent[axis]:
                return None
            continue
        t1, t2 = (-extent[axis] - o[axis]) / d[axis], (extent[axis] - o[axis]) / d[axis]
        t_near, t_far = max(t_near, min(t1, t2)), min(t_far, max(t1, t2))
    if t_near > t_far or t_far < 0:
        return None
    return float(max(t_near, 0.0))


# -- camera ---------------------------------------------------------------------------
def project(points_ego, calibration):
    """Pixels of vehicle-frame points in front of the camera, and their row mask."""
    m = np.asarray(calibration.camera_to_ego)
    optical = (np.asarray(points_ego, dtype=float)[:, :3] - m[:3, 3]) @ m[:3, :3]
    front = optical[:, 2] > 0.1
    o = optical[front]
    u = calibration.cx_px + calibration.fx_px * o[:, 0] / o[:, 2]
    v = calibration.cy_px + calibration.fy_px * o[:, 1] / o[:, 2]
    return np.column_stack((u, v)), front


def rect_of(pixels):
    return (float(pixels[:, 0].min()), float(pixels[:, 1].min()), float(pixels[:, 0].max()), float(pixels[:, 1].max()))


# CARLA 0.9.16 semantic tags (red channel of the instance camera) that are scenery, never a
# placed target. Road, pavement and buildings carry instance codes too; in the first live
# run the road's code won the vote inside the target's box.
BACKGROUND_TAGS = frozenset({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 23, 24, 25, 26, 27, 28})
VEHICLE_TAGS = frozenset({14, 15, 16})  # Car, Truck, Bus


def target_instance(instance_ids, rect, tags=None, allowed_tags=None):
    """The instance covering most pixels inside ``rect``, or None.

    Pixels whose semantic tag is scenery (``BACKGROUND_TAGS``) never count; without tags,
    instance code 0 is treated as background instead. With
    ``allowed_tags``, only those tags count. The result is an
    ``(instance_code, tag)`` key; build the mask with :func:`instance_mask`.
    """
    h, w = instance_ids.shape
    u0, v0 = max(0, int(math.floor(rect[0]))), max(0, int(math.floor(rect[1])))
    u1, v1 = min(w, int(math.ceil(rect[2])) + 1), min(h, int(math.ceil(rect[3])) + 1)
    if u1 <= u0 or v1 <= v0:
        return None
    ids = instance_ids[v0:v1, u0:u1].ravel()
    tag = np.zeros_like(ids) if tags is None else tags[v0:v1, u0:u1].ravel()
    if tags is None:
        keep = ids != 0
    else:
        # With tags, scenery is recognised by its tag, not by instance code 0: in CARLA
        # 0.9.16, vehicles spawned at runtime carry instance code 0 with tag 14 (second
        # live run), so excluding code 0 hid the target car.
        keep = ~np.isin(tag, list(BACKGROUND_TAGS))
        if allowed_tags is not None:
            keep &= np.isin(tag, list(allowed_tags))
    if not keep.any():
        return None
    keys = ids[keep].astype(np.int64) * 256 + tag[keep]
    values, counts = np.unique(keys, return_counts=True)
    best = int(values[np.argmax(counts)])
    return best // 256, best % 256


def instance_mask(instance_ids, tags, key):
    code, tag = key
    mask = instance_ids == code
    if tags is not None:
        mask &= tags == tag
    return mask


def mask_rect(mask):
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    # Pixel edges: a mask pixel spans [x, x+1).
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


EDGES = ("left", "top", "right", "bottom")


def edge_errors(projected, measured, width, height, border_px=2.0):
    """``{edge: absolute error}`` for the left, top, right and bottom edges no border clips."""
    errors = {}
    limits = (0.0, 0.0, float(width), float(height))
    for i, edge in enumerate(EDGES):
        near_border = min(abs(projected[i] - limits[i]), abs(measured[i] - limits[i])) <= border_px
        crosses = (i < 2 and projected[i] < limits[i]) or (i >= 2 and projected[i] > limits[i])
        if not near_border and not crosses:
            errors[edge] = abs(projected[i] - measured[i])
    return errors


def mask_consistency(pixels, mask, dilate_px=2):
    """Share of projected points that land on the (slightly dilated) mask."""
    if len(pixels) == 0:
        return None
    h, w = mask.shape
    if dilate_px:
        padded = np.zeros_like(mask, dtype=bool)
        ys, xs = np.nonzero(mask)
        for dy in range(-dilate_px, dilate_px + 1):
            for dx in range(-dilate_px, dilate_px + 1):
                yy, xx = np.clip(ys + dy, 0, h - 1), np.clip(xs + dx, 0, w - 1)
                padded[yy, xx] = True
        mask = padded
    u, v = np.floor(pixels[:, 0]).astype(int), np.floor(pixels[:, 1]).astype(int)
    keep = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not keep.any():
        return None
    return float(mask[v[keep], u[keep]].mean())


# -- radar ----------------------------------------------------------------------------
def radar_rays(returns, sensor_to_ego):
    """Origins and unit directions (vehicle frame) of radar returns."""
    m = np.asarray(sensor_to_ego, dtype=float)
    az, el = returns[:, 1].astype(float), returns[:, 2].astype(float)
    local = np.column_stack((np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)))
    return np.tile(m[:3, 3], (len(returns), 1)), local @ m[:3, :3].T


def expected_range_rate(
    directions,
    sensor_to_ego,
    ego_velocity_xy,
    yaw_rate_rps,
    target_velocity_xy=(0.0, 0.0),
    lever_arm=True,
    velocity_origin_x_m=0.0,
):
    """``dot(u, v_target - v_sensor)``, with the sensor's velocity optionally including omega x r.

    ``ego_velocity_xy`` is the velocity, in the vehicle frame, of the body point at
    ``(velocity_origin_x_m, 0)``. CARLA reports the actor origin, which sits at
    x = rear-axle offset.
    """
    m = np.asarray(sensor_to_ego, dtype=float)
    vx, vy = ego_velocity_xy
    if lever_arm:
        rx, ry = m[0, 3] - velocity_origin_x_m, m[1, 3]
        vx, vy = vx - yaw_rate_rps * ry, vy + yaw_rate_rps * rx
    rel = np.array([target_velocity_xy[0] - vx, target_velocity_xy[1] - vy, 0.0])
    return np.asarray(directions, dtype=float) @ rel


def p95(values):
    values = [abs(v) for v in values]
    return float(np.percentile(values, 95)) if values else None
