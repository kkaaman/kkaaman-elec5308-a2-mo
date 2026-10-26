"""Where tracked road users may be over the next seconds (plan v2.1, section 10). G3 framework.

Constant velocity over ``horizon_s`` at ``step_s`` (4 s at 0.2 s; 6 s when a lane change
is being considered). Each step's occupied region is the track's footprint, inflated by
the position uncertainty at that time plus ``margin_m``; pedestrians get a wider lateral
envelope because they turn without warning. A track without a measured footprint gets
its class's default size.

The uncertainty is taken in the object's own frame (review of the G3 framework: the
first version added the odometry-frame X variance to the length and the Y variance to
the width, so a northbound car's along-track uncertainty widened it instead of
lengthening it). The position covariance is propagated over the horizon with its
position–velocity cross terms, ``P_pos(t) = P_pp + t (P_pv + P_vp) + t^2 P_vv``, plus the
process (white acceleration) growth, then rotated into the heading frame; the footprint
grows by ``uncertainty_sigmas`` of the longitudinal and lateral sigma plus ``margin_m``.

A prediction is a region the planner must keep the car out of, not a forecast it may
rely on: the regions only grow with time.

``first_footprint_conflict`` tests the predicted regions against the car's oriented
footprint at the time-aligned poses along its planned path (the rear-axle path expanded
by half the width alone left the 3.78 m front overhang unchecked, same review).
"""

import math
from dataclasses import dataclass

import numpy as np

from mapless.contracts import ObjectTrack, Prediction

DEFAULT_SIZE = {  # (length_m, width_m) when the LiDAR has not measured the extent
    "car": (4.5, 2.0),
    "truck": (8.0, 2.6),
    "bike": (2.0, 0.8),
    "pedestrian": (0.6, 0.6),
    "unknown": (2.0, 2.0),
}


@dataclass(frozen=True)
class PredictionConfig:
    horizon_s: float = 4.0
    lane_change_horizon_s: float = 6.0
    step_s: float = 0.2
    margin_m: float = 0.3
    pedestrian_lateral_mps: float = 1.0  # a pedestrian may step sideways this fast
    uncertainty_sigmas: float = 2.0  # position uncertainty counted at this many sigma
    stationary_mps: float = 0.3  # slower tracks keep their measured heading
    # A velocity above stationary_mps counts as motion only once the track has moved this
    # far from where it was first seen: the centroid of a long kerb or hedge cluster
    # jitters as the LiDAR rings cross it differently each sweep (third live stack run,
    # 25 Oct: such tracks read 0.4 m/s and were predicted across the lane). A road user
    # that has yet to move this far is predicted where it stands; the driver's own LiDAR
    # obstacle stop covers it meanwhile.
    min_displacement_m: float = 0.75
    # ... and that motion must be recent and sustained: at least this mean speed over the
    # last second (a walker does 1.2 m/s). The third live M04 run yielded to small static
    # things beside the road whose centroids had drifted 0.75 m as the car went by.
    min_recent_speed_mps: float = 0.5
    # Unmodelled acceleration over the horizon as continuous white noise: position variance
    # grows by q t^3 / 3. The tracker drives its CV model with 1.5 m/s^2 white acceleration
    # at a 0.05 s tick, which is q = 1.5^2 * 0.05 in these units (1 m sigma at 3 s).
    process_noise_m2ps3: float = 1.5**2 * 0.05
    # A track's velocity variance starts at the filter's prior (5 m/s sigma) and settles
    # within a second; propagated as is, a just-confirmed stationary object became a 30 m
    # region at 3 s and stopped the car 26 m short of a parked car (fresh-eyes review of
    # 58adaf3). The velocity uncertainty counted here is bounded by what road users do.
    max_velocity_sigma_mps: float = 1.5
    # Velocity uncertainty counts only once the filter has settled on a velocity (sigma
    # below this): before that the measured velocity stands on its own — an unsettled
    # filter is not evidence of motion, and motion shows in the estimate within a few
    # ticks. A settled track measured stationary is predicted stationary: on the first
    # live stack run every roadside cluster (parked cars, poles) became a region metres
    # wide over the horizon and held the car at the start line. The process noise is not
    # charged to a stationary track either (second live stack run, 25 Oct: the kerb
    # cluster 4.8 m to the right grew a 6 m wide region at 3.6 s and capped the speed to
    # zero at the start line).
    settled_velocity_sigma_mps: float = 1.0

    def __post_init__(self):
        if self.step_s <= 0 or self.horizon_s < self.step_s or self.lane_change_horizon_s < self.horizon_s:
            raise ValueError("PredictionConfig: step_s > 0 and horizon_s <= lane_change_horizon_s")
        if self.process_noise_m2ps3 < 0 or self.uncertainty_sigmas < 0:
            raise ValueError("PredictionConfig: non-negative process_noise_m2ps3 and uncertainty_sigmas")


def position_sigmas(covariance, times, heading_rad, process_noise_m2ps3=0.0, max_velocity_sigma_mps=None, settled_velocity_sigma_mps=None, stationary=False):
    """(sigma_longitudinal, sigma_lateral) at each of ``times`` for a 4x4 ``[x, y, vx, vy]``
    covariance, in the frame of ``heading_rad``.

    ``P_pos(t) = P_pp + t (P_pv + P_vp) + t^2 P_vv + (q t^3 / 3) I``, rotated by -heading.
    The velocity block (and its cross terms, in proportion) is scaled down so that no
    velocity sigma exceeds ``max_velocity_sigma_mps``; it is dropped altogether while the
    filter has not settled (largest velocity sigma above ``settled_velocity_sigma_mps``)
    or for a ``stationary`` track: an unsettled filter is not evidence that a parked car
    may do 5 m/s, and a settled stationary one is not about to. The process noise (the
    unmodelled acceleration a moving road user may apply) is likewise not charged to a
    stationary track: a kerb or a parked car does not drift 1.3 m in 3.6 s, and the
    tracker shows any start within a few ticks.
    """
    P = np.asarray(covariance, dtype=float)
    t = np.asarray(times, dtype=float)
    P_pp, P_pv, P_vv = P[:2, :2], P[:2, 2:], P[2:, 2:]
    largest = math.sqrt(max(float(np.linalg.eigvalsh(P_vv).max()), 0.0))
    if stationary:
        P_vv, P_pv, process_noise_m2ps3 = np.zeros((2, 2)), np.zeros((2, 2)), 0.0
    elif settled_velocity_sigma_mps is not None and largest > settled_velocity_sigma_mps:
        P_vv, P_pv = np.zeros((2, 2)), np.zeros((2, 2))
    elif max_velocity_sigma_mps is not None and largest > max_velocity_sigma_mps:
        scale = max_velocity_sigma_mps / largest
        P_vv = P_vv * scale**2
        P_pv = P_pv * scale
    cross = P_pv + P_pv.T
    c, s = math.cos(heading_rad), math.sin(heading_rad)
    R = np.array([[c, s], [-s, c]])  # odometry -> heading frame
    pos = P_pp[None] + t[:, None, None] * cross[None] + (t**2)[:, None, None] * P_vv[None]
    pos = pos + (process_noise_m2ps3 * t**3 / 3.0)[:, None, None] * np.eye(2)[None]
    local = R[None] @ pos @ R.T[None]
    sigma_long = np.sqrt(np.maximum(local[:, 0, 0], 0.0))
    sigma_lat = np.sqrt(np.maximum(local[:, 1, 1], 0.0))
    return sigma_long, sigma_lat


def decided_class(track, min_probability=0.5):
    """The track's class when the evidence decides it, else None. Without a detector the
    probabilities stay uniform and ``best_class`` breaks the tie on the first name,
    "car": the second live stack M04 run (26 Oct) took the walker for a car that way."""
    best = track.best_class
    return best if track.class_probabilities.get(best, 0.0) >= min_probability else None


def is_moving(track, config=None):
    """Whether a track's velocity is motion: above ``stationary_mps`` and corroborated by
    ``min_displacement_m`` of travel since it was first seen and, once a second of history
    exists, ``min_recent_speed_mps`` of sustained recent motion (centroid drift is not)."""
    c = config or PredictionConfig()
    return (
        track.speed_mps > c.stationary_mps
        and (track.travelled_m is None or track.travelled_m >= c.min_displacement_m)
        and (track.recent_speed_mps is None or track.recent_speed_mps >= c.min_recent_speed_mps)
    )


def predict(track, config=None, lane_change=False):
    """A Prediction for one ObjectTrack."""
    if not isinstance(track, ObjectTrack):
        raise TypeError("track must be an ObjectTrack")
    c = config or PredictionConfig()
    horizon = c.lane_change_horizon_s if lane_change else c.horizon_s
    times = np.arange(c.step_s, horizon + 1e-9, c.step_s)
    moving = is_moving(track, c)
    velocity = track.velocity if moving else np.zeros(2)
    centres = track.position[None, :] + times[:, None] * velocity[None, :]
    if track.footprint is not None:
        length, width, yaw = track.footprint
        heading = math.atan2(*track.velocity[::-1]) if moving else yaw
    else:
        length, width = DEFAULT_SIZE[decided_class(track) or "unknown"]
        heading = math.atan2(*track.velocity[::-1]) if moving else 0.0
    sigma_long, sigma_lat = position_sigmas(
        track.covariance, times, heading, c.process_noise_m2ps3, c.max_velocity_sigma_mps, c.settled_velocity_sigma_mps, stationary=not moving
    )
    inflate = c.uncertainty_sigmas
    half_length = length / 2 + c.margin_m + inflate * sigma_long
    half_width = width / 2 + c.margin_m + inflate * sigma_lat
    if decided_class(track) == "pedestrian":
        half_width = half_width + c.pedestrian_lateral_mps * times
        half_length = np.maximum(half_length, half_width)
    return Prediction(track.track_id, times, centres, np.column_stack((half_length, half_width)), float(heading))


def predict_all(tracks, config=None, lane_change=False):
    return [predict(t, config, lane_change) for t in tracks]


def first_conflict(predictions, path_xy, path_times_s, body_radius_m):
    """The earliest (time_s, track_id) at which a path point comes within ``body_radius_m``
    of a predicted region, or None. ``path_xy`` (N x 2, odometry frame) at ``path_times_s``.

    A disc around each path point: for a point-like path (a waypoint list, a planned
    centre). The stack uses ``first_footprint_conflict`` for the car's own body.
    """
    path = np.asarray(path_xy, dtype=float).reshape(-1, 2)
    times = np.asarray(path_times_s, dtype=float)
    best = None
    for pred in predictions:
        for k, t in enumerate(pred.times_s):
            i = int(np.argmin(np.abs(times - t)))
            if abs(times[i] - t) > pred.times_s[0]:
                continue
            p = path[i] - pred.centres[k]
            c, s = math.cos(pred.heading_rad), math.sin(pred.heading_rad)
            along, across = c * p[0] + s * p[1], -s * p[0] + c * p[1]
            if abs(along) <= pred.half_extents[k, 0] + body_radius_m and abs(across) <= pred.half_extents[k, 1] + body_radius_m:
                if best is None or t < best[0]:
                    best = (float(t), pred.track_id)
                break
    return best


# -- oriented footprints -------------------------------------------------------------------
def box_corners(centre, heading_rad, along_min, along_max, across_half):
    """The four corners (4 x 2) of a box spanning ``[along_min, along_max]`` along
    ``heading_rad`` and ``[-across_half, across_half]`` across it, about ``centre``."""
    c, s = math.cos(heading_rad), math.sin(heading_rad)
    local = np.array([[along_min, -across_half], [along_max, -across_half], [along_max, across_half], [along_min, across_half]])
    return np.asarray(centre, dtype=float) + local @ np.array([[c, s], [-s, c]])


def boxes_overlap(a, b):
    """Separating-axis test for two oriented boxes given as corners (4 x 2 each)."""
    for poly in (a, b):
        for e in (poly[1] - poly[0], poly[3] - poly[0]):
            axis = np.array([-e[1], e[0]])
            pa, pb = a @ axis, b @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def path_headings(path_xy):
    """Heading at each path sample from the segment leaving it (the last one keeps its
    predecessor's; a zero-length segment keeps the previous heading)."""
    path = np.asarray(path_xy, dtype=float).reshape(-1, 2)
    n = len(path)
    if n < 2:
        return np.zeros(n)
    d = np.diff(path, axis=0)
    ok = np.hypot(*d.T) > 1e-9
    if not ok.any():
        return np.zeros(n)
    seg = np.arctan2(d[:, 1], d[:, 0])
    source = np.maximum.accumulate(np.where(ok, np.arange(n - 1), -1))  # last proper segment at or before i
    source[source < 0] = int(np.argmax(ok))
    headings = seg[source]
    return np.append(headings, headings[-1])


def first_occupied_arc(predictions, path_xy, body, from_arc_m=0.0, step_m=0.5, with_track=False):
    """The smallest arc length along ``path_xy`` (from its first sample, at or beyond
    ``from_arc_m``) at which the car's oriented footprint, its rear axle on the path,
    overlaps any step of any prediction, whenever that step falls; or None.

    Timing is ignored on purpose: this answers "where does the path stop being free while
    this road user is predicted on it", for yielding, not "when would the car meet it".
    """
    none = (None, None) if with_track else None
    path = np.asarray(path_xy, dtype=float).reshape(-1, 2)
    if len(path) < 2 or not predictions:
        return none
    seg = np.hypot(*np.diff(path, axis=0).T)
    arcs = np.concatenate(([0.0], np.cumsum(seg)))
    stations = np.arange(max(0.0, from_arc_m), arcs[-1] + 1e-9, step_m)
    if not len(stations):
        return none
    xs, ys = np.interp(stations, arcs, path[:, 0]), np.interp(stations, arcs, path[:, 1])
    heads = path_headings(path)
    idx = np.clip(np.searchsorted(arcs, stations, side="right") - 1, 0, len(path) - 1)
    boxes = [box_corners(o, h, body.rear_m, body.front_m, body.half_width_m) for o, h in zip(zip(xs, ys), heads[idx])]
    regions = [(p.track_id, box_corners(p.centres[k], p.heading_rad, -p.half_extents[k][0], p.half_extents[k][0], p.half_extents[k][1])) for p in predictions for k in range(len(p.times_s))]
    for station, car in zip(stations, boxes):
        for track_id, region in regions:
            if boxes_overlap(car, region):
                return (float(station), track_id) if with_track else float(station)
    return none


def pose_at(path, times, headings, t):
    """(x, y, yaw) on the path at time ``t`` (clamped to the path's span)."""
    if t <= times[0]:
        return path[0, 0], path[0, 1], headings[0]
    if t >= times[-1]:
        return path[-1, 0], path[-1, 1], headings[-1]
    i = int(np.searchsorted(times, t, side="right"))
    i = min(max(i, 1), len(times) - 1)
    span = times[i] - times[i - 1]
    w = 0.0 if span <= 0 else (t - times[i - 1]) / span
    xy = path[i - 1] + w * (path[i] - path[i - 1])
    return xy[0], xy[1], headings[i - 1]


def first_footprint_conflict(predictions, path_xy, path_times_s, body):
    """The earliest (time_s, track_id) at which the car's oriented footprint, placed at
    the time-aligned poses along its path, overlaps a predicted region, or None.

    ``path_xy`` (N x 2, odometry frame, rear-axle positions) at ``path_times_s`` (seconds
    from now; samples at negative times are where the car has been and are skipped).
    ``body`` has ``rear_m``, ``front_m`` and ``half_width_m`` (``VehicleBody``). Each
    prediction step covers the half-step window around its time; the car's footprint is
    tested at the window's ends and at every path sample inside it, so the swept region
    between consecutive samples is covered to within the body's own length.
    """
    path = np.asarray(path_xy, dtype=float).reshape(-1, 2)
    times = np.asarray(path_times_s, dtype=float)
    if len(path) == 0 or len(times) != len(path):
        return None
    headings = path_headings(path)
    end = float(times[-1])
    best = None
    for pred in predictions:
        step = float(pred.times_s[0]) if len(pred.times_s) == 1 else float(pred.times_s[1] - pred.times_s[0])
        for k, t in enumerate(pred.times_s):
            lo = 0.0 if k == 0 else max(0.0, t - step / 2)
            hi = t + step / 2
            if lo > end + step / 2:
                break
            hl, hw = pred.half_extents[k]
            obstacle = box_corners(pred.centres[k], pred.heading_rad, -hl, hl, hw)
            inside = np.flatnonzero((times > lo) & (times < hi))
            poses = [pose_at(path, times, headings, lo), pose_at(path, times, headings, hi)]
            poses += [(path[i, 0], path[i, 1], headings[i]) for i in inside]
            if any(boxes_overlap(box_corners((x, y), yaw, body.rear_m, body.front_m, body.half_width_m), obstacle) for x, y, yaw in poses):
                if best is None or t < best[0]:
                    best = (float(t), pred.track_id)
                break
    return best
