"""Where tracked road users may be over the next seconds (plan v2.1, section 10). G3 framework.

Constant velocity over ``horizon_s`` at ``step_s`` (4 s at 0.2 s; 6 s when a lane change
is being considered). Each step's occupied region is the track's footprint, inflated by
the position uncertainty at that time plus ``margin_m``; pedestrians get a wider lateral
envelope because they turn without warning. A track without a measured footprint gets
its class's default size.

A prediction is a region the planner must keep the car out of, not a forecast it may
rely on: the regions only grow with time.
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

    def __post_init__(self):
        if self.step_s <= 0 or self.horizon_s < self.step_s or self.lane_change_horizon_s < self.horizon_s:
            raise ValueError("PredictionConfig: step_s > 0 and horizon_s <= lane_change_horizon_s")


def predict(track, config=None, lane_change=False):
    """A Prediction for one ObjectTrack."""
    if not isinstance(track, ObjectTrack):
        raise TypeError("track must be an ObjectTrack")
    c = config or PredictionConfig()
    horizon = c.lane_change_horizon_s if lane_change else c.horizon_s
    times = np.arange(c.step_s, horizon + 1e-9, c.step_s)
    centres = track.position[None, :] + times[:, None] * track.velocity[None, :]
    speed = track.speed_mps
    if track.footprint is not None:
        length, width, yaw = track.footprint
        heading = math.atan2(*track.velocity[::-1]) if speed > c.stationary_mps else yaw
    else:
        length, width = DEFAULT_SIZE[track.best_class]
        heading = math.atan2(*track.velocity[::-1]) if speed > c.stationary_mps else 0.0
    # Position uncertainty grows with the velocity uncertainty: sigma_p(t)^2 = P_pp + t^2 P_vv.
    P = track.covariance
    sigma_along = np.sqrt(np.maximum(P[0, 0] + times**2 * P[2, 2], 0.0))
    sigma_across = np.sqrt(np.maximum(P[1, 1] + times**2 * P[3, 3], 0.0))
    inflate = c.uncertainty_sigmas
    half_length = length / 2 + c.margin_m + inflate * sigma_along
    half_width = width / 2 + c.margin_m + inflate * sigma_across
    if track.best_class == "pedestrian":
        half_width = half_width + c.pedestrian_lateral_mps * times
        half_length = np.maximum(half_length, half_width)
    return Prediction(track.track_id, times, centres, np.column_stack((half_length, half_width)), float(heading))


def predict_all(tracks, config=None, lane_change=False):
    return [predict(t, config, lane_change) for t in tracks]


def first_conflict(predictions, path_xy, path_times_s, body_radius_m):
    """The earliest (time_s, track_id) at which a path point comes within ``body_radius_m``
    of a predicted region, or None. ``path_xy`` (N x 2, odometry frame) at ``path_times_s``."""
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
