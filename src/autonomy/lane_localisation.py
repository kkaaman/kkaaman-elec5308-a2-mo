"""Lane-aided pose estimate: GNSS + gyro + wheel speed, corrected by the camera lane offset.

The GNSS-only estimator (``autonomy.localisation``) cannot remove a constant GNSS
offset. With consumer GNSS that offset (about 1 m) becomes position error, and a car
driving on it leaves its lane. This estimator adds the RV-06 lane monitor as a
measurement.

**Measurement.** The monitor gives the signed offset (+left) of the point
``reference_x_m`` ahead of the rear axle, on the ego heading, from the perceived lane
centre. The predicted value is the same point's offset from the mission route, which
follows lane centrelines:

    h(x) = lateral(route, x + L cos(yaw), y + L sin(yaw))

So one measurement constrains lateral position and heading together.

**GNSS offset.** With ``estimate_gnss_offset`` the offset is a world-frame state
``[bx, by]``. The lane observes the offset's component across the road. After a bend,
the other component becomes observable too. That is what removes along-route error,
which the lane alone cannot see. CARLA's offset is exactly constant; the state is given
a small random walk, so a slowly drifting real offset could be followed.

**Wheel-speed scale.** The GNSS-only filter treats wheel-speed error as white noise
growing with speed. A constant scale error (tyre radius) then makes the estimate lag
behind the car along the route: about 0.2 m at 5 m/s for a 0.6% error with consumer
GNSS. With ``estimate_wheel_scale`` the scale error is a state. Its prior is the
filter's ``wheel_scale_sigma``, and the white part of the speed noise is
``wheel_sigma_mps``.

The class is used offline first (``evaluation.lane_localisation_replay``): it replays
a run's logged raw inputs and lane measurements, so it can be compared with the
GNSS-only estimate on the same frames before anything drives on it.
"""

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from autonomy.localisation import FilterTuning, PoseEstimate, wrap


@dataclass(frozen=True)
class LaneAiding:
    """How lane measurements and the GNSS offset are used (filter assumptions)."""

    lane_sigma_m: float = 0.2
    min_confidence: float = 0.3
    gate_sigma: float = 3.0
    reference_x_m: float = 8.0
    route_end_margin_m: float = 2.0
    estimate_gnss_offset: bool = True
    gnss_white_sigma_m: float = 0.6
    gnss_offset_prior_m: float = 1.5
    gnss_offset_walk_m_per_sqrt_s: float = 0.005
    estimate_wheel_scale: bool = True
    wheel_scale_walk_per_sqrt_s: float = 0.0002

    def __post_init__(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool):
                continue
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{item.name} must be finite and non-negative")
        for name in ("lane_sigma_m", "gate_sigma", "reference_x_m", "gnss_white_sigma_m"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    @classmethod
    def load(cls, path):
        document = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if document.get("schema_version") != 1:
            raise ValueError("Unsupported lane localisation config")
        return cls(**document.get("lane", {}))

    def as_dict(self):
        return dict(schema_version=1, lane=asdict(self))


X, Y, YAW, GYRO_BIAS, SCALE, OFFSET_X, OFFSET_Y = range(7)
STATES = 7


class LaneAidedEstimator:
    """EKF over [x, y, yaw, gyro bias, wheel scale error, GNSS offset x, GNSS offset y].

    Core frame. Measured wheel speed = true speed x (1 + scale error).

    States that are switched off stay at zero with zero variance. Without
    ``estimate_gnss_offset``, GNSS fixes use the lumped ``tuning.gnss_sigma_m``, as in
    the GNSS-only filter.
    """

    def __init__(self, tuning=None, aiding=None):
        self.tuning = tuning or FilterTuning()
        self.aiding = aiding or LaneAiding()
        self.x = None
        self.P = None
        self.time_s = None
        self.s_m = None
        self.lane_used = 0
        self.lane_rejected = 0
        self.last_lane = None  # "used", "gated", "route_end", "low_confidence" or None

    @property
    def initialised(self):
        return self.x is not None

    @property
    def gnss_offset_m(self):
        return (float(self.x[OFFSET_X]), float(self.x[OFFSET_Y]))

    @property
    def wheel_scale_error(self):
        return float(self.x[SCALE])

    def _gnss_sigma(self):
        if self.aiding.estimate_gnss_offset:
            return self.aiding.gnss_white_sigma_m
        return self.tuning.gnss_sigma_m

    def _start(self, x, y, yaw, time_s, route, *, from_fix):
        t, a = self.tuning, self.aiding
        self.x = np.zeros(STATES)
        self.x[[X, Y, YAW]] = x, y, yaw
        P = np.zeros((STATES, STATES))
        P[YAW, YAW] = t.initial_yaw_sigma_rad**2
        P[GYRO_BIAS, GYRO_BIAS] = t.initial_bias_sigma_rps**2
        if a.estimate_wheel_scale:
            P[SCALE, SCALE] = t.wheel_scale_sigma**2
        if a.estimate_gnss_offset:
            offset_var = a.gnss_offset_prior_m**2
            P[X, X] = P[Y, Y] = offset_var + a.gnss_white_sigma_m**2
            P[OFFSET_X, OFFSET_X] = P[OFFSET_Y, OFFSET_Y] = offset_var
            if from_fix:
                # Position taken from the fix: its error is the (unknown) offset plus
                # white noise, so position and offset errors are anti-correlated.
                P[X, OFFSET_X] = P[OFFSET_X, X] = -offset_var
                P[Y, OFFSET_Y] = P[OFFSET_Y, Y] = -offset_var
        else:
            P[X, X] = P[Y, Y] = t.gnss_sigma_m**2
        self.P = P
        self.time_s = time_s
        self.s_m = route.project(x, y).s_m

    def initialise_from_fix(self, gnss_xy, route, time_s):
        """First fix: heading from the nearest route point (no heading from GNSS at rest).

        With offset estimation the position is the raw fix. Without it the fix is
        snapped to the route, as in the GNSS-only filter.
        """
        projection = route.project(*gnss_xy)
        rx, ry, heading = route.at(projection.s_m)
        if self.aiding.estimate_gnss_offset:
            self._start(float(gnss_xy[0]), float(gnss_xy[1]), heading, time_s, route, from_fix=True)
        else:
            self._start(rx, ry, heading, time_s, route, from_fix=False)

    def initialise_at(self, x, y, yaw, time_s, route):
        """Start from the mission start, where the car is parked."""
        self._start(x, y, yaw, time_s, route, from_fix=False)

    def predict(self, dt, speed, yaw_rate):
        if dt <= 0:
            return
        t, a = self.tuning, self.aiding
        x, y, yaw, bias, scale = self.x[:5]
        omega = yaw_rate - bias
        heading = yaw + 0.5 * omega * dt
        c, s = math.cos(heading), math.sin(heading)
        v = speed * (1.0 - scale)
        self.x[[X, Y, YAW]] = x + v * dt * c, y + v * dt * s, wrap(yaw + omega * dt)
        F = np.eye(STATES)
        F[X, YAW], F[Y, YAW] = -v * dt * s, v * dt * c
        F[X, GYRO_BIAS], F[Y, GYRO_BIAS] = 0.5 * v * dt * dt * s, -0.5 * v * dt * dt * c
        F[YAW, GYRO_BIAS] = -dt
        if a.estimate_wheel_scale:
            F[X, SCALE], F[Y, SCALE] = -speed * dt * c, -speed * dt * s
            speed_sigma = t.wheel_sigma_mps
        else:
            speed_sigma = t.wheel_sigma_mps + abs(speed) * t.wheel_scale_sigma
        G = np.zeros((STATES, 2))
        G[X, 0], G[Y, 0], G[YAW, 1] = dt * c, dt * s, dt
        Q = G @ np.diag([speed_sigma**2, t.gyro_sigma_rps**2]) @ G.T
        Q[GYRO_BIAS, GYRO_BIAS] += t.bias_walk_rps_per_sqrt_s**2 * dt
        if a.estimate_wheel_scale:
            Q[SCALE, SCALE] += a.wheel_scale_walk_per_sqrt_s**2 * dt
        if a.estimate_gnss_offset:
            Q[OFFSET_X, OFFSET_X] += a.gnss_offset_walk_m_per_sqrt_s**2 * dt
            Q[OFFSET_Y, OFFSET_Y] += a.gnss_offset_walk_m_per_sqrt_s**2 * dt
        self.P = F @ self.P @ F.T + Q

    def _update(self, H, innovation, R):
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.x[YAW] = wrap(self.x[YAW])
        I_KH = np.eye(STATES) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T  # Joseph form

    def update_gnss(self, xy):
        H = np.zeros((2, STATES))
        H[0, X] = H[1, Y] = 1.0
        if self.aiding.estimate_gnss_offset:
            H[0, OFFSET_X] = H[1, OFFSET_Y] = 1.0
        innovation = np.asarray(xy, dtype=float) - H @ self.x
        self._update(H, innovation, np.eye(2) * self._gnss_sigma() ** 2)

    def predicted_lane_offset(self, route):
        """(offset, route heading at the look-ahead point), or None near the route end."""
        L = self.aiding.reference_x_m
        x, y, yaw = self.x[[X, Y, YAW]]
        px, py = x + L * math.cos(yaw), y + L * math.sin(yaw)
        lower = max(0.0, self.s_m - 2.0)
        upper = min(route.length_m, self.s_m + L + 10.0)
        projection = route.project(px, py, minimum_s=lower, maximum_s=upper)
        if projection.s_m > route.length_m - self.aiding.route_end_margin_m:
            return None
        # Heading of the segment the point was projected on: at a vertex,
        # route.at(s) would return the next segment's heading instead.
        tangent = route.tangents[projection.segment]
        return projection.lateral_m, math.atan2(tangent[1], tangent[0])

    def update_lane(self, offset_m, route):
        """Apply one lane-offset measurement; returns why it was used or rejected."""
        predicted = self.predicted_lane_offset(route)
        if predicted is None:
            return "route_end"
        value, theta = predicted
        L = self.aiding.reference_x_m
        H = np.zeros((1, STATES))
        H[0, X], H[0, Y] = -math.sin(theta), math.cos(theta)
        H[0, YAW] = L * math.cos(self.x[YAW] - theta)
        R = np.array([[self.aiding.lane_sigma_m**2]])
        innovation = np.array([offset_m - value])
        S = float((H @ self.P @ H.T + R)[0, 0])
        if abs(innovation[0]) > self.aiding.gate_sigma * math.sqrt(S):
            self.lane_rejected += 1
            return "gated"
        self._update(H, innovation, R)
        self.lane_used += 1
        return "used"

    def step(self, inputs, route, lane=None):
        """Advance to ``inputs.sim_time_s``; returns None until initialised.

        ``lane`` is ``(signed_offset_m, confidence)`` or None for this frame.
        """
        if not self.initialised:
            if inputs.gnss_xy_m is None:
                return None
            self.initialise_from_fix(inputs.gnss_xy_m, route, inputs.sim_time_s)
        else:
            self.predict(
                inputs.sim_time_s - self.time_s, inputs.wheel_speed_mps, inputs.yaw_rate_rps
            )
            self.time_s = inputs.sim_time_s
            if inputs.gnss_xy_m is not None:
                self.update_gnss(inputs.gnss_xy_m)
        self.s_m = route.project(
            self.x[X],
            self.x[Y],
            minimum_s=max(0.0, min(self.s_m - 2.0, route.length_m)),
            maximum_s=min(route.length_m, self.s_m + 10.0),
        ).s_m
        self.last_lane = None
        if lane is not None:
            if lane[1] >= self.aiding.min_confidence:
                self.last_lane = self.update_lane(lane[0], route)
            else:
                self.last_lane = "low_confidence"
        return PoseEstimate(
            float(self.x[X]),
            float(self.x[Y]),
            float(self.x[YAW]),
            float(self.x[GYRO_BIAS]),
            float(math.sqrt(max(0.0, self.P[X, X] + self.P[Y, Y]))),
        )
