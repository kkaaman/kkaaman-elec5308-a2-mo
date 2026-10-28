"""Lead-car following (plan v2.1, section 11). G3 framework.

Given the lead track ahead in the lane, the speed the car should hold this tick:

- desired gap ``gap0_m + headway_s * v`` (4 m + 2 s x speed);
- acceleration limited to ``accel_mps2`` (1 m/s^2) and normal jerk to ``jerk_mps3``
  (2 m/s^3) by rate-limiting the target speed;
- checked against the worst case: the lead braking at ``lead_brake_mps2`` from now, the
  ego reacting after ``response_s`` and braking at ``ego_brake_mps2``. The speed is
  capped so that the ego still stops ``standstill_m`` short.

The stopping distance formula is the plan's: ``v t_response + v^2 / (2 a_min) +
standstill_margin + uncertainty_margin``. Only a track the tracker has confirmed leads
the car: a confirmed one that is coasting (not seen this frame) is followed with its
last state; a tentative one never is (review of the G3 framework: a tentative track that
missed a frame used to read as coasting and lead the car), and one that has been deleted
is no lead, so the planner's own clearance rule takes over.
"""

import math
from dataclasses import dataclass

import numpy as np

from mapless.contracts import TrackStatus
from mapless.obstacles import project_onto_polyline

LEAD_STATUSES = (TrackStatus.CONFIRMED, TrackStatus.COASTING)


@dataclass(frozen=True)
class FollowingConfig:
    gap0_m: float = 4.0
    headway_s: float = 2.0
    accel_mps2: float = 1.0
    jerk_mps3: float = 2.0
    response_s: float = 0.5
    ego_brake_mps2: float = 4.0  # comfortable-firm braking the ego will use
    lead_brake_mps2: float = 6.0  # the lead's worst-case braking
    standstill_m: float = 2.5
    uncertainty_m: float = 1.0
    lane_half_width_m: float = 1.75  # a track this far from the path centre is in the lane

    def __post_init__(self):
        if min(self.gap0_m, self.headway_s, self.accel_mps2, self.jerk_mps3, self.ego_brake_mps2, self.lead_brake_mps2) <= 0:
            raise ValueError("FollowingConfig: positive gaps, limits and decelerations")


def stopping_distance(v, config):
    c = config
    return v * c.response_s + v**2 / (2 * c.ego_brake_mps2) + c.standstill_m + c.uncertainty_m


def worst_case_speed_cap(gap_m, lead_speed_mps, config):
    """The ego speed at which, if the lead brakes at its worst from now, the ego still
    stops ``standstill_m + uncertainty_m`` short after ``response_s``."""
    c = config
    lead_stop = max(lead_speed_mps, 0.0) ** 2 / (2 * c.lead_brake_mps2)
    room = gap_m + lead_stop - c.standstill_m - c.uncertainty_m
    if room <= 0:
        return 0.0
    # v t + v^2 / (2a) = room  ->  v = -a t + sqrt(a^2 t^2 + 2 a room)
    a, t = c.ego_brake_mps2, c.response_s
    return float(-a * t + math.sqrt(a * a * t * t + 2 * a * room))


class FollowingController:
    """Target speed for the lead ahead, rate-limited in acceleration and jerk."""

    def __init__(self, config=None):
        self.config = config or FollowingConfig()
        self._target = None
        self._accel = 0.0

    def reset(self):
        self._target, self._accel = None, 0.0

    def start_from(self, speed_mps):
        """Begin following from ``speed_mps`` rather than the car's speed: the restriction
        already in force when the lead was taken on, so the hand-over from another rule does
        not lift it (third review, R1)."""
        if self._target is None:
            self._target = max(0.0, float(speed_mps))

    def lead_ahead(self, tracks, pose, path_body, max_ahead_m=60.0, s_from_m=0.0, tube_half_m=None):
        """The nearest track within the lane along ``path_body`` (N x 2, vehicle frame), as
        (distance ahead of the rear axle, track). ``s_from_m``: the rear axle's own arc length
        on the line (the corridor line starts behind the car). Tentative and deleted
        tracks are never leads, whatever the caller passes.

        ``tube_half_m``: the track must also reach into the car's swept tube (its near edge,
        half its narrower footprint side inside the centre, within this of the path): a
        bollard or a waiting walker at the lane's edge is not the lead and must not displace
        it (review of 3e9083a, M5)."""
        best = None
        line = np.asarray(path_body, dtype=float)
        for track in tracks:
            if track.status not in LEAD_STATUSES:
                continue
            body = pose.to_body(track.position[None])
            s, lateral, outside = project_onto_polyline(body, line)
            ahead = float(s[0]) - s_from_m
            if outside[0] or abs(lateral[0]) > self.config.lane_half_width_m or not 0.0 < ahead <= max_ahead_m:
                continue
            if tube_half_m is not None:
                half = 0.5 * min(track.footprint[0], track.footprint[1]) if track.footprint is not None else 0.5
                if abs(lateral[0]) - half > tube_half_m:
                    continue
            if best is None or ahead < best[0]:
                best = (ahead, track)
        return best

    def target_speed(self, cruise_mps, ego_speed_mps, gap_m, lead_speed_mps, dt_s):
        """The speed to hold: the cruise speed, the gap-keeping speed and the worst-case cap,
        whichever is lowest, approached within the acceleration and jerk limits."""
        c = self.config
        desired_gap = c.gap0_m + c.headway_s * ego_speed_mps
        # Close a gap error over the headway time, around the lead's own speed.
        gap_speed = max(0.0, lead_speed_mps + (gap_m - desired_gap) / c.headway_s)
        cap = worst_case_speed_cap(gap_m, lead_speed_mps, c)
        wanted = min(cruise_mps, gap_speed, cap)
        current = ego_speed_mps if self._target is None else self._target
        accel = float(np.clip((wanted - current) / max(dt_s, 1e-3), -c.ego_brake_mps2, c.accel_mps2))
        # Jerk limit applies to normal driving; braking harder than the comfort rate is
        # allowed when the worst case demands it, never when merely closing a gap. The worst
        # case demands it whenever the target is above the cap, whichever term is lowest:
        # with the gap term below the cap, the target came down at the jerk limit from far
        # above it (third review, R1: a car cutting in 6 m ahead at 6 m/s, cap 2.9 m/s, and
        # the target fell 0.005 m/s on the first tick).
        if wanted < current and (wanted >= cap - 1e-9 or current > cap + 1e-9):
            accel = float(np.clip(accel, -c.ego_brake_mps2, c.accel_mps2))
        else:
            accel = float(np.clip(accel, self._accel - c.jerk_mps3 * dt_s, self._accel + c.jerk_mps3 * dt_s))
        self._accel = accel
        self._target = max(0.0, current + accel * dt_s)
        return self._target
