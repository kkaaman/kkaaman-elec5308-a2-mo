"""Lead-car following (plan v2.1, section 11). G3 framework.

Given the lead track ahead in the lane, the speed the car should hold this tick:

- desired gap ``gap0_m + headway_s * v`` (4 m + 2 s x speed);
- acceleration limited to ``accel_mps2`` (1 m/s^2) and normal jerk to ``jerk_mps3``
  (2 m/s^3) by rate-limiting the target speed;
- checked against the worst case: the lead braking at ``lead_brake_mps2`` from now, the
  ego reacting after ``response_s`` and braking at ``ego_brake_mps2``. The speed is
  capped so that the ego still stops ``standstill_m`` short;
- a cap only while one of those restricts the car: with both at or above the cruise
  speed the lead is watched, not followed (fifth review, F5-01).

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
    # A lead just past the end of the path line counts, on the line continued: the kerb a
    # lead hides ends the corridor at about its rear, and its centre then lay past the end;
    # it flickered in and out as the lead, and its braking was seen 1.15 s late (live
    # braking lead, c882389). The continuation keeps the line's curvature over its last
    # ``continue_fit_m`` (up to ``continue_max_curvature``) rather than running straight
    # off a bend, and stops where the route turns off (``lead_ahead``'s ``turn_entry_m``):
    # beyond it the line's road is one the car leaves (fifth review, F5-03).
    beyond_end_m: float = 8.0
    continue_fit_m: float = 4.0
    continue_max_curvature: float = 0.15
    # A lead on the continuation whose rear reaches back to within this of the line's end
    # is what ended the line (it hid the kerb beyond). One whose body begins further on did
    # not: the line ended before it, and whether it is on the car's way is uncertain. It is
    # followed for the worst case only, not the gap (fifth review, F5-03).
    ended_by_lead_m: float = 1.5
    # The target closes its error to the wanted speed over this time, not in one tick: with
    # only the jerk limit between them, the target overshot and the car went stop-go behind
    # a slower car (1.9 down to 0.4 m/s and back every 3 s behind one at 1.4 m/s: the first
    # live cut-in with the car staying slower, 6f2aec3). Braking for the worst case is not
    # slowed by it.
    speed_tau_s: float = 1.0

    def __post_init__(self):
        if min(self.gap0_m, self.headway_s, self.accel_mps2, self.jerk_mps3, self.ego_brake_mps2, self.lead_brake_mps2, self.speed_tau_s, self.continue_fit_m) <= 0:
            raise ValueError("FollowingConfig: positive gaps, limits and decelerations")


def stopping_distance(v, config):
    c = config
    return v * c.response_s + v**2 / (2 * c.ego_brake_mps2) + c.standstill_m + c.uncertainty_m


def _half_extent_along(track, heading_rad):
    """Half the track's footprint measured along ``heading_rad`` (odometry frame); 0.5 m
    without one."""
    if track.footprint is None:
        return 0.5
    length, width, yaw = track.footprint
    d = yaw - heading_rad
    return 0.5 * (abs(length * math.cos(d)) + abs(width * math.sin(d)))


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
        self._lead_id = None
        self.last = {}
        self.continuation = None  # how the last lead was placed past the line's end (None: on the line)

    def reset(self):
        self._target, self._accel = None, 0.0
        self._lead_id = None

    def start_from(self, speed_mps):
        """Begin following from ``speed_mps`` rather than the car's speed: the restriction
        already in force when the lead was taken on, so the hand-over from another rule does
        not lift it (third review, R1)."""
        if self._target is None:
            self._target = max(0.0, float(speed_mps))

    @property
    def restricting(self):
        """Whether following sets the car's speed (else it only watches the lead)."""
        return self._target is not None

    def lead_ahead(self, tracks, pose, path_body, max_ahead_m=60.0, s_from_m=0.0, tube_half_m=None, turn_entry_m=None):
        """The nearest track within the lane along ``path_body`` (N x 2, vehicle frame), as
        (distance ahead of the rear axle, track). ``s_from_m``: the rear axle's own arc length
        on the line (the corridor line starts behind the car). Tentative and deleted
        tracks are never leads, whatever the caller passes.

        ``tube_half_m``: the track must also reach into the car's swept tube (its near edge,
        half its narrower footprint side inside the centre, within this of the path): a
        bollard or a waiting walker at the lane's edge is not the lead and must not displace
        it (review of 3e9083a, M5). A track up to ``beyond_end_m`` past the line's end is
        judged on the line continued along its end curvature, no further than
        ``turn_entry_m`` (vehicle x of a junction the route turns off at, while the line is
        the road's own). ``continuation`` then says how: its distance past the end, its
        rear's, and ``uncertain`` when its body begins more than ``ended_by_lead_m`` on."""
        c = self.config
        best = None
        line = np.asarray(path_body, dtype=float)
        ext = self._continuation(line, s_from_m, turn_entry_m) if len(line) >= 2 else None
        line_arcs = float(np.hypot(*np.diff(line, axis=0).T).sum()) if len(line) >= 2 else 0.0
        tracks = [t for t in tracks if t.status in LEAD_STATUSES]
        if not tracks:
            self.continuation = None
            return None
        # All the candidates projected at once (one call per track was a tenth of the
        # speed rules' time, fifth review F5-05).
        bodies = pose.to_body(np.array([t.position for t in tracks]))
        s_all, lat_all, out_all = project_onto_polyline(bodies, line)
        for n, track in enumerate(tracks):
            body = bodies[n : n + 1]
            s, lateral, outside = s_all[n : n + 1], lat_all[n : n + 1], out_all[n : n + 1]
            ahead = float(s[0]) - s_from_m
            note = None
            if outside[0] and len(line) >= 2:
                # Past the line's end: on its continuation (not before its start).
                if ext is None or float((body[0] - line[-1]) @ (line[-1] - line[-2])) <= 0.0:
                    continue
                s_ext, lateral, beyond = project_onto_polyline(body, ext)
                if beyond[0]:
                    continue
                extra = float(s_ext[0])
                ahead = line_arcs + extra - s_from_m
                i = min(int(extra / 0.5), len(ext) - 2)
                d = ext[i + 1] - ext[i]
                rear = extra - _half_extent_along(track, pose.yaw_rad + math.atan2(d[1], d[0]))
                note = dict(extra_m=round(extra, 2), rear_beyond_m=round(rear, 2), uncertain=bool(rear > c.ended_by_lead_m))
            if abs(lateral[0]) > c.lane_half_width_m or not 0.0 < ahead <= max_ahead_m:
                continue
            if tube_half_m is not None:
                half = 0.5 * min(track.footprint[0], track.footprint[1]) if track.footprint is not None else 0.5
                if abs(lateral[0]) - half > tube_half_m:
                    continue
            if best is None or ahead < best[0]:
                best = (ahead, track)
                self.continuation = note
        if best is None:
            self.continuation = None
        return best

    def _continuation(self, line, s_from_m, turn_entry_m):
        """The line continued from its end (N x 2, from ``line[-1]``, 0.5 m steps) along the
        curvature of its last ``continue_fit_m``, ``beyond_end_m`` long or to where the
        route turns off; None when nothing of it is left."""
        c = self.config
        length = c.beyond_end_m
        if turn_entry_m is not None:
            length = min(length, float(turn_entry_m) - (float(np.hypot(*np.diff(line, axis=0).T).sum()) - s_from_m))
        if length <= 0.0:
            return None
        seg = np.diff(line, axis=0)
        lengths = np.hypot(*seg.T)
        keep = lengths > 1e-6
        seg, lengths = seg[keep], lengths[keep]
        if not len(seg):
            return None
        heading = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
        mid = np.cumsum(lengths) - 0.5 * lengths
        fit = mid >= mid[-1] - c.continue_fit_m
        span = float(mid[-1] - mid[fit][0])
        k = 0.0 if span < 1.0 else float(np.clip((heading[-1] - heading[fit][0]) / span, -c.continue_max_curvature, c.continue_max_curvature))
        n = max(1, math.ceil(length / 0.5))
        steps = np.full(n, 0.5)
        steps[-1] = length - 0.5 * (n - 1)
        h = heading[-1] + k * (np.cumsum(steps) - 0.5 * steps)
        pts = line[-1] + np.cumsum(np.column_stack((steps * np.cos(h), steps * np.sin(h))), axis=0)
        return np.vstack((line[-1], pts))

    def target_speed(self, cruise_mps, ego_speed_mps, gap_m, lead_speed_mps, dt_s, held_cap_mps=None, keep_gap=True, lead_id=None):
        """The speed to hold: the cruise speed, the gap-keeping speed and the worst-case cap,
        whichever is lowest, approached within the acceleration and jerk limits; None while
        following does not restrict the car.

        Watching a lead and restricting the car are two decisions (fifth review, F5-01: a
        lead 17 m ahead at 2 m/s was taken on at the start, the target began at the car's
        0.4 m/s and climbed at 1 m/s^2 under the jerk limit, braking the car from 2.0 to
        0.1 m/s; 0.95 s lost). While the gap term and the worst-case cap both allow the
        cruise speed and the car is within the cap, and no restriction is in force
        (``held_cap_mps``, the cap last applied, below cruise), there is no following cap:
        the driver accelerates as without a lead, and ``last`` keeps the lead's numbers.
        Following takes over when either term falls below cruise, from the car's speed or
        the restriction in force, whichever is lower (third review, R1), so the hand-over
        never lifts a cap; it lets go once its own target is back at cruise with both terms
        above it.

        ``keep_gap=False``: the worst case only, not the gap (a lead whose place on the car's
        way is uncertain: ``continuation``).

        ``lead_id``: the followed track. The target and its acceleration belong to that lead:
        when another becomes the lead, following starts again from the car's speed (or a
        restriction set for the new lead, as above), so a near slow lead's low target does
        not carry over to a far one that replaced it (sixth review, F6-05). The new lead's
        own worst case applies from the first tick."""
        c = self.config
        if lead_id is not None and lead_id != self._lead_id:
            self.reset()
            self._lead_id = lead_id
        desired_gap = c.gap0_m + c.headway_s * ego_speed_mps
        # Close a gap error over the headway time, around the lead's own speed.
        gap_speed = max(0.0, lead_speed_mps + (gap_m - desired_gap) / c.headway_s)
        cap = worst_case_speed_cap(gap_m, lead_speed_mps, c)
        kept = gap_speed if keep_gap else math.inf
        wanted = min(cruise_mps, kept, cap)
        allows_cruise = min(kept, cap) >= cruise_mps and cap >= ego_speed_mps
        restricted = held_cap_mps is not None and held_cap_mps < cruise_mps - 1e-6
        self.last = dict(gap_speed_mps=round(float(gap_speed), 2), worst_case_cap_mps=round(float(cap), 2), keep_gap=keep_gap)
        if allows_cruise and (self._target is None and not restricted or self._target is not None and self._target >= cruise_mps - 1e-6):
            self.reset()
            self.last["mode"] = "watching"
            return None
        if self._target is None:
            self._target = max(0.0, min(ego_speed_mps, held_cap_mps) if restricted else ego_speed_mps)
        self.last["mode"] = "restricting"
        current = self._target
        tau_accel = float(np.clip((wanted - current) / c.speed_tau_s, -c.ego_brake_mps2, c.accel_mps2))
        if current > cap + 1e-9:
            # Above the worst case: down to the cap at once, at up to ego_brake_mps2 (third
            # review, R1: jerk-limited, the target fell 0.005 m/s on the first tick, 6 m
            # behind a car cutting in at 6 m/s). Only to the cap: below it, closing the gap
            # is ordinary driving. Going on at ego_brake to the gap term's zero, and then
            # unwinding the jerk limit from -4 m/s^2, stopped the car for 1.4 s behind a
            # cut-in moving at 1.3 m/s (live slower cut-in, c882389). From the cap on, the
            # ordinary approach picks up without a jerk to unwind.
            target = max(cap, current - c.ego_brake_mps2 * dt_s)
            self._accel = min(0.0, tau_accel)
        else:
            # Jerk-limited, closing the error to the wanted speed over speed_tau_s.
            accel = float(np.clip(tau_accel, self._accel - c.jerk_mps3 * dt_s, self._accel + c.jerk_mps3 * dt_s))
            self._accel = accel
            target = current + accel * dt_s
        self._target = max(0.0, target)
        if self._target <= 0.0:
            self._accel = max(self._accel, 0.0)  # standing still: no deceleration wound up
        return self._target
