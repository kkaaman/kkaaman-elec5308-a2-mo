"""Multi-object tracker (plan v2.1, section 10). G3 framework.

One constant-velocity extended Kalman filter per track, state ``[x, y, vx, vy]`` in the
odometry frame. Three kinds of measurement update it:

- **LiDAR clusters** give position and extent (``LidarMeasurement``), and the class
  the camera painted on the cluster, when it did.
- **Radar objects** (``mapless.radar.RadarMeasurement``) give range, bearing and the
  object's own radial speed, lever-arm compensated.
- **Camera-only detections** give class evidence to a track they are associated with;
  they never create a track on their own (no range).

Association gates on space and class first, then a Mahalanobis gate: chi-squared 9.21
for a LiDAR cluster's 2-D position, 11.34 for a radar return's range, bearing and radial
speed together (a return whose speed belongs to another object does not associate: a
parked car's corner return beside a 5 m/s lead set its speed to 0.02 m/s, review of
3e9083a). The gate allows ``manoeuvre_accel_mps2`` over ``manoeuvre_gate_s`` of speed
change, and for a track the LiDAR holds, the return anywhere on its measured extent (a
return comes off a corner or a wheel arch, not the centroid; within 0.8 m only, the rest
spawned radar-only twins every tick). The radial speed is then checked on its own (6.63):
a return that fails it is dropped whole and counted. Assignment is one to one, nearest
normalised distance first (an optimal assignment can replace ``_assign`` without touching
the rest).

Lifecycle: tentative on the first hit; confirmed at 3 hits in 5 frames; coasts up to
``coast_s`` without a hit, predicted forward; deleted after ``delete_s``. Only a track
that was confirmed coasts: a tentative one that misses stays tentative (external review
of the G3 framework: a single cluster then one missed frame became COASTING and was
accepted as a lead car). A tentative track within ``hazard_m`` is still reported to the
planner as a hazard: close hazards act before confirmation. Deleting a track never
clears occupied space (the grid keeps its own evidence).

The tracker is pure arithmetic over measurements it is given; it never sees an actor,
an ID or a pose from the simulator.
"""

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from mapless.contracts import CLASSES, ObjectTrack, TrackStatus
from mapless.ego_motion import Pose2D
from mapless.radar import RadarMeasurement

CHI2_2D = 9.21  # 99% for 2 degrees of freedom
CHI2_1D = 6.63  # 99% for 1 degree of freedom
CHI2_3D = 11.34  # 99% for 3 degrees of freedom


@dataclass(frozen=True)
class LidarMeasurement:
    """A LiDAR cluster in the vehicle frame: its centre, extent and painted class."""

    xy: np.ndarray
    extent: tuple | None = None  # (length_m, width_m, yaw_rad) in the vehicle frame
    class_probabilities: dict | None = None
    points: int = 0
    sigma_m: float = 0.3


@dataclass(frozen=True)
class TrackerConfig:
    process_accel_mps2: float = 1.5  # white acceleration driving the CV model
    initial_speed_sigma_mps: float = 5.0
    confirm_hits: int = 3  # hits ...
    confirm_window: int = 5  # ... within this many frames
    coast_s: float = 0.3
    delete_s: float = 0.75
    hazard_m: float = 12.0  # tentative tracks this close are reported as hazards
    class_veto: float = 0.8  # two confident, different classes do not associate
    radar_gate_m: float = 4.0  # coarse spatial gate before the Mahalanobis gate
    lidar_gate_m: float = 4.0
    class_forgetting: float = 0.9  # per-update weight on the old class belief
    radar_range_sigma_m: float = 0.25
    radar_bearing_sigma_rad: float = math.radians(2.0)
    radar_rate_sigma_mps: float = 0.3
    # Recent motion: net displacement over the last motion_window_s, reported as a mean
    # speed once at least motion_min_span_s of history exists (ObjectTrack.recent_speed_mps).
    motion_window_s: float = 1.0
    motion_min_span_s: float = 0.5
    # A sensor supports a track while it has updated it within this time; one LiDAR hit
    # long ago no longer makes a radar-held track "LiDAR-supported" (review of 405b056).
    support_max_age_s: float = 0.5
    # A radar object no track claims starts a radar-only track only where the radar adds
    # something: beyond the range the LiDAR forms objects at (a car first clustered at
    # 30.5 m on Town05), or moving (ego-compensated radial speed). Inside that range a
    # static unclaimed return is pavement, kerb or building clutter: 155 tracks with the
    # front radar against 61 without, 45 ms of tracking a tick (Town05, 26 Oct).
    radar_spawn_min_range_m: float = 30.0
    radar_spawn_min_speed_mps: float = 0.5
    # ... and a static one only near the line ahead (vehicle frame): far static returns off
    # to the side are buildings and poles (B9: 130 tracks against 58 without the radar).
    radar_static_spawn_half_width_m: float = 8.0
    # One object, two tracks: the radar picks a car up beyond the LiDAR's reach; when the
    # LiDAR clusters it, the cluster fits its own new track better and the radar keeps
    # updating the old one (braking lead B8, 26 Oct: the lead followed as radar-only track
    # 166 to 12 m while its LiDAR twin, 1.7 x 1.0 m with sideways jitter, read as a
    # crossing walker and stopped the car). A radar-only track inside a LiDAR track's
    # footprint grown by ``merge_margin_m`` is merged into it, the older id kept.
    merge_margin_m: float = 1.0
    # A static radar-only track the LiDAR has not confirmed this close is clutter it has
    # carried in from beyond 30 m (B8: 148 tracks, 45 ms), once it is this old.
    radar_clutter_range_m: float = 25.0
    radar_clutter_age_s: float = 0.5
    # A radial speed rejected this many times running, all with the same sign, is the
    # track's motion changing (a manoeuvre), not an outlier: the velocity variance along
    # the line of sight is opened by the innovation and the radial speed is used. B9,
    # 26 Oct: the lead stopped within 0.2 s, the radar read it 1.7 m/s slower nine ticks
    # running, every one rejected, and the track's speed took 0.6 s to fall by LiDAR alone.
    # Only the radial component moves: a walker crossing square on keeps its speed.
    doppler_accept_after: int = 3  # 0.15 s at 20 Hz; two would let an association glitch through
    # ... and only a run within doppler_window_s whose innovations agree (within
    # doppler_agree_mps, or half the first one): three rejections 0.5 s apart, or of
    # different sizes, are not one manoeuvre (review of 3e9083a, H4). The velocity
    # variance is opened by at most (manoeuvre_accel_mps2 x the time since the first),
    # not by the innovation squared, which let one wrong return set any speed.
    doppler_window_s: float = 0.3
    doppler_agree_mps: float = 1.0
    # ... and whose returns sit on the track: the run's mean range and bearing innovations,
    # each in its own sigma, within doppler_position_bias. A neighbour's returns 3.3 m to the
    # side of a radar-only lead agreed with each other for three ticks and took it over;
    # its bearing sat 2.4 sigma out every time, a braking lead's scatter about zero.
    doppler_position_bias: float = 1.5
    manoeuvre_accel_mps2: float = 10.0
    manoeuvre_gate_s: float = 0.1  # the association gate allows this long at manoeuvre_accel_mps2
    # A radar-only twin folds into a LiDAR track only when, for merge_persist ticks
    # running, it lies inside that track's footprint, their radial speeds agree within
    # three sigma plus merge_speed_margin_mps, and the LiDAR track is bigger than a walker
    # (merge_min_extent_m): two cars passing in opposite directions merged into one with
    # one's velocity (external review of the full review of 73ae759), and a walker beside
    # a radar-held car would take the car's speed and the car's track would vanish.
    merge_persist: int = 2
    merge_speed_margin_mps: float = 0.5
    merge_min_extent_m: float = 1.2
    # A walker-sized cluster (under merge_min_extent_m) does not take over a radar-only
    # track moving faster than a cyclist: it is someone beside that car, not the car.
    vru_max_speed_mps: float = 6.0

    def __post_init__(self):
        if self.confirm_hits > self.confirm_window or self.coast_s >= self.delete_s:
            raise ValueError("TrackerConfig: confirm_hits <= confirm_window and coast_s < delete_s")


@dataclass
class _Track:
    track_id: int
    x: np.ndarray
    P: np.ndarray
    time_s: float
    last_hit_s: float
    classes: np.ndarray
    history: deque = field(default_factory=lambda: deque(maxlen=5))
    hits: int = 0
    misses: int = 0
    support: tuple = ()
    footprint: tuple | None = None
    confirmed: bool = False
    origin: np.ndarray | None = None  # first measured position (odometry frame)
    trail: deque = field(default_factory=lambda: deque(maxlen=64))  # (time, x, y) after each update
    seen: dict = field(default_factory=dict)  # sensor -> last time it updated the track
    doppler_rejects: int = 0  # consecutive radar updates with the radial speed rejected
    doppler_sign: float = 0.0  # the sign of the last rejected radial-speed innovation
    doppler_first_s: float = -math.inf  # when the current run of rejections began ...
    doppler_first_y: float = 0.0  # ... its first innovation ...
    doppler_last_y: float = 0.0  # ... and its last
    doppler_position_sum: np.ndarray = field(default_factory=lambda: np.zeros(2))  # normalised range and bearing innovations of the run
    origin_time_s: float = 0.0  # when the track started
    twin_of: int | None = None  # the LiDAR track a radar-only track has lain inside ...
    twin_ticks: int = 0  # ... for this many updates running


def _recent_speed(trail, min_span_s):
    """Net displacement over the trail's span divided by the span, or None if shorter
    than ``min_span_s``: sustained motion, which a static object's centroid drifting as
    the car passes it (a pole, a short hedge) does not show (third live stack M04)."""
    if len(trail) < 2:
        return None
    (t0, x0, y0), (t1, x1, y1) = trail[0], trail[-1]
    span = t1 - t0
    return None if span < min_span_s else float(math.hypot(x1 - x0, y1 - y0) / span)


def _normalised(probabilities):
    p = np.array([max(0.0, float(probabilities.get(k, 0.0))) for k in CLASSES]) if probabilities else np.ones(len(CLASSES))
    return p / p.sum() if p.sum() > 0 else np.ones(len(CLASSES)) / len(CLASSES)


def _assign(cost, gate):
    """One-to-one assignment, cheapest pair first; cost above ``gate`` never assigns."""
    pairs = []
    if cost.size == 0:
        return pairs
    used_rows, used_cols = set(), set()
    order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
    for i, j in order:
        if cost[i, j] > gate:
            break
        if i in used_rows or j in used_cols:
            continue
        pairs.append((int(i), int(j)))
        used_rows.add(i)
        used_cols.add(j)
    return pairs


class Tracker:
    def __init__(self, config=None):
        self.config = config or TrackerConfig()
        self.tracks = {}
        self._next_id = 1
        self._time = None
        # This tick's radar bookkeeping, for the log: objects, those associated, radial
        # speeds rejected (with the innovations), radar-only tracks started, clutter skipped.
        self.radar_stats = {}

    # -- model ---------------------------------------------------------------------------
    def _predict(self, track, dt):
        if dt <= 0:
            return
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = self.config.process_accel_mps2**2
        G = np.array([[dt**2 / 2, 0], [0, dt**2 / 2], [dt, 0], [0, dt]])
        track.x = F @ track.x
        track.P = F @ track.P @ F.T + G @ (q * np.eye(2)) @ G.T
        track.time_s += dt

    @staticmethod
    def _kalman(track, y, H, R):
        """Update with innovation ``y`` (already wrapped for angles)."""
        S = H @ track.P @ H.T + R
        K = track.P @ H.T @ np.linalg.inv(S)
        track.x = track.x + K @ y
        track.P = (np.eye(4) - K @ H) @ track.P
        return float(y @ np.linalg.solve(S, y))

    @staticmethod
    def _radar_model(x, sensor_odo):
        p = x[:2] - sensor_odo
        r = max(float(np.hypot(*p)), 1e-6)
        u = p / r
        return np.array([r, math.atan2(p[1], p[0]), float(x[2:] @ u)])

    def _radar_jacobian(self, x, sensor_odo):
        """The radar model and its exact Jacobian (range, bearing, radial speed over
        ``[x, y, vx, vy]``); was by finite differences, four model calls per pair."""
        p = x[:2] - sensor_odo
        r = max(float(np.hypot(*p)), 1e-6)
        u = p / r
        rate = float(x[2:] @ u)
        base = np.array([r, math.atan2(p[1], p[0]), rate])
        H = np.zeros((3, 4))
        H[0, :2] = u
        H[1, :2] = (-p[1] / r**2, p[0] / r**2)
        H[2, :2] = (x[2:] - rate * u) / r
        H[2, 2:] = u
        return base, H

    # -- update --------------------------------------------------------------------------
    def update(self, time_s, pose, lidar=(), radar=(), camera_classes=(), lidar_view=None):
        """Advance to ``time_s`` and absorb this tick's measurements (vehicle frame, via ``pose``).

        ``camera_classes``: (xy_vehicle, class_probabilities) pairs for detections no
        cluster claimed; they refresh a nearby track's class and nothing else.
        ``lidar_view``: odometry points (N x 2) -> bool array, whether the LiDAR observed
        each place just now; a radar-only track is clutter only where it did. None: everywhere.
        Returns the current ObjectTracks (deleted ones removed).
        """
        if not isinstance(pose, Pose2D):
            raise TypeError("pose must be a Pose2D")
        c = self.config
        dt = 0.0 if self._time is None else time_s - self._time
        self._time = time_s
        for track in self.tracks.values():
            self._predict(track, dt)
        hit = set()
        # LiDAR: position (and extent, class) ------------------------------------------
        lidar = list(lidar)
        if lidar:
            z_odo = pose.to_world(np.array([m.xy for m in lidar]))
            ids = list(self.tracks)
            cost = np.full((len(ids), len(lidar)), np.inf)
            pos = np.array([self.tracks[tid].x[:2] for tid in ids]).reshape(-1, 2)
            near = np.hypot(pos[:, None, 0] - z_odo[None, :, 0], pos[:, None, 1] - z_odo[None, :, 1]) <= c.lidar_gate_m
            for i, j in zip(*np.nonzero(near)):
                t, m = self.tracks[ids[i]], lidar[j]
                if self._class_veto(t, m.class_probabilities):
                    continue
                if "lidar" not in t.seen and m.extent is not None and max(m.extent[0], m.extent[1]) < c.merge_min_extent_m and float(np.hypot(*t.x[2:])) > c.vru_max_speed_mps:
                    continue
                S = t.P[:2, :2] + np.eye(2) * m.sigma_m**2
                y = z_odo[j] - t.x[:2]
                cost[i, j] = float(y @ np.linalg.solve(S, y))
            assigned = _assign(cost, CHI2_2D)
            for i, j in assigned:
                t, m = self.tracks[ids[i]], lidar[j]
                H = np.zeros((2, 4))
                H[0, 0] = H[1, 1] = 1.0
                if "lidar" not in t.seen:
                    # The first LiDAR fix of a radar-held track: from now on the cluster's
                    # centre is its position, not the radar's point (a front corner 1.8 m
                    # out). A change of reference, not motion: the position is set to the
                    # cluster, and its trail and origin move with it. Blended in over many
                    # ticks, the jump read as 1.8 m/s and a parked car stayed "uncertain" for
                    # a second (review of 3e9083a, M6).
                    self._shift_history(t, z_odo[j] - t.x[:2])
                    t.x[:2] = z_odo[j]
                    t.P[:2, :2] = np.eye(2) * m.sigma_m**2
                    t.P[:2, 2:] = t.P[2:, :2] = 0.0
                else:
                    self._kalman(t, z_odo[j] - t.x[:2], H, np.eye(2) * m.sigma_m**2)
                self._absorb(t, m.class_probabilities, "lidar", time_s)
                if m.extent is not None:
                    t.footprint = (m.extent[0], m.extent[1], pose.yaw_rad + m.extent[2])
                hit.add(ids[i])
            taken = {j for _, j in assigned}
            for j, m in enumerate(lidar):
                if j not in taken:
                    tid = self._spawn(z_odo[j], m.sigma_m, m.class_probabilities, time_s, "lidar")
                    t = self.tracks[tid]
                    if m.extent is not None:
                        t.footprint = (m.extent[0], m.extent[1], pose.yaw_rad + m.extent[2])
                    hit.add(tid)
        # Radar: range, bearing, radial speed --------------------------------------------
        radar = list(radar)
        if radar:
            ids = list(self.tracks)
            cost = np.full((len(ids), len(radar)), np.inf)
            models = {}
            pos = np.array([self.tracks[tid].x[:2] for tid in ids]).reshape(-1, 2)
            R = np.diag([c.radar_range_sigma_m**2, c.radar_bearing_sigma_rad**2, c.radar_rate_sigma_mps**2])
            allowance = (c.manoeuvre_accel_mps2 * c.manoeuvre_gate_s) ** 2
            for j, m in enumerate(radar):
                sensor_odo = pose.to_world(m.sensor_xy[None])[0]
                z = np.array([m.range_m, m.bearing_rad + pose.yaw_rad, m.radial_speed_mps])
                obj_odo = pose.to_world(m.xy[None])[0]
                for i in np.flatnonzero(np.hypot(pos[:, 0] - obj_odo[0], pos[:, 1] - obj_odo[1]) <= c.radar_gate_m):
                    t = self.tracks[ids[i]]
                    h, H = self._radar_jacobian(t.x, sensor_odo)
                    Rt = self._extent_noise(t, R, h, time_s)
                    y = z - h
                    y[1] = (y[1] + math.pi) % (2 * math.pi) - math.pi
                    S = H @ t.P @ H.T + Rt
                    gate = S.copy()
                    gate[2, 2] += allowance
                    d2 = float(y @ np.linalg.solve(gate, y))
                    if d2 <= CHI2_3D:
                        cost[i, j] = d2
                        models[(i, j)] = (z, h, H, Rt, y, S, sensor_odo)
            assigned = _assign(cost, CHI2_3D)
            rejected, manoeuvres = [], []
            for i, j in assigned:
                t = self.tracks[ids[i]]
                z, h, H, Rt, y, S, sensor_odo = models[(i, j)]
                d2_rate = float(y[2] ** 2 / S[2, 2])
                sign = math.copysign(1.0, y[2])
                # A track the LiDAR placed this tick takes the radar's radial speed only: a
                # radar return comes off one part of an object (a wheel arch, a corner), not
                # its centre, and pulling a 3 x 2 m object's position between that and the
                # LiDAR's centroid every tick read as 0.4-0.5 m/s of motion and a conflict
                # (B10, 26 Oct). The LiDAR gives position and extent; the radar, closing speed.
                rows = slice(2, 3) if t.seen.get("lidar") == time_s else slice(0, 3)
                tol = max(c.doppler_agree_mps, 0.5 * min(abs(t.doppler_first_y), abs(float(y[2]))))
                position = np.array([y[0] / math.sqrt(S[0, 0]), y[1] / math.sqrt(S[1, 1])])
                run = time_s - t.doppler_first_s <= c.doppler_window_s + 1e-9 and t.doppler_sign == sign and abs(y[2] - t.doppler_first_y) <= tol and abs(y[2] - t.doppler_last_y) <= tol
                if d2_rate <= CHI2_1D:
                    self._kalman(t, y[rows], H[rows], Rt[rows, rows])
                    t.doppler_rejects = 0
                elif run and t.doppler_rejects + 1 >= c.doppler_accept_after and np.all(np.abs((t.doppler_position_sum + position) / (t.doppler_rejects + 1)) <= c.doppler_position_bias):
                    # Persistent, one-sided and consistent: the object's motion changed. Open
                    # the velocity variance along the line of sight (H's radial-speed row is
                    # the unit line-of-sight vector over vx, vy) by what it could have changed
                    # since the run began, and take the update.
                    u = H[2, 2:]
                    bound = (c.manoeuvre_accel_mps2 * max(time_s - t.doppler_first_s, c.manoeuvre_gate_s)) ** 2
                    t.P[2:, 2:] += min(float(y[2] ** 2), bound) * np.outer(u, u)
                    self._kalman(t, y[rows], H[rows], Rt[rows, rows])
                    manoeuvres.append((t.track_id, round(float(y[2]), 2)))
                    t.doppler_rejects = 0
                else:
                    # Radial speed inconsistent with the track: the whole return is dropped
                    # (its position may be another object's too) and counted. The radar
                    # measures motion along its line of sight only, so agreement says
                    # nothing about motion across it (a walker crossing square on reads ~0).
                    if run:
                        t.doppler_rejects += 1
                        t.doppler_position_sum = t.doppler_position_sum + position
                    else:
                        t.doppler_rejects, t.doppler_first_s, t.doppler_first_y = 1, time_s, float(y[2])
                        t.doppler_position_sum = position
                    t.doppler_sign, t.doppler_last_y = sign, float(y[2])
                    rejected.append((t.track_id, round(float(y[2]), 2), t.doppler_rejects))
                self._absorb(t, None, "radar", time_s)
                hit.add(ids[i])
            # Unassigned radar objects start tracks where the radar adds something: beyond
            # the LiDAR's object range, or moving. Static near returns no track claims are clutter.
            assigned_cols = {j for _, j in assigned}
            spawned = skipped = 0
            for j, m in enumerate(radar):
                if j not in assigned_cols:
                    static = abs(m.radial_speed_mps) < c.radar_spawn_min_speed_mps
                    if static and (m.range_m < c.radar_spawn_min_range_m or abs(float(m.xy[1])) > c.radar_static_spawn_half_width_m):
                        skipped += 1
                        continue
                    obj_odo = pose.to_world(m.xy[None])[0]
                    tid = self._spawn(obj_odo, 1.0, None, time_s, "radar")
                    hit.add(tid)
                    spawned += 1
            self.radar_stats = dict(objects=len(radar), associated=len(assigned), doppler_rejected=rejected[:5], doppler_manoeuvres=manoeuvres[:5], spawned=spawned, clutter_skipped=skipped)
        else:
            self.radar_stats = {}
        self._merge_radar_twins(time_s, pose)
        self._drop_near_radar_clutter(pose, time_s, lidar_view)
        # Camera-only class evidence -------------------------------------------------------
        for xy, probabilities in camera_classes:
            odo = pose.to_world(np.asarray(xy, dtype=float)[None])[0]
            nearest = min(self.tracks.values(), key=lambda t: np.hypot(*(t.x[:2] - odo)), default=None)
            if nearest is not None and np.hypot(*(nearest.x[:2] - odo)) <= c.lidar_gate_m:
                self._absorb(nearest, probabilities, "camera", time_s, counts=False)
        # Lifecycle ------------------------------------------------------------------------
        for tid in list(self.tracks):
            t = self.tracks[tid]
            if tid in hit:
                t.history.append(True)
                t.misses = 0
            else:
                t.history.append(False)
                t.misses += 1
            if sum(t.history) >= c.confirm_hits:
                t.confirmed = True
            if time_s - t.last_hit_s > c.delete_s:
                del self.tracks[tid]
        for track in self.tracks.values():
            track.trail.append((time_s, float(track.x[0]), float(track.x[1])))
            while track.trail and time_s - track.trail[0][0] > c.motion_window_s + 1e-9:
                track.trail.popleft()
        return self.snapshot()

    def _fresh(self, track, sensor, now):
        return now - track.seen.get(sensor, -math.inf) <= self.config.support_max_age_s + 1e-9

    def _extent_noise(self, track, R, h, now):
        """``R`` with the range and bearing loosened by the half-extent of a LiDAR-held
        track's footprint along and across the line of sight ``h[1]``: its return may come
        off any part of it."""
        if track.footprint is None or not self._fresh(track, "lidar", now):
            return R
        length, width, yaw = track.footprint
        rel = yaw - float(h[1])
        along = 0.5 * (length * abs(math.cos(rel)) + width * abs(math.sin(rel)))
        across = 0.5 * (length * abs(math.sin(rel)) + width * abs(math.cos(rel)))
        out = R.copy()
        out[0, 0] += along**2
        out[1, 1] += (across / max(float(h[0]), 1.0)) ** 2
        return out

    def _merge_radar_twins(self, now, pose=None):
        """Fold a radar-only track into a LiDAR-updated one when, for ``merge_persist``
        updates running, it lies inside that track's footprint (grown by ``merge_margin_m``),
        their radial speeds agree, and the LiDAR track is bigger than a walker. The older id
        is kept, so a lead keeps its history; the merged track takes the LiDAR position,
        footprint, trail and origin, and the two velocity estimates combined (the radar's
        speed was thrown away when the fresh LiDAR twin's was kept: review of 3e9083a, M1)."""
        c = self.config
        lidar_now = [t for t in self.tracks.values() if t.seen.get("lidar") == now]
        twins = [t for t in self.tracks.values() if not self._fresh(t, "lidar", now) and self._fresh(t, "radar", now)]
        if not lidar_now:
            for r in twins:
                r.twin_of, r.twin_ticks = None, 0
            return
        sensor = np.array([pose.x_m, pose.y_m]) if pose is not None else None
        merges = 0
        # Why a radar-only track next to a LiDAR one was not folded in, for the log (fourth
        # review: duplicate tracks on one car survive the merge; which gate keeps them?).
        blocked = []
        for r in twins:
            best = nearest = None
            for cand in lidar_now:
                if cand.track_id not in self.tracks:
                    continue
                rel = r.x[:2] - cand.x[:2]
                d = float(np.hypot(*rel))
                if cand.footprint is None or max(cand.footprint[0], cand.footprint[1]) < c.merge_min_extent_m:
                    why = "the LiDAR track is smaller than merge_min_extent_m" if cand.footprint is not None else "the LiDAR track has no extent"
                else:
                    length, width, yaw = cand.footprint
                    along = abs(rel[0] * math.cos(yaw) + rel[1] * math.sin(yaw)) - length / 2
                    across = abs(-rel[0] * math.sin(yaw) + rel[1] * math.cos(yaw)) - width / 2
                    if along > c.merge_margin_m or across > c.merge_margin_m:
                        why = f"outside its footprint by {max(along, across) - c.merge_margin_m:.1f} m"
                    elif not self._speeds_agree(r, cand, sensor):
                        why = "radial speeds disagree"
                    else:
                        why = None
                        if best is None or d < best[0]:
                            best = (d, cand)
                if why is not None and (nearest is None or d < nearest[0]):
                    nearest = (d, cand, why)
            if best is None:
                r.twin_of, r.twin_ticks = None, 0
                if nearest is not None and nearest[0] <= 5.0:
                    blocked.append([r.track_id, nearest[1].track_id, round(nearest[0], 1), nearest[2]])
                continue
            lt = best[1]
            r.twin_ticks = r.twin_ticks + 1 if r.twin_of == lt.track_id else 1
            r.twin_of = lt.track_id
            if r.twin_ticks < c.merge_persist:
                blocked.append([r.track_id, lt.track_id, round(best[0], 1), f"persisting ({r.twin_ticks} of {c.merge_persist} ticks)"])
                continue
            keep, gone = (r, lt) if r.track_id < lt.track_id else (lt, r)

            def span(track):
                return track.trail[-1][0] - track.trail[0][0] if len(track.trail) > 1 else 0.0

            src = r if span(r) > span(lt) else lt
            offset = lt.x[:2] - src.x[:2]  # before keep.x is overwritten (keep may be src)
            # Two estimates of one velocity, from independent measurements: combined.
            info_r, info_l = np.linalg.inv(r.P[2:, 2:]), np.linalg.inv(lt.P[2:, 2:])
            Pv = np.linalg.inv(info_r + info_l)
            v = Pv @ (info_r @ r.x[2:] + info_l @ lt.x[2:])
            P = np.zeros((4, 4))
            P[:2, :2], P[2:, 2:] = lt.P[:2, :2], Pv
            keep.x, keep.P = np.concatenate((lt.x[:2], v)), P
            keep.footprint = lt.footprint
            # The longer history, moved to where the LiDAR has the object: the radar track's
            # trail and origin as they were, a return off its front 1.8 m away, made a parked
            # car "uncertain" for a second and inflated its travelled distance for good
            # (review of 3e9083a, M6); the fresh LiDAR twin's alone would be too short to show
            # it standing.
            keep.trail, keep.origin, keep.origin_time_s = deque(src.trail, maxlen=src.trail.maxlen), None if src.origin is None else src.origin.copy(), src.origin_time_s
            self._shift_history(keep, offset)
            if np.ptp(lt.classes) > np.ptp(keep.classes):
                keep.classes = lt.classes.copy()
            for sensor_name, when in gone.seen.items():
                keep.seen[sensor_name] = max(keep.seen.get(sensor_name, -math.inf), when)
            keep.support = tuple(sorted(set(keep.support) | set(gone.support)))
            keep.hits += gone.hits
            keep.last_hit_s = max(keep.last_hit_s, gone.last_hit_s)
            keep.confirmed = keep.confirmed or gone.confirmed
            keep.doppler_rejects = min(keep.doppler_rejects, gone.doppler_rejects)
            keep.twin_of, keep.twin_ticks = None, 0
            del self.tracks[gone.track_id]
            merges += 1
        if merges and self.radar_stats is not None:
            self.radar_stats = dict(self.radar_stats, merged=merges)
        if blocked and self.radar_stats is not None:
            self.radar_stats = dict(self.radar_stats, twins_kept=blocked[:8])  # [radar id, LiDAR id, distance, why]

    @staticmethod
    def _shift_history(track, offset):
        """Move a track's trail and origin by ``offset`` (odometry): its reference point
        changed, the object did not move."""
        dx, dy = float(offset[0]), float(offset[1])
        track.trail = deque(((ts, x + dx, y + dy) for ts, x, y in track.trail), maxlen=track.trail.maxlen)
        if track.origin is not None:
            track.origin = track.origin + np.array([dx, dy])

    def _speeds_agree(self, a, b, sensor):
        """Radial speeds (along the line of sight from ``sensor``, odometry) within three
        sigma of each other plus ``merge_speed_margin_mps``."""
        rel = a.x[:2] - (sensor if sensor is not None else np.zeros(2))
        u = rel / max(float(np.hypot(*rel)), 1e-6)
        diff = float((a.x[2:] - b.x[2:]) @ u)
        sigma = math.sqrt(float(u @ a.P[2:, 2:] @ u) + float(u @ b.P[2:, 2:] @ u))
        return abs(diff) <= 3.0 * sigma + self.config.merge_speed_margin_mps

    def _drop_near_radar_clutter(self, pose, now, lidar_view=None):
        """Delete static radar-only tracks within ``radar_clutter_range_m`` of the car that
        the LiDAR has not confirmed after ``radar_clutter_age_s``, where the LiDAR observed
        the place (``lidar_view``): it saw the ground there, or something its own checks
        hold. Where it saw nothing (occluded, between rings), the radar is the only
        evidence and the track stays (external review: missing LiDAR confirmation is not
        clutter when the LiDAR could not have seen it)."""
        c = self.config
        dropped = 0
        for tid in list(self.tracks):
            t = self.tracks[tid]
            if self._fresh(t, "lidar", now) or "lidar" in t.seen or not self._fresh(t, "radar", now):
                continue
            if now - t.origin_time_s < c.radar_clutter_age_s or float(np.hypot(*t.x[2:])) >= c.radar_spawn_min_speed_mps:
                continue
            if float(np.hypot(*pose.to_body(t.x[None, :2])[0])) >= c.radar_clutter_range_m:
                continue
            if lidar_view is not None and not bool(np.asarray(lidar_view(t.x[None, :2])).reshape(-1)[0]):
                continue
            del self.tracks[tid]
            dropped += 1
        if dropped and self.radar_stats is not None:
            self.radar_stats = dict(self.radar_stats, clutter_dropped=dropped)

    def _class_veto(self, track, probabilities):
        if not probabilities:
            return False
        p = _normalised(probabilities)
        c = self.config.class_veto
        return bool(track.classes.max() >= c and p.max() >= c and track.classes.argmax() != p.argmax())

    def _absorb(self, track, probabilities, sensor, time_s, counts=True):
        if probabilities:
            track.classes = self.config.class_forgetting * track.classes + (1 - self.config.class_forgetting) * _normalised(probabilities)
            track.classes /= track.classes.sum()
        if counts:
            track.hits += 1
            track.last_hit_s = time_s
        track.support = tuple(sorted(set(track.support) | {sensor}))
        track.seen[sensor] = time_s

    def _spawn(self, xy_odo, sigma_m, probabilities, time_s, sensor):
        c = self.config
        tid = self._next_id
        self._next_id += 1
        P = np.diag([sigma_m**2, sigma_m**2, c.initial_speed_sigma_mps**2, c.initial_speed_sigma_mps**2])
        track = _Track(tid, np.array([xy_odo[0], xy_odo[1], 0.0, 0.0]), P, time_s, time_s, _normalised(probabilities), origin=np.array(xy_odo[:2], dtype=float), origin_time_s=time_s)
        track.history = deque(maxlen=c.confirm_window)
        track.hits = 1
        track.support = (sensor,)
        track.seen = {sensor: time_s}
        self.tracks[tid] = track
        return tid

    # -- output --------------------------------------------------------------------------
    def snapshot(self):
        c = self.config
        out = []
        for t in self.tracks.values():
            since = (self._time or t.time_s) - t.last_hit_s
            # ``confirmed`` is independent of this frame's observation: a confirmed track
            # unseen this frame coasts; an unconfirmed one stays tentative until deleted.
            if since > c.delete_s:
                status = TrackStatus.DELETED
            elif not t.confirmed:
                status = TrackStatus.TENTATIVE
            elif since > 1e-9:
                status = TrackStatus.COASTING
            else:
                status = TrackStatus.CONFIRMED
            out.append(
                ObjectTrack(
                    track_id=t.track_id,
                    time_s=t.time_s,
                    state=t.x.copy(),
                    covariance=t.P.copy(),
                    class_probabilities=dict(zip(CLASSES, t.classes.tolist())),
                    status=status,
                    hits=t.hits,
                    misses=t.misses,
                    support=tuple(sorted(k for k, when in t.seen.items() if (self._time or t.time_s) - when <= c.support_max_age_s + 1e-9)) if t.seen else t.support,
                    footprint=t.footprint,
                    travelled_m=None if t.origin is None else float(np.hypot(*(t.x[:2] - t.origin))),
                    recent_speed_mps=_recent_speed(t.trail, c.motion_min_span_s),
                    doppler_rejects=t.doppler_rejects,
                )
            )
        return out

    def live(self, tracks=None):
        """Tracks the planner may act on: seen this frame, or unseen for at most coast_s
        (confirmed ones coasting, tentative ones still tentative); a track unseen for
        longer is kept only for re-association, not for planning."""
        now = self._time or 0.0
        out = []
        for track in self.snapshot() if tracks is None else tracks:
            if track.status is TrackStatus.DELETED or track.track_id not in self.tracks:
                continue
            if now - self.tracks[track.track_id].last_hit_s > self.config.coast_s:
                continue
            out.append(track)
        return out

    def hazards(self, pose, within_m=None):
        """Tracks the planner must act on: confirmed ones (coasting at most coast_s), plus
        tentative ones within hazard_m ahead: close hazards act before confirmation."""
        limit = self.config.hazard_m if within_m is None else within_m
        out = []
        for track in self.live():
            body = pose.to_body(track.position[None])[0]
            near = 0.0 <= body[0] and np.hypot(*body) <= limit
            if track.status in (TrackStatus.CONFIRMED, TrackStatus.COASTING) or near:
                out.append(track)
        return out

    def reset(self):
        self.tracks.clear()
        self._time = None


__all__ = ["CHI2_1D", "CHI2_2D", "CHI2_3D", "LidarMeasurement", "RadarMeasurement", "Tracker", "TrackerConfig"]
