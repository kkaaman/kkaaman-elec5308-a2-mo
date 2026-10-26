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
for the 2-D position and 6.63 for the 1-D radial velocity. The velocity innovation is
checked on its own: a radar return whose position fits but whose radial speed does
not updates the position only. Assignment is one to one, nearest normalised distance
first (an optimal assignment can replace ``_assign`` without touching the rest).

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
    origin_time_s: float = 0.0  # when the track started


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
    def update(self, time_s, pose, lidar=(), radar=(), camera_classes=()):
        """Advance to ``time_s`` and absorb this tick's measurements (vehicle frame, via ``pose``).

        ``camera_classes``: (xy_vehicle, class_probabilities) pairs for detections no
        cluster claimed; they refresh a nearby track's class and nothing else.
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
                S = t.P[:2, :2] + np.eye(2) * m.sigma_m**2
                y = z_odo[j] - t.x[:2]
                cost[i, j] = float(y @ np.linalg.solve(S, y))
            assigned = _assign(cost, CHI2_2D)
            for i, j in assigned:
                t, m = self.tracks[ids[i]], lidar[j]
                H = np.zeros((2, 4))
                H[0, 0] = H[1, 1] = 1.0
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
            for j, m in enumerate(radar):
                sensor_odo = pose.to_world(m.sensor_xy[None])[0]
                z = np.array([m.range_m, m.bearing_rad + pose.yaw_rad, m.radial_speed_mps])
                obj_odo = pose.to_world(m.xy[None])[0]
                for i in np.flatnonzero(np.hypot(pos[:, 0] - obj_odo[0], pos[:, 1] - obj_odo[1]) <= c.radar_gate_m):
                    t = self.tracks[ids[i]]
                    h, H = self._radar_jacobian(t.x, sensor_odo)
                    R = np.diag([c.radar_range_sigma_m**2, c.radar_bearing_sigma_rad**2, c.radar_rate_sigma_mps**2])
                    y = z - h
                    y[1] = (y[1] + math.pi) % (2 * math.pi) - math.pi
                    S = H @ t.P @ H.T + R
                    d2_pos = float(y[:2] @ np.linalg.solve(S[:2, :2], y[:2]))
                    cost[i, j] = d2_pos
                    models[(i, j)] = (z, h, H, R, y, S, sensor_odo)
            assigned = _assign(cost, CHI2_2D)
            rejected = []
            for i, j in assigned:
                t = self.tracks[ids[i]]
                z, h, H, R, y, S, sensor_odo = models[(i, j)]
                d2_rate = float(y[2] ** 2 / S[2, 2])
                if d2_rate <= CHI2_1D:
                    self._kalman(t, y, H, R)
                    t.doppler_rejects = 0
                else:
                    # Radial speed inconsistent with the track: use range and bearing only,
                    # and count it. The radar measures motion along its line of sight only,
                    # so agreement says nothing about motion across it (a walker crossing
                    # square on reads ~0); a persistent disagreement is logged, not
                    # resolved for either sensor (external review, 26 Oct).
                    self._kalman(t, y[:2], H[:2], R[:2, :2])
                    t.doppler_rejects += 1
                    rejected.append((t.track_id, round(float(y[2]), 2), t.doppler_rejects))
                self._absorb(t, None, "radar", time_s)
                hit.add(ids[i])
            # Unassigned radar objects start tracks where the radar adds something: beyond
            # the LiDAR's object range, or moving. Static near returns no track claims are clutter.
            assigned_cols = {j for _, j in assigned}
            spawned = skipped = 0
            for j, m in enumerate(radar):
                if j not in assigned_cols:
                    if m.range_m < c.radar_spawn_min_range_m and abs(m.radial_speed_mps) < c.radar_spawn_min_speed_mps:
                        skipped += 1
                        continue
                    obj_odo = pose.to_world(m.xy[None])[0]
                    tid = self._spawn(obj_odo, 1.0, None, time_s, "radar")
                    hit.add(tid)
                    spawned += 1
            self.radar_stats = dict(objects=len(radar), associated=len(assigned), doppler_rejected=rejected[:5], spawned=spawned, clutter_skipped=skipped)
        else:
            self.radar_stats = {}
        self._merge_radar_twins(time_s)
        self._drop_near_radar_clutter(pose, time_s)
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

    def _merge_radar_twins(self, now):
        """Fold each radar-only track lying inside a LiDAR-updated track's footprint (grown
        by ``merge_margin_m``; a disc of that radius without one) into it. The older id is
        kept, so a lead keeps its history; the merged track takes the LiDAR position and
        footprint, the velocity of whichever had the smaller velocity variance, and both
        tracks' support and hit counts."""
        c = self.config
        lidar_now = [t for t in self.tracks.values() if t.seen.get("lidar") == now]
        if not lidar_now:
            return
        twins = [t for t in self.tracks.values() if not self._fresh(t, "lidar", now) and self._fresh(t, "radar", now)]
        merges = 0
        for r in twins:
            best = None
            for cand in lidar_now:
                if cand.track_id not in self.tracks:
                    continue
                rel = r.x[:2] - cand.x[:2]
                if cand.footprint is not None:
                    length, width, yaw = cand.footprint
                    along = abs(rel[0] * math.cos(yaw) + rel[1] * math.sin(yaw)) - length / 2
                    across = abs(-rel[0] * math.sin(yaw) + rel[1] * math.cos(yaw)) - width / 2
                    inside = along <= c.merge_margin_m and across <= c.merge_margin_m
                else:
                    inside = float(np.hypot(*rel)) <= c.merge_margin_m
                d = float(np.hypot(*rel))
                if inside and (best is None or d < best[0]):
                    best = (d, cand)
            if best is None:
                continue
            lt = best[1]
            keep, gone = (r, lt) if r.track_id < lt.track_id else (lt, r)
            vel_from = r if np.trace(r.P[2:, 2:]) < np.trace(lt.P[2:, 2:]) else lt
            x = np.concatenate((lt.x[:2], vel_from.x[2:]))
            P = np.zeros((4, 4))
            P[:2, :2], P[2:, 2:] = lt.P[:2, :2], vel_from.P[2:, 2:]
            keep.x, keep.P = x, P
            keep.footprint = lt.footprint
            if np.ptp(lt.classes) > np.ptp(keep.classes):
                keep.classes = lt.classes.copy()
            for sensor, when in gone.seen.items():
                keep.seen[sensor] = max(keep.seen.get(sensor, -math.inf), when)
            keep.support = tuple(sorted(set(keep.support) | set(gone.support)))
            keep.hits += gone.hits
            keep.last_hit_s = max(keep.last_hit_s, gone.last_hit_s)
            keep.confirmed = keep.confirmed or gone.confirmed
            keep.doppler_rejects = min(keep.doppler_rejects, gone.doppler_rejects)
            del self.tracks[gone.track_id]
            merges += 1
        if merges and self.radar_stats is not None:
            self.radar_stats = dict(self.radar_stats, merged=merges)

    def _drop_near_radar_clutter(self, pose, now):
        """Delete static radar-only tracks within ``radar_clutter_range_m`` of the car that
        the LiDAR has not confirmed after ``radar_clutter_age_s``: what the radar adds is
        range and moving objects, and the LiDAR forms the objects this close."""
        c = self.config
        dropped = 0
        for tid in list(self.tracks):
            t = self.tracks[tid]
            if self._fresh(t, "lidar", now) or "lidar" in t.seen or not self._fresh(t, "radar", now):
                continue
            if now - t.origin_time_s < c.radar_clutter_age_s or float(np.hypot(*t.x[2:])) >= c.radar_spawn_min_speed_mps:
                continue
            if float(np.hypot(*pose.to_body(t.x[None, :2])[0])) < c.radar_clutter_range_m:
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


__all__ = ["CHI2_1D", "CHI2_2D", "LidarMeasurement", "RadarMeasurement", "Tracker", "TrackerConfig"]
