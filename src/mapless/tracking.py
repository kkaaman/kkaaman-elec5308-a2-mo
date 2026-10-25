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

    def _radar_jacobian(self, x, sensor_odo, eps=1e-4):
        base = self._radar_model(x, sensor_odo)
        H = np.zeros((3, 4))
        for k in range(4):
            dx = np.zeros(4)
            dx[k] = eps
            d = self._radar_model(x + dx, sensor_odo) - base
            d[1] = (d[1] + math.pi) % (2 * math.pi) - math.pi
            H[:, k] = d / eps
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
            for i, tid in enumerate(ids):
                t = self.tracks[tid]
                for j, m in enumerate(lidar):
                    if np.hypot(*(t.x[:2] - z_odo[j])) > c.lidar_gate_m or self._class_veto(t, m.class_probabilities):
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
            for j, m in enumerate(radar):
                sensor_odo = pose.to_world(m.sensor_xy[None])[0]
                z = np.array([m.range_m, m.bearing_rad + pose.yaw_rad, m.radial_speed_mps])
                for i, tid in enumerate(ids):
                    t = self.tracks[tid]
                    obj_odo = pose.to_world(m.xy[None])[0]
                    if np.hypot(*(t.x[:2] - obj_odo)) > c.radar_gate_m:
                        continue
                    h, H = self._radar_jacobian(t.x, sensor_odo)
                    R = np.diag([c.radar_range_sigma_m**2, c.radar_bearing_sigma_rad**2, c.radar_rate_sigma_mps**2])
                    y = z - h
                    y[1] = (y[1] + math.pi) % (2 * math.pi) - math.pi
                    S = H @ t.P @ H.T + R
                    d2_pos = float(y[:2] @ np.linalg.solve(S[:2, :2], y[:2]))
                    cost[i, j] = d2_pos
                    models[(i, j)] = (z, h, H, R, y, S, sensor_odo)
            assigned = _assign(cost, CHI2_2D)
            for i, j in assigned:
                t = self.tracks[ids[i]]
                z, h, H, R, y, S, sensor_odo = models[(i, j)]
                d2_rate = float(y[2] ** 2 / S[2, 2])
                if d2_rate <= CHI2_1D:
                    self._kalman(t, y, H, R)
                else:
                    # Radial speed inconsistent with the track: use range and bearing only.
                    self._kalman(t, y[:2], H[:2], R[:2, :2])
                self._absorb(t, None, "radar", time_s)
                hit.add(ids[i])
            # Unassigned radar objects start tracks too (a car beyond the LiDAR's range).
            assigned_cols = {j for _, j in assigned}
            for j, m in enumerate(radar):
                if j not in assigned_cols:
                    obj_odo = pose.to_world(m.xy[None])[0]
                    tid = self._spawn(obj_odo, 1.0, None, time_s, "radar")
                    hit.add(tid)
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
        return self.snapshot()

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

    def _spawn(self, xy_odo, sigma_m, probabilities, time_s, sensor):
        c = self.config
        tid = self._next_id
        self._next_id += 1
        P = np.diag([sigma_m**2, sigma_m**2, c.initial_speed_sigma_mps**2, c.initial_speed_sigma_mps**2])
        track = _Track(tid, np.array([xy_odo[0], xy_odo[1], 0.0, 0.0]), P, time_s, time_s, _normalised(probabilities), origin=np.array(xy_odo[:2], dtype=float))
        track.history = deque(maxlen=c.confirm_window)
        track.hits = 1
        track.support = (sensor,)
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
                    support=t.support,
                    footprint=t.footprint,
                    travelled_m=None if t.origin is None else float(np.hypot(*(t.x[:2] - t.origin))),
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
