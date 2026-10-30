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
normalised distance first, or with ``TrackerConfig.assignment`` "optimal" the least total
cost per group of gated pairs (``_assign_optimal``).

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
    plan_xy: np.ndarray | None = None  # the cluster's points in plan (vehicle frame), for a remembered shape (``Tracker``)


@dataclass(frozen=True)
class TrackerConfig:
    process_accel_mps2: float = 1.5  # white acceleration driving the CV model
    # One-to-one association (ninth review, N9-02): "greedy", the cheapest gated pair first,
    # or "optimal", the least total cost with a pair left unassigned costing the gate
    # (``_assign_optimal``). Benchmarked on the recorded runs before any change of default.
    assignment: str = "greedy"
    # A radar return whose radial speed is rejected: "drop" (the whole return; eighth review,
    # F8-01) or "position" (its range and bearing still update a track the LiDAR did not
    # place this tick, observing nothing: no hit, no freshness). Tenth review, 2A: compared
    # on the braking matrix and the recorded runs before any change of default.
    doppler_reject_policy: str = "drop"
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
    # Motion coherence (fifth review, 4.3), over the last coherence_window updates once
    # coherence_min_updates exist: incoherent when the position residual about a
    # constant-acceleration fit through the measured positions exceeds
    # coherence_max_residual_m, or when fewer than coherence_min_continuity of the updates
    # measured it and the residual exceeds coherence_patchy_residual_m (or fewer than
    # coherence_min_points were measured). Town05 recorded frames (c882389): the followed
    # car's residual stayed under 0.08 m with every update measured; the building faces
    # and fence stretches that stopped the car ran 0.1-0.9 m, measured 5-60% of the time.
    coherence_window: int = 20
    coherence_min_updates: int = 10
    coherence_min_continuity: float = 0.6
    coherence_max_residual_m: float = 0.25
    coherence_patchy_residual_m: float = 0.1
    coherence_min_points: int = 4
    coherence_min_extent_m: float = 2.5  # judged only for tracks this long (StackConfig.vehicle_min_m is 3 m)
    # Movement history (sixth review, F6-04): a track has moved when its recent speed (net
    # displacement over motion_window_s) reaches moving_mps; ``still_s`` counts from then.
    moving_mps: float = 0.3
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
    # ... or up to merge_depth_m beyond it along the line of sight, when the LiDAR has the
    # object only as a face (its extent along the line of sight under merge_face_depth_m):
    # the LiDAR sees a car's rear, the radar's return comes off its body behind it. On the
    # Town05 braking lead (c882389) 177 of the 216 ticks a twin was kept, it lay outside the
    # rear face's footprint. A car queued behind the lead starts at least ~5 m beyond its
    # rear face and stays a track of its own.
    merge_depth_m: float = 3.5
    merge_face_depth_m: float = 1.5
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
    # Remembered shape (live crossing car, r8, b15b76f). A cluster's mean sits where its
    # points are densest, on the faces in view, not at the object's centre: a car that had
    # crossed and stopped 1.9 m beside the path read 0.3-0.6 m/s towards it as the ego drew
    # level and saw only its end face (the mean moved 1.2 m in 2 s, the car not at all), and
    # the ego braked from 3-4 m/s to a stop for it in three runs out of three. A vehicle-
    # sized track is measured instead at the centre of its remembered box placed against
    # the points in view: along each of the box's axes, the middle of the points' span
    # where the span is the box's size or the sensor lies within it (a face square on),
    # else the end nearer the sensor plus half the size (the far side is out of view). The
    # remembered size is the shape_min_views-th largest measured, so one or two merged
    # clusters do not set it; its axes are those of the last view that showed that length
    # (an L-shape fit, ``_rectangle``). Vehicle-sized: shape_min_length_m
    # (StackConfig.vehicle_min_m) to shape_max_length_m long, wider than a group of walkers
    # (StackConfig.group_max_width_m) and no wider than a lorry, not decided a pedestrian
    # or a bike: a 5.9 x 4.1 m roadside blob on Town05 is no box to measure against.
    shape_min_views: int = 3
    shape_min_length_m: float = 3.0
    shape_max_length_m: float = 6.0
    shape_min_width_m: float = 1.45
    shape_max_width_m: float = 2.6
    shape_full_view_m: float = 0.3  # a view within this of the remembered length shows the length: its axis is the box's
    # A cluster is measured against the box when it shows one of the box's sides whole (its
    # span along one axis within shape_face_m of the box's size) or at least shape_part
    # of it along either axis. A smaller fragment of the object (a ring or two of it, the
    # rest occluded) may be any part of it: it associates where it lies on the remembered
    # box and does not move the track. Placed against the box, a 0.4 m fragment of that
    # Town05 blob put its centre metres off; the blob's next whole cluster then fitted a new
    # track better, and the new track's jump read as a car at 4.9 m/s. (Taken for a
    # fragment, a 5.7 m stretch of a 6.7 m fence corner went to a clutter track instead,
    # which slid at 5 m/s towards the path: hence the share.)
    shape_face_m: float = 0.5
    shape_part: float = 0.5
    shape_refit_every: int = 5  # a full shape memory is refitted every this many updates (``_worth_fitting``) ...
    shape_fits_per_tick: int = 4  # ... and no more than this many clusters a tick are fitted, the nearest first
    shape_fragment_cost: float = 4.0  # added to a fragment's gating cost, so a whole cluster of the object is taken first
    # The leading edge (``ObjectTrack.edge_speed_mps``): a cluster's extent along
    # edge_directions directions (its support function) is kept per LiDAR update for
    # clusters edge_min_extent_m long or longer; the edge's advance along the velocity is
    # read over the last edge_window_s, at least edge_min_span_s and edge_min_updates
    # updates. The crossing car that had stopped 6 m left of the path (PC 1, Town05, r11
    # seed 2, 84fb071) read 0.5-0.7 m/s towards it as the car drew level, its centroid
    # sliding to the near end face, and the car braked from 3.9 m/s to a stop: its leading
    # edge read 0.00-0.08 m/s; a crossing car's, an oncoming car's and a cut-in's read
    # 0.85-1.0 of their speed in the recorded runs.
    edge_directions: int = 16
    edge_min_extent_m: float = 1.5  # a car seen end-on is a 1.8-2 m face
    edge_window_s: float = 1.0
    edge_min_span_s: float = 0.5
    edge_min_updates: int = 5

    def __post_init__(self):
        if self.confirm_hits > self.confirm_window or self.coast_s >= self.delete_s:
            raise ValueError("TrackerConfig: confirm_hits <= confirm_window and coast_s < delete_s")
        if self.shape_min_views < 1 or not 0 < self.shape_min_length_m <= self.shape_max_length_m:
            raise ValueError("TrackerConfig: shape_min_views >= 1 and 0 < shape_min_length_m <= shape_max_length_m")
        if self.assignment not in ("greedy", "optimal"):
            raise ValueError("TrackerConfig.assignment: greedy or optimal")
        if self.doppler_reject_policy not in ("drop", "position"):
            raise ValueError("TrackerConfig.doppler_reject_policy: drop or position")
        if self.edge_directions < 4 or self.edge_min_span_s > self.edge_window_s:
            raise ValueError("TrackerConfig: edge_directions >= 4 and edge_min_span_s <= edge_window_s")


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
    measured: deque = field(default_factory=lambda: deque(maxlen=64))  # (time, x, y, measured this update, footprint long side) after each update
    nis: deque = field(default_factory=lambda: deque(maxlen=10))  # normalised innovation squared of the last LiDAR position updates
    seen: dict = field(default_factory=dict)  # sensor -> last time it updated the track
    doppler_rejects: int = 0  # consecutive radar updates with the radial speed rejected
    doppler_sign: float = 0.0  # the sign of the last rejected radial-speed innovation
    doppler_first_s: float = -math.inf  # when the current run of rejections began ...
    doppler_first_y: float = 0.0  # ... its first innovation ...
    doppler_last_y: float = 0.0  # ... and its last
    doppler_position_sum: np.ndarray = field(default_factory=lambda: np.zeros(2))  # normalised range and bearing innovations of the run
    origin_time_s: float = 0.0  # when the track started
    max_excursion: float = 0.0  # the farthest it has been from where it was first seen (m)
    last_moving_s: float | None = None  # when it was last seen moving (``moving_mps``)
    twin_of: int | None = None  # the LiDAR track a radar-only track has lain inside ...
    twin_ticks: int = 0  # ... for this many updates running
    long_sides: list = field(default_factory=list)  # the longest measured long sides, longest first (shape memory)
    short_sides: list = field(default_factory=list)  # ... and short sides
    shape_yaw: float | None = None  # odometry yaw of the last view that showed the remembered length
    shape: tuple | None = None  # the remembered box (length, width, odometry yaw) it is measured against, once it has one
    visible_long: float | None = None  # the long side of the last cluster measured (the shape memory aside)
    radar_rejected_s: float | None = None  # when a radar return associated with it last had its radial speed rejected (observed nothing)
    edges: deque = field(default_factory=lambda: deque(maxlen=40))  # (time, the cluster's support along each edge direction, odometry frame) per LiDAR update


def _remembered_shape(track, config):
    """(length, width, odometry yaw) of a track's remembered box, or None (``TrackerConfig.shape_*``)."""
    c = config
    n = c.shape_min_views
    if track.shape_yaw is None or len(track.long_sides) < n or len(track.short_sides) < n:
        return None
    length = track.long_sides[n - 1]
    width = min(track.short_sides[n - 1], length)
    if not c.shape_min_length_m <= length <= c.shape_max_length_m or not c.shape_min_width_m <= width <= c.shape_max_width_m:
        return None
    best = CLASSES[int(np.argmax(track.classes))]
    if best in ("pedestrian", "bike") and float(track.classes.max()) >= 0.5:
        return None
    return length, width, track.shape_yaw


_RECT_COARSE = np.radians(np.arange(0.0, 90.0, 3.0))
_RECT_FINE = np.radians(np.arange(-3.0, 3.01, 0.5))


def _worth_fitting(track, long_side, config):
    """Whether a cluster with this principal-axis long side is fitted into the track's shape
    memory: a confirmed track's, vehicle-sized at all (from 0.6 of the shortest box to a
    metre over the longest); while the memory fills, then every ``shape_refit_every``-th
    update (the box's axes follow a turning car at that rate) or when the view is clearly
    longer than any before. Every view of every such cluster cost 8 ms a tick on a Town10HD street
    (29 fits: walls, hedges, parked cars)."""
    c = config
    if not track.confirmed or not 0.6 * c.shape_min_length_m <= long_side <= c.shape_max_length_m + 1.0:
        return False
    keep = c.shape_min_views + 2
    # (The principal axis's extent runs over an L's diagonal, up to ~0.5 m over its box.)
    return len(track.long_sides) < keep or track.hits % c.shape_refit_every == 0 or long_side > track.long_sides[0] + 0.6


_RECT_MAX_POINTS = 48


def _rectangle(points, d0=0.05):
    """(length, width, yaw) of the rectangle the plan points lie along, yaw on its longer
    side: an L-shape fit, the orientation whose rectangle has the points closest to its
    edges (Zhang et al., 2017: the sum over points of one over the distance to the nearest
    edge, floored at ``d0``). A cluster's principal axis runs between an L's two faces (the
    synthetic crossing car, 4.2 x 2.0 m, read 4.4-4.6 m long 9-40 degrees off), and the
    smallest rectangle is no better: an L's hull is a right triangle, as small along its
    hypotenuse as along its legs."""
    if len(points) > _RECT_MAX_POINTS:
        points = points[:: int(math.ceil(len(points) / _RECT_MAX_POINTS))]  # the outline, evenly thinned
    best = None
    for angles in (_RECT_COARSE, None):
        if angles is None:
            angles = best + _RECT_FINE
        c, s = np.cos(angles), np.sin(angles)
        along = points @ np.vstack((c, s))
        across = points @ np.vstack((-s, c))
        d1 = np.minimum(along.max(axis=0) - along, along - along.min(axis=0))
        d2 = np.minimum(across.max(axis=0) - across, across - across.min(axis=0))
        score = (1.0 / np.maximum(np.minimum(d1, d2), d0)).sum(axis=0)
        k = int(np.argmax(score))
        best = float(angles[k])
    a, b = float(np.ptp(along[:, k])), float(np.ptp(across[:, k]))
    return (a, b, best) if a >= b else (b, a, best + math.pi / 2)


def _remember_extent(track, extent_odo, config):
    """Fold one cluster's (length, width, odometry yaw) into the track's shape memory."""
    length, width, yaw = (float(v) for v in extent_odo)
    keep = config.shape_min_views + 2
    track.long_sides = sorted(track.long_sides + [length], reverse=True)[:keep]
    track.short_sides = sorted(track.short_sides + [width], reverse=True)[:keep]
    n = config.shape_min_views
    if len(track.long_sides) >= n and length >= track.long_sides[n - 1] - config.shape_full_view_m:
        track.shape_yaw = yaw


def _shaped_measurement(points_odo, shape, sensor_xy, face_m=0.5, part=0.5):
    """(centre, footprint, face) of a remembered box (length, width, yaw) placed against
    ``points_odo``, the cluster's points in plan (``TrackerConfig.shape_*``); ``face``:
    the points span one of the box's sides whole, within ``face_m``, or ``part`` of it
    along either axis (else a fragment)."""
    length, width, yaw = shape
    cy, sy = math.cos(yaw), math.sin(yaw)
    axes = np.array([[cy, -sy], [sy, cy]])  # columns: along the box, across it
    proj = points_odo @ axes
    lows, highs = proj.min(axis=0), proj.max(axis=0)
    at = sensor_xy @ axes
    middles, spans = [], []
    for lo, hi, a, size in zip(lows.tolist(), highs.tolist(), at.tolist(), (length, width), strict=True):
        if hi - lo < size and not lo <= a <= hi:
            middles.append(lo + 0.5 * size if a < lo else hi - 0.5 * size)
        else:
            middles.append(0.5 * (lo + hi))
        spans.append(hi - lo)
    centre = axes @ np.array(middles)
    face = spans[0] >= min(length - face_m, part * length) or spans[1] >= min(width - face_m, part * width)
    return centre, (max(length, spans[0]), max(width, spans[1]), float(yaw)), face


def _off_box(xy, centre, shape):
    """How far ``xy`` lies outside a box (length, width, yaw) at ``centre``, as a vector (0 inside)."""
    length, width, yaw = shape
    u = np.array([math.cos(yaw), math.sin(yaw)])
    v = np.array([-u[1], u[0]])
    rel = np.asarray(xy, dtype=float) - centre
    a, b = float(rel @ u), float(rel @ v)
    da = math.copysign(max(0.0, abs(a) - 0.5 * length), a)
    db = math.copysign(max(0.0, abs(b) - 0.5 * width), b)
    return da * u + db * v


def _recent_speed(trail, min_span_s):
    """Net displacement over the trail's span divided by the span, or None if shorter
    than ``min_span_s``: sustained motion, which a static object's centroid drifting as
    the car passes it (a pole, a short hedge) does not show (third live stack M04)."""
    if len(trail) < 2:
        return None
    (t0, x0, y0), (t1, x1, y1) = trail[0], trail[-1]
    span = t1 - t0
    return None if span < min_span_s else float(math.hypot(x1 - x0, y1 - y0) / span)


def _edge_directions(n):
    a = np.arange(n) * (2.0 * math.pi / n)
    return np.column_stack((np.cos(a), np.sin(a)))


def _record_edges(track, points_odo, time_s, config, directions):
    """Keep a cluster's support function (its farthest extent along each direction) for
    ``_edge_speed``, for clusters ``edge_min_extent_m`` long or longer."""
    if len(points_odo) < 3 or float(np.ptp(points_odo, axis=0).max()) < config.edge_min_extent_m:
        return
    track.edges.append((time_s, (points_odo @ directions.T).max(axis=0)))


def _support_at(support, angle):
    """The support function sampled at evenly spaced directions, interpolated at ``angle``."""
    n = len(support)
    k = (angle % (2.0 * math.pi)) / (2.0 * math.pi / n)
    i0 = int(math.floor(k)) % n
    f = k - math.floor(k)
    return (1.0 - f) * float(support[i0]) + f * float(support[(i0 + 1) % n])


def _edge_speed(track, now, config):
    """How fast the leading edge (the cluster's extent along the velocity) advanced over the
    last ``edge_window_s``, m/s, or None (no velocity, too little history, no cluster now)."""
    c = config
    speed = float(np.hypot(*track.x[2:4]))
    if speed < 0.1 or not track.edges or abs(track.edges[-1][0] - now) > 1e-9:
        return None
    window = [e for e in track.edges if now - e[0] <= c.edge_window_s + 1e-9]
    if len(window) < c.edge_min_updates or now - window[0][0] < c.edge_min_span_s - 1e-9:
        return None
    angle = math.atan2(float(track.x[3]), float(track.x[2]))
    (t0, h0), (t1, h1) = window[0], window[-1]
    return (_support_at(h1, angle) - _support_at(h0, angle)) / (t1 - t0)


def covariance_intersection(a, Pa, b, Pb, steps=21):
    """Fuse two estimates of one quantity whose correlation is unknown: ``P^-1 = w Pa^-1 +
    (1 - w) Pb^-1`` with ``w`` in [0, 1] minimising ``trace(P)`` (a grid of ``steps``).
    Never more confident than the evidence allows for any correlation (Julier and
    Uhlmann, 1997); with ``w`` at 0 or 1 it is the better estimate alone."""
    ia, ib = np.linalg.inv(Pa), np.linalg.inv(Pb)
    best = None
    for w in np.linspace(0.0, 1.0, steps):
        info = w * ia + (1.0 - w) * ib
        P = np.linalg.inv(info)
        if best is None or np.trace(P) < best[0]:
            best = (float(np.trace(P)), w, P)
    _, w, P = best
    x = P @ (w * ia @ np.asarray(a, dtype=float) + (1.0 - w) * ib @ np.asarray(b, dtype=float))
    return x, P


def _with_velocity_block(P, Pv):
    """``P`` (4 x 4) with its velocity block replaced by ``Pv``, the cross terms kept, halved
    as often as it takes for the result to stay positive definite (to zero at worst)."""
    out = np.array(P, dtype=float)
    out[2:, 2:] = Pv
    cross = out[:2, 2:].copy()
    for _ in range(8):
        out[:2, 2:], out[2:, :2] = cross, cross.T
        try:
            np.linalg.cholesky(0.5 * (out + out.T))
            return 0.5 * (out + out.T)
        except np.linalg.LinAlgError:
            cross = 0.5 * cross
    out[:2, 2:] = out[2:, :2] = 0.0
    return out


def _coherence(measured, config, footprint=None, classes=None):
    """``motion_coherent`` and ``coherence`` for ObjectTrack from a track's updates (see
    ``TrackerConfig.coherence_*``). Only for a track that may be vehicle-sized (a footprint
    ``coherence_min_extent_m`` long, or no footprint and a car or truck class at even
    odds): the gate is for vehicles, and the fit is a third of tracking's time otherwise."""
    c = config
    if footprint is not None:
        if max(footprint[0], footprint[1]) < c.coherence_min_extent_m:
            return dict(motion_coherent=None, coherence=None)
    elif classes is None or float(classes[CLASSES.index("car")] + classes[CLASSES.index("truck")]) < 0.5:
        return dict(motion_coherent=None, coherence=None)
    window = list(measured)[-c.coherence_window :]
    if len(window) < c.coherence_min_updates:
        return dict(motion_coherent=None, coherence=None)
    hits = [(t, x, y) for t, x, y, m, _ in window if m]
    continuity = len(hits) / len(window)
    lengths = [ext for *_, ext in window if ext is not None]
    spread = None if len(lengths) < c.coherence_min_points else round(float((max(lengths) - min(lengths)) / max(float(np.median(lengths)), 0.5)), 2)
    residual = None
    if len(hits) >= c.coherence_min_points:
        arr = np.asarray(hits, dtype=float)
        tt = arr[:, 0] - arr[0, 0]
        A = np.column_stack((np.ones_like(tt), tt, 0.5 * tt * tt)) if len(hits) >= 5 else np.column_stack((np.ones_like(tt), tt))
        # Least squares by the normal equations (3 x 3; over at most a second the columns
        # are well conditioned), not an SVD per track per tick.
        AtA = A.T @ A
        try:
            coef = np.linalg.solve(AtA, A.T @ arr[:, 1:])
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(A, arr[:, 1:], rcond=None)[0]
        residual = float(np.sqrt(np.sum((arr[:, 1:] - A @ coef) ** 2) / max(1, len(hits) - A.shape[1])))
    if residual is None:
        coherent = continuity >= c.coherence_min_continuity
    else:
        coherent = residual <= c.coherence_max_residual_m and (continuity >= c.coherence_min_continuity or residual <= c.coherence_patchy_residual_m)
    return dict(motion_coherent=bool(coherent), coherence=(round(continuity, 2), None if residual is None else round(residual, 3), spread))


def _normalised(probabilities):
    p = np.array([max(0.0, float(probabilities.get(k, 0.0))) for k in CLASSES]) if probabilities else np.ones(len(CLASSES))
    return p / p.sum() if p.sum() > 0 else np.ones(len(CLASSES)) / len(CLASSES)


def _assign(cost, gate, method="greedy"):
    """One-to-one assignment; cost above ``gate`` never assigns. "greedy": cheapest pair
    first. "optimal": the least total cost, a pair left unassigned costing ``gate``
    (``_assign_optimal``)."""
    if method == "optimal":
        return _assign_optimal(cost, gate)
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


def _hungarian(a):
    """Rows-to-columns assignment of the square matrix ``a`` (finite) with the least total
    cost (Kuhn-Munkres with potentials, O(n^3)). Returns the column of each row."""
    n = a.shape[0]
    u, v = np.zeros(n + 1), np.zeros(n + 1)
    p, way = np.zeros(n + 1, dtype=int), np.zeros(n + 1, dtype=int)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv, used = np.full(n + 1, np.inf), np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], np.inf, 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = a[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    col = np.zeros(n, dtype=int)
    for j in range(1, n + 1):
        if p[j]:
            col[p[j] - 1] = j - 1
    return col


def _assign_optimal(cost, gate):
    """Gated global nearest neighbour (ninth review, N9-02): the one-to-one assignment with
    the least total cost, each track or measurement left unassigned costing ``gate``, solved
    per connected group of gated pairs (a group is a few tracks and clusters around one
    object or two, so the exact solve stays cheap). Greedy took the cheapest pair first: a
    track's best measurement could be another's only one."""
    pairs = []
    if cost.size == 0:
        return pairs
    ok = cost <= gate
    rows, cols = cost.shape
    parent = list(range(rows + cols))

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for i, j in zip(*np.nonzero(ok)):
        parent[find(int(i))] = find(rows + int(j))
    groups = {}
    for i, j in zip(*np.nonzero(ok)):
        groups.setdefault(find(int(i)), (set(), set()))
        groups[find(int(i))][0].add(int(i))
        groups[find(int(i))][1].add(int(j))
    big = 1e6
    for r_set, c_set in groups.values():
        r, c = sorted(r_set), sorted(c_set)
        if len(r) == 1 or len(c) == 1:
            sub = cost[np.ix_(r, c)]
            k = int(np.argmin(sub))
            pairs.append((r[k // len(c)], c[k % len(c)]))
            continue
        nr, nc = len(r), len(c)
        a = np.full((nr + nc, nc + nr), big)
        sub = cost[np.ix_(r, c)]
        a[:nr, :nc] = np.where(sub <= gate, sub, big)
        a[np.arange(nr), nc + np.arange(nr)] = gate  # row i left unassigned
        a[nr + np.arange(nc), np.arange(nc)] = gate  # column j left unassigned
        a[nr:, nc:] = 0.0
        col = _hungarian(a)
        for k in range(nr):
            if col[k] < nc and sub[k, col[k]] <= gate:
                pairs.append((r[k], c[col[k]]))
    return pairs


class Tracker:
    def __init__(self, config=None):
        self.config = config or TrackerConfig()
        self.tracks = {}
        self._next_id = 1
        self._time = None
        self._edge_dirs = _edge_directions(self.config.edge_directions)
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
            has_plan = [m.plan_xy is not None and len(m.plan_xy) >= 3 for m in lidar]
            plans = {}

            def plan(j):
                # A cluster's points in the odometry frame, converted once and only if needed.
                if j not in plans:
                    plans[j] = pose.to_world(np.asarray(lidar[j].plan_xy, dtype=float)[:, :2])
                return plans[j]

            sensor = np.array([pose.x_m, pose.y_m])
            ids = list(self.tracks)
            cost = np.full((len(ids), len(lidar)), np.inf)
            shaped = {}  # (i, j) -> (centre, footprint): the cluster against track i's remembered box (centre None: a fragment)
            pos = np.array([self.tracks[tid].x[:2] for tid in ids]).reshape(-1, 2)
            near = np.hypot(pos[:, None, 0] - z_odo[None, :, 0], pos[:, None, 1] - z_odo[None, :, 1]) <= c.lidar_gate_m
            for i, j in zip(*np.nonzero(near)):
                t, m = self.tracks[ids[i]], lidar[j]
                if self._class_veto(t, m.class_probabilities):
                    continue
                if "lidar" not in t.seen and m.extent is not None and max(m.extent[0], m.extent[1]) < c.merge_min_extent_m and float(np.hypot(*t.x[2:])) > c.vru_max_speed_mps:
                    continue
                z, penalty = z_odo[j], 0.0
                if t.shape is not None and has_plan[j]:
                    centre, fp, face = _shaped_measurement(plan(j), t.shape, sensor, c.shape_face_m, c.shape_part)
                    if face:
                        shaped[(i, j)], z = (centre, fp), centre
                    else:
                        # A fragment: gated by how far its mean lies off the remembered box,
                        # behind any whole cluster of the object (one to one, cheapest first).
                        shaped[(i, j)], z = (None, t.shape), t.x[:2] + _off_box(z_odo[j], t.x[:2], t.shape)
                        penalty = c.shape_fragment_cost
                S = t.P[:2, :2] + np.eye(2) * m.sigma_m**2
                y = z - t.x[:2]
                cost[i, j] = float(y @ np.linalg.solve(S, y)) + penalty
            assigned = _assign(cost, CHI2_2D, c.assignment)
            # The shape fits this tick, at most shape_fits_per_tick, nearest the car first.
            wanted = []
            for i, j in assigned:
                t, m = self.tracks[ids[i]], lidar[j]
                if has_plan[j] and m.extent is not None and _worth_fitting(t, max(m.extent[0], m.extent[1]), c):
                    wanted.append((float(np.hypot(*(z_odo[j] - sensor))), i, j))
            fit_now = {(i, j) for _, i, j in sorted(wanted)[: c.shape_fits_per_tick]}
            for i, j in assigned:
                t, m = self.tracks[ids[i]], lidar[j]
                H = np.zeros((2, 4))
                H[0, 0] = H[1, 1] = 1.0
                z, footprint = z_odo[j], None
                if m.extent is not None:
                    footprint = (m.extent[0], m.extent[1], pose.yaw_rad + m.extent[2])
                    t.visible_long = max(m.extent[0], m.extent[1])
                    if (i, j) in fit_now:
                        _remember_extent(t, _rectangle(plan(j)), c)
                fragment = (i, j) in shaped and shaped[(i, j)][0] is None
                if fragment:
                    footprint = t.shape  # the box stands; the fragment says nothing of where on it
                elif has_plan[j] and "lidar" in t.seen and (t.shape is not None or t.shape_yaw is not None):
                    # The box the cluster is measured against: the one now remembered (None:
                    # none, or no longer a vehicle's). When it changes (first known, resized,
                    # turned, given up) the reference point changes with it: the track, its
                    # trail and origin move by the difference this cluster shows, a change of
                    # reference, not motion (as at a radar-held track's first LiDAR fix, below).
                    shape = _remembered_shape(t, c)
                    measured = shaped[(i, j)][0] if (i, j) in shaped else z_odo[j]
                    target, fp = z_odo[j], footprint
                    if shape is not None and shape == t.shape and (i, j) in shaped:
                        target, fp = shaped[(i, j)]  # the same box: measured against it already
                    elif shape is not None:
                        centre, fp_new, face = _shaped_measurement(plan(j), shape, sensor, c.shape_face_m, c.shape_part)
                        if face:
                            target, fp = centre, fp_new
                        else:
                            shape = t.shape  # shown only in part against the new box: wait for a whole view
                            target, fp = (shaped[(i, j)] if (i, j) in shaped else (z_odo[j], footprint))
                    offset = target - measured
                    if float(offset @ offset) > 1e-12:
                        self._shift_history(t, offset)
                        t.x[:2] = t.x[:2] + offset
                    t.shape, z, footprint = shape, target, fp
                elif (i, j) in shaped:
                    z, footprint = shaped[(i, j)]
                if "lidar" not in t.seen:
                    # The first LiDAR fix of a radar-held track: from now on the cluster's
                    # centre is its position, not the radar's point (a front corner 1.8 m
                    # out). A change of reference, not motion: the position is set to the
                    # cluster, and its trail and origin move with it. Blended in over many
                    # ticks, the jump read as 1.8 m/s and a parked car stayed "uncertain" for
                    # a second (review of 3e9083a, M6).
                    self._shift_history(t, z - t.x[:2])
                    t.x[:2] = z
                    t.P[:2, :2] = np.eye(2) * m.sigma_m**2
                    t.P[:2, 2:] = t.P[2:, :2] = 0.0
                elif not fragment:
                    t.nis.append(float(cost[i, j]))
                    self._kalman(t, z - t.x[:2], H, np.eye(2) * m.sigma_m**2)
                self._absorb(t, m.class_probabilities, "lidar", time_s)
                if footprint is not None:
                    t.footprint = footprint
                hit.add(ids[i])
                if has_plan[j] and not fragment and m.extent is not None and max(m.extent[0], m.extent[1]) >= c.edge_min_extent_m:
                    _record_edges(t, plan(j), time_s, c, self._edge_dirs)  # a fragment's extent is not the object's
            taken = {j for _, j in assigned}
            for j, m in enumerate(lidar):
                if j not in taken:
                    tid = self._spawn(z_odo[j], m.sigma_m, m.class_probabilities, time_s, "lidar")
                    t = self.tracks[tid]
                    if m.extent is not None:
                        t.footprint = (m.extent[0], m.extent[1], pose.yaw_rad + m.extent[2])
                        t.visible_long = max(m.extent[0], m.extent[1])
                        if has_plan[j] and _worth_fitting(t, t.visible_long, c):
                            _remember_extent(t, _rectangle(plan(j)), c)
                        if has_plan[j] and t.visible_long >= c.edge_min_extent_m:
                            _record_edges(t, plan(j), time_s, c, self._edge_dirs)
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
            assigned = _assign(cost, CHI2_3D, c.assignment)
            rejected, manoeuvres = [], []
            disowned = []  # (LiDAR-placed track, the radial speed of its rejected return) this tick
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
                    if t.seen.get("lidar") == time_s and t.footprint is not None and max(t.footprint[0], t.footprint[1]) >= c.merge_min_extent_m:
                        disowned.append((t, radar[j].radial_speed_mps))
                    # ... and observes nothing (eighth review, F8-01): no hit, no support, no
                    # freshness. A track whose returns are all rejected coasts and is deleted
                    # as if unseen; counted, those returns kept a confirmed radar-only track
                    # alive, supported and a lead candidate indefinitely.
                    if c.doppler_reject_policy == "position" and t.seen.get("lidar") != time_s:
                        # The tenth review's alternative (2A): the range and bearing kept, the
                        # radial speed dropped, still observing nothing (no hit, no freshness).
                        self._kalman(t, y[:2], H[:2], Rt[:2, :2])
                    t.radar_rejected_s = time_s
                    continue
                self._absorb(t, None, "radar", time_s)
                hit.add(ids[i])
            # Unassigned radar objects start tracks where the radar adds something: beyond
            # the LiDAR's object range, or moving. Static near returns no track claims are clutter.
            assigned_cols = {j for _, j in assigned}
            spawned = skipped = same_object = 0
            sensor = np.array([pose.x_m, pose.y_m])
            for j, m in enumerate(radar):
                if j not in assigned_cols:
                    static = abs(m.radial_speed_mps) < c.radar_spawn_min_speed_mps
                    if static and (m.range_m < c.radar_spawn_min_range_m or abs(float(m.xy[1])) > c.radar_static_spawn_half_width_m):
                        skipped += 1
                        continue
                    obj_odo = pose.to_world(m.xy[None])[0]
                    if any(abs(m.radial_speed_mps - rs) <= c.doppler_agree_mps and self._on_object(obj_odo, t, sensor) for t, rs in disowned):
                        # Another return off an object whose own return this tick had its
                        # radial speed rejected, agreeing with that one: the object's motion
                        # changed (its returns agree with each other, not with the track), not
                        # a second object (ninth review, N9-01). Started as a track, it took
                        # the next returns and the LiDAR cluster from the braking lead, which
                        # then coasted at 2.6 m/s: its stop was seen 0.3 s later (PC 2,
                        # r10_lead_pair, a73f5e0). Left alone, the returns go to the track and
                        # its rejection run is accepted as a manoeuvre.
                        same_object += 1
                        continue
                    tid = self._spawn(obj_odo, 1.0, None, time_s, "radar")
                    hit.add(tid)
                    spawned += 1
            self.radar_stats = dict(objects=len(radar), associated=len(assigned), doppler_rejected=rejected[:5], doppler_manoeuvres=manoeuvres[:5], spawned=spawned, clutter_skipped=skipped, same_object=same_object)
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
        for tid, track in self.tracks.items():
            track.trail.append((time_s, float(track.x[0]), float(track.x[1])))
            if track.origin is not None:
                track.max_excursion = max(track.max_excursion, float(np.hypot(*(track.x[:2] - track.origin))))
            recent = _recent_speed(track.trail, c.motion_min_span_s)
            if recent is not None and recent >= c.moving_mps:
                track.last_moving_s = time_s
            # The cluster's own long side (a remembered box's is constant): ``_coherence``'s spread.
            track.measured.append((time_s, float(track.x[0]), float(track.x[1]), tid in hit, track.visible_long if tid in hit and track.footprint is not None else None))
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
                    if (along > c.merge_margin_m or across > c.merge_margin_m) and not self._behind_face(r, cand, sensor):
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
            # Two estimates of one velocity, combined conservatively: the two tracks may
            # share information (one object's motion through one filter model, a common
            # ego-motion), so not as independent ones, which made the merged velocity
            # look more certain than either (fifth review, F5-04), but by covariance
            # intersection, consistent whatever their correlation. The LiDAR track's
            # position and its position-velocity cross terms are kept, scaled down only as
            # far as the new velocity block needs to stay a covariance.
            v, Pv = covariance_intersection(r.x[2:], r.P[2:, 2:], lt.x[2:], lt.P[2:, 2:])
            keep.x, keep.P = np.concatenate((lt.x[:2], v)), _with_velocity_block(lt.P, Pv)
            keep.footprint, keep.visible_long = lt.footprint, lt.visible_long
            # ... and the LiDAR track's reference point with it: its shape memory.
            keep.long_sides, keep.short_sides, keep.shape_yaw, keep.shape = list(lt.long_sides), list(lt.short_sides), lt.shape_yaw, lt.shape
            # The longer history, moved to where the LiDAR has the object: the radar track's
            # trail and origin as they were, a return off its front 1.8 m away, made a parked
            # car "uncertain" for a second and inflated its travelled distance for good
            # (review of 3e9083a, M6); the fresh LiDAR twin's alone would be too short to show
            # it standing.
            keep.trail, keep.origin, keep.origin_time_s = deque(src.trail, maxlen=src.trail.maxlen), None if src.origin is None else src.origin.copy(), src.origin_time_s
            keep.measured = deque(src.measured, maxlen=src.measured.maxlen)
            keep.max_excursion = max(r.max_excursion, lt.max_excursion)
            keep.last_moving_s = max((v for v in (r.last_moving_s, lt.last_moving_s) if v is not None), default=None)
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

    def _on_object(self, xy, lidar_track, sensor):
        """Whether the odometry point ``xy`` lies on ``lidar_track``'s object as the twin
        merge reads it: within its footprint grown by ``merge_margin_m``, or behind its face
        (``_behind_face``)."""
        c = self.config
        length, width, yaw = lidar_track.footprint
        rel = np.asarray(xy, dtype=float) - lidar_track.x[:2]
        along = abs(rel[0] * math.cos(yaw) + rel[1] * math.sin(yaw)) - length / 2
        across = abs(-rel[0] * math.sin(yaw) + rel[1] * math.cos(yaw)) - width / 2
        return (along <= c.merge_margin_m and across <= c.merge_margin_m) or self._behind_face_xy(xy, lidar_track, sensor)

    def _behind_face(self, radar_track, lidar_track, sensor):
        return self._behind_face_xy(radar_track.x[:2], lidar_track, sensor)

    def _behind_face_xy(self, xy, lidar_track, sensor):
        """Whether the odometry point ``xy`` lies behind ``lidar_track``'s face as the sensor sees it:
        the LiDAR track's extent along the line of sight is a face (under
        ``merge_face_depth_m``), and the point is no more than ``merge_depth_m`` beyond it
        along that line and within its half-extent across it plus ``merge_margin_m``."""
        c = self.config
        if sensor is None:
            return False
        los = lidar_track.x[:2] - sensor
        dist = float(np.hypot(*los))
        if dist < 1e-6:
            return False
        u = los / dist
        length, width, yaw = lidar_track.footprint
        rel = yaw - math.atan2(u[1], u[0])
        deep = 0.5 * (length * abs(math.cos(rel)) + width * abs(math.sin(rel)))  # half-extent along the line of sight
        wide = 0.5 * (length * abs(math.sin(rel)) + width * abs(math.cos(rel)))  # ... and across it
        if 2.0 * deep > c.merge_face_depth_m:
            return False
        off = np.asarray(xy, dtype=float) - lidar_track.x[:2]
        beyond = float(off @ u) - deep
        aside = abs(float(off @ np.array([-u[1], u[0]])))
        return -c.merge_margin_m <= beyond <= c.merge_depth_m and aside <= wide + c.merge_margin_m

    @staticmethod
    def _shift_history(track, offset):
        """Move a track's trail and origin by ``offset`` (odometry): its reference point
        changed, the object did not move."""
        dx, dy = float(offset[0]), float(offset[1])
        track.trail = deque(((ts, x + dx, y + dy) for ts, x, y in track.trail), maxlen=track.trail.maxlen)
        track.measured = deque(((ts, x + dx, y + dy, m, ext) for ts, x, y, m, ext in track.measured), maxlen=track.measured.maxlen)
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
                    **_coherence(t.measured, c, t.footprint, t.classes),
                    lidar_nis=round(float(np.mean(t.nis)), 2) if t.nis else None,
                    max_excursion_m=round(t.max_excursion, 3) if t.origin is not None else None,
                    still_s=round((self._time or t.time_s) - (t.last_moving_s if t.last_moving_s is not None else t.origin_time_s), 3),
                    edge_speed_mps=None if (edge := _edge_speed(t, self._time if self._time is not None else t.time_s, c)) is None else round(edge, 3),
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
