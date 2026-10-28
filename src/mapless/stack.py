"""``MaplessStack``: the G2/G3 stages around the G1 driver (plan v2.1, sections 9–11). Framework.

Per tick, in order:

1. **Drive** — ``MaplessDriver.step`` (health, ego motion, terrain grid, kerb corridor,
   obstacle clearance, junction turns, planner, controller). It is the G1 car and keeps
   its own safety rules; the stack only reads its inputs and may lower its speed.
2. **Segment and paint** — every ``segment_every`` ticks each admitted camera is
   segmented (``road_segmentation``) and that same frame's sweep painted (``painting``);
   the road model accumulates the result and ages it between those ticks. A sweep is
   never painted with another frame's segmentation (external review of the G3 framework:
   painting every tick projected the current LiDAR into the previous image).
3. **Detect and cluster** — obstacle points are clustered; detector boxes claim their
   foreground clusters (``detections``).
4. **Track** — clusters, radar objects and camera classes update the tracker
   (``tracking``); radar enters only as admitted (ablation A/B/C: camera+LiDAR; +front
   radar; +rear radars).
5. **Predict and follow** — predictions over the horizon (``prediction``); the nearest
   confirmed track in the lane sets a following speed (``following``), and a predicted
   conflict with the car's own oriented footprint along its planned path within
   ``conflict_horizon_s`` caps the speed to one it can stop from short of where that road
   user may be (zero once that is close). Both reach the driver as
   ``speed_cap_mps`` for the next tick, and the record says so: ``cap_applied_mps`` is
   what this tick's command was planned under, ``speed_cap_next_mps`` what was just
   computed. The driver's own obstacle stop remains the backstop.
6. **Record** — a ``DecisionRecord`` with stage timings, coverage and the local-frame
   generation.

When the driver rebuilds its local frame after an odometry outage (``frame_rebuilt`` in
its diagnostics), everything the stack placed in the old frame goes with it: tracks,
semantic evidence, following history and the cached segmentation (same review: they
survived the rebuild and were read against the new frame). While the driver is stopping
to rebuild, the stack adds nothing.

The stack never sees the simulator: everything comes in through the ``SensorFrame``.
Status: the plumbing runs on synthetic frames (tests); the learned segmenter and the
detector are swapped in by configuration once trained/benchmarked (G2), and the radar
admission is an experiment (G3).
"""

import math
import time
from dataclasses import dataclass, field

import numpy as np

from mapless.contracts import DecisionRecord
from mapless.detections import (
    DetectionsConfig,
    Detector,
    NoDetector,
    cluster_and_split,
    join,
    plan_extent,
)
from mapless.driver import DriveMode, MaplessDriver
from mapless.following import LEAD_STATUSES, FollowingConfig, FollowingController
from mapless.frame import SensorFrame, SensorState
from mapless.obstacles import project_onto_polyline
from mapless.painting import PaintingConfig, SegmentedFrame, paint
from mapless.prediction import (
    PredictionConfig,
    decided_class,
    first_footprint_conflict,
    first_occupied_arc,
    is_moving,
    motion_state,
    path_headings,
    predict_all,
)
from mapless.radar import RadarConfig, measure
from mapless.road_model import RoadModelBuilder, RoadModelConfig
from mapless.road_segmentation import RoadSegmenter, load_segmenter
from mapless.terrain import CellLabel
from mapless.tracking import Tracker, TrackerConfig


@dataclass(frozen=True)
class Admission:
    """Which sensors the stack consumes (section 4, sensor admission; section 10, ablation)."""

    cameras: bool = True
    lidar: bool = True
    front_radar: bool = False
    rear_radars: bool = False

    @classmethod
    def preset(cls, name):
        return {
            "A": cls(cameras=True, lidar=True, front_radar=False, rear_radars=False),
            "B": cls(cameras=True, lidar=True, front_radar=True, rear_radars=False),
            "C": cls(cameras=True, lidar=True, front_radar=True, rear_radars=True),
        }[name.upper()]


@dataclass(frozen=True)
class StackConfig:
    admission: Admission = field(default_factory=Admission)
    segmenter: str | None = None  # None/"constant" or a checkpoint path
    detector: str | None = None  # None or "yolo"
    painting: PaintingConfig = field(default_factory=PaintingConfig)
    detections: DetectionsConfig = field(default_factory=DetectionsConfig)
    road_model: RoadModelConfig = field(default_factory=RoadModelConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    radar: RadarConfig = field(default_factory=RadarConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    following: FollowingConfig = field(default_factory=FollowingConfig)
    conflict_horizon_s: float = 3.0
    # A predicted conflict caps the speed to one the car can stop from, at
    # conflict_decel_mps2, conflict_margin_m short of the first place the path is
    # predicted occupied by that road user (zero when that place is that close already).
    conflict_decel_mps2: float = 2.0
    conflict_margin_m: float = 2.0
    # Yield to a vulnerable road user crossing the path ahead (M04): while one is predicted
    # on the path, at walking pace at most, and stopping yield_margin_m short of where it
    # is predicted. "Vulnerable" is the pedestrian/bike class, or, without a detector, a
    # LiDAR footprint no larger than vulnerable_max_m.
    yield_mps: float = 0.8
    yield_margin_m: float = 2.0
    vulnerable_max_m: float = 2.0
    # A pedestrian standing at the roadside, its near edge more than roadside_clear_m beyond
    # the car's tube: passed at roadside_mps at most, not stopped for. Its walking-pace
    # envelope reached the path, so a walker waiting on the pavement held the car at a
    # standstill for good (third review, R5). Once it moves it is a crosser again.
    roadside_mps: float = 3.0
    roadside_clear_m: float = 0.5
    # A pedestrian-sized moving track is a crossing candidate only within reach of the
    # road (the kerb, keep_right_m, plus vru_reach_m) and either in the path band or
    # heading for it at approach_min_mps or more. Long walls and fences beside the road
    # slide past as the car moves (the visible stretch's centre travels with it): 1.5-3.5
    # m/s "walkers" 7-15 m to the right in the fourth live M04 run (26 Oct).
    vru_reach_m: float = 2.0
    approach_min_mps: float = 0.3
    vehicle_min_m: float = 3.0  # a footprint this long (or a decided car/truck) is a vehicle
    # The lead's body beyond its rear face: other tracks there are pieces of it (a car's
    # LiDAR returns split into several clusters), not road users that can conflict.
    lead_body_m: float = 5.0  # for a lead as wide as a car's rear (or a vehicle)
    lead_body_min_width_m: float = 1.2
    lead_body_margin_m: float = 0.8  # beyond its half-width (at least that of a car, 0.9 m); A10: a piece at 1.5 m
    # ... but only tracks that have not been road users of their own: one seen outside the
    # lead's body for lead_piece_independent_s, or classed a pedestrian or bike, is kept
    # (external review of the full review of 73ae759: a walker beside the lead was dropped).
    lead_piece_independent_s: float = 0.5
    # A lead seen this many times is known, not new: once it is no longer moving along or
    # across the path it is a stopped obstacle, not an "uncertain" small object (B9: the
    # lead's rear face, 1.7 m wide, read "uncertain" while it came to rest and its own
    # walking-pace envelope stopped the car 10 m short, twice).
    lead_known_hits: int = 20
    # Vehicles drive on the road: a vehicle-sized track beyond the kerb plus vru_reach_m
    # whose heading towards the path runs into a kerb, a railing or another object within
    # reach_check_m cannot get there (B8/A9/B9: walls 13-23 m right, their visible
    # stretch sliding at 4-6 m/s, read as cars heading for the path, out of the radar's view).
    reach_check_m: float = 30.0
    # ... and "runs into" means a wall: cells with persistent tall evidence
    # (reach_wall_min_evidence, ~1 s of steady hits; a car passing through leaves ~10) that
    # no live track covers, across a car's width (reach_wall_half_width_m each side of the
    # run), for reach_block_ticks running. A kerb or pavement is not a wall (a car leaves a
    # driveway over it), nor a parked car or another crossing car's trail (review of
    # 3e9083a, M2; external review: an uncertain obstruction must not erase the vehicle).
    reach_wall_min_evidence: float = 20.0
    reach_wall_half_width_m: float = 1.0
    reach_block_ticks: int = 3
    # A static radar-only track near the car is clutter only where the LiDAR hit something
    # (ground or not) within lidar_view_age_s; where it saw nothing the radar is all there is.
    lidar_view_age_s: float = 0.3
    # The yield is for road users crossing the path: moving across it at approach_min_mps
    # or more (a car driving ahead in the lane, its rear face under 2 m across, is a lead),
    # no faster than a pedestrian or cyclist (vru_max_speed_mps), and qualifying for
    # yield_persist_s first (single-tick fragments braked the car hard; live cut-in B3).
    vru_max_speed_mps: float = 6.0  # a cyclist (review of 3e9083a, H3: 2.5 left a runner at 2.6 m/s)
    # A slow cluster with no class up to group_max_m across is people on foot (a group),
    # yielded to and never a "vehicle" a wall can make unreachable (review of 3e9083a, M7).
    group_max_m: float = 4.0
    offroad_min_size_m: float = 0.5  # beyond the kerb's reach a mover must be at least this long (not a one-cell cluster)
    group_min_width_m: float = 0.6
    # ... and no wider than a car's width: a car cutting in slowly, seen as 3.9 x 1.6 m,
    # was yielded to as a group at 0.8 m/s and the car slowed to 0.25 m/s (live cut-in
    # with the car staying slower, 6f2aec3). It is the conflict check's.
    group_max_width_m: float = 1.4
    group_max_mps: float = 2.0
    # A vehicle-sized track beyond the kerb's reach stands on or beside road: within
    # offroad_road_window_m of it the grid has seen at least offroad_road_min_cells points of
    # ROAD, or too little of the area at all to say (offroad_known_min_cells). Building
    # faces 14-22 m off a Town05 street, 3.5-9 m long and 0.3-2.4 m across, "approaching"
    # at 4-7 m/s, had no road within 3 m and stopped the car (live, 6370b98).
    offroad_road_window_m: float = 3.0
    offroad_road_min_cells: int = 3
    offroad_known_min_cells: int = 20
    # A vehicle beside the path slower than this, not coming towards it, is parked or
    # queueing: its drift is the car's changing view of it (see ``parked_beside``).
    parked_max_mps: float = 1.0
    # ... and whose near side is this far outside the car's swept body. The track sits on
    # the visible points' centroid, near the car's near face, so centre less half-width
    # puts the near side up to a few decimetres too close: shoulder-parked cars about 1 m
    # clear of the body read 0.0-0.1 m (live, 7eaca3d). The driver's LiDAR obstacle stop
    # (its tube 0.35 m wider than the body) is the check for a corner that pokes in.
    parked_clearance_m: float = -0.1  # (negative: the tracked near side may be 0.1 m inside the body's edge; see parked_beside)
    yield_persist_s: float = 0.3
    road_model_every: int = 4  # build the road model every this many ticks (5 Hz at 0.05 s; it is a slow product)
    segment_every: int = 2


@dataclass(frozen=True)
class StackStep:
    command: object
    mode: DriveMode
    reason: str
    diagnostics: dict
    tracks: tuple
    predictions: tuple
    road_model: object
    decision: DecisionRecord


class MaplessStack:
    def __init__(self, suite, route, driver_config, config=None, dt_s=0.05, segmenter=None, detector=None):
        self.config = c = config or StackConfig()
        self.driver = MaplessDriver(suite, route, driver_config, dt_s)
        self.suite = suite
        self.segmenter = segmenter if isinstance(segmenter, RoadSegmenter) else load_segmenter(c.segmenter)
        if detector is None:
            detector = NoDetector() if c.detector in (None, "none") else self._make_detector(c.detector)
        if not isinstance(detector, Detector):
            raise TypeError("detector must be a Detector")
        self.detector = detector
        self.road_model = RoadModelBuilder(self.driver.grid, c.road_model)
        self.tracker = Tracker(c.tracker)
        self.following = FollowingController(c.following)
        self.lidar = suite.lidars[0].name if suite.lidars else None
        self.radar_roles = {}
        for spec in suite.radars:
            self.radar_roles[spec.name] = "front" if abs(spec.mount.yaw_deg) < 90 else "rear"
        self._tick = 0
        self._segmentations = None  # the last SegmentedFrame (its frame id says which sweep it may paint)
        self._model = None
        self.frame_generation = 0  # bumped on every local-frame rebuild
        self._cap_reason = ""  # what produced the cap the driver plans the next command under
        self._lead = None  # the lead behind the last cap, for the log
        self._cause = None  # the track behind a conflict or yield cap, for the log
        self._clearing = None  # a vehicle not stopped for while clearing a left turn's opposing lane (S5)
        self._parked = []  # vehicles beside the path whose only motion is the car's passing view of them
        self._oncoming = None  # the stack's verdict on a left turn's opposing lane, for the log
        self._structures = []  # clusters too big for a road user, this tick, for the log
        self._unreachable = []  # vehicle candidates a kerb or object separates from the path
        self._yield_since = {}  # track id -> first time it qualified for a yield
        self._outside_lead_s = {}  # track id -> time seen outside the lead's body (a road user of its own)
        self._blocked_ticks = {}  # track id -> ticks running its run to the path met a wall
        self.records = []

    @staticmethod
    def _make_detector(name):
        if name == "yolo":
            from mapless.detections import YoloDetector  # noqa: PLC0415

            return YoloDetector()
        raise ValueError(f"unknown detector {name!r}")

    def admitted_radars(self, frame):
        a = self.config.admission
        out = []
        for radar in frame.radars:
            role = self.radar_roles.get(radar.name, "front")
            if (role == "front" and a.front_radar) or (role == "rear" and a.rear_radars):
                out.append(radar)
        return out

    def step(self, frame):
        if not isinstance(frame, SensorFrame):
            raise TypeError("step(SensorFrame)")
        c = self.config
        timing = {}
        # The cap the driver plans this tick's command under is last tick's result.
        cap_applied = self.driver.speed_cap_mps
        begun = time.perf_counter()
        step = self.driver.step(frame)
        timing["drive"] = (time.perf_counter() - begun) * 1000
        motion, sweep = self.driver.last_motion, self.driver.last_classified
        self._tick += 1
        t = frame.sim_time_s
        tracks, predictions = (), ()
        cap, cap_reason = None, ""
        self._lead = self._cause = self._clearing = self._oncoming = None
        self._parked = []
        self._structures = []
        self._unreachable = []
        coverage = {s.name: s.state.value for s in frame.status}
        # -- local frame rebuild -----------------------------------------------------------
        rebuilt = bool(step.diagnostics.get("frame_rebuilt"))
        if rebuilt:
            self.reset_spatial()
            self.frame_generation += 1
        # While the driver stops to rebuild, the odometry frame lags the car: nothing is placed in it.
        frame_usable = rebuilt or not step.diagnostics.get("rebuilding")
        perceived = motion is not None and motion.route is not None and motion.odometry_trusted and frame_usable and sweep is not None and c.admission.lidar
        if not perceived and not rebuilt and self.driver.speed_cap_mps is not None:
            # No perception this tick (a missed or stale sweep, untrusted odometry): keep
            # the restriction already in force rather than clear it; only a fresh
            # evaluation may lift it. The driver's own stale-LiDAR fault is the backstop
            # for a longer outage (external review of 405b056: one missed tick cleared a
            # stop cap for the tick after).
            cap = self.driver.speed_cap_mps
            cap_reason = (self._cap_reason or "speed cap").removesuffix(" (held: no fresh LiDAR)") + " (held: no fresh LiDAR)"
        if perceived:
            pose = motion.odometry
            # -- segment and paint ---------------------------------------------------------
            begun = time.perf_counter()
            cameras = [im for im in frame.cameras if frame.status_of(im.name).state is SensorState.OK] if c.admission.cameras else []
            painted = None
            if cameras and self._tick % c.segment_every == 0:
                # Segment and paint the same frame: the sweep and the images were taken together.
                self._segmentations = SegmentedFrame(frame.frame_id, t, [self.segmenter.segment(im) for im in cameras])
                painted = paint(sweep.xyz, cameras, self._segmentations, c.painting, frame_id=frame.frame_id)
            if painted is not None:
                self.road_model.add(sweep.xyz, painted, pose, t)
            timing["paint"] = (time.perf_counter() - begun) * 1000
            # -- detect, cluster, join -------------------------------------------------------
            begun = time.perf_counter()
            # Scenery chained into one cluster is structure, not a road user: never tracked
            # (it stays OBSTACLE points for the driver's LiDAR stop and the corridor).
            clusters, structures = cluster_and_split(sweep, c.detections)
            self._structures = sorted(
                (tuple(round(float(v), 1) for v in (*sweep.xyz[m, :2].mean(axis=0), *plan_extent(sweep.xyz[m, :2]))) + (len(m),) for m in structures),
                key=lambda item: -item[-1],
            )
            boxes = {im.name: self.detector.detect(im) for im in cameras} if cameras else {}
            measurements, camera_only = join(sweep, clusters, cameras, boxes, c.detections, c.painting)
            timing["detect"] = (time.perf_counter() - begun) * 1000
            # -- radar -----------------------------------------------------------------------
            begun = time.perf_counter()
            radar_measurements = []
            for radar in self.admitted_radars(frame):
                if frame.status_of(radar.name).state is SensorState.OK:
                    radar_measurements.extend(measure(radar, motion.speed_mps, motion.yaw_rate_rps, c.radar))
            # -- track and predict ----------------------------------------------------------
            tracks = tuple(self.tracker.update(t, pose, measurements, radar_measurements, camera_only, lidar_view=self._lidar_saw))
            live = self.tracker.live(tracks)
            # Only tracks the tracker vouches for are predicted onto the path: a fresh
            # tentative track carries the filter's initial velocity variance (5 m/s), which
            # the prediction inflates to a 30 m box at 3 s, and every new parked-car cluster
            # tapped the brakes for a few ticks (fresh-eyes review of 58adaf3). Tentative
            # clusters are the driver's LiDAR obstacle stop's job, the documented backstop.
            # Only LiDAR-supported tracks are predicted. A radar-only track has a 2 degree
            # bearing (1 m at 30 m) and no measured extent: at a standstill with the front
            # radar admitted (fourth live stack run, 25 Oct), returns from the pavement and
            # the buildings spawned hundreds of such tracks whose jitter read as motion and
            # whose default 2 x 2 m boxes were predicted across the lane. Radar-only tracks
            # still lead the car (following) and are what the radar is for: a car beyond
            # the LiDAR's reach, which the LiDAR takes over long before the horizon.
            predictions = tuple(predict_all(predictable_tracks(live), c.prediction))
            timing["track"] = (time.perf_counter() - begun) * 1000
            # -- follow and conflicts --------------------------------------------------------
            begun = time.perf_counter()
            cap, cap_reason = self._speed_cap(motion, pose, live, predictions, t)
            gate = self.driver.last_gate
            self.driver.oncoming_override = None if gate is None else oncoming_verdict(live, gate, self.driver.config.junction, c)
            self._oncoming = None if self.driver.oncoming_override is None else self.driver.oncoming_override["note"]
            timing["plan"] = (time.perf_counter() - begun) * 1000
            # -- road model ------------------------------------------------------------------
            if self._tick % c.road_model_every == 0:
                begun = time.perf_counter()
                self._model = self.road_model.build(pose, t)
                timing["road_model"] = (time.perf_counter() - begun) * 1000
        self.driver.speed_cap_mps = cap  # a request for the next tick's plan
        proposed = "WAIT" if step.mode is DriveMode.STOPPING else ("STOP" if step.mode is DriveMode.FAULT else "FOLLOW")
        executed, veto = proposed, ""
        # ``executed`` describes this tick's command, so only the cap that shaped it (last
        # tick's) can have vetoed it; the cap just computed is recorded as the next request.
        if cap_applied is not None and cap_applied <= 0.0 and proposed == "FOLLOW":
            executed, veto = "STOP", self._cap_reason or "speed cap"
        self._cap_reason = cap_reason
        constraints = tuple(k for k in ("obstacle_m", "stop_reason", "turn") if step.diagnostics.get(k))
        record = DecisionRecord(
            t,
            proposed,
            executed,
            veto,
            constraints,
            coverage,
            {k: round(v, 2) for k, v in timing.items()},
            cap_applied_mps=cap_applied,
            speed_cap_next_mps=cap,
            frame_generation=self.frame_generation,
        )
        self.records.append(record)
        near = near_tracks(live if perceived else (), motion.odometry if perceived else None)
        diagnostics = dict(
            step.diagnostics,
            tracks=len(tracks),
            # (x, y, length, width, points) of the largest clusters too big for a road user.
            structures=self._structures[:3],
            radar_stats=self.tracker.radar_stats if perceived else {},
            unreachable=self._unreachable[:5],
            speed_cap_mps=cap_applied,
            speed_cap_next_mps=cap,
            speed_cap_reason=cap_reason,  # logged every tick: a hold at a standstill never reaches the veto
            lead=self._lead,
            # The live tracks around the car, for the evaluator's actor matching (F1).
            tracks_near=near[0],
            tracks_near_omitted=near[1],
            cap_track=self._cause,
            clearing_for=self._clearing,
            parked_beside=self._parked[:5],
            oncoming=self._oncoming,
            frame_generation=self.frame_generation,
            stack_timing_ms=record.timing_ms,
        )
        return StackStep(step.command, step.mode, step.reason, diagnostics, tracks, predictions, self._model, record)

    def _speed_cap(self, motion, pose, tracks, predictions, t):
        """Following speed for the lead in the lane, and zero on a predicted conflict."""
        c = self.config
        plan = self.driver.last_plan
        # The path the driver planned along this tick, in this tick's frame: in a turn, the
        # turn's (review of 3e9083a, H5: the kerb corridor's last good line, from an older
        # frame, made a car crossing 9 m ahead mid-turn the "lead" and dropped it).
        corridor = self.driver.last_corridor
        if corridor is None or not corridor.ok or plan is None:
            # No path to place a lead or a conflict on this tick: the checks did not run, so
            # the restriction already in force stands (review of 3e9083a, M8: returning no
            # cap cleared a zero cap). The driver brakes or faults on its own account.
            self._lead = self._cause = None
            held = self.driver.speed_cap_mps
            if held is None:
                return None, ""
            return held, (self._cap_reason or "speed cap").removesuffix(" (held: no path this tick)") + " (held: no path this tick)"
        path_body = corridor.centre_body
        # The corridor line starts behind the rear axle: arc lengths are measured from the car.
        s_car = float(project_onto_polyline(np.zeros((1, 2)), path_body)[0][0])
        cap, reason = None, ""
        self._lead = self._cause = None
        # Only tracks the tracker vouches for lead the car (a 3-point tentative cluster must not).
        leads = [tr for tr in self.tracker.live(tracks) if tr.status in LEAD_STATUSES]
        tube_half = self.driver.config.body.half_width_m + self.driver.config.obstacles.margin_m
        lead = self.following.lead_ahead(leads, pose, path_body, s_from_m=s_car, tube_half_m=tube_half)
        lead_id = None
        if lead is not None:
            s, track = lead
            lead_id = track.track_id
            state = motion_state(track, c.prediction)
            vehicle = is_vehicle(track, c)
            # Motion along and across the path at the lead, not the car's own axis (review
            # of 405b056), and the gap to the lead's rear, not its centre.
            tangent = _path_heading_at(path_body, s + s_car)  # ``s`` is from the car; the line starts behind it
            v_body = pose.to_body(np.array([track.position + track.velocity]))[0] - pose.to_body(track.position[None])[0]
            along = float(v_body[0] * math.cos(tangent) + v_body[1] * math.sin(tangent))
            across = float(-v_body[0] * math.sin(tangent) + v_body[1] * math.cos(tangent))
            gap = s - self.driver.config.body.front_m - _rear_extent(track, pose.yaw_rad + tangent)
            at = pose.to_body(track.position[None])[0]
            self._lead = dict(
                track_id=track.track_id, gap_m=round(float(gap), 2), speed_mps=round(along, 2), across_mps=round(across, 2),
                # Where the stack had it (vehicle frame): the evaluator matches the lead to the
                # scenario's actor by position before it scores speeds (third review, F1).
                x_m=round(float(at[0]), 2), y_m=round(float(at[1]), 2),
                state=state, support=list(track.support), status=track.status.name, doppler_rejects=track.doppler_rejects,
                footprint=None if track.footprint is None else [round(float(track.footprint[0]), 2), round(float(track.footprint[1]), 2)],
            )
            role = lead_role(track, along, across, c, wide=_across_width(track, pose.yaw_rad + tangent) >= c.lead_body_min_width_m)
            self._lead["role"] = role
            if role == "crossing":
                # In the lane but moving across it (a crossing walker reads ~0 m/s along the
                # path, a diagonal one more): not a lead to follow or to hand to the
                # obstacle stop; the conflict and yield checks keep it (first live M04).
                self.following.reset()
                lead_id = None
                self._lead["crossing"] = True
            elif role == "follow":
                if self.driver.speed_cap_mps is not None:
                    # Taken on under a restriction (a conflict cap for the car cutting in):
                    # following starts from it rather than from the car's speed (third
                    # review, R1: a 1.4 m/s conflict cap became a 6 m/s following target the
                    # tick the car crossing in reached the tube).
                    self.following.start_from(min(motion.speed_mps, self.driver.speed_cap_mps))
                cap = self.following.target_speed(self.driver.config.planner.cruise_mps, motion.speed_mps, gap, along, self.driver.dt_s)
                reason = "lead at a standstill" if cap is not None and cap <= 0.0 else "following the lead"
                if not vehicle:
                    lead_id = None  # a walker or cyclist ahead is followed and still checked for crossing
            elif role == "obstacle_stop":
                # A lead at a standstill is an obstacle in the lane: the driver's own LiDAR
                # obstacle stop (stop point 2.5 m short, then a held brake released only on
                # observed clearance) is the measured behaviour for it. A second stop rule
                # here stopped the car 4 m earlier, short of the driver's stop point, so the
                # driver never reached its hold (first live stack stop run, 26 Oct).
                self.following.reset()
                reason = "stationary lead: the obstacle stop's"
            else:
                # A small object in the lane whose motion is not yet known: neither followed
                # nor delegated; the conflict check gives it an uncertain envelope.
                self.following.reset()
                lead_id = None
        else:
            self.following.reset()
        if self._lead is not None:
            # What protects the car from the lead (third review, F3): the conflict check,
            # unless another rule covers it, following (a vehicle moving along the path:
            # its worst-case cap) or the driver's LiDAR obstacle stop (a lead not moving
            # along, across or towards the car: ``lead_role``).
            self._lead["checked_by"] = "conflict check" if lead_id is None else ("following" if role == "follow" else "obstacle stop")
        if predictions and plan is not None and plan.trajectory is not None:
            # Time the path at the speed the car would drive without any cap (the planned
            # speeds already carry this tick's cap, which fed back on itself: a cap of zero
            # made every conflict look far away and the cap flickered tick to tick).
            samples = plan.trajectory.samples
            xy = np.array([[smp.x_m, smp.y_m] for smp in samples])
            speed = max(motion.speed_mps, min(self.driver.config.planner.cruise_mps, max(plan.curve_speed_mps, 0.5)))
            seg = np.hypot(*np.diff(xy, axis=0).T)
            arcs = np.concatenate(([0.0], np.cumsum(seg)))
            # Time zero is the car itself, not the path's first sample (which lies behind it).
            s_now = float(project_onto_polyline(np.array([[pose.x_m, pose.y_m]]), xy)[0][0])
            times = (arcs - s_now) / speed
            # The car's whole oriented body along the path, not a disc about the rear axle.
            others = conflict_candidates(tracks, predictions, lead_id, c.prediction)
            admitted = {}
            keep_right = self.driver.corridor.keep_right_m
            if keep_right is not None:
                others = plausible_crossers(tracks, others, xy, keep_right, c, ego_speed_mps=motion.speed_mps, grid=self.driver.grid, why=admitted)
            if keep_right is not None:
                others, self._unreachable = road_reachable(tracks, others, xy, self.driver.grid, keep_right, c, history=self._blocked_ticks)
            others, self._parked = parked_beside(tracks, others, xy, self.driver.config.body, c)
            if lead is not None and role != "crossing":
                # Also for a followed walker or cyclist, which stays in the crossing checks
                # itself (review of 3e9083a, M9: pieces 2.7 m beyond a narrow followed lead
                # were yielded to, because the filter ran only for a vehicle lead).
                others = outside_lead_body(tracks, others, lead[1], lead[0], pose, path_body, s_car, c, followed=role == "follow", history=self._outside_lead_s, dt_s=self.driver.dt_s)
            body = self.driver.config.body
            clearing = self.driver.clearing_opposing_lane
            for p in others:
                # Every road user with a timed conflict within the horizon, not just the
                # earliest (review of 3e9083a, H2: a car crossing first hid one pulling out
                # beside the car, cap 2.83 m/s where it should be 0).
                hit = first_footprint_conflict([p], xy, times, body)
                if hit is None or hit[0] > c.conflict_horizon_s:
                    continue
                # Stop short of where the road user may be, not at once: a speed from which
                # the car stops conflict_margin_m before the first place the path is
                # predicted occupied by it. A conflict up close is still a zero cap; one
                # 3-tick fragment 9 m ahead no longer brakes the car from 2 m/s to rest
                # (A10, 26 Oct). The cap is recomputed every tick as the car closes.
                arc = first_occupied_arc([p], xy, body, from_arc_m=s_now)
                track = next((t for t in tracks if t.track_id == p.track_id), None)
                if clearing and arc is not None and arc <= s_now + 0.5 and track is not None and is_vehicle(track, c) and motion_state(track, c.prediction) == "moving":
                    # In the opposing lane of a left turn, a vehicle predicted where the car
                    # already is: stopping there keeps the car in its way; it clears the
                    # lane instead (full review of 73ae759, S5). The driver's obstacle stop
                    # still stops for anything in the path ahead.
                    self._clearing = _track_note(tracks, p.track_id, pose)
                    continue
                conflict_cap = conflict_speed_cap([p], xy, s_now, body, c)
                why = "predicted conflict on the path"
                passing = roadside_cap(track, xy, s_now, body, tube_half, c)
                if passing is not None and passing > conflict_cap:
                    conflict_cap, why = passing, "passing a road user standing at the roadside"
                if cap is None or conflict_cap < cap:
                    cap, reason = conflict_cap, why
                    self._cause = dict(_track_note(tracks, p.track_id, pose), admitted=admitted.get(p.track_id, "not filtered"))
            yield_cap, yield_id = crossing_yield(tracks, others, xy, s_now, self.driver.config.body, self.driver.config.planner.decel_mps2, c, self._yield_since, t)
            if yield_cap is not None and (cap is None or yield_cap < cap):
                cap, reason = yield_cap, "yielding to a road user crossing the path"
                self._cause = dict(_track_note(tracks, yield_id, pose), admitted=admitted.get(yield_id, "not filtered"))
        return cap, reason

    def _lidar_saw(self, xy_odo):
        """For odometry points: did the LiDAR hit anything (ground included) in its cell or a
        neighbouring one (0.25 m) within ``lidar_view_age_s``? A ray that reached the ground
        there passed through where an object would stand (external review: within 0.5 m
        was too loose a proxy)."""
        grid = self.driver.grid
        pts = np.asarray(xy_odo, dtype=float).reshape(-1, 2)
        if grid.time_s is None or not len(pts):
            return np.zeros(len(pts), dtype=bool)
        offsets = np.array([[0.0, 0.0], [0.25, 0.0], [-0.25, 0.0], [0.0, 0.25], [0.0, -0.25]])
        seen = grid.seen_at((pts[:, None, :] + offsets[None]).reshape(-1, 2)).reshape(len(pts), -1)
        return np.any(seen >= grid.time_s - self.config.lidar_view_age_s, axis=1)

    def reset_spatial(self):
        """Drop everything kept in the odometry frame: tracks, semantic evidence, the
        following history and the cached segmentation (the frame they were placed in is gone)."""
        self.tracker.reset()
        self.road_model.reset()
        self.following.reset()
        self._yield_since = {}
        self._outside_lead_s = {}
        self._blocked_ticks = {}
        self._segmentations, self._model = None, None
        self.driver.speed_cap_mps = None
        self.driver.oncoming_override = None

    def reset(self):
        self.reset_spatial()
        self.frame_generation = 0


def near_tracks(tracks, pose, ahead_m=60.0, behind_m=10.0, side_m=12.0, limit=40):
    """The tracks around the car, nearest first, as [id, x, y, vx, vy, length, width,
    status, support] (vehicle frame, m and m/s; the extent None when not measured; status C,
    T or K for confirmed, tentative or coasting; support the sensors' initials, C camera,
    L LiDAR, R radar): the log's compact view
    for matching tracks to the scenario's actors on the evaluator's side (third review, F1).
    Past ``limit`` the farthest are left out, and the log says how many
    (``tracks_near_omitted``, fourth review: a crowded scene must not lose an actor
    silently)."""
    if pose is None or not tracks:
        return [], 0
    rows = []
    for track in tracks:
        at = pose.to_body(track.position[None])[0]
        if not (-behind_m <= at[0] <= ahead_m and abs(at[1]) <= side_m):
            continue
        v = pose.to_body(np.array([track.position + track.velocity]))[0] - at
        fp = track.footprint
        rows.append(
            (
                float(np.hypot(*at)),
                [int(track.track_id), round(float(at[0]), 2), round(float(at[1]), 2), round(float(v[0]), 2), round(float(v[1]), 2)]
                + ([None, None] if fp is None else [round(float(fp[0]), 2), round(float(fp[1]), 2)])
                + [track.status.name[0], "".join(sorted(s[0].upper() for s in track.support))],
            )
        )
    rows.sort(key=lambda r: r[0])
    return [r[1] for r in rows[:limit]], max(0, len(rows) - limit)


def predictable_tracks(tracks):
    """The tracks predicted onto the path: vouched for by the tracker and seen by the LiDAR."""
    return [tr for tr in tracks if tr.status in LEAD_STATUSES and "lidar" in tr.support]


def conflict_candidates(tracks, predictions, lead_id, config):
    """The predictions the conflict check tests: road users that may move into the path.

    - The lead is the following rule's (moving) or the driver's obstacle stop's (stopped):
      a stationary car in the lane is always a "conflict", and timing the path at cruise
      speed put it 6 m in front of the bumper (first live stack stop run, 26 Oct).
    - A static vehicle or object beside the lane is the static world, which the driver's
      LiDAR corridor check clears with its own margins; its predicted box clipped the swept
      path past a car parked at the kerb and held the car at a standstill (first live
      stack cut-in run, 26 Oct).
    - Moving and uncertain tracks are tested: too little evidence of motion is not
      evidence of standing still (external review of 405b056); an uncertain track is
      predicted where it stands with a walking-pace envelope.
    - A decided pedestrian is tested even when standing: a walker can step out.
    """
    by_id = {t.track_id: t for t in tracks}
    out = []
    for p in predictions:
        track = by_id.get(p.track_id)
        if p.track_id == lead_id or track is None:
            continue
        # Static things are the driver's corridor check's; moving and uncertain ones (their
        # envelope is bounded at walking pace) may enter the path (review of 405b056).
        if motion_state(track, config) != "static" or decided_class(track) == "pedestrian":
            out.append(p)
    return out


def oncoming_verdict(tracks, gate, junction, config):
    """The stack's view of the opposing lane a left turn is about to cross (S5), for the
    driver's gate: dict(clear, excused, note).

    Tracks in the lane (``gate.band``, 0.5 m either side) from the crossing on: one moving
    towards it that can reach it (``gate.zone_x``) within ``gate.required_s``, at its own
    speed plus 1 m/s, blocks; so does one whose motion is not known yet within the reach of
    ``junction.oncoming_mps``. Static ones (a parked car, one waiting to turn), ones driving
    away and ones too far or too slow are excused: the driver's LiDAR check does not count
    their cells, which on its own it must (it cannot tell a parked car from a stopped one
    about to go). The radar's tracks reach beyond the LiDAR's 40 m.
    """
    lo, hi = gate.band
    x_lo, x_hi = gate.zone_x
    reach = junction.oncoming_mps * gate.required_s
    excused, blocking = [], []
    for track in tracks:
        pos = gate.frame.to_body(track.position[None])[0]
        if not lo - 0.5 <= pos[1] <= hi + 0.5 or pos[0] < x_lo - 3.0:
            continue
        vel = gate.frame.to_body(np.array([track.position + track.velocity]))[0] - pos
        if track.footprint is not None:
            length, width, yaw = track.footprint
        else:
            length, width, yaw = 4.5, 2.0, gate.frame.yaw_rad
        half = 0.5 * (length * abs(math.cos(yaw - gate.frame.yaw_rad)) + width * abs(math.sin(yaw - gate.frame.yaw_rad)))
        gap = float(pos[0]) - half - x_hi
        toward = -float(vel[0])
        state = motion_state(track, config.prediction)
        footprint = ((float(track.position[0]), float(track.position[1])), length, width, yaw)
        if state == "static" or (state == "moving" and toward < -config.prediction.stationary_mps):
            excused.append(footprint)
        elif state == "uncertain":
            if gap <= reach:
                blocking.append((track.track_id, gap, None))
            else:
                excused.append(footprint)
        elif gap <= 0.0 or (gap / max(toward + 1.0, 1e-3)) < gate.required_s:
            blocking.append((track.track_id, gap, toward))
        else:
            excused.append(footprint)
    if blocking:
        tid, gap, toward = min(blocking, key=lambda b: b[1])
        speed = "" if toward is None else f" at {toward:.1f} m/s"
        note = f"track {tid} in the oncoming lane {max(0.0, gap):.0f} m from the crossing{speed}"
        return dict(clear=False, excused=excused, note=note)
    return dict(clear=True, excused=excused, note=f"oncoming lane: {len(excused)} tracks excused")


_AROUND = np.array([[0.0, 0.0], [0.25, 0.0], [-0.25, 0.0], [0.0, 0.25], [0.0, -0.25]])  # a cell and its neighbours


def plausible_crossers(tracks, candidates, path_xy, keep_right_m, config, ego_speed_mps=0.0, grid=None, why=None):
    """Drop candidates that are not clearly vehicles and cannot be crossing. Vehicles (a
    decided car/truck class, or a footprint ``vehicle_min_m`` or longer) are kept: the timed
    conflict check handles them. Of the rest, kept are:

    - those in the path band;
    - within the kerb plus ``vru_reach_m`` of the path: a pedestrian, one whose motion is
      not known yet (its envelope is bounded), or one heading for the path at
      ``approach_min_mps`` or more;
    - beyond that reach: one heading for the path at ``approach_min_mps`` or more, unless
      it shows the sliding-wall signature (moving along the path, the car's way, faster than
      towards it and at half the car's speed or more) or stands on structure (``grid``:
      persistent tall evidence, ``reach_wall_min_evidence``, where it is). Walls and fences
      beside the road slide past as the car moves (the visible stretch's centre travels with
      it: 1.5-3.5 m/s "walkers" 7-15 m to the right in the fourth live M04 run, some of
      them heading for the path); a cyclist out of a side road comes across open road
      (review of 3e9083a, H3: the reach alone dropped one at 5 m/s until it was 3.75 m from
      the path, and the car hit it).

    A clipped wall segment can be any length under ``vehicle_min_m`` (2.19 m in the fourth
    live M04 run).

    ``why`` (a dict, optional) gets each kept track's admitting rule, for the log (fourth
    review: which gate let a track become a predicted conflict)."""
    by_id = {t.track_id: t for t in tracks}
    why = {} if why is None else why
    path = np.asarray(path_xy, dtype=float)
    if len(path) < 2:
        return list(candidates)
    heads = path_headings(path)
    arcs = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))))
    out = []
    for p in candidates:
        track = by_id.get(p.track_id)
        if track is None:
            out.append(p)
            continue
        s, lateral, _ = project_onto_polyline(track.position[None], path)
        lat = float(lateral[0])
        within = abs(lat) <= keep_right_m + config.vru_reach_m
        seg = int(np.clip(np.searchsorted(arcs, float(s[0]), side="right") - 1, 0, len(path) - 1))
        tangent = np.array([math.cos(heads[seg]), math.sin(heads[seg])])
        normal = np.array([-tangent[1], tangent[0]])  # +left
        towards = -math.copysign(1.0, lat) * float(track.velocity @ normal)  # speed towards the path line
        along = float(track.velocity @ tangent)
        if is_vehicle(track, config):
            # A "vehicle" only by length, as thin as a wall and standing on persistent tall
            # evidence off the road, is a wall's visible stretch sliding (an 8.8 x 0.32 m
            # one 20 m out at 5.5 m/s stopped the car on the live stack turn, 7eaca3d).
            thin = track.footprint is not None and min(track.footprint[0], track.footprint[1]) < config.offroad_min_size_m and decided_class(track) is None
            if not within and thin and grid is not None and float(grid.tall_evidence(track.position[None] + _AROUND).max()) >= config.reach_wall_min_evidence:
                continue
            if not within and grid is not None and not road_near(grid, track.position, config):
                continue  # beyond reach and nowhere near road: a building's face, not a car
            if not within and towards < config.approach_min_mps:
                # Beyond reach and not heading for the path: it can enter it only by turning
                # first, and is kept the tick it does. Fence stretches 5-6 m right, 4-8 m long,
                # "oncoming" at 5 m/s along the road, reached the path only through their
                # envelopes and capped the car to 0 (Town05, 6370b98, recorded frames).
                continue
            why[p.track_id] = "vehicle " + ("within reach" if within else "beyond reach, road near")
            out.append(p)
            continue
        if within and (motion_state(track, config.prediction) == "uncertain" or decided_class(track) == "pedestrian"):
            # Within reach and motion unknown (its envelope is bounded), or a pedestrian,
            # who can step out from a standstill (review of 3e9083a, M5): keep it.
            why[p.track_id] = "within reach, " + ("pedestrian" if decided_class(track) == "pedestrian" else "motion uncertain")
            out.append(p)
            continue
        if abs(lat) <= config.following.lane_half_width_m + 0.5:
            why[p.track_id] = "in the lane band"
            out.append(p)
            continue
        if towards < config.approach_min_mps:
            continue
        if not within:
            if track.footprint is None or max(track.footprint[0], track.footprint[1]) < config.offroad_min_size_m or track.speed_mps > config.vru_max_speed_mps:
                # Off the road a road user on foot or on a bike is more than a point and no
                # faster than a cyclist: one-cell clusters at 6-10 m/s, 13-23 m out at a
                # junction, capped the car to 0 a dozen times on the live stack turn (7eaca3d).
                continue
            if track.recent_speed_mps is None or track.recent_speed_mps < config.prediction.min_recent_speed_mps:
                # Off the road, the motion must be sustained first: a 0.55 m fragment 18 m
                # out, a few ticks old, "heading for the path" at 4.7 m/s capped the car to
                # 0 on the first live stack turn (ff1910b). A cyclist at 5 m/s has 2.5 m of
                # history half a second on, still metres before the old kerb + 2 m reach.
                continue
            if ego_speed_mps > 0.5 and along >= 0.5 * ego_speed_mps and along > towards:
                continue  # the sliding-wall signature
            if (abs(lat) - (config.following.lane_half_width_m + 0.5)) / towards > config.conflict_horizon_s + 1.0:
                continue  # cannot reach the lane within the horizon: a 0.3 m "runner" 23 m out (live, ff1910b)
            if grid is not None:
                if float(grid.tall_evidence(track.position[None] + _AROUND).max()) >= config.reach_wall_min_evidence:
                    continue  # standing on a wall or a fence: its visible stretch, not a road user
        why[p.track_id] = ("beyond reach" if not within else "within reach") + f", approaching at {towards:.1f} m/s"
        out.append(p)
    return out


def road_near(grid, position, config):
    """Whether the grid shows road within ``offroad_road_window_m`` of ``position``
    (odometry), or has seen too little of the area to say: a vehicle beyond the kerb's reach
    must stand on or beside a road (``plausible_crossers``). A car's own cells read
    OCCUPIED, so it is the road around it that counts."""
    half = config.offroad_road_window_m
    steps = np.arange(-half, half + 1e-9, 0.5)
    window = np.asarray(position, dtype=float).reshape(1, 2) + np.array([[dx, dy] for dx in steps for dy in steps])
    labels = grid.labels(window)
    if int((labels != CellLabel.UNKNOWN).sum()) < config.offroad_known_min_cells:
        return True
    return int((labels == CellLabel.ROAD).sum()) >= config.offroad_road_min_cells


def parked_beside(tracks, candidates, path_xy, body, config):
    """``(kept, dropped_ids)``: drop vehicles beside the path whose only motion is the car's
    passing view of them.

    As the car drives past a parked car, the part of it the LiDAR sees changes and its
    centroid drifts the car's way at a fraction of the car's speed (0.3-0.46 m/s on the
    first live stack turn, on ff1910b). That reads as "uncertain" motion, whose envelope
    reached the lane from 2.4 m beside it: the car stopped and went every 2 s past a row
    of parked cars. Dropped here: a vehicle whose near side is more than
    ``parked_clearance_m`` outside the car's swept body, slower than ``parked_max_mps``, not moving towards the path at
    ``approach_min_mps`` and not against the car's direction. A car pulling out comes
    across, and is kept the moment it does.
    """
    path = np.asarray(path_xy, dtype=float)
    if len(path) < 2:
        return list(candidates), []
    by_id = {t.track_id: t for t in tracks}
    heads = path_headings(path)
    arcs = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))))
    kept, dropped = [], []
    for p in candidates:
        track = by_id.get(p.track_id)
        if track is None or not is_vehicle(track, config) or is_group(track, config) or track.footprint is None:
            kept.append(p)
            continue
        s, lateral, _ = project_onto_polyline(track.position[None], path)
        seg = int(np.clip(np.searchsorted(arcs, float(s[0]), side="right") - 1, 0, len(path) - 1))
        tangent = np.array([math.cos(heads[seg]), math.sin(heads[seg])])
        normal = np.array([-tangent[1], tangent[0]])
        lat = float(lateral[0])
        towards = -math.copysign(1.0, lat) * float(track.velocity @ normal)
        along = float(track.velocity @ tangent)
        # Parked along the road: the near side is its half-width in from its centre. The
        # footprint's yaw from a partial view is not to be trusted here: an L of rear and
        # side read 14 degrees off, a 2.9 m "across" extent, and a parked car 2.4 m beside
        # the path seemed to reach into the car's own body (live, ff1910b). A corner that
        # truly pokes out is the driver's LiDAR obstacle stop's (its tube 0.35 m wider).
        near_side = abs(lat) - 0.5 * min(track.footprint[0], track.footprint[1])
        if (
            near_side > body.half_width_m + config.parked_clearance_m
            and float(np.hypot(*track.velocity)) < config.parked_max_mps
            and towards < config.approach_min_mps
            and along > -config.prediction.stationary_mps
        ):
            dropped.append(p.track_id)
            continue
        kept.append(p)
    return kept, dropped


def road_reachable(tracks, candidates, path_xy, grid, keep_right_m, config, history=None):
    """``(kept, dropped_ids)``: drop vehicle candidates beyond the kerb plus ``vru_reach_m``
    whose straight run along their velocity towards the path meets a wall before it
    reaches the lane, within ``reach_check_m``.

    A wall is persistent tall evidence (``reach_wall_min_evidence``) that no live track's
    footprint covers, across ``reach_wall_half_width_m`` either side of the run: a railing,
    a building. A kerb or pavement is not one (a car leaves a driveway over it), nor a
    parked car (a track) or another crossing car's fading trail (review of 3e9083a, M2).
    With ``history`` (track id -> ticks running, kept by the caller) a candidate is
    dropped only after ``reach_block_ticks`` running. Cells within the track's own
    footprint are not checked. Walkers and cyclists cross kerbs and are left alone; so is
    a vehicle whose run reaches the lane without meeting a wall, or that never reaches it.
    """
    path = np.asarray(path_xy, dtype=float)
    if len(path) < 2:
        return list(candidates), []
    by_id = {t.track_id: t for t in tracks}
    kept, dropped = [], []
    band = config.following.lane_half_width_m + 0.5
    blocked_now = set()
    for p in candidates:
        track = by_id.get(p.track_id)
        if track is None or not is_vehicle(track, config) or is_group(track, config):
            kept.append(p)
            continue
        _, lat, _ = project_onto_polyline(track.position[None], path)
        speed = float(np.hypot(*track.velocity))
        if abs(float(lat[0])) <= keep_right_m + config.vru_reach_m or speed < config.prediction.stationary_mps:
            kept.append(p)
            continue
        own = 0.5 * max(track.footprint[0], track.footprint[1]) + 0.5 if track.footprint is not None else 1.5
        steps = np.arange(own, config.reach_check_m, 0.25)
        heading = track.velocity / speed
        pts = track.position[None] + steps[:, None] * heading[None]
        _, lat_pts, _ = project_onto_polyline(pts, path)
        enters = np.flatnonzero(np.abs(lat_pts) <= band)
        if enters.size == 0:
            kept.append(p)  # never reaches the lane on its line: the timed check decides
            continue
        if _wall_across(pts[: enters[0]], heading, grid, tracks, track.track_id, config):
            blocked_now.add(p.track_id)
            if history is None:
                dropped.append(p.track_id)
                continue
            history[p.track_id] = history.get(p.track_id, 0) + 1
            if history[p.track_id] >= config.reach_block_ticks:
                dropped.append(p.track_id)
                continue
        kept.append(p)
    if history is not None:
        for tid in list(history):
            if tid not in blocked_now:
                del history[tid]
    return kept, dropped


def _wall_across(run, heading, grid, tracks, own_id, config):
    """Whether the run (odometry points) meets a wall: a sample whose cells have persistent
    tall evidence uncovered by any other track, at the run and across it either side."""
    if not len(run):
        return False
    normal = np.array([-heading[1], heading[0]])
    offsets = np.linspace(-config.reach_wall_half_width_m, config.reach_wall_half_width_m, 5)
    pts = (run[:, None, :] + offsets[None, :, None] * normal[None, None, :]).reshape(-1, 2)
    tall = grid.tall_evidence(pts).reshape(len(run), len(offsets)) >= config.reach_wall_min_evidence
    if not tall.any():
        return False
    covered = np.zeros(len(pts), dtype=bool)
    for other in tracks:
        if other.track_id == own_id or other.footprint is None:
            continue
        length, width, yaw = other.footprint
        rel = pts - other.position[None]
        along = np.abs(rel[:, 0] * math.cos(yaw) + rel[:, 1] * math.sin(yaw))
        across = np.abs(-rel[:, 0] * math.sin(yaw) + rel[:, 1] * math.cos(yaw))
        covered |= (along <= length / 2 + 0.3) & (across <= width / 2 + 0.3)
    wall = tall & ~covered.reshape(len(run), len(offsets))
    return bool(np.any(wall.all(axis=1)))


def outside_lead_body(tracks, candidates, lead, lead_ahead_m, pose, path_body, s_car, config, followed=False, history=None, dt_s=0.05):
    """Drop candidates that are pieces of the lead: inside its body, from its rear face
    (less 0.3 m) to ``lead_body_m`` beyond it when the lead is a vehicle or as wide as a
    car's rear, else to its own far edge plus 0.5 m, and within its half-width (at least a
    car's) plus ``lead_body_margin_m`` of its line, and never road users of their own.

    Braking-lead runs on 6b1f705 (26 Oct): 0.3 m pieces of the lead's roof and wheels,
    1.5 to 4 m beyond its rear face, jittered sideways at 0.5 to 2.6 m/s and stopped the
    car for "predicted conflicts" on every approach, once with the lead itself duplicated
    as a fresh LiDAR-only track (a 2.4 m/s initial velocity). Nothing can cross the path
    there without first leaving the lead's body, where the check still sees it.

    A road user of its own is a track classed pedestrian or bike, or one seen outside the
    body for ``lead_piece_independent_s`` (``history``: track id -> seconds, kept by the
    caller and updated here by ``dt_s``). A walker beside the lead came from somewhere;
    the lead's own pieces are born inside it (external review of the full review of
    73ae759: an independent pedestrian was dropped). The lead itself is never dropped.
    """
    line = np.asarray(path_body, dtype=float)
    ahead_l = float(lead_ahead_m)
    _, lat_l, _ = project_onto_polyline(pose.to_body(lead.position[None]), line)
    lat_l = float(lat_l[0])
    tangent = pose.yaw_rad + _path_heading_at(line, ahead_l + s_car)
    rear_ext = _rear_extent(lead, tangent)
    across = _across_width(lead, tangent)
    # A lead followed along the lane is a body (a car, a cyclist) however little of it the
    # LiDAR has measured (B10 on Town05: a piece 2.7 m beyond a followed lead's rear face,
    # its footprint narrower than a car's rear, stopped the car).
    body_like = followed or is_vehicle(lead, config) or across >= config.lead_body_min_width_m
    rear = ahead_l - rear_ext
    far = rear + (config.lead_body_m if body_like else 2.0 * rear_ext + 0.5)
    half = max(0.5 * across, 0.9) + config.lead_body_margin_m
    by_id = {t.track_id: t for t in tracks}
    out = []
    for p in candidates:
        track = by_id.get(p.track_id)
        if track is None or p.track_id == lead.track_id:
            out.append(p)
            continue
        s, lat, _ = project_onto_polyline(pose.to_body(track.position[None]), line)
        ahead = float(s[0]) - s_car
        inside = rear - 0.3 <= ahead <= far and abs(float(lat[0]) - lat_l) <= half
        if history is not None and not inside:
            history[p.track_id] = history.get(p.track_id, 0.0) + dt_s
        own = decided_class(track) in ("pedestrian", "bike") or (history is not None and history.get(p.track_id, 0.0) >= config.lead_piece_independent_s - 1e-9)
        if inside and not own:
            continue
        out.append(p)
    return out


def lead_role(track, along_mps, across_mps, config, wide=False):
    """What the lead in the lane is to the stack: "crossing" (a non-vehicle moving across
    the path at approach_min_mps or more, kept for the conflict and yield checks),
    "follow" (moving along the path), "obstacle_stop" (static, or a vehicle not moving
    along: the driver's LiDAR stop), or "uncertain" (a small object whose motion is not
    known: the conflict check's). ``along_mps``/``across_mps``: its velocity along and
    across the path at the lead (external review of 405b056: the car's own axis let a
    diagonal walker be followed and dropped from the crossing checks). ``wide``: as wide
    across the path as a car's rear (``lead_body_min_width_m``), which a walker is not."""
    state = motion_state(track, config.prediction)
    vehicle = is_vehicle(track, config) or wide
    if not vehicle and state != "static" and abs(across_mps) >= config.approach_min_mps:
        return "crossing"
    if along_mps > config.prediction.stationary_mps and state == "moving":
        return "follow"
    if vehicle and state == "moving" and (along_mps < -config.prediction.stationary_mps or abs(across_mps) >= config.approach_min_mps):
        # Coming towards the car, reversing or crossing: not a stopped obstacle to hand to
        # the driver's stop, which looks only along the path (review of 3e9083a, M4: an
        # oncoming car in the lane at -4 m/s left every check). The conflict check keeps it.
        return "crossing"
    if state == "static" or vehicle:
        return "obstacle_stop"
    if track.hits >= config.lead_known_hits and abs(along_mps) <= config.prediction.stationary_mps and abs(across_mps) < config.approach_min_mps:
        return "obstacle_stop"
    return "uncertain"


def _path_heading_at(path, s):
    """The heading of the polyline ``path`` (N x 2) at arc length ``s``."""
    path = np.asarray(path, dtype=float)
    seg = np.hypot(*np.diff(path, axis=0).T)
    arcs = np.concatenate(([0.0], np.cumsum(seg)))
    i = int(np.clip(np.searchsorted(arcs, s, side="right") - 1, 0, len(path) - 2))
    d = path[i + 1] - path[i]
    return math.atan2(d[1], d[0])


def _across_width(track, along_heading_rad):
    """The track's footprint measured across ``along_heading_rad`` (odometry frame), 0 without one."""
    if track.footprint is None:
        return 0.0
    length, width, yaw = track.footprint
    rel = yaw - along_heading_rad
    return length * abs(math.sin(rel)) + width * abs(math.cos(rel))


def _rear_extent(track, along_heading_rad):
    """Half the track's footprint measured along ``along_heading_rad`` (odometry frame):
    the distance from its centre back to its rear along the path. Without a measured
    footprint, the "unknown" default (2 x 2 m)."""
    if track.footprint is None:
        return 1.0
    length, width, yaw = track.footprint
    rel = yaw - along_heading_rad
    return 0.5 * (length * abs(math.cos(rel)) + width * abs(math.sin(rel)))


def _track_note(tracks, track_id, pose):
    """The log's account of the track behind a conflict or yield (vehicle frame)."""
    track = next((t for t in tracks if t.track_id == track_id), None)
    if track is None:
        return dict(track_id=track_id)
    body = pose.to_body(track.position[None])[0]
    v = pose.to_body(np.array([track.position + track.velocity]))[0] - body
    r = lambda x: None if x is None else round(float(x), 2)  # noqa: E731
    return dict(
        track_id=track_id, x_m=r(body[0]), y_m=r(body[1]), vx_mps=r(v[0]), vy_mps=r(v[1]),
        footprint=None if track.footprint is None else [r(track.footprint[0]), r(track.footprint[1])],
        travelled_m=r(track.travelled_m), recent_speed_mps=r(track.recent_speed_mps),
        support=list(track.support), status=track.status.name, doppler_rejects=track.doppler_rejects,
    )


def is_vehicle(track, config):
    """A car or truck by a decided class, or a LiDAR footprint ``vehicle_min_m`` or longer."""
    cls = decided_class(track)
    if cls is not None:
        return cls in ("car", "truck")
    return track.footprint is not None and max(track.footprint[0], track.footprint[1]) >= config.vehicle_min_m


def is_vulnerable(track, config):
    """A pedestrian or bike by a decided class, or, without one, a LiDAR footprint no
    larger than ``vulnerable_max_m`` (the walker's cluster is well under a metre)."""
    cls = decided_class(track)
    if cls in ("pedestrian", "bike"):
        return True
    if cls in ("car", "truck"):
        return False
    return track.footprint is not None and (max(track.footprint[0], track.footprint[1]) <= config.vulnerable_max_m or is_group(track, config))


def is_group(track, config):
    """People on foot together: no decided class, at most ``group_max_m`` across, moving no
    faster than ``group_max_mps`` (review of 3e9083a, M7: a 2.4 m group was not yielded to,
    and one of 3 m or more counted as a vehicle a wall could make unreachable)."""
    if decided_class(track) is not None or track.footprint is None:
        return False
    # A blob, not a line: a 3.4 x 0.37 m stretch of railing sliding past was yielded to as
    # a "group" on the first live stack turn (ff1910b).
    if not config.group_min_width_m <= min(track.footprint[0], track.footprint[1]) <= config.group_max_width_m:
        return False
    return max(track.footprint[0], track.footprint[1]) <= config.group_max_m and max(track.speed_mps, track.recent_speed_mps or 0.0) <= config.group_max_mps


def roadside_cap(track, path_xy, s_now, body, tube_half_m, config):
    """The cap for passing a pedestrian standing at the roadside, or None when the track is
    not one: vulnerable (``is_vulnerable``), not moving (``motion_state``), its near edge
    more than ``roadside_clear_m`` beyond the tube (``tube_half_m`` of the path) and ahead.
    The car slows to ``roadside_mps`` by the time its front draws level, at
    ``conflict_decel_mps2`` (third review, R5)."""
    if track is None or not is_vulnerable(track, config) or motion_state(track, config.prediction) == "moving":
        return None
    path = np.asarray(path_xy, dtype=float)
    s, lateral, outside = project_onto_polyline(track.position[None], path)
    if outside[0]:
        return None  # past the path's ends its "lateral" is the distance to an end point
    half = 0.5 * min(track.footprint[0], track.footprint[1]) if track.footprint is not None else 0.3
    if abs(float(lateral[0])) - half <= tube_half_m + config.roadside_clear_m:
        return None
    ahead = float(s[0]) - s_now - body.front_m
    return math.sqrt(config.roadside_mps**2 + 2.0 * config.conflict_decel_mps2 * max(0.0, ahead))


def conflict_speed_cap(predictions, path_xy, s_now, body, config):
    """The speed from which the car stops, at ``conflict_decel_mps2``, ``conflict_margin_m``
    short of the first arc where its footprint meets any of ``predictions`` (the road user
    of a predicted conflict); 0 when that is already that close, or not found."""
    arc = first_occupied_arc(predictions, path_xy, body, from_arc_m=s_now)
    room = 0.0 if arc is None else max(0.0, arc - s_now - config.conflict_margin_m)
    return math.sqrt(2.0 * config.conflict_decel_mps2 * room)


def crossing_yield_cap(tracks, candidates, path_xy, s_now, body, decel_mps2, config):
    """The speed cap for yielding to vulnerable road users predicted on the path ahead:
    walking pace, and a stop ``yield_margin_m`` short of the first place the path is
    predicted occupied; None when nothing is to be yielded to.

    The conflict check asks whether the car meets a road user within its horizon; at
    2 m/s a walker crossing 15 m ahead has cleared before the car arrives, so the car
    never slowed (first live stack M04, 26 Oct). A crossing pedestrian ahead in the
    path is yielded to wherever it is, as a driver would.
    """
    return crossing_yield(tracks, candidates, path_xy, s_now, body, decel_mps2, config)[0]


def crossing_yield(tracks, candidates, path_xy, s_now, body, decel_mps2, config, eligible_since=None, now_s=None):
    """``crossing_yield_cap`` with the track yielded to: (cap, track_id), or (None, None).

    ``eligible_since`` (track id -> first time it qualified, kept by the caller) and
    ``now_s`` apply ``yield_persist_s``; without them every qualifying track counts.
    """
    by_id = {t.track_id: t for t in tracks}
    path = np.asarray(path_xy, dtype=float)
    qualifying = []
    for p in candidates:
        t = by_id.get(p.track_id)
        if t is None or not is_moving(t, config.prediction) or not is_vulnerable(t, config):
            continue
        if max(t.speed_mps, t.recent_speed_mps or 0.0) > config.vru_max_speed_mps:
            continue
        if len(path) >= 2 and abs(_across_path(t, path)) < config.approach_min_mps:
            continue  # moving along the path: a lead, not a crosser
        qualifying.append(p)
    if eligible_since is not None and now_s is not None:
        ids = {p.track_id for p in qualifying}
        for tid in list(eligible_since):
            if tid not in ids:
                del eligible_since[tid]
        for tid in ids:
            eligible_since.setdefault(tid, now_s)
        qualifying = [p for p in qualifying if now_s - eligible_since[p.track_id] >= config.yield_persist_s - 1e-9]
    arc, track_id = first_occupied_arc(qualifying, path_xy, body, from_arc_m=s_now, with_track=True)
    if arc is None:
        return None, None
    room = max(0.0, arc - s_now - config.yield_margin_m)
    return min(config.yield_mps, math.sqrt(2.0 * decel_mps2 * room)), track_id


def _across_path(track, path):
    """The track's speed across ``path`` (odometry frame) at its nearest point, +left."""
    s, _, _ = project_onto_polyline(track.position[None], path)
    heading = _path_heading_at(path, float(s[0]))
    return float(-track.velocity[0] * math.sin(heading) + track.velocity[1] * math.cos(heading))


def record_rows(records):
    return [r.as_dict() for r in records]


