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
   ``conflict_horizon_s`` caps the speed to zero. Both reach the driver as
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

import time
from dataclasses import dataclass, field

import numpy as np

from mapless.contracts import DecisionRecord
from mapless.detections import DetectionsConfig, Detector, NoDetector, cluster_obstacles, join
from mapless.driver import DriveMode, MaplessDriver
from mapless.following import LEAD_STATUSES, FollowingConfig, FollowingController
from mapless.frame import SensorFrame, SensorState
from mapless.obstacles import project_onto_polyline
from mapless.painting import PaintingConfig, SegmentedFrame, paint
from mapless.prediction import PredictionConfig, first_footprint_conflict, predict_all
from mapless.radar import RadarConfig, measure
from mapless.road_model import RoadModelBuilder, RoadModelConfig
from mapless.road_segmentation import RoadSegmenter, load_segmenter
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
        coverage = {s.name: s.state.value for s in frame.status}
        # -- local frame rebuild -----------------------------------------------------------
        rebuilt = bool(step.diagnostics.get("frame_rebuilt"))
        if rebuilt:
            self.reset_spatial()
            self.frame_generation += 1
        # While the driver stops to rebuild, the odometry frame lags the car: nothing is placed in it.
        frame_usable = rebuilt or not step.diagnostics.get("rebuilding")
        if motion is not None and motion.route is not None and motion.odometry_trusted and frame_usable and sweep is not None and c.admission.lidar:
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
            clusters = cluster_obstacles(sweep, c.detections)
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
            tracks = tuple(self.tracker.update(t, pose, measurements, radar_measurements, camera_only))
            live = self.tracker.live(tracks)
            # Only tracks the tracker vouches for are predicted onto the path: a fresh
            # tentative track carries the filter's initial velocity variance (5 m/s), which
            # the prediction inflates to a 30 m box at 3 s, and every new parked-car cluster
            # tapped the brakes for a few ticks (fresh-eyes review of 58adaf3). Tentative
            # clusters are the driver's LiDAR obstacle stop's job, the documented backstop.
            predictions = tuple(predict_all([tr for tr in live if tr.status in LEAD_STATUSES], c.prediction))
            timing["track"] = (time.perf_counter() - begun) * 1000
            # -- follow and conflicts --------------------------------------------------------
            begun = time.perf_counter()
            cap, cap_reason = self._speed_cap(motion, pose, live, predictions, t)
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
        diagnostics = dict(
            step.diagnostics,
            tracks=len(tracks),
            speed_cap_mps=cap_applied,
            speed_cap_next_mps=cap,
            speed_cap_reason=cap_reason,  # logged every tick: a hold at a standstill never reaches the veto
            frame_generation=self.frame_generation,
            stack_timing_ms=record.timing_ms,
        )
        return StackStep(step.command, step.mode, step.reason, diagnostics, tracks, predictions, self._model, record)

    def _speed_cap(self, motion, pose, tracks, predictions, t):
        """Following speed for the lead in the lane, and zero on a predicted conflict."""
        c = self.config
        plan = self.driver.last_plan
        corridor = self.driver.corridor.previous
        if corridor is None or not corridor.ok:
            return None
        path_body = corridor.centre_body
        # The corridor line starts behind the rear axle: arc lengths are measured from the car.
        s_car = float(project_onto_polyline(np.zeros((1, 2)), path_body)[0][0])
        cap, reason = None, ""
        # Only tracks the tracker vouches for lead the car (a 3-point tentative cluster must not).
        leads = [tr for tr in self.tracker.live(tracks) if tr.status in LEAD_STATUSES]
        lead = self.following.lead_ahead(leads, pose, path_body, s_from_m=s_car)
        if lead is not None:
            s, track = lead
            gap = s - self.driver.config.body.front_m
            along = pose.to_body(np.array([track.position + track.velocity]))[0] - pose.to_body(track.position[None])[0]
            cap = self.following.target_speed(self.driver.config.planner.cruise_mps, motion.speed_mps, gap, float(along[0]), self.driver.dt_s)
            reason = "lead at a standstill" if cap is not None and cap <= 0.0 else "following the lead"
        else:
            self.following.reset()
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
            hit = first_footprint_conflict(predictions, xy, times, self.driver.config.body)
            if hit is not None and hit[0] <= c.conflict_horizon_s:
                cap, reason = 0.0, "predicted conflict on the path"
        return cap, reason

    def reset_spatial(self):
        """Drop everything kept in the odometry frame: tracks, semantic evidence, the
        following history and the cached segmentation (the frame they were placed in is gone)."""
        self.tracker.reset()
        self.road_model.reset()
        self.following.reset()
        self._segmentations, self._model = None, None
        self.driver.speed_cap_mps = None

    def reset(self):
        self.reset_spatial()
        self.frame_generation = 0


def record_rows(records):
    return [r.as_dict() for r in records]


