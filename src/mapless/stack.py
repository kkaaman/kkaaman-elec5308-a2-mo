"""``MaplessStack``: the G2/G3 stages around the G1 driver (plan v2.1, sections 9–11). Framework.

Per tick, in order:

1. **Drive** — ``MaplessDriver.step`` (health, ego motion, terrain grid, kerb corridor,
   obstacle clearance, junction turns, planner, controller). It is the G1 car and keeps
   its own safety rules; the stack only reads its inputs and may lower its speed.
2. **Segment and paint** — each admitted camera is segmented (``road_segmentation``)
   and the sweep painted (``painting``); the road model accumulates the result.
3. **Detect and cluster** — obstacle points are clustered; detector boxes claim their
   foreground clusters (``detections``).
4. **Track** — clusters, radar objects and camera classes update the tracker
   (``tracking``); radar enters only as admitted (ablation A/B/C: camera+LiDAR; +front
   radar; +rear radars).
5. **Predict and follow** — predictions over the horizon (``prediction``); the nearest
   confirmed track in the lane sets a following speed (``following``), and a predicted
   conflict on the car's path within ``conflict_horizon_s`` caps the speed to zero.
   Both reach the driver as ``speed_cap_mps`` for the next tick; the driver's own
   obstacle stop remains the backstop.
6. **Record** — a ``DecisionRecord`` with stage timings and coverage.

The stack never sees the simulator: everything comes in through the ``SensorFrame``.
Status: the plumbing runs on synthetic frames (tests); the learned segmenter and the
detector are swapped in by configuration once trained/benchmarked (G2), and the radar
admission is an experiment (G3).
"""

import time
from dataclasses import dataclass, field

import numpy as np

from mapless.contracts import DecisionRecord, TrackStatus
from mapless.detections import DetectionsConfig, Detector, NoDetector, cluster_obstacles, join
from mapless.driver import DriveMode, MaplessDriver
from mapless.following import FollowingConfig, FollowingController
from mapless.frame import SensorFrame, SensorState
from mapless.obstacles import project_onto_polyline
from mapless.painting import PaintingConfig, paint
from mapless.prediction import PredictionConfig, first_conflict, predict_all
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
        self._segmentations = []
        self._model = None
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
        begun = time.perf_counter()
        step = self.driver.step(frame)
        timing["drive"] = (time.perf_counter() - begun) * 1000
        motion, sweep = self.driver.last_motion, self.driver.last_classified
        self._tick += 1
        t = frame.sim_time_s
        tracks, predictions = (), ()
        cap = None
        coverage = {s.name: s.state.value for s in frame.status}
        if motion is not None and motion.route is not None and motion.odometry_trusted and sweep is not None and c.admission.lidar:
            pose = motion.odometry
            # -- segment and paint ---------------------------------------------------------
            begun = time.perf_counter()
            cameras = [im for im in frame.cameras if frame.status_of(im.name).state is SensorState.OK] if c.admission.cameras else []
            if cameras and self._tick % c.segment_every == 0:
                self._segmentations = [self.segmenter.segment(im) for im in cameras]
            painted = paint(sweep.xyz, cameras, self._segmentations, c.painting) if cameras and self._segmentations else None
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
            predictions = tuple(predict_all(live, c.prediction))
            timing["track"] = (time.perf_counter() - begun) * 1000
            # -- follow and conflicts --------------------------------------------------------
            begun = time.perf_counter()
            cap = self._speed_cap(motion, pose, live, predictions, t)
            timing["plan"] = (time.perf_counter() - begun) * 1000
            # -- road model ------------------------------------------------------------------
            if self._tick % c.road_model_every == 0:
                begun = time.perf_counter()
                self._model = self.road_model.build(pose, t)
                timing["road_model"] = (time.perf_counter() - begun) * 1000
        self.driver.speed_cap_mps = cap
        proposed = "WAIT" if step.mode is DriveMode.STOPPING else ("STOP" if step.mode is DriveMode.FAULT else "FOLLOW")
        executed, veto = proposed, ""
        if cap is not None and cap <= 0.0 and proposed == "FOLLOW":
            executed, veto = "STOP", "predicted conflict on the path"
        constraints = tuple(k for k in ("obstacle_m", "stop_reason", "turn") if step.diagnostics.get(k))
        record = DecisionRecord(t, proposed, executed, veto, constraints, coverage, {k: round(v, 2) for k, v in timing.items()})
        self.records.append(record)
        diagnostics = dict(step.diagnostics, tracks=len(tracks), speed_cap_mps=cap, stack_timing_ms=record.timing_ms)
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
        cap = None
        # Only tracks the tracker vouches for lead the car (a 3-point tentative cluster must not).
        leads = [tr for tr in self.tracker.live(tracks) if tr.status in (TrackStatus.CONFIRMED, TrackStatus.COASTING)]
        lead = self.following.lead_ahead(leads, pose, path_body, s_from_m=s_car)
        if lead is not None:
            s, track = lead
            gap = s - self.driver.config.body.front_m
            along = pose.to_body(np.array([track.position + track.velocity]))[0] - pose.to_body(track.position[None])[0]
            cap = self.following.target_speed(self.driver.config.planner.cruise_mps, motion.speed_mps, gap, float(along[0]), self.driver.dt_s)
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
            times = np.concatenate(([0.0], np.cumsum(seg / speed)))
            hit = first_conflict(predictions, xy, times, self.driver.config.body.half_width_m)
            if hit is not None and hit[0] <= c.conflict_horizon_s:
                cap = 0.0
        return cap

    def reset(self):
        self.tracker.reset()
        self.road_model.reset()
        self.following.reset()
        self._segmentations, self._model = [], None


def record_rows(records):
    return [r.as_dict() for r in records]


