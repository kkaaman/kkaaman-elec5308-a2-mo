"""Perception-to-control composition using the unchanged A1 PathFollower."""

import math
import time
from collections import deque
from dataclasses import asdict, dataclass, field

from navigation_control import PathFollower
from navigation_control.types import ControlCommand, ControllerConfig, ControlStatus, Waypoint
from obstacle_perception.association import associate
from obstacle_perception.box_depth import ground_fallback
from obstacle_perception.track_filter import TrackFilter, TrackFilterConfig
from obstacle_perception.visibility import DepthVisibility, RouteClearanceMemory
from planning.behaviour import Behaviour, BehaviourConfig, BehaviourState
from planning.corridor import RouteGeometry
from planning.speed_governor import GovernorConfig, SpeedGovernor


@dataclass(frozen=True)
class BaselineConfig:
    dt_s: float = 0.05
    front_bumper_from_rear_axle_m: float = 3.81152
    ego_half_width_m: float = 1.08173
    lateral_padding_m: float = 0.15
    visible_horizon_m: float = 35.0
    planned_min_brake: float = 0.4
    # Stationary-scene memory of corridor cells seen clear (tight-turn FOV).
    visibility_memory_s: float = 15.0
    # Detector input size (long side, px) and confidence. The Perception stage
    # measures range/latency with these; freezing copies the measured values.
    detector_image_size: int = 640
    detector_confidence: float = 0.25
    controller: ControllerConfig = field(
        default_factory=lambda: ControllerConfig(
            wheelbase_m=3.00465,
            max_steering_angle_rad=math.radians(20),
            lookahead_m=4.0,
            kp=0.45,
            ki=0.15,
            stopping_deceleration_mps2=1.0,
            hold_brake=0.5,
        )
    )
    governor: GovernorConfig = field(default_factory=GovernorConfig)
    behaviour: BehaviourConfig = field(default_factory=BehaviourConfig)
    tracks: TrackFilterConfig = field(default_factory=TrackFilterConfig)

    def __post_init__(self):
        if not 0.4 <= self.planned_min_brake <= 1:
            raise ValueError("Planned brake must remain in the measured range [0.4, 1]")
        for name in (
            "dt_s",
            "front_bumper_from_rear_axle_m",
            "ego_half_width_m",
            "lateral_padding_m",
            "visible_horizon_m",
            "visibility_memory_s",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.dt_s > 0.1:
            raise ValueError("Baseline requires dt <= 0.1 seconds")
        if (
            type(self.detector_image_size) is not int
            or not 320 <= self.detector_image_size <= 1920
            or self.detector_image_size % 32
        ):
            raise ValueError("Detector image size must be a multiple of 32 in [320, 1920]")
        if (
            isinstance(self.detector_confidence, bool)
            or not isinstance(self.detector_confidence, (float, int))
            or not 0.05 <= self.detector_confidence <= 0.95
        ):
            raise ValueError("Detector confidence must lie in [0.05, 0.95]")

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        for name, kind in (
            ("controller", ControllerConfig),
            ("governor", GovernorConfig),
            ("behaviour", BehaviourConfig),
            ("tracks", TrackFilterConfig),
        ):
            if name in values:
                values[name] = kind(**values[name])
        return cls(**values)


class BaselinePolicy:
    """Motion-policy slot in SkeletonStack; no simulator or evaluation truth.

    Route is installed once on reset. A stopped obstacle area is retained until
    current depth observes clearance, even when tracker identities disappear.

    Every observation, confirmed or not, limits speed and blocks clearance.
    Only N-of-M confirmed observations (or a confirmed unknown-depth box over
    the route) can force STOPPING; a single flickering frame cannot. Emergency
    braking still reacts to any observation inside the emergency margin.
    """

    def __init__(self, points_xy_m, config=None, *, navigation_only=False):
        self.config = config or BaselineConfig()
        self.route = RouteGeometry(points_xy_m)
        self.controller = PathFollower(self.config.controller)
        self.filter = TrackFilter(self.config.tracks)
        self.behaviour = Behaviour(self.config.behaviour)
        self.governor = SpeedGovernor(self.config.governor)
        self.memory = RouteClearanceMemory(self.config.visibility_memory_s)
        self.navigation_only = navigation_only
        self.last = {}
        self.path_installations = 0

    def reset(self, *, episode_id, graph):
        self.episode_id = episode_id
        self.controller.set_path(Waypoint(*p) for p in self.route.points)
        self.path_installations += 1
        self.filter.reset(episode_id)
        self.behaviour.reset()
        self.governor.reset()
        self.memory.reset()
        self._invalid_history = deque(maxlen=max(1, self.config.tracks.window_frames - 1))
        self._last_time = None
        self._progress = 0.0
        self._blocked_areas = []
        self._finished = False
        self.last = {}

    def __call__(self, frame, batch):
        config, state = self.config, frame.ego.state
        if frame.episode_id != self.episode_id:
            raise ValueError("Policy episode mismatch")
        dt = config.dt_s if self._last_time is None else frame.sim_time_s - self._last_time
        if not math.isclose(dt, config.dt_s, abs_tol=1e-5):
            raise ValueError("Unexpected control timestep")
        self._last_time = frame.sim_time_s
        projection = self.route.project(state.x_m, state.y_m)
        if projection.s_m > self._progress + max(2.0, state.speed_mps * dt * 2) and self.last:
            raise ValueError("Route progress jump")
        if projection.distance_m > 2.0:
            raise ValueError("Vehicle left the supported route corridor")
        self._progress = max(self._progress, projection.s_m)
        half_width = config.ego_half_width_m + config.lateral_padding_m
        association_started = time.perf_counter()
        associated = associate(frame, batch) if not self.navigation_only else None
        association_s = time.perf_counter() - association_started
        filter_started = time.perf_counter()
        obstacles = self.filter.update(associated.footprints, batch.metadata) if associated else ()
        filter_s = time.perf_counter() - filter_started
        planning_started = time.perf_counter()
        visibility = DepthVisibility(frame)
        front = config.front_bumper_from_rear_axle_m
        measured = []
        for item in obstacles:
            gap = self.route.gap(item, self._progress, front, half_width)
            if gap is not None and gap < config.visible_horizon_m:
                measured.append((gap, item))
        confirmed = [(gap, item) for gap, item in measured if item.confirmed]
        for _, item in confirmed:
            centre = tuple(sum(p[i] for p in item.corners_world_m) / 4 for i in (0, 1))
            # Remember spatial areas across ID changes, not a scenario actor ID.
            if all(math.dist(centre, old[0]) > 1.5 for old in self._blocked_areas):
                self._blocked_areas.append((centre, item))
        # A stationary obstacle seen before a bend remains a spatial speed
        # constraint after tracker expiry. It does not force an early blind
        # stop 30 m away, and does not disappear merely because its ID expired.
        retained, retained_gaps = [], []
        for centre, item in self._blocked_areas:
            gap = self.route.gap(item, self._progress, front, half_width)
            if gap is None or visibility.corridor_area_clear(
                item, self.route, self._progress, half_width
            ):
                continue
            retained.append((centre, item))
            if gap < config.visible_horizon_m:
                retained_gaps.append((gap, item))
        self._blocked_areas = retained
        stopping = confirmed + retained_gaps
        nearest_confirmed = min((gap for gap, _ in stopping), default=None)
        nearest = min((gap for gap, _ in measured + retained_gaps), default=None)
        visible = (
            None
            if self.navigation_only
            else visibility.route_horizon(
                self.route,
                self._progress,
                front,
                half_width,
                config.visible_horizon_m,
                memory=self.memory,
                time_s=frame.sim_time_s,
            )
        )
        unknown = []
        if associated is not None:
            unknown = [
                detection
                for detection in associated.invalid_detections
                if visibility.box_may_overlap_route(
                    detection.box,
                    self.route,
                    self._progress,
                    front,
                    half_width,
                    config.visible_horizon_m,
                )
            ]
        unknown_confirmed = self._confirm_unknown(unknown)
        # An unconfirmed unknown-depth box limits speed by its assumed-ground
        # range; it is never evidence of clear space.
        unknown_gap = None
        for detection in unknown:
            estimate = ground_fallback(detection.box, frame.calibration)
            if estimate.point_ego_m is not None:
                gap = float(estimate.point_ego_m[0]) - front
                unknown_gap = gap if unknown_gap is None else min(unknown_gap, gap)
        clear = not measured and not retained_gaps and not unknown
        if not self.navigation_only:
            clear = clear and visible >= min(8.0, config.visible_horizon_m)
            clear = clear and all(
                visibility.corridor_area_clear(item, self.route, self._progress, half_width)
                for _, item in self._blocked_areas
            )
        must_stop = unknown_confirmed or (
            nearest_confirmed is not None
            and nearest_confirmed <= config.governor.stop_margin_m + 0.1
        )
        must_stop = must_stop or (visible is not None and visible <= config.governor.stop_margin_m)
        mode = self.behaviour.update(
            time_s=frame.sim_time_s,
            speed=state.speed_mps,
            obstacle=bool(stopping),
            must_stop=must_stop,
            clear=clear,
            observation_valid=not unknown_confirmed,
            finished=self._finished,
        )
        if mode in (BehaviourState.RESUME, BehaviourState.CRUISE) and clear:
            self._blocked_areas.clear()
        hold = mode in (
            BehaviourState.STOPPING,
            BehaviourState.HOLD,
            BehaviourState.FINISHED,
            BehaviourState.FAULT_STOP,
        )
        # The assumed-ground range of an unknown-depth box behaves like reduced
        # visibility: it caps speed but, being unmeasured, cannot by itself
        # trigger emergency braking. Measured gaps keep emergency precedence.
        speed_visible = visible
        if unknown_gap is not None:
            speed_visible = unknown_gap if visible is None else min(visible, unknown_gap)
        target, emergency = self.governor.limit(
            speed=state.speed_mps,
            dt=dt,
            curvature=self.route.curvature_ahead(self._progress, config.visible_horizon_m),
            obstacle_gap=nearest,
            visible_m=speed_visible,
            hold=hold,
        )
        planning_s = time.perf_counter() - planning_started
        control_started = time.perf_counter()
        result = self.controller.update(state, target, dt)
        self._finished = result.status == ControlStatus.FINISHED
        if self._finished:
            self.behaviour.state = BehaviourState.FINISHED
        command = result.command
        if emergency or mode == BehaviourState.FAULT_STOP:
            command = ControlCommand(command.steering_angle_rad, 0, 1)
        elif mode == BehaviourState.HOLD:
            command = ControlCommand(0, 0, 1)
        # Planned obstacle braking stays inside the measured brake range, but
        # only for confirmed obstacles that actually require slowing.
        if nearest_confirmed is not None and command.brake > 0 and target < state.speed_mps:
            command = ControlCommand(
                command.steering_angle_rad, 0, max(command.brake, config.planned_min_brake)
            )
        self.last = dict(
            state=self.behaviour.state.value,
            reason=self.behaviour.reason,
            target_mps=target,
            internal_target_mps=result.diagnostics.internal_target_speed_mps,
            speed_mps=state.speed_mps,
            signed_cte_m=projection.lateral_m,
            cte_m=projection.distance_m,
            progress_m=self._progress,
            distance_to_goal_m=result.diagnostics.distance_to_goal_m,
            finished=self._finished,
            obstacle_gap_m=nearest,
            confirmed_gap_m=nearest_confirmed,
            unknown_in_route=len(unknown),
            unknown_confirmed=unknown_confirmed,
            unknown_ground_gap_m=unknown_gap,
            visible_m=visible,
            invalid_depth_count=associated.invalid_depth_count if associated else 0,
            clear_frames=self.behaviour.clear_count,
            emergency=emergency,
            path_installations=self.path_installations,
            obstacles=[asdict(o) for o in obstacles],
            detections=asdict(batch),
            policy_stage_s=dict(
                association=association_s,
                filter=filter_s,
                planning=planning_s,
                controller=time.perf_counter() - control_started,
            ),
            controller=asdict(result.diagnostics),
        )
        return command

    def _confirm_unknown(self, detections):
        """N-of-M persistence for unknown-depth boxes over the route.

        A box matches an earlier frame by the same tracker ID, or by class and
        IoU >= 0.3 when anonymous. The history covers the last M-1 frames.
        """
        required = self.config.tracks.required_hits
        confirmed = False
        for detection in detections:
            hits = 1
            for previous in self._invalid_history:
                if any(_same_unknown(detection, old) for old in previous):
                    hits += 1
            confirmed = confirmed or hits >= required
        self._invalid_history.append(tuple(detections))
        return confirmed


def _iou(a, b):
    width = min(a.x_max, b.x_max) - max(a.x_min, b.x_min)
    height = min(a.y_max, b.y_max) - max(a.y_min, b.y_min)
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    union = a.width_px * a.height_px + b.width_px * b.height_px - inter
    return inter / union if union > 0 else 0.0


def _same_unknown(current, previous):
    if current.class_name != previous.class_name:
        return False
    if current.track_id is not None and previous.track_id is not None:
        return current.track_id == previous.track_id
    return _iou(current.box, previous.box) >= 0.3
