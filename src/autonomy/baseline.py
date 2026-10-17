"""Perception-to-control composition using the unchanged A1 PathFollower."""

import math
import time
from dataclasses import asdict, dataclass, field

from navigation_control import PathFollower
from navigation_control.types import ControlCommand, ControllerConfig, ControlStatus, Waypoint
from obstacle_perception.association import associate
from obstacle_perception.track_filter import TrackFilter, TrackFilterConfig
from obstacle_perception.visibility import DepthVisibility
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
    """

    def __init__(self, points_xy_m, config=None, *, navigation_only=False):
        self.config = config or BaselineConfig()
        self.route = RouteGeometry(points_xy_m)
        self.controller = PathFollower(self.config.controller)
        self.filter = TrackFilter(self.config.tracks)
        self.behaviour = Behaviour(self.config.behaviour)
        self.governor = SpeedGovernor(self.config.governor)
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
        gaps = [
            (self.route.gap(o, self._progress, config.front_bumper_from_rear_axle_m, half_width), o)
            for o in obstacles
        ]
        gaps = [
            (gap, item) for gap, item in gaps if gap is not None and gap < config.visible_horizon_m
        ]
        for _, item in gaps:
            centre = tuple(sum(p[i] for p in item.corners_world_m) / 4 for i in (0, 1))
            # Remember spatial areas across ID changes, not a scenario actor ID.
            if all(math.dist(centre, old[0]) > 1.5 for old in self._blocked_areas):
                self._blocked_areas.append((centre, item))
        # A stationary obstacle seen before a bend remains a spatial speed
        # constraint after tracker expiry. It does not force an early blind
        # stop 30 m away, and does not disappear merely because its ID expired.
        retained = []
        for centre, item in self._blocked_areas:
            gap = self.route.gap(
                item, self._progress, config.front_bumper_from_rear_axle_m, half_width
            )
            if gap is None or visibility.corridor_area_clear(
                item, self.route, self._progress, half_width
            ):
                continue
            retained.append((centre, item))
            if gap < config.visible_horizon_m:
                gaps.append((gap, item))
        self._blocked_areas = retained
        nearest = min((gap for gap, _ in gaps), default=None)
        visible = (
            None
            if self.navigation_only
            else visibility.route_horizon(
                self.route,
                self._progress,
                config.front_bumper_from_rear_axle_m,
                half_width,
                config.visible_horizon_m,
            )
        )
        invalid = associated is not None and any(
            visibility.box_may_overlap_route(
                box,
                self.route,
                self._progress,
                config.front_bumper_from_rear_axle_m,
                half_width,
                config.visible_horizon_m,
            )
            for box in associated.invalid_boxes
        )
        clear = not gaps and not invalid
        if not self.navigation_only:
            clear = clear and visible >= min(8.0, config.visible_horizon_m)
            clear = clear and all(
                visibility.corridor_area_clear(item, self.route, self._progress, half_width)
                for _, item in self._blocked_areas
            )
        must_stop = invalid or (
            nearest is not None and nearest <= config.governor.stop_margin_m + 0.1
        )
        must_stop = must_stop or (visible is not None and visible <= config.governor.stop_margin_m)
        mode = self.behaviour.update(
            time_s=frame.sim_time_s,
            speed=state.speed_mps,
            obstacle=bool(gaps),
            must_stop=must_stop,
            clear=clear,
            observation_valid=not invalid,
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
        target, emergency = self.governor.limit(
            speed=state.speed_mps,
            dt=dt,
            curvature=self.route.curvature_ahead(self._progress, config.visible_horizon_m),
            obstacle_gap=nearest,
            visible_m=visible,
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
        if nearest is not None and command.brake > 0:
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
