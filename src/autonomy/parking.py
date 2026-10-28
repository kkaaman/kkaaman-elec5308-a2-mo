"""Opt-in forward parking episode: depth -> governor -> controller -> PARK."""

import math
import time
from bisect import bisect_right
from dataclasses import dataclass

from autonomy.interfaces import ControlPacket, Diagnostics, FrameBundle
from autonomy.parking_contracts import (
    DriveMode,
    MotionState,
    ParkingConfig,
    ParkingPhase,
    angle_error,
    finite,
)
from navigation_control import ControllerConfig, ControlStatus, PathFollower
from navigation_control.path import ReferencePath
from navigation_control.types import ControlCommand
from obstacle_perception.parking_clearance import Clearance, DepthParkingClearance
from planning.behaviour import ClearanceGuard
from planning.parking import plan_forward_parking
from planning.parking_geometry import bay_polygon, contains, footprint
from planning.speed_governor import GovernorConfig, SpeedGovernor, stopping_distance

BRAKE = ControlCommand(0, 0, 1)


@dataclass(frozen=True)
class ParkingEvent:
    frame_id: int
    sim_time_s: float
    event: str


@dataclass(frozen=True)
class ParkingStep:
    packet: ControlPacket
    motion: MotionState
    phase: ParkingPhase
    position_error_m: float
    heading_error_rad: float
    progress_m: float
    target_speed_mps: float
    clearance: str
    obstacle_gap_m: float | None
    reason: str


class ParkingEpisode:
    """One immutable path per episode; PARK is an application mode, not a gear RPC.

    The caller applies returned ControlPackets through its World/adapter and
    supplies ego state and aligned calibrated depth. No scene actor geometry or
    detector result is accepted as a substitute for depth clearance. Construct
    a new episode to leave PARK; no per-frame path reinstall/reset is provided.
    """

    def __init__(self, initial_frame, goal, vehicle, drivable_area, config=None):
        if not isinstance(initial_frame, FrameBundle):
            raise TypeError("Expected FrameBundle")
        self.config = ParkingConfig() if config is None else config
        self.goal, self.vehicle = goal, vehicle
        if self.config.ground_exclusion_m >= vehicle.height_m:
            raise ValueError("Ground exclusion must be below body height")
        self.plan = plan_forward_parking(
            initial_frame.ego.state, goal, vehicle, drivable_area, self.config
        )
        self.episode_id = initial_frame.episode_id
        self._initial_state = initial_frame.ego.state
        self._initial_frame_id = initial_frame.frame_id
        self._initial_time = initial_frame.sim_time_s
        self._last_frame = None
        self._last_time = None
        self._phase = ParkingPhase.READY
        self._mode = DriveMode.DRIVE
        self._failure = None
        self._progress = 0.0
        self._events = [ParkingEvent(initial_frame.frame_id, initial_frame.sim_time_s, "path_installed")]
        config = self.config
        self._path = ReferencePath(self.plan.waypoints)
        self._follower = PathFollower(ControllerConfig(
            vehicle.wheelbase_m, vehicle.steering_limit_rad,
            lookahead_m=config.lookahead_m, kp=1.2, ki=0.0, kd=0.0,
            stopping_deceleration_mps2=config.deceleration_mps2,
            goal_tolerance_m=goal.position_tolerance_m,
            stopped_speed_mps=config.stopped_speed_mps, hold_brake=config.hold_brake,
        ))
        self._follower.set_path(self.plan.waypoints)
        self._guard = ClearanceGuard()
        self._governor = SpeedGovernor(GovernorConfig(
            cruise_mps=config.max_speed_mps, deceleration_mps2=config.deceleration_mps2,
            lateral_acceleration_mps2=0.5, response_budget_s=config.response_budget_s,
            stop_margin_m=config.stop_margin_m, acceleration_mps2=config.acceleration_mps2,
            emergency_margin_m=config.emergency_margin_m,
        ))

    @property
    def mode(self):
        return self._mode

    @property
    def phase(self):
        return self._phase

    @property
    def events(self):
        return tuple(self._events)

    @property
    def route_install_count(self):
        return 1

    def _result(self, frame, now, command=BRAKE, *, phase=None, reason="", valid=True,
                target=0.0, clearance="unknown", gap=None):
        if phase is not None and phase != self._phase:
            self._phase = phase
            self._events.append(ParkingEvent(frame.frame_id, frame.sim_time_s, phase.value))
        if not valid:
            self._guard.invalidate()
            self._governor.reset()
        state = frame.ego.state
        diagnostics = Diagnostics(observation_valid=valid, reasons=(reason,) if reason else ())
        packet = ControlPacket(
            self.episode_id, frame.frame_id, frame.sim_time_s,
            frame.sim_time_s + self.config.command_ttl_s, command, diagnostics,
            valid_until_monotonic_s=now + self.config.command_ttl_s,
        )
        return ParkingStep(
            packet, MotionState.from_forward(state, self._mode), self._phase,
            math.hypot(state.x_m - self.goal.x_m, state.y_m - self.goal.y_m),
            abs(angle_error(state.yaw_rad, self.goal.yaw_rad)), self._progress,
            target, clearance, gap, reason,
        )

    def _fault(self, frame, now, reason):
        self._failure = self._failure or reason
        return self._result(frame, now, phase=ParkingPhase.FAULT, reason=self._failure, valid=False)

    def step(self, frame, *, now_monotonic_s=None):
        """Return full brake on unknown/stale data and latch structural/control faults.

        An explicit host time supports deterministic replay. Production callers
        omit it so the function also enforces its command-delivery time budget.
        """
        if not isinstance(frame, FrameBundle):
            raise TypeError("Expected FrameBundle; caller must brake on malformed packets")
        now = time.monotonic() if now_monotonic_s is None else now_monotonic_s
        finite("now_monotonic_s", now, minimum=0)
        previous_mode, previous_phase = self._mode, self._phase
        previous_events = len(self._events)
        result = self._step(frame, now)
        if now_monotonic_s is None:
            finished_at = time.monotonic()
            capture = frame.capture_started_monotonic_s
            capture = frame.received_monotonic_s if capture is None else capture
            timeout = finished_at - now > self.config.command_ttl_s
            expired = (previous_mode == DriveMode.DRIVE
                       and result.packet.diagnostics.observation_valid
                       and finished_at - capture > self.config.max_observation_age_s)
            if timeout or expired:
                # _step's state changes are not published until the result's
                # clocks pass. A failed delivery must never publish a new PARK.
                self._mode, self._phase = previous_mode, previous_phase
                del self._events[previous_events:]
                return self._fault(frame, finished_at,
                                   "processing_timeout" if timeout else "observation_expired_during_step")
        return result

    def _step(self, frame, now):
        config, goal, state = self.config, self.goal, frame.ego.state
        if self._failure:
            return self._fault(frame, now, self._failure)
        if frame.episode_id != self.episode_id:
            return self._fault(frame, now, "foreign_episode")
        if self._last_frame is None:
            if (frame.frame_id != self._initial_frame_id or frame.sim_time_s != self._initial_time
                    or state != self._initial_state):
                return self._fault(frame, now, "initial_frame_changed")
        elif (frame.frame_id != self._last_frame + 1
              or not math.isclose(frame.sim_time_s - self._last_time, config.dt_s, abs_tol=1e-8)):
            return self._fault(frame, now, "nonconsecutive_frame_or_time")
        self._last_frame, self._last_time = frame.frame_id, frame.sim_time_s
        if self._mode == DriveMode.PARK:
            return self._result(frame, now, phase=ParkingPhase.PARKED,
                                reason="park_latched", clearance="not_rechecked")
        if frame.sim_time_s - self._initial_time > config.max_duration_s:
            return self._fault(frame, now, "parking_timeout")
        capture = frame.capture_started_monotonic_s
        capture = frame.received_monotonic_s if capture is None else capture
        if now < frame.received_monotonic_s or now - capture > config.max_observation_age_s:
            return self._result(frame, now, phase=ParkingPhase.HOLD,
                                reason="stale_or_future_observation", valid=False)
        actual_body = footprint(state, self.vehicle, goal.clearance_m)
        if not contains(self.plan.drivable_area, actual_body):
            return self._fault(frame, now, "body_outside_manoeuvring_area")
        projection = self._path.project(state.x_m, state.y_m)
        self._progress = max(self._progress, projection.progress_m)
        index = min(int(bisect_right(
            self._path.arc_lengths, projection.progress_m
        )) - 1, len(self.plan.segments) - 1)
        expected_heading = self.plan.headings_rad[index]
        if (projection.distance_m > config.tracking_margin_m
                or abs(angle_error(state.yaw_rad, expected_heading)) > config.tracking_heading_rad):
            return self._fault(frame, now, "tracking_envelope_exceeded")
        observer = DepthParkingClearance(frame, self.vehicle, config)
        current_clearance = observer.check(actual_body)
        gap = None
        unknown = current_clearance == Clearance.UNKNOWN
        if current_clearance == Clearance.BLOCKED:
            gap = 0.0
        horizon = max(config.preview_m, stopping_distance(
            state.speed_mps, config.deceleration_mps2, config.response_budget_s
        ) + config.stop_margin_m + config.sample_step_m)
        for segment in self.plan.segments[index:]:
            if segment.start_s_m > self._progress + horizon:
                break
            clearance = observer.check(segment.sweep)
            if clearance != Clearance.CLEAR:
                unknown |= clearance == Clearance.UNKNOWN
                gap = max(0.0, segment.start_s_m - self._progress) if gap is None else gap
                break
        clear = self._guard.update(time_s=frame.sim_time_s, obstacle_present=gap is not None,
                                   observation_valid=not unknown)
        if unknown:
            return self._result(frame, now, phase=ParkingPhase.HOLD,
                                reason="unobserved_swept_volume", valid=False, gap=gap)
        hold = gap is None and not clear
        target, emergency = self._governor.limit(
            speed_mps=state.speed_mps, dt_s=config.dt_s, obstacle_gap_m=gap,
            hold=hold,
        )
        near_goal = math.hypot(state.x_m - goal.x_m, state.y_m - goal.y_m) <= goal.position_tolerance_m
        heading_ok = abs(angle_error(state.yaw_rad, goal.yaw_rad)) <= goal.heading_tolerance_rad
        body_fits = contains(bay_polygon(goal.bay), actual_body)
        completion = clear and gap is None and near_goal and heading_ok and body_fits
        if (near_goal and state.speed_mps <= config.stopped_speed_mps
                and self._progress >= self.plan.length_m - goal.position_tolerance_m
                and not (heading_ok and body_fits)):
            return self._fault(frame, now, "endpoint_pose_or_body_unsatisfied")
        stopped_at_blocker = (gap is not None and target == 0
                              and state.speed_mps <= config.stopped_speed_mps)
        if hold or emergency or stopped_at_blocker:
            self._governor.reset()
            return self._result(frame, now, phase=ParkingPhase.HOLD,
                                reason="clearance_confirmation" if hold else "blocked_bay",
                                clearance="clear_pending" if hold else "blocked", gap=gap)
        try:
            result = self._follower.update(state, target, config.dt_s, completion_allowed=completion)
        except (ValueError, RuntimeError) as error:
            return self._fault(frame, now, "controller_rejected: " + str(error))
        if result.status == ControlStatus.FINISHED:
            self._mode = DriveMode.PARK
            self._events.append(ParkingEvent(frame.frame_id, frame.sim_time_s, "drive_to_park"))
            return self._result(frame, now, phase=ParkingPhase.PARKED,
                                reason="goal_validated", clearance="clear")
        command = result.command
        if state.speed_mps >= config.max_speed_mps:
            command = ControlCommand(command.steering_angle_rad, 0, config.hold_brake)
        phase = ParkingPhase.STOPPING if result.status == ControlStatus.STOPPING else ParkingPhase.APPROACH
        return self._result(frame, now, command, phase=phase, target=result.diagnostics.internal_target_speed_mps,
                            reason="approaching_blocker" if gap is not None else "following",
                            clearance="blocked" if gap is not None else "clear", gap=gap)
