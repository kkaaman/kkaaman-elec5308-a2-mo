"""Fixed-step forward bicycle dynamics behind autonomy.interfaces.World."""

import math
import time
import uuid
from dataclasses import dataclass, replace

from autonomy.interfaces import ControlPacket, EgoPose, FrameBundle, LaneGraph
from navigation_control.types import ControlCommand, VehicleState

from .synthetic import SyntheticScenario

FULL_BRAKE = ControlCommand(0.0, 0.0, 1.0)


@dataclass(frozen=True)
class KinematicConfig:
    """Illustrative SI parameters, not calibrated vehicle physics.

    Steering is a physical front-wheel-equivalent angle. It is independent of
    the path follower's (possibly lower) controller steering cap.
    """

    dt_s: float = 0.05
    wheelbase_m: float = 2.8
    max_steering_angle_rad: float = 0.6
    throttle_acceleration_mps2: float = 2.0
    brake_deceleration_mps2: float = 4.0

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.max_steering_angle_rad >= math.pi / 2:
            raise ValueError("max_steering_angle_rad must be less than pi/2")


class KinematicWorld:
    """Single-owner, deterministic planar world with scripted sensor inputs.

    A command is held for one dt. Acceleration is constant until speed reaches
    zero; rear-axle motion follows the exact constant-curvature arc for the
    travelled distance. No reversing, tire slip, drag, collision or actuator lag.

    Invalid packets select full brake and raise without advancing simulation.
    Since this world has no background clock, braking physically takes effect
    only when the caller advances it with a valid brake packet.
    """

    def __init__(
        self, config=None, *, initial_state=None, graph=None, scenario=None, clock=time.monotonic
    ):
        config = KinematicConfig() if config is None else config
        initial_state = VehicleState(0, 0, 0, 0) if initial_state is None else initial_state
        graph = LaneGraph((), ()) if graph is None else graph
        scenario = SyntheticScenario() if scenario is None else scenario
        if not isinstance(config, KinematicConfig) or not isinstance(initial_state, VehicleState):
            raise TypeError("Expected KinematicConfig and VehicleState")
        if not isinstance(graph, LaneGraph) or not isinstance(scenario, SyntheticScenario):
            raise TypeError("Expected LaneGraph and SyntheticScenario")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.config = replace(config)
        self.scenario = replace(scenario)
        self.graph = LaneGraph(graph.nodes, graph.edges)
        self._initial_state = replace(initial_state)
        self._clock = clock
        self._state = self._initial_state
        self._episode = None
        self._frame_id = 0
        self._closed = False
        self.last_command = FULL_BRAKE

    def _frame(self, episode, frame_id, state):
        rgb, depth = self.scenario.sensor_arrays(frame_id)
        return FrameBundle(
            episode,
            frame_id,
            frame_id * self.config.dt_s,
            self._clock(),
            EgoPose(state),
            rgb,
            self.scenario.calibration,
            depth,
            frame_id if depth is not None else None,
        )

    def reset(self, *, seed):
        """Restore initial state and scripts, with a fresh anti-replay episode ID.

        There is currently no stochastic model. The seed is validated and
        recorded in the ID; identical inputs reproduce the same physical trace.
        """
        if self._closed:
            raise RuntimeError("World is closed")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        episode = f"kinematic-{seed}-{uuid.uuid4().hex}"
        frame = self._frame(episode, 0, self._initial_state)
        self._episode, self._frame_id, self._state = episode, 0, self._initial_state
        self.last_command = FULL_BRAKE
        return frame, self.graph

    def _advance(self, command):
        state, config = self._state, self.config
        acceleration = (
            command.throttle * config.throttle_acceleration_mps2
            - command.brake * config.brake_deceleration_mps2
        )
        duration = config.dt_s
        if acceleration < 0:
            duration = min(duration, state.speed_mps / -acceleration)
        distance = state.speed_mps * duration + 0.5 * acceleration * duration**2
        speed = max(0.0, state.speed_mps + acceleration * config.dt_s)
        curvature = math.tan(command.steering_angle_rad) / config.wheelbase_m
        turn = curvature * distance
        half_turn = turn / 2
        chord = distance * (math.sin(half_turn) / half_turn if half_turn else 1.0)
        return VehicleState(
            state.x_m + chord * math.cos(state.yaw_rad + half_turn),
            state.y_m + chord * math.sin(state.yaw_rad + half_turn),
            state.yaw_rad + turn,
            speed,
        )

    def step(self, packet):
        """Apply a current ControlPacket for one dt and return aligned sensors.

        Invalid packets select full brake and raise without consuming a tick.
        """
        if self._closed or self._episode is None:
            raise RuntimeError("Reset an open world before stepping")
        try:
            if not isinstance(packet, ControlPacket):
                raise TypeError("Expected ControlPacket")
            packet = replace(
                packet, command=replace(packet.command), diagnostics=replace(packet.diagnostics)
            )
            now = self._frame_id * self.config.dt_s
            if (
                packet.source_frame_id != self._frame_id
                or packet.source_sim_time_s != now
                or not packet.is_current(self._episode, now)
            ):
                raise ValueError("Rejected foreign, expired or wrong-frame/time packet")
            command = replace(
                packet.command,
                steering_angle_rad=max(
                    -self.config.max_steering_angle_rad,
                    min(self.config.max_steering_angle_rad, packet.command.steering_angle_rad),
                ),
            )
            state = self._advance(command)
            frame = self._frame(self._episode, self._frame_id + 1, state)
        except Exception:
            self.last_command = FULL_BRAKE
            raise
        self.last_command = command
        self._state, self._frame_id = state, frame.frame_id
        return frame

    def close(self):
        """Select full brake and release episode state; repeated calls are safe."""
        self.last_command = FULL_BRAKE
        self._episode = None
        self._closed = True
