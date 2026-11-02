"""G5: a learned cruise advisor (2 Nov).

The policy picks the cruise target (2, 3 or 4 m/s) five times a second from the driver's
own view of the road: how far the corridor is valid and observed, how much kerb it holds,
how it bends ahead and how much it changed since last tick, the sight-limited speed, the
caps in force, an obstacle ahead, a junction crossing, the distance to the goal, its own
last choice. Every restriction still binds under it (``MaplessDriver.cruise_target_mps``):
curvature, sight distance, the stack's following and conflict caps, holds, the goal. It
never steers or brakes; it only says how fast the car may wish to go.

Training runs the real driver on the synthetic world (``sim_harness.mapless_road``):
random roads of straights and arcs, widths, kerb distances, parked cars, sensor-noise
seeds. No CARLA town is involved, Town10HD least of all (the plan's rule: no model trains
on it); evaluator truth (the car's place in its lane, collisions) shapes the reward only.

``features`` is the one place the observation is built, for training and live alike.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

ACTIONS_MPS = (2.0, 3.0, 4.0)
DECISION_TICKS = 4  # 5 Hz at the 20 Hz control step
FEATURE_NAMES = (
    "speed", "corridor_ok", "valid", "road_support", "evidence", "dy_5", "dy_15",
    "curv_near", "curv_far", "lane_offset", "sight_speed", "cap", "obstacle", "crossing",
    "goal", "stopping", "prev_2", "prev_3", "prev_4",
)


def _clip(x, lo=0.0, hi=1.0):
    return float(min(max(x, lo), hi))


def _curvature(line, s0, s1):
    """Mean |curvature| (1/m) of the polyline between arc lengths ``s0`` and ``s1`` ahead
    of its start, from the heading change over that stretch; 0 when it does not reach."""
    line = np.asarray(line, dtype=float)
    if len(line) < 3:
        return 0.0
    seg = np.diff(line, axis=0)
    arcs = np.concatenate(([0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))))
    if arcs[-1] < s0 + 2.0:
        return 0.0
    s1 = min(s1, arcs[-1])
    i0, i1 = int(np.searchsorted(arcs, s0)), int(np.searchsorted(arcs, s1)) - 1
    i0, i1 = min(i0, len(seg) - 1), max(min(i1, len(seg) - 1), 0)
    h0, h1 = math.atan2(seg[i0, 1], seg[i0, 0]), math.atan2(seg[i1, 1], seg[i1, 0])
    dh = (h1 - h0 + math.pi) % (2 * math.pi) - math.pi
    return abs(dh) / max(arcs[i1 + 1] - arcs[i0], 1.0)


def features(diagnostics, corridor, mode, previous_action, cruise_max_mps=4.0):
    """The policy's observation: a vector in [-1, 1] (``FEATURE_NAMES``), from the driver's
    diagnostics and its corridor only — what the car itself computes."""
    d = diagnostics or {}
    line = corridor.centre_body if corridor is not None and getattr(corridor, "ok", False) else None
    if line is not None and len(line) >= 2:
        ahead = line[line[:, 0] >= -1.0] if (line[:, 0] >= -1.0).sum() >= 3 else line
        near, far = _curvature(ahead, 0.0, 10.0), _curvature(ahead, 10.0, 25.0)
    else:
        near = far = 0.0
    change = d.get("corridor_change") or {}
    cap = d.get("speed_cap_mps")
    obstacle = d.get("obstacle_m")
    goal = d.get("goal_distance_m")
    prev = [1.0 if previous_action == i else 0.0 for i in range(len(ACTIONS_MPS))]
    return np.array([
        _clip((d.get("speed_mps") or 0.0) / cruise_max_mps),
        1.0 if d.get("corridor_ok") else 0.0,
        _clip((d.get("corridor_valid_m") or 0.0) / 30.0),
        _clip((d.get("corridor_road_support_m") or 0.0) / 30.0),
        _clip((d.get("corridor_evidence") or 0.0) / 30.0),
        _clip(abs(change.get("dy_5m") or 0.0)),
        _clip(abs(change.get("dy_15m") or 0.0)),
        _clip(near * 20.0),  # 1: a 20 m radius
        _clip(far * 20.0),
        _clip(d.get("lane_offset_m") or 0.0, -1.0, 1.0),
        _clip((d.get("sight_speed_mps") or 0.0) / 5.0),
        1.0 if cap is None else _clip(cap / cruise_max_mps),
        1.0 if obstacle is None else _clip(obstacle / 30.0),
        1.0 if d.get("crossing") else 0.0,
        1.0 if goal is None else _clip(goal / 50.0),
        1.0 if mode in ("stopping", "fault") else 0.0,
        *prev,
    ], dtype=np.float32)


# -- the synthetic training world ----------------------------------------------------------
@dataclass(frozen=True)
class EpisodeSpec:
    pieces: tuple  # Straight / Arc pieces of the kerb line
    road_width_m: float
    lane_centre_m: float
    parked: tuple
    seed: int
    goal_m: float


def random_episode(rng, seed):
    """A random road for one episode: 3-6 pieces, straights of 15-60 m and arcs of 15-80 m
    radius turning 20-90 degrees either way, starting and ending straight; road 7-11 m wide,
    the lane centre 1.75-3.5 m from the kerb; a few cars parked at the kerb."""
    from sim_harness.mapless_road import Arc, ParkedCar, Straight  # noqa: PLC0415

    pieces = [Straight(float(rng.uniform(30.0, 45.0)))]
    for _ in range(int(rng.integers(2, 5))):
        if rng.random() < 0.55:
            radius = float(rng.uniform(15.0, 80.0)) * (1 if rng.random() < 0.5 else -1)
            pieces.append(Arc(radius, math.radians(float(rng.uniform(20.0, 90.0)))))
        pieces.append(Straight(float(rng.uniform(15.0, 60.0))))
    pieces.append(Straight(40.0))
    length = sum(p.length_m if isinstance(p, Straight) else abs(p.radius_m) * p.angle_rad for p in pieces)
    width = float(rng.uniform(7.0, 11.0))
    lane = float(rng.uniform(1.75, min(3.5, width / 2 - 0.5)))
    # Cars parked at the kerb (centre 1.0 m out, 1.9 m wide) only where the lane runs clear of
    # them: the policy learns to pass them, the obstacle stop is not its business.
    parked = () if lane < 3.0 else tuple(
        ParkedCar(float(s), 1.0) for s in sorted(rng.uniform(25.0, length - 30.0, size=int(rng.integers(0, 3))))
    )
    return EpisodeSpec(tuple(pieces), width, lane, parked, int(seed), float(length - 25.0))


def make_driver_config(lane_centre_m, cruise_mps=2.0):
    """The live harness's driver configuration (``carla_adapter.mapless_drive.driver_config``,
    whose controller values these are), on the synthetic world's car."""
    from mapless.corridor import CorridorConfig  # noqa: PLC0415
    from mapless.driver import DriverConfig  # noqa: PLC0415
    from mapless.planner import PlannerConfig  # noqa: PLC0415
    from mapless.terrain import VehicleBody  # noqa: PLC0415
    from navigation_control.local import LocalControlConfig  # noqa: PLC0415
    from navigation_control.types import ControllerConfig  # noqa: PLC0415

    controller = ControllerConfig(wheelbase_m=3.00465, max_steering_angle_rad=math.radians(20), lookahead_m=4.0, kp=0.45, ki=0.15, stopping_deceleration_mps2=1.0, hold_brake=0.5)
    return DriverConfig(body=VehicleBody(), control=LocalControlConfig(controller), corridor=CorridorConfig(keep_right_m=lane_centre_m), planner=PlannerConfig(cruise_mps=cruise_mps))


@dataclass(frozen=True)
class RewardConfig:
    progress_per_m: float = 1.0
    time_per_decision: float = 0.05
    lane_free_m: float = 0.30  # truth lane offset tolerated without cost
    lane_weight: float = 4.0  # per decision, times (offset - free)^2
    lane_fail_m: float = 0.80  # truth offset ending the episode (the corner near the lane edge)
    fail_penalty: float = 30.0
    switch_penalty: float = 0.05
    goal_bonus: float = 5.0


class CruiseEnv:
    """A Gymnasium environment (``gymnasium`` imported lazily so the module loads without
    it): one episode is one random synthetic road driven to its goal by the real driver;
    one step is ``DECISION_TICKS`` control ticks under the chosen cruise target."""

    metadata = {"render_modes": []}

    def __init__(self, seed=0, reward=None, max_decisions=600, suite_path=None, fixed_action=None):
        import gymnasium as gym  # noqa: PLC0415

        from mapless.sensor_suite import load_sensor_suite  # noqa: PLC0415

        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(len(FEATURE_NAMES),), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(ACTIONS_MPS))
        self.reward_config = reward or RewardConfig()
        self.max_decisions = max_decisions
        self.rng = np.random.default_rng(seed)
        self.fixed_action = fixed_action
        from pathlib import Path  # noqa: PLC0415

        root = Path(__file__).resolve().parents[2]
        self.suite = load_sensor_suite(suite_path or root / "configs" / "sensor_suites" / "mapless_v2.json")
        self.world = self.driver = None

    # Gymnasium API
    def reset(self, seed=None, options=None):
        from mapless.driver import MaplessDriver  # noqa: PLC0415
        from sim_harness.mapless_road import KerbLine, MaplessRoadWorld, RoadScene  # noqa: PLC0415

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        spec = (options or {}).get("spec") or random_episode(self.rng, int(self.rng.integers(1 << 30)))
        self.spec = spec
        scene = RoadScene(KerbLine(list(spec.pieces)), road_width_m=spec.road_width_m, lane_centre_m=spec.lane_centre_m, parked=spec.parked)
        self.world = MaplessRoadWorld(scene, self.suite, start_s_m=10.0, seed=spec.seed)
        self.driver = MaplessDriver(self.suite, self.world.route(spec.goal_m), make_driver_config(spec.lane_centre_m), 0.05)
        self.previous = 0
        self.decisions = 0
        self.last_s = self.world.truth().s_m
        self.max_offset = 0.0
        self.time_s = 0.0
        step = self.driver.step(self.world.frame())
        self.world.apply(step.command)
        self.last_step = step
        return self._observe(), {}

    def _observe(self):
        st = self.last_step
        return features(st.diagnostics, self.driver.last_corridor, st.mode.value, self.previous)

    def step(self, action):
        action = int(self.fixed_action if self.fixed_action is not None else action)
        rc = self.reward_config
        reward = -rc.time_per_decision - (rc.switch_penalty if action != self.previous else 0.0)
        self.driver.cruise_target_mps = ACTIONS_MPS[action]
        terminated = truncated = False
        info = {}
        for _ in range(DECISION_TICKS):
            step = self.driver.step(self.world.frame())
            self.world.apply(step.command)
            self.last_step = step
            self.time_s += 0.05
            truth = self.world.truth()
            off = abs(truth.lane_offset_m)
            self.max_offset = max(self.max_offset, off)
            reward -= rc.lane_weight * max(0.0, off - rc.lane_free_m) ** 2 / DECISION_TICKS
            if truth.collision or off > rc.lane_fail_m:
                reward -= rc.fail_penalty
                terminated = True
                info["failure"] = "collision" if truth.collision else "lane"
                break
            if step.diagnostics.get("route_finished") or step.mode.value == "finished" or truth.s_m >= self.spec.goal_m - 1.0:
                reward += rc.goal_bonus
                terminated = True
                info["finished"] = True
                break
            if step.mode.value == "fault":
                reward -= rc.fail_penalty
                terminated = True
                info["failure"] = "fault: " + step.reason[:60]
                break
        progress = truth.s_m - self.last_s
        self.last_s = truth.s_m
        reward += rc.progress_per_m * progress
        self.previous = action
        self.decisions += 1
        if not terminated and self.decisions >= self.max_decisions:
            truncated = True
        info.update(time_s=round(self.time_s, 2), max_offset_m=round(self.max_offset, 3), s_m=round(truth.s_m, 1))
        return self._observe(), float(reward), terminated, truncated, info

    def close(self):
        self.world = self.driver = None


def make_env(seed, **kwargs):
    """A ``CruiseEnv`` wrapped as a ``gymnasium.Env`` subclass (for stable-baselines3)."""
    import gymnasium as gym  # noqa: PLC0415

    class _Env(gym.Env, CruiseEnv):
        def __init__(self):
            CruiseEnv.__init__(self, seed=seed, **kwargs)

        def reset(self, seed=None, options=None):
            gym.Env.reset(self, seed=seed)
            return CruiseEnv.reset(self, seed=seed, options=options)

    return _Env()


# -- the deployed policy: numpy only ------------------------------------------------------
class CruisePolicy:
    """The trained Q-network as plain arrays (``export_policy``): the live harness needs no
    torch or stable-baselines3, and inference is a few hundred multiply-adds."""

    def __init__(self, path):
        data = np.load(path, allow_pickle=False)
        self.weights = [data[f"w{i}"] for i in range(int(data["layers"]))]
        self.biases = [data[f"b{i}"] for i in range(int(data["layers"]))]
        self.features = tuple(str(x) for x in data["features"])
        self.actions = tuple(float(x) for x in data["actions"])
        if self.features != FEATURE_NAMES or self.actions != ACTIONS_MPS:
            raise ValueError("cruise policy built for other features or actions")
        self.provenance = str(data["provenance"]) if "provenance" in data.files else ""

    def q_values(self, obs):
        x = np.asarray(obs, dtype=np.float32)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w.T + b
            if i < len(self.weights) - 1:
                x = np.maximum(x, 0.0)
        return x

    def act(self, obs):
        return int(np.argmax(self.q_values(obs)))


def export_policy(model, path, provenance=""):
    """Write a stable-baselines3 DQN's Q-network (Linear/ReLU MLP) as ``CruisePolicy`` arrays."""
    import torch  # noqa: PLC0415

    linears = [m for m in model.q_net.q_net.modules() if isinstance(m, torch.nn.Linear)]
    arrays = {}
    for i, layer in enumerate(linears):
        arrays[f"w{i}"] = layer.weight.detach().cpu().numpy().astype(np.float32)
        arrays[f"b{i}"] = layer.bias.detach().cpu().numpy().astype(np.float32)
    np.savez(path, layers=len(linears), features=np.array(FEATURE_NAMES), actions=np.array(ACTIONS_MPS), provenance=np.array(provenance), **arrays)


def rule_action(obs):
    """The hand-written comparator the policy must beat: 4 m/s when the corridor is valid
    far ahead, near-straight and nothing restricts; 3 m/s on a gentle bend; else 2."""
    o = dict(zip(FEATURE_NAMES, obs))
    if o["stopping"] or o["obstacle"] < 0.7 or o["cap"] < 0.75 or o["crossing"]:
        return 0
    if o["valid"] >= 0.6 and o["curv_near"] < 0.15 and o["curv_far"] < 0.2:
        return 2
    if o["valid"] >= 0.4 and o["curv_near"] < 0.35:
        return 1
    return 0
