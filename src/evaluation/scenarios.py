"""Deterministic S1-S5 obstacle scenarios (BL-B-06). Simulator-free and evaluation-only.

A scenario file fixes everything that decides where and when its obstacle exists:
- the obstacle blueprint;
- its pose, as route arc length ``route_s_m`` and signed lateral offset ``lateral_m``
  (positive = left of travel) on the frozen mission route;
- its removal rule.

Placement is a pure function of the route and the file, so a clean run reproduces
the same pose. Removal is a pure function of simulation time and true ego speed,
so it happens at the same point of every run that behaves the same.

The seeds are repeatability labels; the scripted world uses no random draws.
Nothing here is imported by perception, planning or control.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

SCENARIO_IDS = ("S1", "S2", "S3", "S4", "S5")
OBSTACLE_KINDS = {"vehicle": "vehicle.", "pedestrian": "walker.pedestrian."}
EXPECTED_BEHAVIOURS = ("stop_hold_resume", "no_stop", "persistent_hold")
REMOVAL_RULES = ("after_steady_hold", "never")
STOPPED_SPEED_MPS = 0.1  # docs/acceptance.md: stopped means speed <= 0.1 m/s


def _unique_keys(pairs):
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate key in scenario file")
    return dict(pairs)


def _finite(name, value, *, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return float(value)


def _exact_keys(name, mapping, required, optional=()):
    if not isinstance(mapping, dict):
        raise ValueError(f"{name} must be an object")
    missing = set(required) - set(mapping)
    unknown = set(mapping) - set(required) - set(optional)
    if missing or unknown:
        raise ValueError(f"{name}: missing {sorted(missing)}, unknown {sorted(unknown)}")


@dataclass(frozen=True)
class Obstacle:
    kind: str
    blueprint: str
    route_s_m: float
    lateral_m: float
    yaw_offset_deg: float


@dataclass(frozen=True)
class Removal:
    rule: str
    steady_hold_s: float | None = None


@dataclass(frozen=True)
class ScenarioDefinition:
    scenario_id: str
    description: str
    mission_config: str
    obstacle: Obstacle
    expected_behaviour: str
    removal: Removal
    max_episode_s: float
    seeds: tuple
    sha256: str


def load_scenario(path):
    """Read and validate one scenario file; returns a frozen definition with its hash."""
    path = Path(path)
    raw = path.read_bytes()
    document = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_keys)
    _exact_keys(
        str(path.name),
        document,
        (
            "schema_version",
            "scenario_id",
            "description",
            "mission_config",
            "obstacle",
            "expected_behaviour",
            "removal",
            "max_episode_s",
            "seeds",
        ),
    )
    if document["schema_version"] != 1:
        raise ValueError("Unsupported scenario schema")
    scenario_id = document["scenario_id"]
    if scenario_id not in SCENARIO_IDS or path.stem != scenario_id:
        raise ValueError("Scenario id must be S1-S5 and match the file name")
    if not str(document["description"]).strip():
        raise ValueError("Scenario needs a description")

    o = document["obstacle"]
    _exact_keys("obstacle", o, ("kind", "blueprint", "route_s_m", "lateral_m", "yaw_offset_deg"))
    if o["kind"] not in OBSTACLE_KINDS or not str(o["blueprint"]).startswith(
        OBSTACLE_KINDS[o["kind"]]
    ):
        raise ValueError("Obstacle kind and blueprint disagree")
    obstacle = Obstacle(
        o["kind"],
        o["blueprint"],
        _finite("route_s_m", o["route_s_m"], minimum=0.0),
        _finite("lateral_m", o["lateral_m"]),
        _finite("yaw_offset_deg", o["yaw_offset_deg"]),
    )

    r = document["removal"]
    if not isinstance(r, dict) or r.get("rule") not in REMOVAL_RULES:
        raise ValueError(f"removal.rule must be one of {REMOVAL_RULES}")
    if r["rule"] == "after_steady_hold":
        _exact_keys("removal", r, ("rule", "steady_hold_s"))
        removal = Removal(
            "after_steady_hold", _finite("steady_hold_s", r["steady_hold_s"], minimum=0.05)
        )
    else:
        _exact_keys("removal", r, ("rule",))
        removal = Removal("never")

    expected = document["expected_behaviour"]
    if expected not in EXPECTED_BEHAVIOURS:
        raise ValueError(f"expected_behaviour must be one of {EXPECTED_BEHAVIOURS}")
    if (expected == "stop_hold_resume") != (removal.rule == "after_steady_hold"):
        raise ValueError("Only stop_hold_resume scenarios remove their obstacle")

    seeds = document["seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("seeds must be distinct non-negative integers")
    return ScenarioDefinition(
        scenario_id,
        document["description"],
        document["mission_config"],
        obstacle,
        expected,
        removal,
        _finite("max_episode_s", document["max_episode_s"], minimum=1.0),
        tuple(seeds),
        hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
    )


def load_scenarios(directory):
    """All five scenarios from a directory, keyed by id."""
    directory = Path(directory)
    return {sid: load_scenario(directory / f"{sid}.json") for sid in SCENARIO_IDS}


def route_heading(path, progress_m):
    """Heading (rad, core frame) of the route segment that contains ``progress_m``."""
    s = min(max(progress_m, 0.0), path.length_m)
    index = 0
    while index < len(path.points) - 2 and path.arc_lengths[index + 1] <= s:
        index += 1
    a, b = path.points[index], path.points[index + 1]
    return math.atan2(b.y_m - a.y_m, b.x_m - a.x_m)


def placement_pose(path, obstacle):
    """Obstacle actor-origin pose (x, y, yaw in the core frame) from its route coordinates."""
    if not 0.0 <= obstacle.route_s_m <= path.length_m:
        raise ValueError("Obstacle route_s_m lies outside the mission route")
    centre = path.point_at(obstacle.route_s_m)
    heading = route_heading(path, obstacle.route_s_m)
    x = centre.x_m - obstacle.lateral_m * math.sin(heading)
    y = centre.y_m + obstacle.lateral_m * math.cos(heading)
    yaw = math.atan2(
        math.sin(heading + math.radians(obstacle.yaw_offset_deg)),
        math.cos(heading + math.radians(obstacle.yaw_offset_deg)),
    )
    return x, y, yaw


HOLD_EVIDENCE_MAX_GAP_M = 15.0  # an obstacle further ahead than this is not what the ego holds for


def obstacle_hold_evidence(truth, max_gap_m=HOLD_EVIDENCE_MAX_GAP_M):
    """Whether a per-tick truth record (``ego_truth``) shows an obstacle the ego could be
    holding for: present, in the ego's corridor and within ``max_gap_m`` ahead."""
    gap = truth.get("truth_front_gap_m")
    return bool(truth.get("truth_in_corridor")) and gap is not None and -1e-9 <= gap <= max_gap_m


class RemovalSchedule:
    """Decide the scripted removal tick from simulation time, true ego speed and the
    obstacle-in-path evidence.

    ``after_steady_hold``: remove on the first tick at which the ego has been at or
    below ``STOPPED_SPEED_MPS`` for ``steady_hold_s`` of continuous simulation time
    **while the obstacle is in its path** (``obstacle_in_path``, from
    ``obstacle_hold_evidence`` on the evaluation truth). Any tick without that
    evidence restarts the hold, so the standstill at mission start, or a fault stop
    before the obstacle, never removes it (review of PR #98). A tick gap larger
    than 1.5 * dt restarts the hold too, so a dropped frame cannot shorten it.
    ``never``: the obstacle stays for the whole episode.

    Evaluation-only: it reads the true speed and truth and is applied by the world,
    never by the autonomy stack.
    """

    def __init__(self, removal, dt_s):
        self.removal = removal
        self.dt_s = _finite("dt_s", dt_s, minimum=1e-6)
        self.hold_since_s = None
        self.previous_s = None
        self.removed_at_s = None

    def update(self, sim_time_s, speed_mps, obstacle_in_path):
        """Return True exactly once, on the removal tick."""
        _finite("sim_time_s", sim_time_s)
        _finite("speed_mps", speed_mps)
        if not isinstance(obstacle_in_path, bool):
            raise TypeError("obstacle_in_path must be a bool (obstacle_hold_evidence on the truth)")
        if self.removal.rule == "never" or self.removed_at_s is not None:
            return False
        if self.previous_s is not None and sim_time_s - self.previous_s > 1.5 * self.dt_s:
            self.hold_since_s = None
        self.previous_s = sim_time_s
        if abs(speed_mps) > STOPPED_SPEED_MPS or not obstacle_in_path:
            self.hold_since_s = None
            return False
        if self.hold_since_s is None:
            self.hold_since_s = sim_time_s
        if sim_time_s - self.hold_since_s >= self.removal.steady_hold_s - 1e-9:
            self.removed_at_s = sim_time_s
            return True
        return False
