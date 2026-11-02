"""The baseline route and its S0–S5 obstacle scenarios on the mapless harness (G2's gate:
S0–S5 × 5 seeds on the baseline route, with no map or truth in the policy).

The simulator side resolves the frozen baseline's lane chain (``configs/missions/*.json``,
``planning.mission``) on the live map, as the baseline does, and turns it into what the
mapless car is allowed: a sparse sat-nav route (GPS points every ``spacing_m``, a STRAIGHT
at each junction's entry, LANEFOLLOW at its exit and at the goal). The obstacle of a
scenario (``configs/scenarios/S*.json``: class, route s, lateral offset, removal after a
hold, placement after the turn exit) is placed as the baseline places it
(``scenario_obstacle_s``). The route polyline is the evaluator's truth lane, never the
car's.

What differs from the baseline's runner is on purpose: the car gets no ego pose, lane
model or obstacle truth; it drives the chain on its kerbs and crosses the junction on its
own latched lane line (``mapless.crossing``).
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mapless.satnav import Command, RoutePoint, SatNavRoute

OBSTACLE_BLUEPRINTS = {"car": "vehicle.tesla.model3", "person": "walker.pedestrian.0001"}
SCENARIO_IDS = tuple(f"S{i}" for i in range(6))


@dataclass(frozen=True)
class MissionSite:
    name: str
    points_xy: np.ndarray  # the lane chain's centreline, right-handed metres (x, -carla y), ~1 m apart
    arcs: np.ndarray  # arc length at each point
    junctions: tuple  # (entry_s, exit_s, junction_id) per junction crossed, along the chain
    turn_exit_s_m: float
    lane_width_m: float
    kerb_distance_m: float | None
    kerb_source: str
    right_lanes: list
    segment_ids: tuple
    obstacle_s_m: float | None  # the route definition's own placement for S1–S4 (else the scenario's)

    @property
    def length_m(self):
        return float(self.arcs[-1])

    def at(self, s_m, lateral_m=0.0):
        """(x, y, heading) on the chain at arc length ``s_m``, ``lateral_m`` to its left."""
        s = float(np.clip(s_m, 0.0, self.length_m))
        i = int(np.clip(np.searchsorted(self.arcs, s, side="right") - 1, 0, len(self.points_xy) - 2))
        d = self.points_xy[i + 1] - self.points_xy[i]
        heading = math.atan2(d[1], d[0])
        p = self.points_xy[i] + (s - self.arcs[i]) * d / max(np.hypot(*d), 1e-9) + lateral_m * np.array([-math.sin(heading), math.cos(heading)])
        return float(p[0]), float(p[1]), heading


def load_scenario(scenario_id, scenarios_dir):
    """A baseline obstacle scenario (``configs/scenarios/S0.json`` … ``S5.json``)."""
    if scenario_id not in SCENARIO_IDS:
        raise ValueError(f"acceptance scenario is one of {', '.join(SCENARIO_IDS)}")
    path = Path(scenarios_dir) / f"{scenario_id}.yaml"
    if not path.exists():
        path = Path(scenarios_dir) / f"{scenario_id}.json"
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    if document.get("id") != scenario_id or document.get("schema_version") != 1:
        raise ValueError(f"{path}: not scenario {scenario_id}")
    return document


def kerb_across_lanes(carla, wp):
    """Distance from the lane centre to the kerb on the right, across whatever lanes lie
    between (the baseline route starts on road 1 lane 1 with lane 2 beside it; the kerb
    corridor calibrates across them, 7.8 m on the r12 runs), the lanes crossed, and the
    evidence; None past six lanes."""
    distance, crossed = wp.lane_width / 2, []
    right = wp.get_right_lane()
    for _ in range(6):
        if right is None:
            return distance, crossed, "road edge"
        if right.lane_type == carla.LaneType.Sidewalk:
            return distance, crossed, "sidewalk"
        crossed.append(str(right.lane_type).split(".")[-1])
        distance += right.lane_width
        right = right.get_right_lane()
    return None, crossed, "too many side lanes"


def resolve_mission(carla, carla_map, definition_path, spacing_m=1.0):
    """The lane chain of ``definition_path`` on the live map, as a MissionSite. Uses the
    baseline's own topology export and mission builder (simulator side; the car never
    sees them)."""
    from carla_adapter.topology import export_lane_topology  # noqa: PLC0415
    from planning.corridor import RouteGeometry  # noqa: PLC0415
    from planning.mission import load_route_definition, mission_from_definition  # noqa: PLC0415

    definition = load_route_definition(definition_path)
    topology = export_lane_topology(carla_map, sampling_distance_m=spacing_m, navigation_speed_cap_mps=5.0)
    mission = mission_from_definition(topology, definition)
    points = np.asarray([p[:2] for p in mission.points_xyz_m], dtype=float)
    route = RouteGeometry(points)
    by_id = {lane.segment_id: lane for lane in topology.lanes}
    junctions = []
    ids = list(mission.segment_ids)
    for i, key in enumerate(ids):
        lane = by_id[key]
        if lane.junction_id is None or (i > 0 and by_id[ids[i - 1]].junction_id is not None):
            continue  # not a junction lane, or the continuation of one already entered
        entry_s = route.project(*lane.points_xyz_m[0][:2]).s_m
        j = i
        while j + 1 < len(ids) and by_id[ids[j + 1]].junction_id is not None:
            j += 1
        exit_lane = by_id[ids[j + 1]] if j + 1 < len(ids) else None
        exit_s = route.length_m if exit_lane is None else route.project(*exit_lane.points_xyz_m[0][:2]).s_m
        junctions.append((float(entry_s), float(exit_s), int(lane.junction_id)))
    arcs = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))))
    x0, y0 = points[0]
    wp = carla_map.get_waypoint(carla.Location(x=float(x0), y=float(-y0), z=float(mission.points_xyz_m[0][2])), project_to_road=True, lane_type=carla.LaneType.Driving)
    kerb, crossed, source = kerb_across_lanes(carla, wp)
    summary = (mission.map_validation or {}).get("route_definition") or {}
    return MissionSite(
        name=definition["name"],
        points_xy=points,
        arcs=arcs,
        junctions=tuple(junctions),
        turn_exit_s_m=float(mission.turn_exit_s_m),
        lane_width_m=float(wp.lane_width),
        kerb_distance_m=kerb,
        kerb_source=source,
        right_lanes=crossed,
        segment_ids=tuple(ids),
        obstacle_s_m=None if summary.get("obstacle_s_m") is None else float(summary["obstacle_s_m"]),
    )


def obstacle_placement(site, scenario):
    """(s, lateral, blueprint) of the scenario's obstacle on the chain, or None (S0): the
    baseline's rule (``planning.mission.scenario_obstacle_s``): after the turn exit when the
    scenario says so, else the route definition's own placement, else the scenario's."""
    kind = scenario.get("obstacle_class")
    if kind is None:
        return None
    if scenario.get("after_turn_m") is not None:
        s = site.turn_exit_s_m + float(scenario["after_turn_m"])
    elif site.obstacle_s_m is not None:
        s = site.obstacle_s_m
    else:
        s = float(scenario["obstacle_s_m"])
    if s > site.length_m - 15.0:
        raise ValueError(f"obstacle at {s:.1f} m is too near the mission's end ({site.length_m:.1f} m)")
    return float(s), float(scenario.get("lateral_m", 0.0) or 0.0), OBSTACLE_BLUEPRINTS[kind]


def junction_command(site, entry_s, exit_s, exit_in_m=5.0, turn_min_deg=60.0):
    """The sat-nav command at a junction of the chain: LEFT or RIGHT when the chain's
    heading turns by ``turn_min_deg`` or more between the entry and ``exit_in_m`` into the
    exit lane (left positive), else STRAIGHT. Town10HD 468 on ``a1_test01`` is a crossing
    with a curving exit lane: well under the threshold, STRAIGHT as before (2 Nov)."""
    h0 = site.at(entry_s)[2]
    h1 = site.at(min(exit_s + exit_in_m, site.length_m))[2]
    turn = math.degrees((h1 - h0 + math.pi) % (2 * math.pi) - math.pi)
    if turn >= turn_min_deg:
        return Command.LEFT
    if turn <= -turn_min_deg:
        return Command.RIGHT
    return Command.STRAIGHT


def mission_route(carla_map, site, spacing_m=25.0, exit_in_m=5.0, name=None):
    """The sat-nav route for the chain: LANEFOLLOW points every ``spacing_m``, the
    junction's command at each junction's entry (``junction_command``: STRAIGHT, or LEFT /
    RIGHT where the chain turns) and LANEFOLLOW ``exit_in_m`` into its exit lane, and the
    goal at the chain's end. GPS fixes come from the map's projection, as the harness's
    other routes do."""
    import carla  # noqa: PLC0415

    marks = []  # (s, command)
    for entry_s, exit_s, _ in site.junctions:
        marks.append((entry_s, junction_command(site, entry_s, exit_s, exit_in_m)))
        marks.append((min(exit_s + exit_in_m, site.length_m - 1.0), Command.LANEFOLLOW))
    stations = []
    last = 0.0
    for s in np.arange(0.0, site.length_m, spacing_m):
        # Ordinary points keep clear of a junction's marks: none within 10 m of the entry
        # or between the entry and the exit point.
        if any(entry_s - 10.0 <= s <= exit_s + exit_in_m + 1.0 for entry_s, exit_s, _ in site.junctions):
            continue
        stations.append((float(s), Command.LANEFOLLOW))
        last = float(s)
    stations += marks
    if site.length_m - last > 0.5:
        stations.append((site.length_m, Command.LANEFOLLOW))
    stations.sort(key=lambda sc: sc[0])
    points = []
    for s, command in stations:
        x, y, _ = site.at(s)
        fix = carla_map.transform_to_geolocation(carla.Location(x=x, y=-y, z=0.0))
        points.append(RoutePoint(fix.latitude, fix.longitude, command))
    return SatNavRoute(name or site.name, tuple(points), provenance=dict(source="baseline mission lane chain", spacing_m=spacing_m, junctions=[dict(entry_s_m=round(a, 1), exit_s_m=round(b, 1), junction_id=j, command=junction_command(site, a, b, exit_in_m).value) for a, b, j in site.junctions]))


def brake_taps(log, dt_s, start_s_m=5.0, goal_m=12.0):
    """Stops the drive did not need (REPORT): runs of ``stopping`` ticks between
    ``start_s_m`` and ``goal_m`` short of the chain's end that are not an obstacle stop or
    hold or the goal — the speed cap's reason on the first tick, where, how long, how slow. r14 on
    Town10HD: "predicted conflict on the path" at s 52.8 and 71.3 in run after run, a full
    brake to a standstill for 0.3 s each; scenery the tracker read as a moving road user."""
    end_s = max(r["truth"]["s_m"] for r in log) - goal_m
    taps, current = [], None
    for r in log:
        s_m = r["truth"]["s_m"]
        tapping = r["mode"] == "stopping" and start_s_m < s_m < end_s and not r["reason"].startswith("holding") and r["reason"] not in ("goal", "obstacle")
        if tapping:
            reason = str((r.get("driver") or {}).get("speed_cap_reason") or r["reason"] or "")
            if current is None:
                current = dict(s_m=round(s_m, 1), time_s=round(r["time_s"], 1), reason=reason[:60], ticks=1, min_speed_mps=r["truth"]["speed_mps"])
            else:
                current["ticks"] += 1
                current["min_speed_mps"] = min(current["min_speed_mps"], r["truth"]["speed_mps"])
                if reason and not current["reason"]:
                    current["reason"] = reason[:60]
        elif current is not None:
            taps.append(current)
            current = None
    if current is not None:
        taps.append(current)
    for t in taps:
        t["min_speed_mps"] = round(t["min_speed_mps"], 2)
        t["seconds"] = round(t.pop("ticks") * dt_s, 2)
    return dict(count=len(taps), standstills=sum(t["min_speed_mps"] < 0.1 for t in taps), taps=taps[:12])


def score_mission(scenario, log, end, hold_ticks, removed_at_s, dt_s, goal_reach_m=3.0, hold_band_m=(1.5, 5.0), plan_band_m=(2.5, 6.0), hold_speed_max_mps=0.1, obstacle_s_m=None, roadside_window_m=(30.0, 20.0)):
    """The gates of a baseline scenario on the mapless run, as (name, passed, values) rows.
    ``passed`` None: could not be judged (INCOMPLETE).

    - S0 and S3: the goal reached, and (S3) no stop for the roadside car.
    - S1, S2 and S5: a hold behind the obstacle within the band, at a standstill; the
      obstacle removed after the hold and the goal then reached (``resume``).
    - S4: the hold, kept until the run ended (the obstacle is never removed).
    The band is the harness's obstacle-stop band (G1b); the plan's 2.5–6 m is reported
    beside it. ``obstacle_s_m`` is where the obstacle stands along the chain (the run's
    record); S3's gate counts a stop within ``roadside_window_m`` (before, after) of it —
    the start-up brake and the stop at the goal are not stops for the roadside car (r14:
    every S3 failed on those two ticks). Brake taps elsewhere are reported as
    ``brake_taps`` (r14: a predicted conflict with scenery at two spots, a full brake to a
    standstill for 0.3 s each)."""
    rows = []
    last = log[-1]
    kind = scenario.get("obstacle_class")
    removed = scenario.get("remove_after_hold_s") is not None
    goal = last["truth"]["goal_distance_m"]
    held_rows = [r for r in log if r["mode"] == "stopping" and r["reason"].startswith("holding") and r["truth"]["speed_mps"] < 0.05]
    if kind is None or float(scenario.get("lateral_m") or 0.0) >= 2.0:
        rows.append(("goal", end == "finished" and goal <= goal_reach_m, dict(end=end, goal_distance_m=goal)))
        if kind is not None:
            s_car = obstacle_s_m if obstacle_s_m is not None else scenario.get("obstacle_s_m")
            window = (s_car - roadside_window_m[0], s_car + roadside_window_m[1]) if s_car is not None else None
            near = [r for r in log if window is not None and window[0] <= r["truth"]["s_m"] <= window[1]]
            # A stop for a red light (G4 M07) is not a stop for the roadside car.
            stopped = [r for r in near if r["mode"] == "stopping" and (r.get("driver") or {}).get("light_hold_m") is None]
            speeds = [r["truth"]["speed_mps"] for r in near]
            rows.append(("no_stop_for_roadside_car", None if window is None else not stopped, dict(stopping_ticks=len(stopped), min_speed_mps=round(min(speeds), 2) if speeds else None, window_s_m=[round(w, 1) for w in window] if window else None, obstacle_s_m=s_car)))
        rows.append(("brake_taps", None, brake_taps(log, dt_s)))
        return rows
    gaps = [r["truth"]["gap_m"] for r in held_rows if r["truth"].get("gap_m") is not None]
    if not gaps:
        rows.append(("obstacle_stop", False, dict(end=end, note="never held behind the obstacle", held_s=hold_ticks * dt_s)))
    else:
        # The whole hold is judged, not its first tick (thirteenth review, 2: a stop that
        # began at 1.55 m and crept to 1.35 m passed on its first gap): every held tick's
        # gap within the band, and the car at a standstill throughout.
        gap, low, high = gaps[0], min(gaps), max(gaps)
        inside = sum(hold_band_m[0] <= g <= hold_band_m[1] for g in gaps)
        speeds = max(r["truth"]["speed_mps"] for r in held_rows)
        rows.append(
            (
                "obstacle_stop",
                hold_band_m[0] <= low and high <= hold_band_m[1] and speeds <= hold_speed_max_mps,
                dict(
                    gap_m=round(gap, 2), gap_min_m=round(low, 2), gap_max_m=round(high, 2), gap_creep_m=round(gap - low, 2), inside_band_share=round(inside / len(gaps), 3),
                    band_m=list(hold_band_m), within_plan_band=plan_band_m[0] <= low and high <= plan_band_m[1], plan_band_m=list(plan_band_m), hold_speed_max_mps=round(speeds, 3), held_s=round(len(held_rows) * dt_s, 2),
                ),
            )
        )
    if removed:
        if removed_at_s is None:
            rows.append(("resume", None if not gaps else False, dict(note="the obstacle was never removed (no hold long enough)", end=end)))
        else:
            after = [r for r in log if r["time_s"] > removed_at_s]
            moved = next((r["time_s"] for r in after if r["truth"]["speed_mps"] > 0.5), None)
            rows.append(("resume", moved is not None and end == "finished" and goal <= goal_reach_m, dict(removed_at_s=round(removed_at_s, 2), moved_off_s=None if moved is None else round(moved - removed_at_s, 2), end=end, goal_distance_m=goal)))
    else:
        rows.append(("held", end == "held at the obstacle", dict(end=end, held_s=round(hold_ticks * dt_s, 2))))
    rows.append(("brake_taps", None, brake_taps(log, dt_s)))  # r15: reported for the obstacle scenarios too
    return rows
