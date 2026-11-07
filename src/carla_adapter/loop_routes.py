"""Evaluator-side search for a long showcase mission (3 Nov): from the baseline route's start
round Town10HD to its goal area, on the lane topology alone (no simulator needed).

The car keeps its lane (no lane changes), so a route is a legal lane chain: each junction
traversed by the one exit its manoeuvre names
(``planning.junction_planner.choose_exit``), every non-junction road used once, legs of at
least ``min_leg_m`` before each junction, no point crossing the route's own earlier path,
and validated by the baseline's own builder (``planning.mission.mission_along_lanes``). It
ends where ``a1_test01`` ends (the goal area, after junction 468); from a1_test01's own
start lane only straight on through 468 is legal, so every start lane is tried. Ranked by length towards
``--target-m``, the longest straight (top speed), left and right turns, and junction count.

    python scripts/find_loop_routes.py --output <folder>

Writes ``loops.json`` (each candidate: its legs, junctions with route s, the straights) and
``loop_c<n>.json`` mission files to drive with ``--mission``. The car still gets only the
sparse sat-nav route; this file never reaches it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from planning.baseline35_corridor import RouteGeometry
from planning.junction_planner import choose_exit
from planning.lane_graph import LaneTopology
from planning.mission import mission_along_lanes

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOPOLOGY = ROOT / "tests" / "fixtures" / "Town10HD.lane_graph.json"
DEFAULT_MISSION = ROOT / "configs" / "missions" / "a1_test01.json"


def lane_length(lane):
    p = np.asarray(lane.points_xyz_m, dtype=float)[:, :2]
    return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 0.0


def _segments_cross(a, b, c, d):
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    return orient(a, b, c) * orient(a, b, d) < -1e-9 and orient(c, d, a) * orient(c, d, b) < -1e-9


def crosses(path_pts, new_pts, skip=3):
    """Whether polyline ``new_pts`` crosses ``path_pts`` (its last ``skip`` points excepted:
    they join)."""
    if len(path_pts) < skip + 2 or len(new_pts) < 2:
        return False
    old = np.asarray(path_pts[:-skip], dtype=float)
    new = np.asarray(new_pts, dtype=float)
    lo, hi = new.min(axis=0) - 1.0, new.max(axis=0) + 1.0
    keep = np.all((old >= lo) & (old <= hi), axis=1)
    idx = np.flatnonzero(keep)
    for i in idx:
        if i + 1 >= len(old):
            continue
        a, b = old[i], old[i + 1]
        for j in range(len(new) - 1):
            if _segments_cross(a, b, new[j], new[j + 1]):
                return True
    return False


def search(topology, start_id, goal_id, max_m=2600.0, min_leg_m=25.0, max_junctions=14, limit=200000):
    """Every legal chain from ``start_id`` to ``goal_id`` within the rules (module doc)."""
    by_id = {lane.segment_id: lane for lane in topology.lanes}
    road = {k: k.split(":")[0] for k in by_id}
    found, steps = [], [0]

    def leg_from(key):
        """Non-junction lanes from ``key`` on to the next junction approach (or a dead end)."""
        ids, length = [key], lane_length(by_id[key])
        while True:
            nxt = [s for s in by_id[ids[-1]].successor_ids if by_id[s].junction_id is None]
            if len(nxt) != 1 or len(by_id[ids[-1]].successor_ids) != 1 or nxt[0] in ids:
                return ids, length
            ids.append(nxt[0])
            length += lane_length(by_id[nxt[0]])

    def points_of(ids):
        out = []
        for k in ids:
            out.extend(p[:2] for p in by_id[k].points_xyz_m)
        return out

    def dfs(ids, manoeuvres, length, roads, pts, legs, junctions):
        steps[0] += 1
        if steps[0] > limit:
            return
        leg_ids, leg_m = leg_from(ids[-1])
        chain = ids + leg_ids[1:]
        if goal_id in leg_ids:
            cut = chain[: chain.index(goal_id) + 1]
            found.append(dict(segment_ids=cut, manoeuvres=list(manoeuvres), legs=legs + [round(leg_m, 1)], junctions=list(junctions), length_m=round(length + leg_m, 1)))
            return
        if length + leg_m > max_m or len(manoeuvres) >= max_junctions or (legs and leg_m < min_leg_m):
            return
        last = chain[-1]
        if not by_id[last].successor_ids:
            return
        for manoeuvre in ("straight", "left", "right"):
            try:
                exit_ = choose_exit(topology, last, manoeuvre)
            except ValueError:
                continue
            through = list(exit_.segment_ids[1:])  # connectors ..., the exit lane
            exit_lane = through[-1]
            if road[exit_lane] in roads:
                continue
            new_pts = points_of(through)
            here = pts + points_of(leg_ids[1:])
            if crosses(here, new_pts):
                continue
            jid = by_id[through[0]].junction_id
            dfs(chain + through, manoeuvres + [manoeuvre], length + leg_m + sum(lane_length(by_id[k]) for k in through[:-1]),
                roads | {road[exit_lane]}, here + new_pts, legs + [round(leg_m, 1)], junctions + [(jid, manoeuvre, round(math.degrees(exit_.heading_change_rad), 1))])

    first = by_id[start_id]
    dfs([start_id], [], 0.0, {road[start_id]}, [p[:2] for p in first.points_xyz_m], [], [])
    return found, steps[0]


def describe(topology, candidate, definition):
    """The validated mission's junctions (route s), straights and length, or None if the
    baseline's builder refuses it."""
    try:
        mission = mission_along_lanes(topology, candidate["segment_ids"], candidate["manoeuvres"], start_xy_m=tuple(definition["start_xy_m"]), end_xy_m=tuple(definition["end_xy_m"]))
    except ValueError as error:
        return None, str(error)
    by_id = {lane.segment_id: lane for lane in topology.lanes}
    pts = np.asarray([p[:2] for p in mission.points_xyz_m], dtype=float)
    route = RouteGeometry(pts)
    ids = list(mission.segment_ids)
    junctions = []
    for i, key in enumerate(ids):
        lane = by_id[key]
        if lane.junction_id is None or (i > 0 and by_id[ids[i - 1]].junction_id is not None):
            continue
        j = i
        while j + 1 < len(ids) and by_id[ids[j + 1]].junction_id is not None:
            j += 1
        entry = route.project(*lane.points_xyz_m[0][:2]).s_m
        exit_ = route.project(*by_id[ids[j + 1]].points_xyz_m[0][:2]).s_m if j + 1 < len(ids) else route.length_m
        junctions.append(dict(junction_id=lane.junction_id, entry_s_m=round(entry, 1), exit_s_m=round(exit_, 1)))
    for j, (jid, manoeuvre, turn) in zip(junctions, candidate["junctions"]):
        j.update(manoeuvre=manoeuvre, turn_deg=turn)
    # Straights between junctions, by route s (the curvature small throughout).
    bounds = [0.0] + [x for j in junctions for x in (j["entry_s_m"], j["exit_s_m"])] + [route.length_m]
    legs = [(bounds[k], bounds[k + 1]) for k in range(0, len(bounds) - 1, 2)]
    heading = np.unwrap(np.arctan2(*np.diff(pts, axis=0).T[::-1]))
    arcs = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))))
    straights = []
    for a, b in legs:
        sel = (arcs[:-1] >= a) & (arcs[:-1] <= b)
        if sel.sum() < 3:
            continue
        bend = float(np.degrees(np.ptp(heading[sel])))
        straights.append(dict(from_s_m=round(a, 1), to_s_m=round(b, 1), length_m=round(b - a, 1), bend_deg=round(bend, 1)))
    return dict(length_m=round(route.length_m, 1), junctions=junctions, legs=straights, longest_leg_m=max((s["length_m"] for s in straights if s["bend_deg"] < 20), default=0.0)), ""


def score(info, target_m):
    turns = [j for j in info["junctions"] if j["manoeuvre"] != "straight"]
    kinds = {j["manoeuvre"] for j in turns}
    return (
        -abs(info["length_m"] - target_m) / 100.0
        + info["longest_leg_m"] / 40.0
        + min(len(turns), 4)
        + 2.0 * ("left" in kinds and "right" in kinds)
        + 0.3 * len(info["junctions"])
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY, help="a LaneTopology document (the repo's Town10HD export by default)")
    p.add_argument("--mission", type=Path, default=DEFAULT_MISSION, help="the route whose start and goal the loop keeps")
    p.add_argument("--target-m", type=float, default=1200.0)
    p.add_argument("--max-m", type=float, default=2600.0)
    p.add_argument("--min-leg-m", type=float, default=8.0, help="the shortest lane between two junctions (Town10HD has 9-11 m between paired junctions)")
    p.add_argument("--start", default="any", help="the start lane (road:section:lane), or any: every non-junction lane (the car keeps its lane, so from a1_test01's own start lane only straight on through 468 is legal)")
    p.add_argument("--keep", type=int, default=5, help="mission files written for the best this many")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    topology = LaneTopology.load(args.topology)
    definition = json.loads(args.mission.read_text(encoding="utf-8"))
    goal_id = definition["segment_ids"][-1]
    by_id = {lane.segment_id: lane for lane in topology.lanes}
    starts = [lane.segment_id for lane in topology.lanes if lane.junction_id is None] if args.start == "any" else [args.start]
    ranked = []
    for start_id in starts:
        found, steps = search(topology, start_id, goal_id, max_m=args.max_m, min_leg_m=args.min_leg_m)
        for c in found:
            here = dict(definition)
            if start_id != definition["segment_ids"][0]:
                pts = by_id[start_id].points_xyz_m
                here["start_xy_m"] = list(pts[min(2, len(pts) - 1)][:2])  # a few metres into the lane
            info, why = describe(topology, c, here)
            if info is not None and info["length_m"] >= 300.0:
                ranked.append(dict(score=round(score(info, args.target_m), 2), start_lane=start_id, start_xy_m=[round(v, 2) for v in here["start_xy_m"]], **info, segment_ids=c["segment_ids"], manoeuvres=c["manoeuvres"]))
    print(f"{len(ranked)} routes to {goal_id} of 300 m or more", file=sys.stderr)
    ranked.sort(key=lambda r: -r["score"])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "loops.json").write_text(json.dumps(ranked, indent=1), encoding="utf-8")
    for i, r in enumerate(ranked[: args.keep]):
        name = f"loop_c{i + 1}"
        doc = dict(
            schema_version=1, name=name,
            description=f"Round Town10HD from {definition['name']}'s start to its goal area, {r['length_m']:.0f} m: " + ", ".join(f"{j['manoeuvre']} at {j['junction_id']}" for j in r["junctions"]),
            source="scripts/find_loop_routes.py (evaluator-side lane topology)",
            segment_ids=r["segment_ids"], manoeuvres=r["manoeuvres"],
            start_xy_m=r["start_xy_m"], end_xy_m=definition["end_xy_m"],
        )
        (args.output / f"{name}.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    for i, r in enumerate(ranked[:10]):
        print(f"#{i + 1} score {r['score']:6.2f}  from {r['start_lane']} {r['length_m']:6.0f} m  longest straight {r['longest_leg_m']:.0f} m  " + " ".join(f"[{j['junction_id']} {j['manoeuvre']} s{j['entry_s_m']:.0f}]" for j in r["junctions"]))
    return ranked


if __name__ == "__main__":
    main()
