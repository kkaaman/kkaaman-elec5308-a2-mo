"""Evaluator-side route search for the G3 showcase (2 Nov): continuations of a mission's
lane chain through the next junctions, with each junction's turn, its traffic lights and
the straight road around it, and where a right-side parking lane or shoulder could hold a
parallel-parking bay. CARLA's topology builds and scores the route; it never reaches the
car (which gets the sparse sat-nav route only).

    python scripts/find_showcase_routes.py --map Town10HD_Opt --mission configs/missions/a1_test01.json --junctions 2 --output <folder>

Writes ``candidates.json`` (every chain found, best first) and ``<name>.json`` mission
files for the best few, in the mission schema (``segment_ids`` road:section:lane,
``manoeuvres`` per junction), to drive with ``--mission``.
"""

import argparse
import json
import sys
from pathlib import Path


def heading_deg(transform):
    return transform.rotation.yaw


def wrap(angle):
    return (angle + 180.0) % 360.0 - 180.0


def segment_id(wp):
    return f"{wp.road_id}:{wp.section_id}:{wp.lane_id}"


def follow_lane(wp, step_m=1.0, max_m=400.0):
    """Waypoints along the lane from ``wp`` to the next junction entry (exclusive), or
    ``max_m``. Returns (points, the last waypoint before the junction or None)."""
    pts = [wp]
    travelled = 0.0
    while travelled < max_m:
        nxt = pts[-1].next(step_m)
        if not nxt:
            return pts, None
        n = nxt[0]
        if n.is_junction:
            return pts, n
        pts.append(n)
        travelled += step_m
    return pts, None


def through_junction(entry, step_m=1.0):
    """Every way through the junction from its first waypoint ``entry``: lists of
    waypoints from the entry to the first waypoint after the junction."""
    out = []
    for start in [entry] + [w for w in entry.next(0.5) if w.is_junction and w.road_id != entry.road_id]:
        path = [start]
        for _ in range(400):
            nxt = path[-1].next(step_m)
            if not nxt:
                break
            path.append(nxt[0])
            if not nxt[0].is_junction:
                break
        if not path[-1].is_junction:
            out.append(path)
    # distinct by connecting road
    seen, unique = set(), []
    for p in out:
        key = p[0].road_id
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def branches(last_before, step_m=1.0):
    """The junction's exits reachable from the lane ending at ``last_before``."""
    paths = []
    for first in last_before.next(step_m * 2):
        if first.is_junction:
            paths.extend(through_junction(first, step_m))
    seen, unique = set(), []
    for p in paths:
        key = (p[0].road_id, p[-1].road_id, p[-1].lane_id)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def right_side(wp):
    """Lane types to the right of ``wp`` up to the kerb: Parking / Shoulder / Driving..."""
    kinds = []
    r = wp.get_right_lane()
    for _ in range(4):
        if r is None or r.lane_id * wp.lane_id < 0:
            break
        kinds.append(str(r.lane_type))
        r = r.get_right_lane()
    return kinds


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--map", default="Town10HD_Opt")
    p.add_argument("--mission", type=Path, default=Path("configs/missions/a1_test01.json"))
    p.add_argument("--junctions", type=int, default=2, help="how many more junctions to pass")
    p.add_argument("--min-leg-m", type=float, default=35.0, help="the shortest straight lane before and after a turn")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    args = p.parse_args(argv)
    import carla  # noqa: PLC0415

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.get_world()
    if args.map.split("/")[-1] not in world.get_map().name:
        world = client.load_world(args.map)
    cmap = world.get_map()
    mission = json.loads(args.mission.read_text(encoding="utf-8"))
    ex, ey = mission["end_xy_m"]
    end = cmap.get_waypoint(carla.Location(x=ex, y=-ey, z=0.0), project_to_road=True, lane_type=carla.LaneType.Driving)
    print(f"mission ends on {segment_id(end)} at ({end.transform.location.x:.1f}, {end.transform.location.y:.1f}), heading {heading_deg(end.transform):.0f}", file=sys.stderr)

    def lights_in(junction_id):
        try:
            return len(world.get_traffic_lights_in_junction(junction_id))
        except Exception:  # noqa: BLE001 - older API
            return None

    chains = []

    def extend(wp, legs, segs, depth):
        pts, last_lane_wp = follow_lane(wp)
        leg_m = len(pts) - 1.0
        here_segs = segs + [s for s in dict.fromkeys(segment_id(w) for w in pts) if s not in segs]
        parking = sorted({k for w in pts[:: max(1, len(pts) // 8)] for k in right_side(w)})
        leg = dict(length_m=leg_m, segments=list(dict.fromkeys(segment_id(w) for w in pts)), right_side=parking, start=[round(pts[0].transform.location.x, 1), round(pts[0].transform.location.y, 1)])
        if depth == args.junctions or last_lane_wp is None:
            chains.append(dict(legs=legs + [leg], segment_ids=here_segs, end=[round(pts[-1].transform.location.x, 1), round(pts[-1].transform.location.y, 1)]))
            return
        for path in branches(pts[-1]):
            turn = wrap(heading_deg(path[-1].transform) - heading_deg(pts[-1].transform))
            kind = "straight" if abs(turn) < 30 else ("left" if turn < 0 else "right")  # CARLA yaw: clockwise positive (left-handed frame)
            junction = path[0].get_junction()
            jleg = dict(junction_id=junction.id if junction else None, turn_deg=round(turn, 1), manoeuvre=kind, lights=lights_in(junction.id) if junction else None, connector=segment_id(path[0]), approach_m=leg_m)
            extend(path[-1], legs + [leg, jleg], here_segs + [segment_id(path[0])], depth + 1)

    extend(end, [], [], 0)

    def score(chain):
        js = [leg for leg in chain["legs"] if "junction_id" in leg]
        lanes = [leg for leg in chain["legs"] if "length_m" in leg]
        turns = sum(1 for j in js if j["manoeuvre"] != "straight")
        squareish = sum(1 for j in js if j["manoeuvre"] != "straight" and 70 <= abs(j["turn_deg"]) <= 110)
        short = sum(1 for leg in lanes[:-1] if leg["length_m"] < args.min_leg_m)
        kinds = {j["manoeuvre"] for j in js}
        parking = any("Parking" in k or "Shoulder" in k for k in lanes[-1]["right_side"])
        return (squareish * 3 + turns + ("left" in kinds and "right" in kinds) * 2 + parking * 2 - short * 4, lanes[-1]["length_m"])

    chains.sort(key=score, reverse=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "candidates.json").write_text(json.dumps([dict(score=score(c)[0], **c) for c in chains], indent=1), encoding="utf-8")
    for i, c in enumerate(chains[:5]):
        js = [leg for leg in c["legs"] if "junction_id" in leg]
        name = f"showcase_c{i + 1}"
        definition = dict(
            schema_version=1, name=name,
            description=f"{mission['name']} continued through {len(js)} junction(s): " + ", ".join(f"{j['manoeuvre']} {abs(j['turn_deg']):.0f} deg at {j['junction_id']} (lights {j['lights']})" for j in js),
            source="scripts/find_showcase_routes.py (evaluator-side topology)",
            segment_ids=list(dict.fromkeys(mission["segment_ids"] + c["segment_ids"])),
            manoeuvres=list(mission.get("manoeuvres", [])) + [j["manoeuvre"] for j in js],
            start_xy_m=mission["start_xy_m"],
            end_xy_m=[c["end"][0], -c["end"][1]],
        )
        (args.output / f"{name}.json").write_text(json.dumps(definition, indent=2), encoding="utf-8")
    for i, c in enumerate(chains[:12]):
        js = [leg for leg in c["legs"] if "junction_id" in leg]
        lanes = [leg for leg in c["legs"] if "length_m" in leg]
        print(f"#{i + 1} score {score(c)[0]:>3}  " + "  ->  ".join(
            [f"{lanes[0]['length_m']:.0f} m"] + [f"[J{j['junction_id']} {j['manoeuvre']} {j['turn_deg']:+.0f} lights {j['lights']}] {lanes[k + 1]['length_m']:.0f} m" for k, j in enumerate(js)]
        ) + f"   last leg right side: {lanes[-1]['right_side']}")
    print(f"{len(chains)} chains; written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
