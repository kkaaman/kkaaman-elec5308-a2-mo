"""Read-only live map audit of legal junction routes; no driving acceptance."""

import argparse
import json
from collections import Counter
from pathlib import Path

from planning.junction_planner import choose_exit, junction_exits
from planning.route_manager import stitch_route


def main():
    from carla_adapter.topology import connect_map, export_lane_topology

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    map_, version = connect_map("127.0.0.1", 2000, 60, "Town10HD_Opt")
    topology = export_lane_topology(map_)
    args.output.mkdir(parents=True, exist_ok=False)
    topology.save(args.output / "topology.json")
    by_id = {lane.segment_id: lane for lane in topology.lanes}
    accepted, rejected = [], []
    for lane in topology.lanes:
        if lane.junction_id is not None or not any(
            by_id[key].junction_id is not None for key in lane.successor_ids
        ):
            continue
        try:
            exits = junction_exits(topology, lane.segment_id)
        except ValueError as error:
            rejected.append(dict(approach=lane.segment_id, reason=str(error)))
            continue
        for maneuver in sorted({exit.maneuver for exit in exits}):
            try:
                route = choose_exit(topology, lane.segment_id, maneuver)
                points = stitch_route(topology, route.segment_ids)
                accepted.append(
                    dict(
                        approach=lane.segment_id,
                        maneuver=maneuver,
                        segments=route.segment_ids,
                        points=len(points),
                    )
                )
            except ValueError as error:
                rejected.append(
                    dict(approach=lane.segment_id, maneuver=maneuver, reason=str(error))
                )
    report = dict(
        status="MEASURED_REVIEW_REQUIRED",
        server=version,
        map=topology.map_name,
        topology_sha256=topology.sha256,
        scope="read-only installed-map legal route geometry; unsupported or ambiguous configurations rejected; no vehicle driven",
        accepted=accepted,
        rejected=rejected,
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            dict(
                status=report["status"],
                accepted=len(accepted),
                rejected=len(rejected),
                maneuvers=dict(Counter(row["maneuver"] for row in accepted)),
                output=str(args.output),
            ),
            indent=2,
        )
    )
    if not accepted:
        raise RuntimeError("No usable live-map junction routes; inspect rejection report")


if __name__ == "__main__":
    main()
