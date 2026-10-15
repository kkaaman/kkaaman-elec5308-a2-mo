"""Export CARLA lane topology, or validate a JSON document without CARLA."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from planning.lane_graph import LaneTopology


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="export a live map or local OpenDRIVE file")
    export.add_argument("--host", default="127.0.0.1")
    export.add_argument("--port", type=int, default=2000)
    export.add_argument("--timeout", type=float, default=10.0)
    export.add_argument("--map", default="Town10HD", help="expected live map, or local OpenDRIVE map name")
    export.add_argument("--xodr", type=Path, help="local real OpenDRIVE file; no server needed")
    export.add_argument("--sampling-distance", type=float, default=2.0)
    export.add_argument("--speed-cap", type=float, default=5.0, help="configured navigation cap in m/s")
    export.add_argument("--max-connection-gap", type=float, default=0.25)
    export.add_argument("--output", type=Path, required=True)
    commands.add_parser("validate", help="check JSON without a CARLA SDK").add_argument("path", type=Path)
    verify = commands.add_parser("verify-map", help="compare a saved graph with real CARLA map geometry")
    verify.add_argument("path", type=Path)
    verify.add_argument("--xodr", type=Path, help="local source map; otherwise use the active simulator map")
    verify.add_argument("--host", default="127.0.0.1")
    verify.add_argument("--port", type=int, default=2000)
    verify.add_argument("--timeout", type=float, default=10.0)
    audit = commands.add_parser("audit", help="independent raw OpenDRIVE audit; no CARLA SDK needed")
    audit.add_argument("path", type=Path)
    audit.add_argument("--xodr", type=Path, required=True)
    audit.add_argument("--report", type=Path, help="save detailed audit JSON")
    args = parser.parse_args(argv)
    try:
        if args.command == "audit":
            from planning.lane_graph_audit import audit_lane_topology

            document = LaneTopology.load(args.path)
            report = audit_lane_topology(args.xodr.read_text(encoding="utf-8"), document)
            result = json.dumps(report, indent=2) + "\n"
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(result, encoding="utf-8")
            print(result, end="")
            return 0
        if args.command == "verify-map":
            from carla_adapter.topology import (
                connect_map,
                load_opendrive,
                validate_lane_topology_on_map,
            )

            document = LaneTopology.load(args.path)
            map_ = load_opendrive(args.xodr, document.map_name) if args.xodr else connect_map(args.host, args.port, args.timeout, document.map_name)[0]
            print(json.dumps(validate_lane_topology_on_map(map_, document), indent=2))
            return 0
        if args.command == "validate":
            document = LaneTopology.load(args.path)
        else:
            from carla_adapter.topology import connect_map, export_lane_topology, load_opendrive

            map_ = load_opendrive(args.xodr, args.map) if args.xodr else connect_map(args.host, args.port, args.timeout, args.map)[0]
            document = export_lane_topology(map_, sampling_distance_m=args.sampling_distance,
                                            navigation_speed_cap_mps=args.speed_cap,
                                            max_connection_gap_m=args.max_connection_gap)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            document.save(args.output)
        graph = document.graph
        print(json.dumps({"map": document.map_name, "sha256": document.sha256,
                          "opendrive_sha256": document.opendrive_sha256,
                          "lanes": len(document.lanes), "nodes": len(graph.nodes), "edges": len(graph.edges),
                          "junction_lanes": sum(lane.junction_id is not None for lane in document.lanes)}, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"lane topology: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
