"""Validate the frozen route and optionally export an annotated map overlay."""

import argparse
import json
import math
from html import escape
from pathlib import Path

from planning.baseline_mission import load_frozen_mission
from planning.lane_graph import LaneTopology


def route_overlay(mission, topology):
    """Return a vector map with lane context and the frozen rear-axle reference."""
    all_points = [p for lane in topology.lanes for p in lane.points_xyz_m]
    lo_x, hi_x = min(p[0] for p in all_points), max(p[0] for p in all_points)
    lo_y, hi_y = min(p[1] for p in all_points), max(p[1] for p in all_points)
    width, height, margin = 1050, 900, 70
    scale = min((width - 2 * margin) / (hi_x - lo_x), (height - 2 * margin) / (hi_y - lo_y))

    def xy(point):
        return margin + (point[0] - lo_x) * scale, height - margin - (point[1] - lo_y) * scale

    def line(points, colour, thickness):
        vertices = " ".join(f"{x:.2f},{y:.2f}" for x, y in map(xy, points))
        return f'<polyline points="{vertices}" fill="none" stroke="{colour}" stroke-width="{thickness}"/>'

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="Arial" fill="#17263b">',
             '<text x="30" y="30" font-size="22">Town10HD frozen S0 mission</text>',
             f'<text x="30" y="53" font-size="13">{mission.length_m:.1f} m | rear-axle reference | +Y left | route {mission.route_sha256[:12]}</text>']
    parts.extend(line(lane.points_xyz_m, "#cbd3dd", 1) for lane in topology.lanes)
    parts.append(line(mission.route["points_xyz_m"], "#1766b3", 4))
    for point, label, colour in [(mission.route["points_xyz_m"][0], "START", "#17804b"), (mission.route["goal_xyz_m"], "GOAL", "#b13232")]:
        x, y = xy(point)
        parts.extend([f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" fill="{colour}"/>', f'<text x="{x + 12:.2f}" y="{y - 10:.2f}" font-size="15" fill="{colour}">{label}</text>'])
    by_id = {lane.segment_id: lane for lane in topology.lanes}
    for index, key in enumerate(mission.route["segment_ids"]):
        lane = by_id[key]
        x, y = xy(lane.points_xyz_m[len(lane.points_xyz_m) // 2])
        parts.append(f'<text x="{x + 8:.2f}" y="{y + 18:.2f}" font-size="12">{index + 1}: {escape(key)}</text>')
    parts.append('</g></svg>')
    return "\n".join(parts) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mission", type=Path, default=Path("configs/mission_baseline.json"))
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--verify-map", action="store_true")
    parser.add_argument("--xodr", type=Path)
    args = parser.parse_args(argv)
    mission = load_frozen_mission(args.mission)
    topology = LaneTopology.load(args.mission.resolve().parent / mission.config["topology_file"])
    map_report = None
    if args.verify_map or args.xodr:
        from carla_adapter.topology import (
            connect_map,
            load_opendrive,
            validate_lane_topology_on_map,
        )

        map_ = load_opendrive(args.xodr, mission.map_name) if args.xodr else connect_map("127.0.0.1", 2000, 30, mission.map_name)[0]
        map_report = validate_lane_topology_on_map(map_, topology)
    if args.overlay:
        args.overlay.parent.mkdir(parents=True, exist_ok=True)
        args.overlay.write_text(route_overlay(mission, topology), encoding="utf-8")
    print(json.dumps({"status": "PASS", "scope": "frozen route geometry and identity; not driving acceptance", "length_m": mission.length_m,
                      "route_sha256": mission.route_sha256, "config_sha256": mission.config_sha256, "segment_ids": mission.route["segment_ids"],
                      "map_validation": map_report, "spawn_yaw_deg": math.degrees(mission.spawn_pose["yaw_rad"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
