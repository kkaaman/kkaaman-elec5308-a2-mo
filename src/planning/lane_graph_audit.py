"""Independent OpenDRIVE audit of Town10HD graph data, without CARLA.

This intentionally does not call the exporter or CARLA Waypoint queries.
Road/lane/junction links define adjacency; line/arc geometry and lane-width
polynomials define centre positions. Unsupported map features fail explicitly.
"""
from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from collections import Counter

from planning.lane_graph import LaneTopology, lane_segment_id


def _require(condition, message):
    if not condition:
        raise ValueError(f"OpenDRIVE audit: {message}")


def _active(records, coordinate, attribute="s"):
    matches = [record for record in records if float(record.attrib[attribute]) <= coordinate + 1e-10]
    _require(matches, f"missing polynomial/geometry at {coordinate}")
    return max(matches, key=lambda record: float(record.attrib[attribute]))


def _polynomial(records, coordinate, attribute="s"):
    record = _active(records, coordinate, attribute)
    ds = coordinate - float(record.attrib[attribute])
    return sum(float(record.attrib.get(c, 0)) * ds**i for i, c in enumerate("abcd"))


def _centre(road, section, lane_id, s):
    geometry = _active(road.findall("planView/geometry"), s)
    ds = s - float(geometry.attrib["s"])
    x, y, heading = (float(geometry.attrib[k]) for k in ("x", "y", "hdg"))
    if geometry.find("line") is not None:
        x += ds * math.cos(heading)
        y += ds * math.sin(heading)
    else:
        arc = geometry.find("arc")
        _require(arc is not None, "only line/arc geometry is supported")
        curvature = float(arc.attrib["curvature"])
        _require(curvature != 0, "zero-curvature arc")
        end_heading = heading + curvature * ds
        x += (math.sin(end_heading) - math.sin(heading)) / curvature
        y -= (math.cos(end_heading) - math.cos(heading)) / curvature
        heading = end_heading
    offsets = road.findall("lanes/laneOffset")
    lateral = _polynomial(offsets, s) if offsets else 0.0
    relative_s = s - float(section.attrib["s"])
    side = "left" if lane_id > 0 else "right"
    lane_nodes = {int(lane.attrib["id"]): lane for lane in section.findall(f"{side}/lane")}
    width = None
    for index in range(1, abs(lane_id) + 1):
        key = index if lane_id > 0 else -index
        _require(key in lane_nodes, f"missing intermediate lane {key}")
        width = _polynomial(lane_nodes[key].findall("width"), relative_s, "sOffset")
        _require(width > 0, f"nonpositive lane width on road {road.attrib['id']}")
        lateral += (1 if lane_id > 0 else -1) * width * (0.5 if key == lane_id else 1)
    # Export Y reflection cancels CARLA's reflection of OpenDRIVE Y.
    return (x - lateral * math.sin(heading), y + lateral * math.cos(heading), 0.0), width


def _distance_to_segment(point, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    fraction = max(0, min(1, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / (dx * dx + dy * dy)))
    return math.hypot(point[0] - a[0] - fraction * dx, point[1] - a[1] - fraction * dy)


def _reachable(adjacency, start):
    seen, pending = {start}, [start]
    while pending:
        for target in adjacency[pending.pop()]:
            if target not in seen:
                seen.add(target)
                pending.append(target)
    return seen


def audit_lane_topology(opendrive: str, document: LaneTopology, *, dense_step_m: float = 0.25) -> dict:
    """Audit the saved document against raw flat, right-hand-traffic OpenDRIVE.

    Dense probes measure polyline approximation, not just sample coordinates.
    Lane width checks concern the lane centre path; vehicle footprint/swept
    clearance remains the responsibility of the corridor/planner workstream.
    """
    _require(isinstance(dense_step_m, (int, float)) and not isinstance(dense_step_m, bool)
             and math.isfinite(dense_step_m) and 0 < dense_step_m <= 0.25, "dense step must be in (0, 0.25] metres")
    source_hash = hashlib.sha256(opendrive.replace("\r\n", "\n").encode("utf-8")).hexdigest()
    _require(source_hash == document.opendrive_sha256, "source map hash differs")
    try:
        root = ET.fromstring(opendrive)
    except ET.ParseError as exc:
        raise ValueError(f"OpenDRIVE audit: malformed XML: {exc}") from exc
    roads = {int(road.attrib["id"]): road for road in root.findall("road")}
    junctions = {int(j.attrib["id"]): j for j in root.findall("junction")}
    definitions = {}
    for road_id, road in roads.items():
        _require(road.attrib.get("rule", "RHT") == "RHT", "LHT maps are not supported by this independent audit")
        for record in road.findall("elevationProfile/elevation") + road.findall("lateralProfile/superelevation"):
            _require(all(float(record.attrib.get(c, 0)) == 0 for c in "abcd"), "non-flat/banked maps need a 3D audit")
        _require(all(g.find("line") is not None or g.find("arc") is not None for g in road.findall("planView/geometry")),
                 "unsupported reference-line geometry")
        sections = road.findall("lanes/laneSection")
        for index, section in enumerate(sections):
            for lane in section.findall("./*/lane"):
                if lane.attrib.get("type") == "driving":
                    key = lane_segment_id(road_id, index, int(lane.attrib["id"]))
                    definitions[key] = (road, sections, index, lane)
    saved = {lane.segment_id: lane for lane in document.lanes}
    _require(set(saved) == set(definitions), "driving lane coverage differs")
    xml_adjacency = {}
    for key, (road, sections, index, lane) in definitions.items():
        road_id, lane_id = int(road.attrib["id"]), int(lane.attrib["id"])
        forward = lane_id < 0
        link_name = "successor" if forward else "predecessor"
        lane_link = lane.find(f"link/{link_name}")
        targets = set()
        next_index = index + (1 if forward else -1)
        if 0 <= next_index < len(sections):
            if lane_link is not None:
                targets.add(lane_segment_id(road_id, next_index, int(lane_link.attrib["id"])))
        else:
            road_link = road.find(f"link/{link_name}")
            if road_link is not None and road_link.attrib["elementType"] == "road":
                if lane_link is not None:
                    target_road = int(road_link.attrib["elementId"])
                    target_sections = roads[target_road].findall("lanes/laneSection")
                    target_index = 0 if road_link.attrib["contactPoint"] == "start" else len(target_sections) - 1
                    targets.add(lane_segment_id(target_road, target_index, int(lane_link.attrib["id"])))
            elif road_link is not None and road_link.attrib["elementType"] == "junction":
                junction = junctions[int(road_link.attrib["elementId"])]
                for connection in junction.findall("connection"):
                    if int(connection.attrib["incomingRoad"]) == road_id:
                        target_road = int(connection.attrib["connectingRoad"])
                        target_sections = roads[target_road].findall("lanes/laneSection")
                        target_index = 0 if connection.attrib["contactPoint"] == "start" else len(target_sections) - 1
                        for lane_link in connection.findall("laneLink"):
                            if int(lane_link.attrib["from"]) == lane_id:
                                targets.add(lane_segment_id(target_road, target_index, int(lane_link.attrib["to"])))
        _require(targets <= set(definitions), f"non-driving/unresolved XML target from {key}")
        _require(targets == set(saved[key].successor_ids), f"XML successors disagree for {key}: {targets} != {saved[key].successor_ids}")
        xml_adjacency[key] = targets
        record = saved[key]
        _require((record.exit_s_m > record.entry_s_m) == forward, f"travel direction disagrees for {key}")
        lo = float(sections[index].attrib["s"])
        hi = float(sections[index + 1].attrib["s"]) if index + 1 < len(sections) else float(road.attrib["length"])
        inset = min(0.001, (hi - lo) / 4)
        ends = sorted((record.entry_s_m, record.exit_s_m))
        tolerance = max(1e-5, abs(hi) * 1e-7)
        _require(abs(ends[0] - lo - inset) <= tolerance and abs(ends[1] - hi + inset) <= tolerance,
                 f"section boundary coverage differs for {key}")
        _require(len(record.points_xyz_m) - 1 == max(1, math.ceil((hi - lo - 2 * inset) / document.sampling_distance_m)),
                 f"sampling count differs for {key}")
        junction_id = int(road.attrib.get("junction", -1))
        _require(record.junction_id == (junction_id if junction_id >= 0 else None), f"junction ID disagrees for {key}")
    max_sample_error, max_chord_error, min_boundary_margin = 0.0, 0.0, math.inf
    sample_count, dense_count = 0, 0
    worst_sample, worst_chord = None, None
    lengths = {}
    for key, record in saved.items():
        road, sections, index, _ = definitions[key]
        section = sections[index]
        points = record.points_xyz_m
        count = len(points) - 1
        s_values = [record.entry_s_m + (record.exit_s_m - record.entry_s_m) * i / count for i in range(count + 1)]
        lengths[key] = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
        for s, point in zip(s_values, points):
            expected, _ = _centre(road, section, record.lane_id, s)
            error = math.dist(expected, point)
            _require(error <= 1e-4, f"raw XML centre differs at {key}, s={s}, error={error}")
            if error > max_sample_error:
                max_sample_error, worst_sample = error, {"lane": key, "s_m": s}
            sample_count += 1
        for i, (a, b) in enumerate(zip(points, points[1:])):
            intervals = max(1, math.ceil(abs(s_values[i + 1] - s_values[i]) / dense_step_m))
            for j in range(intervals + 1):
                s = s_values[i] + (s_values[i + 1] - s_values[i]) * j / intervals
                expected, width = _centre(road, section, record.lane_id, s)
                deviation = _distance_to_segment(expected, a, b)
                margin = width / 2 - deviation
                _require(margin > 0, f"polyline shortcut leaves lane at {key}, s={s}")
                min_boundary_margin = min(min_boundary_margin, margin)
                if deviation > max_chord_error:
                    max_chord_error, worst_chord = deviation, {"lane": key, "s_m": s}
                dense_count += 1
    graph = document.graph
    nodes = {node.node_id: node for node in graph.nodes}
    edges = {(edge.start_id, edge.end_id): edge for edge in graph.edges}
    expected_pairs = set()
    connection_gaps = []
    for key, lane in saved.items():
        for name, point in ((lane.entry_node_id, lane.points_xyz_m[0]), (lane.exit_node_id, lane.points_xyz_m[-1])):
            _require((nodes[name].x_m, nodes[name].y_m, nodes[name].z_m) == point, f"core node changed geometry: {name}")
        pair = lane.entry_node_id, lane.exit_node_id
        expected_pairs.add(pair)
        _require(edges[pair].points_xy_m == tuple(p[:2] for p in lane.points_xyz_m), f"core edge lost lane samples: {key}")
        for target in xml_adjacency[key]:
            expected_pairs.add((lane.exit_node_id, saved[target].entry_node_id))
            connection_gaps.append(math.dist(lane.points_xyz_m[-1], saved[target].points_xyz_m[0]))
    _require(set(edges) == expected_pairs, "core graph adds/drops connections")
    node_adjacency = {key: set() for key in nodes}
    for a, b in edges:
        node_adjacency[a].add(b)
    reachable = {key: _reachable(xml_adjacency, key) for key in saved}
    for key, lane in saved.items():
        core_reached = _reachable(node_adjacency, lane.entry_node_id)
        reached_lanes = {target for target, item in saved.items() if item.entry_node_id in core_reached}
        _require(reached_lanes == reachable[key], f"core reachability changed from {key}")
    remaining, components = set(saved), []
    while remaining:
        key = min(remaining)
        component = {target for target in reachable[key] if key in reachable[target]}
        components.append(sorted(component))
        remaining -= component
    component_index = {key: i for i, group in enumerate(components) for key in group}
    component_links = sorted({(component_index[a], component_index[b]) for a in xml_adjacency for b in xml_adjacency[a]
                             if component_index[a] != component_index[b]})
    return {"status": "passed", "graph_sha256": document.sha256, "opendrive_sha256": source_hash,
            "method": "raw OpenDRIVE XML links and independent line/arc + lane-width mathematics; no CARLA/exporter calls",
            "roads_in_source": len(roads), "junctions_in_source": len(junctions),
            "driving_lane_sections": len(saved), "successors_checked": sum(map(len, xml_adjacency.values())),
            "junction_lane_counts": dict(sorted(Counter(lane.junction_id for lane in saved.values() if lane.junction_id is not None).items())),
            "sample_points_checked": sample_count, "max_raw_xml_point_error_m": max_sample_error,
            "worst_sample": worst_sample, "dense_probe_step_m": dense_step_m, "dense_probes_checked": dense_count,
            "max_polyline_centre_deviation_m": max_chord_error, "worst_chord": worst_chord,
            "minimum_centre_path_boundary_margin_m": min_boundary_margin,
            "maximum_successor_gap_m": max(connection_gaps, default=0),
            "graph_nodes_checked": len(nodes), "graph_edges_checked": len(edges),
            "all_origin_reachability_checks": len(saved), "reachable_ordered_lane_pairs_excluding_self": sum(len(v) - 1 for v in reachable.values()),
            "strongly_connected_component_sizes": [len(group) for group in components],
            "strongly_connected_components": components, "component_links": component_links,
            "total_lane_polyline_length_m": sum(lengths.values())}
