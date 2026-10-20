"""CARLA 0.9.16 topology queries and coordinate conversion (BL-A-02).

get_topology() supplies lane-section successor relationships. OpenDRIVE section
bounds allow sampling every section in travel order without choosing next[0]
or skipping short sections. All geometry comes from CARLA's Waypoint API.
"""
from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import carla

from planning.lane_graph import LaneSegment, LaneTopology, _positive, lane_segment_id


class TopologyError(ValueError):
    """The map cannot be exported without losing or inventing connectivity."""


def _key(waypoint) -> tuple[int, int, int]:
    return waypoint.road_id, waypoint.section_id, waypoint.lane_id


def _driving(waypoint) -> bool:
    return str(waypoint.lane_type) == "Driving"


def _section_bounds(opendrive: str) -> dict[tuple[int, int, int], tuple[float, float]]:
    result = {}
    try:
        root = ET.fromstring(opendrive)
    except ET.ParseError as exc:
        raise TopologyError(f"invalid OpenDRIVE XML: {exc}") from exc
    for road in root.findall("road"):
        road_id = int(road.attrib["id"])
        length = float(road.attrib["length"])
        sections = road.findall("lanes/laneSection")
        for index, section in enumerate(sections):
            lo = float(section.attrib["s"])
            hi = float(sections[index + 1].attrib["s"]) if index + 1 < len(sections) else length
            if not math.isfinite(lo) or not math.isfinite(hi) or lo < 0 or hi <= lo:
                raise TopologyError(f"invalid section extent on road {road_id}")
            for lane in section.findall("./*/lane"):
                if lane.attrib.get("type") == "driving":
                    result[(road_id, index, int(lane.attrib["id"]))] = (lo, hi)
    return result


def _waypoint(map_, key, s):
    waypoint = map_.get_waypoint_xodr(key[0], key[2], s)
    if waypoint is None or _key(waypoint) != key or not _driving(waypoint):
        raise TopologyError(f"cannot sample driving lane {key} at s={s}")
    return waypoint


def _sample_lane(map_, key, bounds, spacing):
    lo, hi = bounds
    # Stay inside sections: exact boundaries can select the adjacent section.
    # Millimetre insets also keep explicit connector edges nondegenerate in XY.
    inset = min(0.001, (hi - lo) / 4)
    midpoint = _waypoint(map_, key, (lo + hi) / 2)
    probe = min(0.01, (hi - lo) / 4)
    candidates = [wp for wp in midpoint.next(probe) if _key(wp) == key]
    if len(candidates) != 1 or abs(candidates[0].s - midpoint.s) < 1e-9:
        raise TopologyError(f"ambiguous travel direction for lane {key}")
    increasing = candidates[0].s > midpoint.s
    start, end = (lo + inset, hi - inset) if increasing else (hi - inset, lo + inset)
    count = max(1, math.ceil(abs(end - start) / spacing))
    if count > 100_000:
        raise TopologyError(f"excessive sampling count for lane {key}")
    samples = [_waypoint(map_, key, start + (end - start) * i / count) for i in range(count + 1)]
    return samples, inset


def export_lane_topology(
    map_: carla.Map,
    *,
    sampling_distance_m: float = 2.0,
    navigation_speed_cap_mps: float = 5.0,
    max_connection_gap_m: float = 0.25,
) -> LaneTopology:
    """Export all driving lane sections, rejecting unresolved connections.

    No lane changes, world ticks, actor queries or simulator mutations occur.
    Query order never defines IDs or chooses a route. A real carla.Map built
    from a local .xodr file uses exactly the same implementation as a live map.
    """
    for name, value in (("sampling_distance_m", sampling_distance_m),
                        ("navigation_speed_cap_mps", navigation_speed_cap_mps),
                        ("max_connection_gap_m", max_connection_gap_m)):
        _positive(name, value)
    opendrive = map_.to_opendrive().replace("\r\n", "\n")
    bounds = _section_bounds(opendrive)
    successors = {key: set() for key in bounds}
    entries = set()
    for entry, target in map_.get_topology():
        if not _driving(entry):
            continue
        key, target_key = _key(entry), _key(target)
        if key not in bounds:
            raise TopologyError(f"topology lane absent from OpenDRIVE: {key}")
        entries.add(key)
        # A terminal section is returned paired with its own last waypoint.
        if target_key != key:
            if not _driving(target) or target_key not in bounds:
                raise TopologyError(f"non-driving or unknown successor: {key} -> {target_key}")
            successors[key].add(target_key)
    if entries != set(bounds):
        raise TopologyError(f"topology is missing driving sections: {sorted(set(bounds) - entries)}")
    lanes = []
    for key in sorted(bounds):
        samples, inset = _sample_lane(map_, key, bounds[key], sampling_distance_m)
        observed = {_key(wp) for wp in samples[-1].next(inset + min(0.0001, inset / 2))}
        if observed != successors[key]:
            raise TopologyError(f"successor query disagrees with topology for {key}: {observed} != {successors[key]}")
        # Independently check each sampled step follows the same directed lane.
        for a, b in zip(samples, samples[1:]):
            reachable = a.next(abs(b.s - a.s))
            if not any(_key(wp) == key and abs(wp.s - b.s) < 1e-5 for wp in reachable):
                raise TopologyError(f"sample is not forward reachable on {key} at s={b.s}")
        points = []
        for wp in samples:
            location = wp.transform.location
            point = tuple(0.0 if round(v, 6) == 0 else round(v, 6) for v in (location.x, -location.y, location.z))
            points.append(point)
        junction = samples[0].junction_id if samples[0].is_junction else None
        if any((wp.junction_id if wp.is_junction else None) != junction for wp in samples):
            raise TopologyError(f"junction identity changes inside lane section {key}")
        lanes.append(LaneSegment(*key, round(samples[0].s, 8), round(samples[-1].s, 8),
                                 tuple(points), tuple(lane_segment_id(*s) for s in sorted(successors[key])), junction))
    document = LaneTopology(map_.name.rsplit("/", 1)[-1], hashlib.sha256(opendrive.encode("utf-8")).hexdigest(),
                            sampling_distance_m, navigation_speed_cap_mps, tuple(lanes), max_connection_gap_m)
    # Exercise the existing contract before handing this map to the core.
    document.graph
    return document


def _map_name_matches(actual: str, expected: str, *, allow_town10hd_alias=False) -> bool:
    """Recognize the two Town10HD assets only with a separately verified hash."""
    actual = actual.rsplit("/", 1)[-1]
    return actual == expected or (
        allow_town10hd_alias and {actual, expected} == {"Town10HD", "Town10HD_Opt"}
    )


def validate_lane_topology_on_map(map_: carla.Map, document: LaneTopology) -> dict:
    """Check a saved graph against source waypoints, not another graph copy.

    Every XYZ sample is compared with CARLA's lane-centre geometry; every
    successor is checked by stepping across the lane-section boundary.
    """
    opendrive = map_.to_opendrive().replace("\r\n", "\n")
    if hashlib.sha256(opendrive.encode("utf-8")).hexdigest() != document.opendrive_sha256:
        raise TopologyError("graph and source OpenDRIVE hashes differ")
    if not _map_name_matches(map_.name, document.map_name, allow_town10hd_alias=True):
        raise TopologyError("graph and source map names differ")
    bounds = _section_bounds(opendrive)
    if {tuple((lane.road_id, lane.section_id, lane.lane_id)) for lane in document.lanes} != set(bounds):
        raise TopologyError("graph does not cover all driving lane sections")
    point_count, successor_count, max_error = 0, 0, 0.0
    for lane in document.lanes:
        key = (lane.road_id, lane.section_id, lane.lane_id)
        lo, hi = bounds[key]
        inset = min(0.001, (hi - lo) / 4)
        ends = sorted((lane.entry_s_m, lane.exit_s_m))
        # The SDK's get_waypoint_xodr s argument is float32; account for its
        # quantization when comparing saved samples with double XML extents.
        s_tolerance = max(1e-5, abs(hi) * 1e-7)
        if abs(ends[0] - lo - inset) > s_tolerance or abs(ends[1] - hi + inset) > s_tolerance:
            raise TopologyError(f"saved section omits lane boundary {key}")
        count = len(lane.points_xyz_m) - 1
        if count != max(1, math.ceil((hi - lo - 2 * inset) / document.sampling_distance_m)):
            raise TopologyError(f"saved sampling count disagrees for {key}")
        previous = None
        for i, point in enumerate(lane.points_xyz_m):
            s = lane.entry_s_m + (lane.exit_s_m - lane.entry_s_m) * i / count
            wp = _waypoint(map_, key, s)
            location = wp.transform.location
            error = math.dist(point, (location.x, -location.y, location.z))
            max_error = max(max_error, error)
            if error > 1e-4:
                raise TopologyError(f"saved polyline leaves lane centre {key} at s={s}")
            junction = wp.junction_id if wp.is_junction else None
            if junction != lane.junction_id:
                raise TopologyError(f"saved junction identity disagrees for {key}")
            if previous is not None:
                if not any(_key(w) == key and abs(w.s - wp.s) < 1e-5 for w in previous.next(abs(wp.s - previous.s))):
                    raise TopologyError(f"saved sample is not forward reachable on {key}")
            previous = wp
            point_count += 1
        increasing = lane.exit_s_m > lane.entry_s_m
        boundary_distance = hi - lane.exit_s_m if increasing else lane.exit_s_m - lo
        end = _waypoint(map_, key, lane.exit_s_m)
        expected = {lane_segment_id(*_key(wp)) for wp in end.next(boundary_distance + min(0.0001, (hi - lo) / 8))}
        if expected != set(lane.successor_ids):
            raise TopologyError(f"saved successors disagree for {key}")
        successor_count += len(expected)
    document.graph
    return {"map": document.map_name, "graph_sha256": document.sha256,
            "lanes_checked": len(document.lanes), "points_checked": point_count,
            "successors_checked": successor_count, "max_point_error_m": max_error}


def connect_map(
    host: str, port: int, timeout_s: float, expected_map: str, *,
    expected_opendrive_sha256: str | None = None,
) -> tuple[carla.Map, str]:
    """Read the active map; require matching versions and optional frozen content.

    Map names remain strict unless a frozen fingerprint is supplied. Town10HD
    and Town10HD_Opt may then identify the same road network only when its
    normalized OpenDRIVE content matches that fingerprint exactly.
    """
    import carla

    _positive("timeout_s", timeout_s)
    client = carla.Client(host, port)
    client.set_timeout(timeout_s)
    client_version, server_version = client.get_client_version(), client.get_server_version()
    if client_version != server_version:
        raise TopologyError(f"client/server version mismatch: {client_version} != {server_version}")
    map_ = client.get_world().get_map()
    if not _map_name_matches(
        map_.name, expected_map, allow_town10hd_alias=expected_opendrive_sha256 is not None,
    ):
        raise TopologyError(f"expected active map {expected_map}, got {map_.name}")
    if expected_opendrive_sha256 is not None:
        opendrive = map_.to_opendrive().replace("\r\n", "\n")
        if hashlib.sha256(opendrive.encode("utf-8")).hexdigest() != expected_opendrive_sha256:
            raise TopologyError("active map OpenDRIVE hash differs from the frozen mission")
    return map_, server_version


def load_opendrive(path, map_name: str) -> carla.Map:
    """Build the CARLA map from a genuine local OpenDRIVE file."""
    from pathlib import Path

    import carla

    return carla.Map(map_name, Path(path).read_text(encoding="utf-8"))
