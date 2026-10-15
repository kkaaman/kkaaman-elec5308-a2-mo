"""Immutable lane stitching; install the resulting path once per mission."""

import math

from planning.junction_planner import _heading


def stitch_route(topology, segment_ids, max_heading_jump_deg=30.0):
    ids = tuple(segment_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Route must be nonempty, open and contain no repeated sections")
    if not math.isfinite(max_heading_jump_deg) or not 0 < max_heading_jump_deg <= 90:
        raise ValueError("Invalid heading discontinuity limit")
    by_id = {lane.segment_id: lane for lane in topology.lanes}
    if any(key not in by_id for key in ids):
        raise ValueError("Unknown route section")
    points = []
    previous = None
    for key in ids:
        lane = by_id[key]
        if previous is not None:
            if key not in previous.successor_ids:
                raise ValueError("Illegal route transition")
            delta = _heading(lane.points_xyz_m) - _heading(previous.points_xyz_m, True)
            delta = abs(math.degrees(math.atan2(math.sin(delta), math.cos(delta))))
            if delta > max_heading_jump_deg:
                raise ValueError("Discontinuous boundary heading; refusing shortcut")
        for point in lane.points_xyz_m:
            if not points or point != points[-1]:
                points.append(point)
        previous = lane
    return tuple(points)
