"""RV-06 replay: lane monitor versus the fixed route on recorded run frames.

Frames come from a run's ``video.avi`` (one frame per logged sample, written
after the command was applied), so the monitor adds no latency to the live loop
and can never influence control. Ego pose is the logged rear-axle state.
"""

import math
from dataclasses import asdict

from visualisation.lane_monitor import (
    REFERENCE_X_M,
    compare_to_map,
    map_lookahead_offset,
    monitor_from_mask,
    summarize_monitor,
)

CURVE_HEADING_CHANGE_RAD = 0.15


def route_turning(route, progress_m, ahead_m=REFERENCE_X_M + 4.0):
    """True where the route heading changes by > ~9 degrees over the look-ahead."""
    _, _, start = route.at(progress_m)
    _, _, end = route.at(min(route.length_m, progress_m + ahead_m))
    change = math.atan2(math.sin(end - start), math.cos(end - start))
    return abs(change) > CURVE_HEADING_CHANGE_RAD


def replay(rows, frames, segmenter, calibration, route, *, every=2):
    """Run the diagnostic monitor on aligned (row, RGB frame) pairs."""
    if type(every) is not int or every < 1:
        raise ValueError("every must be a positive integer")
    records = []
    for index, (row, rgb) in enumerate(zip(rows, frames)):
        if index % every:
            continue
        departure, path, ground = monitor_from_mask(segmenter.predict(rgb), calibration)
        ego = row["ego"]
        map_offset = map_lookahead_offset(route, ego["x_m"], ego["y_m"], ego["yaw_rad"])
        records.append(
            dict(
                index=index,
                frame_id=row["frame_id"],
                sim_time_s=row["sim_time_s"],
                progress_m=row.get("progress_m"),
                curve=route_turning(route, row.get("progress_m", 0.0)),
                map_offset_m=map_offset,
                departure=asdict(departure),
                valid=departure.valid,
                comparison=compare_to_map(departure, map_offset),
                ground_points=len(ground),
                path=path,
            )
        )
    return records


def report(records, *, rows, frames_read, reference_x_m=REFERENCE_X_M):
    return dict(
        status="REPLAY_MEASURED_REVIEW_REQUIRED",
        reference_point=(
            f"{reference_x_m} m ahead of the rear axle on the ego heading; lane and"
            " map offsets both positive when that point is left of the centreline/route"
        ),
        summary=summarize_monitor(records),
        logged_rows=rows,
        video_frames=frames_read,
        aligned=rows == frames_read,
        control_coupling=False,
        scope=(
            "Recorded-frame replay of the existing LaneUNet; flat-ground projection."
            " The monitor is a HUD/diagnostic only and is not an input to control."
        ),
        records=[{k: v for k, v in r.items() if k != "path"} for r in records],
    )
