"""Evaluation-only stop-margin truth and HOLD/resume metrics (BL-B-07). Simulator-free.

Reference points, matching ``tests/fixtures/acceptance/golden.json``:
- **Ego:** the rear-axle position is projected onto the continuous mission route,
  giving ``s_ego``. The front bumper is at ``s_ego + front_offset_m``, where
  ``front_offset_m`` is the rear-axle-to-front-face distance from the vehicle's
  bounding box (``ego_front_offset_m``). The offset is applied exactly once.
- **Obstacle:** the corners of its bounding-box footprint, sampled every
  ``SAMPLE_STEP_M`` along its edges. The obstacle surface is the smallest route arc
  length among samples inside the evaluation corridor (``|lateral| <= half_width_m``)
  and ahead of the ego rear axle.
- **Gap:** ``truth_front_gap_m = surface_s - (s_ego + front_offset_m)``. The value
  is None when no sample is in the corridor ahead, meaning the obstacle is not in
  the ego's path.

These values come from the simulator's actor poses and boxes. They are logged for
scoring only and must never reach perception, planning or control.
"""

import math

SAMPLE_STEP_M = 0.05
EVALUATION_CLEARANCE_M = 0.15  # corridor = ego half-width + this clearance, each side


def ego_front_offset_m(rear_axle_offset_m, box_location_x_m, box_extent_x_m):
    """Rear axle to front bumper face, from the actor origin's rear-axle offset and box."""
    value = rear_axle_offset_m + box_location_x_m + box_extent_x_m
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Front offset must be positive")
    return value


def footprint_samples(corners, step_m=SAMPLE_STEP_M):
    """Corners plus evenly spaced points along each footprint edge."""
    if len(corners) < 3:
        raise ValueError("Footprint needs at least three corners")
    points = []
    for a, b in zip(corners, corners[1:] + corners[:1]):
        length = math.dist(a, b)
        count = max(1, math.ceil(length / step_m))
        for i in range(count):
            t = i / count
            points.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    if not all(math.isfinite(v) for p in points for v in p):
        raise ValueError("Nonfinite footprint")
    return points


def signed_lateral_m(path, x_m, y_m, projection=None):
    """Signed distance to the route, positive to the left of travel."""
    projection = projection or path.project(x_m, y_m)
    s = projection.progress_m
    before = path.point_at(max(0.0, s - 0.05))
    after = path.point_at(min(path.length_m, s + 0.05))
    cross = (after.x_m - before.x_m) * (y_m - projection.point.y_m) - (after.y_m - before.y_m) * (
        x_m - projection.point.x_m
    )
    return projection.distance_m if cross >= 0 else -projection.distance_m


def route_surface_gap(path, ego_x_m, ego_y_m, front_offset_m, corners, half_width_m):
    """Evaluation truth for one tick; see the module docstring for the reference points."""
    ego_s = path.project(ego_x_m, ego_y_m).progress_m
    in_path = []
    laterals = []
    for x, y in footprint_samples(corners):
        projection = path.project(x, y)
        lateral = signed_lateral_m(path, x, y, projection)
        laterals.append(lateral)
        if abs(lateral) <= half_width_m and projection.progress_m >= ego_s:
            in_path.append(projection.progress_m)
    surface_s = min(in_path) if in_path else None
    return {
        "truth_front_gap_m": None if surface_s is None else surface_s - ego_s - front_offset_m,
        "truth_in_corridor": surface_s is not None,
        "truth_ego_s_m": ego_s,
        "truth_surface_s_m": surface_s,
        "truth_obstacle_lateral_m": [min(laterals), max(laterals)],
        "truth_corridor_half_width_m": half_width_m,
    }


def hold_metrics(
    rows,
    *,
    dt_s,
    stopped_speed_mps=0.1,
    stopped_duration_s=1.0,
    removal_time_s=None,
):
    """Stopped margin, HOLD speed, continuous HOLD and resume delay from logged rows.

    ``rows``: dicts with ``sim_time_s``, true ``speed_mps`` and ``truth_front_gap_m``
    (None while no obstacle is in the path).

    - **HOLD start:** the first time the ego has been at or below
      ``stopped_speed_mps`` for ``stopped_duration_s`` while an obstacle is in its
      path.
    - **HOLD end:** the scripted removal, or the last row with the obstacle in the
      path (S4).
    - **Margins:** min and max truth gap over the HOLD rows.
    - **HOLD speed:** the maximum true speed over the HOLD rows, so any creep counts.
    - **Resume delay:** from removal to the first row above ``stopped_speed_mps``.

    The goal stop is never mistaken for a HOLD, because it has no obstacle in the
    path. Missing evidence gives None, never zero.
    """
    ordered = sorted(rows, key=lambda r: r["sim_time_s"])
    end = removal_time_s
    if end is None:
        present = [r["sim_time_s"] for r in ordered if r.get("truth_front_gap_m") is not None]
        end = (present[-1] + dt_s) if present else None
    start = None
    since = previous = None
    for row in ordered:
        t = row["sim_time_s"]
        if end is not None and t >= end:
            break
        if previous is not None and t - previous > 1.5 * dt_s:
            since = None
        previous = t
        stopped = (
            abs(row["speed_mps"]) <= stopped_speed_mps and row.get("truth_front_gap_m") is not None
        )
        since = (t if since is None else since) if stopped else None
        if since is not None and t - since >= stopped_duration_s - 1e-9:
            start = since
            break
    hold = [
        r for r in ordered if start is not None and start <= r["sim_time_s"] < (end or math.inf)
    ]
    gaps = [r["truth_front_gap_m"] for r in hold if r.get("truth_front_gap_m") is not None]
    # Longest run of consecutive stopped ticks; each tick stands for dt of simulation.
    longest = 0.0
    first = last = None
    for row in hold:
        t = row["sim_time_s"]
        if (abs(row["speed_mps"]) > stopped_speed_mps
                or row.get("truth_front_gap_m") is None):
            first = last = None
            continue
        if last is None or t - last > 1.5 * dt_s:
            first = t
        last = t
        longest = max(longest, last - first + dt_s)
    resume = None
    if removal_time_s is not None:
        moving = [
            r["sim_time_s"]
            for r in ordered
            if r["sim_time_s"] >= removal_time_s and abs(r["speed_mps"]) > stopped_speed_mps
        ]
        resume = (moving[0] - removal_time_s) if moving else None
    return {
        "hold_start_s": start,
        "hold_end_s": end if start is not None else None,
        "hold_samples": len(hold),
        "stopped_margin_min_m": min(gaps) if gaps else None,
        "stopped_margin_max_m": max(gaps) if gaps else None,
        "hold_max_speed_mps": max(abs(r["speed_mps"]) for r in hold) if hold else None,
        "continuous_hold_s": longest if hold else None,
        "removal_time_s": removal_time_s,
        "resume_delay_s": resume,
        "definition": (
            f"HOLD starts once speed <= {stopped_speed_mps} m/s for {stopped_duration_s} s with an"
            " obstacle in the path and ends at the scripted removal (or when the obstacle"
            " leaves the path); resume delay runs from removal to the first speed above"
            f" {stopped_speed_mps} m/s"
        ),
    }
