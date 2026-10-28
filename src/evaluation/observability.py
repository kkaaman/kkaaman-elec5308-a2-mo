"""Observability-limited speed (fifth review, 4.2): how far ahead an obstacle was first
seen, and the speed that distance supports.

A car may drive only as fast as it can stop within the distance at which it reliably sees
what is ahead: ``d >= v * response + v^2 / (2 a) + margin``. A small object seen only at
8 m supports a lower speed than a parked car seen at 30 m. This module measures the first
reliable detection of a run's obstacle from the drive log, against the evaluator's truth
gap, and turns it into that speed. Repeated over obstacle sizes and approach speeds (the
``stop`` scenario with ``--obstacle-blueprint``), it gives the supported-speed envelope
per size; nothing here sets the car's speed.

A detection is the driver's own LiDAR obstacle in its path (``obstacle_m``, from the rear
axle) whose bumper gap agrees with the true gap within ``tolerance_m``, held for
``persist_ticks`` ticks running; the first such tick counts.
"""

import math


def supported_speed(distance_m, decel_mps2, response_s, margin_m):
    """The highest speed from which the car stops ``margin_m`` short of something
    ``distance_m`` ahead, reacting after ``response_s`` and braking at ``decel_mps2``."""
    room = float(distance_m) - float(margin_m)
    if room <= 0.0:
        return 0.0
    a, t = float(decel_mps2), float(response_s)
    return -a * t + math.sqrt(a * a * t * t + 2.0 * a * room)


def first_detection(log, front_m, tolerance_m=1.5, persist_ticks=3):
    """Index of the first tick of a run of ``persist_ticks`` with the obstacle seen where
    the truth has it, or None. Each row needs ``driver.obstacle_m`` and ``truth.gap_m``."""
    run = 0
    for k, row in enumerate(log):
        seen = (row.get("driver") or {}).get("obstacle_m")
        gap = (row.get("truth") or {}).get("gap_m")
        if seen is not None and gap is not None and abs((seen - front_m) - gap) <= tolerance_m:
            run += 1
            if run >= persist_ticks:
                return k - persist_ticks + 1
        else:
            run = 0
    return None


def observability(log, front_m, cruise_mps, decel_mps2=1.0, response_s=0.25, margin_m=2.5, tolerance_m=1.5, persist_ticks=3):
    """REPORT: the first reliable detection (time, true bumper gap, speed then) and the
    speed it supports at the planner's deceleration, ``response_s`` (the measured brake
    lag, 0.1-0.25 s, BL-E-03) and the stop gap as the margin; and whether the cruise speed
    was within it."""
    assumption = dict(decel_mps2=decel_mps2, response_s=response_s, margin_m=margin_m)
    k = first_detection(log, front_m, tolerance_m, persist_ticks)
    if k is None:
        return dict(detected=False, assumption=assumption, note="the obstacle was never seen where the truth has it")
    row = log[k]
    gap = row["truth"]["gap_m"]
    supported = supported_speed(gap, decel_mps2, response_s, margin_m)
    return dict(
        detected=True,
        first_seen_s=round(float(row["time_s"]), 2),
        first_seen_gap_m=round(float(gap), 2),
        speed_then_mps=round(float(row["truth"]["speed_mps"]), 2),
        supported_mps=round(supported, 2),
        cruise_mps=cruise_mps,
        cruise_supported=bool(cruise_mps <= supported + 1e-9),
        assumption=assumption,
    )
