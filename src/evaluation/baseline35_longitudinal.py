"""Measured braking budgets; absent/failed trials cannot justify a speed cap."""

import math


def braking_summary(rows):
    trials = {}
    for row in rows:
        trials.setdefault(row["trial"], []).append(row)
    measurements = []
    for key, trace in trials.items():
        braking = [r for r in trace if r["phase"] == "brake"]
        if len(braking) < 3 or braking[-1]["speed_mps"] > 0.1:
            continue
        start = braking[0]
        v0 = start["speed_mps"]
        if v0 < 1.5:
            continue
        distance = braking[-1]["distance_m"] - start["distance_m"]
        lag = next(
            (r["sim_time_s"] - start["sim_time_s"] for r in braking if r["speed_mps"] <= v0 - 0.1),
            None,
        )
        if distance <= 0 or lag is None:
            continue
        # Effective distance deceleration includes the measured actuator lag.
        effective = v0 * v0 / (2 * distance)
        measurements.append(
            dict(
                trial=key,
                initial_speed_mps=v0,
                brake=start["brake"],
                stopping_distance_m=distance,
                response_lag_s=lag,
                effective_deceleration_mps2=effective,
            )
        )
    required = {(v, b) for v in (2, 5, 8) for b in (0.4, 0.7, 1.0)}
    observed = {(r["target_mps"], r["brake"]) for r in rows if r["phase"] == "brake"}
    complete = observed == required and len(measurements) == 9
    decel = min((x["effective_deceleration_mps2"] for x in measurements), default=None)
    conservative = 0.7 * decel if decel is not None else None
    if conservative is not None and (not math.isfinite(conservative) or conservative <= 0):
        complete = False
    return dict(
        status="MEASURED_PASS" if complete else "INCOMPLETE",
        trials=measurements,
        deceleration_mps2=conservative,
        response_budget_s=max((x["response_lag_s"] for x in measurements), default=0)
        + 0.2
        + 0.1
        + 0.1,
        scope="Dry-road fixed-vehicle braking >=0.4; 30% deceleration margin plus camera, processing and confirmation budgets",
    )
