import numpy as np
import pytest

from evaluation.longitudinal import braking_summary


def braking_rows():
    rows = []
    for speed in (2, 5, 8):
        for brake in (0.4, 0.7, 1):
            for t in np.linspace(0, speed / 2, 20):
                rows.append(
                    dict(
                        trial=f"{speed}-{brake}",
                        phase="brake",
                        target_mps=speed,
                        brake=brake,
                        speed_mps=max(0, speed - 2 * t),
                        sim_time_s=float(t),
                        distance_m=float(speed * t - t * t),
                    )
                )
    return rows


def test_measured_budget_uses_all_trials_and_conservative_reserve():
    report = braking_summary(braking_rows())

    assert report["status"] == "MEASURED_PASS"
    assert report["deceleration_mps2"] == pytest.approx(1.4)
    assert report["measured_response_lag_s"] == pytest.approx(4 / 19)
    assert report["response_budget_s"] == report["measured_response_lag_s"]

    incomplete = braking_rows()[:-20]
    assert braking_summary(incomplete)["status"] == "INCOMPLETE"