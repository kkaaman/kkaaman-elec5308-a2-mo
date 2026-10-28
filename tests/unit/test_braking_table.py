"""Third external review (e93154f), section 4: achieved braking against the stack's
assumed decelerations, on synthetic logs."""

import pytest

from evaluation.braking_table import against_assumptions, episodes, table


def braking_log(v0=6.0, decel=3.0, lag_ticks=2, brake=1.0, dt=0.05):
    """Cruise at ``v0`` for 1 s, then brake: ``lag_ticks`` with no deceleration, then ``decel``."""
    rows, v = [], v0
    for k in range(200):
        braking = k >= 20
        if braking and k >= 20 + lag_ticks:
            v = max(0.0, v - decel * dt)
        rows.append(dict(time_s=k * dt, truth=dict(speed_mps=v), command=[0.0, 0.0, brake if braking and v > 0 else 0.0], driver=dict(stop_reason="obstacle")))
        if braking and v == 0.0:
            rows.append(dict(time_s=(k + 1) * dt, truth=dict(speed_mps=0.0), command=[0.0, 0.0, brake], driver={}))
            break
    return rows


def test_a_braking_episode_reports_lag_deceleration_and_stopping_distance():
    (e,) = episodes(braking_log())
    assert e["speed_mps"] == pytest.approx(6.0)
    assert e["lag_s"] == pytest.approx(0.1, abs=1e-6)  # the speed first drops two ticks after the brake
    assert e["peak_decel_mps2"] == pytest.approx(3.0, abs=0.01)
    assert e["stopping_distance_m"] == pytest.approx(6.0**2 / (2 * 3.0) + 6.0 * 0.05, abs=0.1)  # one tick at 6 m/s first
    assert e["why"] == "obstacle"
    (row,) = table([e])
    assert row["brake"] == "0.90-1.01" and row["episodes"] == 1


def test_assumed_decelerations_are_checked_against_the_achieved_peak_at_full_brake():
    check = against_assumptions(episodes(braking_log(decel=3.0)))
    rules = check["rules"]
    assert check["achieved_peak_decel_p10_mps2"] == pytest.approx(3.0, abs=0.01)
    assert [r["achieved"] for r in rules.values()] == [True, True, False]  # 1, 2 and 4 m/s^2 assumed
    assert check["response_ok"] is True
    assert "BL-E-03" in against_assumptions(episodes(braking_log(brake=0.5)))["note"]
