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


def test_assumed_stops_are_checked_against_the_stops_full_brake_achieved():
    """A 3 m/s^2 stop from 6 m/s after one tick of lag: 6.3 m. The 1 and 2 m/s^2 ramps
    (18 m, 9 m) hold; the following rule's 4 m/s^2 after 0.5 s (7.5 m) holds too, by 1.2 m,
    though its deceleration alone is not reached; at 1.5 m/s^2 it does not."""
    check = against_assumptions(episodes(braking_log(decel=3.0)))
    rules = list(check["rules"].values())
    assert [r["achieved"] for r in rules] == [True, True, True]
    assert rules[2]["margin_m"] == pytest.approx(6.0 * 0.5 + 6.0**2 / 8.0 - 6.3, abs=0.1)
    assert check["worst_effective_decel_mps2"] == pytest.approx(6.0**2 / (2 * 6.3), abs=0.05)
    assert check["response_ok"] is True and check["implausible_excluded"] == 0
    assert [r["achieved"] for r in against_assumptions(episodes(braking_log(decel=1.5)))["rules"].values()] == [True, False, False]
    assert "BL-E-03" in against_assumptions(episodes(braking_log(brake=0.5)))["note"]
    assert against_assumptions(episodes(braking_log(decel=40.0)))["note"].startswith("no stop")  # over 1 g: implausible

def test_the_rules_checked_are_the_ones_the_run_drove_with():
    """Fourth review, 7.4: the assumptions come from the run report, not constants; a run
    with a stronger assumed braking fails where the default holds."""
    from evaluation.braking_table import assumptions  # noqa: PLC0415

    eps = episodes(braking_log(decel=3.0))
    strong = assumptions(1.0, 2.0, 8.0, 0.5)
    rules = list(against_assumptions(eps, strong)["rules"].values())
    assert [r["achieved"] for r in rules] == [True, True, False]
    assert list(against_assumptions(eps, assumptions(1.0))["rules"]) == ["planner ramp (PlannerConfig.decel_mps2)"]
    excluded = against_assumptions(episodes(braking_log(decel=40.0)))["excluded"]
    assert len(excluded) == 1 and excluded[0]["why"].startswith("over 1 g")
