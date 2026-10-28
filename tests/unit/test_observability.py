"""Fifth review, 4.2: the first reliable detection of the stop scenario's obstacle, and the
speed its distance supports."""

import pytest

from evaluation.observability import first_detection, observability, supported_speed


def test_supported_speed_meets_the_stopping_distance_exactly():
    # 4 m/s at 1 m/s^2 with 0.25 s and a 2.5 m margin needs 1 + 8 + 2.5 = 11.5 m.
    assert supported_speed(11.5, 1.0, 0.25, 2.5) == pytest.approx(4.0)
    assert supported_speed(2.0, 1.0, 0.25, 2.5) == 0.0
    assert supported_speed(30.0, 1.0, 0.25, 2.5) > supported_speed(12.0, 1.0, 0.25, 2.5)


def _log(first_seen_gap, start_gap=40.0, speed=3.0, front=3.8, flicker_at=None):
    rows, gap, t = [], start_gap, 0.0
    while gap > 3.0:
        seen = gap <= first_seen_gap or (flicker_at is not None and abs(gap - flicker_at) < 0.1)
        rows.append(dict(time_s=t, truth=dict(gap_m=gap, speed_mps=speed), driver=dict(obstacle_m=gap + front if seen else None)))
        gap -= speed * 0.05
        t += 0.05
    return rows


def test_the_first_detection_is_the_first_held_one_where_the_truth_has_it():
    log = _log(9.0, flicker_at=25.0)  # one stray tick at 25 m does not count
    k = first_detection(log, 3.8)
    assert log[k]["truth"]["gap_m"] == pytest.approx(9.0, abs=0.16)
    out = observability(log, 3.8, cruise_mps=3.0)
    assert out["detected"] and out["first_seen_gap_m"] == pytest.approx(9.0, abs=0.16)
    # 9 m supports about 3.2 m/s: the 3 m/s cruise is within it, 4 m/s would not be.
    assert out["supported_mps"] == pytest.approx(supported_speed(out["first_seen_gap_m"], 1.0, 0.25, 2.5), abs=0.01)
    assert out["cruise_supported"] is True
    assert observability(log, 3.8, cruise_mps=4.0)["cruise_supported"] is False
    # Seen at the wrong place (a different object), or never: no detection.
    wrong = [dict(r, driver=dict(obstacle_m=r["truth"]["gap_m"] + 3.8 + 5.0)) for r in log]
    assert observability(wrong, 3.8, cruise_mps=3.0)["detected"] is False
