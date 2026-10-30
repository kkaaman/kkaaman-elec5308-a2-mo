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


def test_t10_the_reliable_range_is_where_the_detection_held_from_not_where_it_first_appeared():
    """Tenth review, 5: the first detection is the best case. A cone seen at 24 m, lost
    again for 5 ticks at 20 m and missed every third tick to 14 m, then held: it held (every
    1 s window 80% seen, no run of misses over 3 ticks) from 14-16 m, and the speed that
    supports is lower than the first sighting's."""
    log = _log(24.0)
    for k, row in enumerate(log):
        g = row["truth"]["gap_m"]
        if 19.5 <= g <= 20.3 or (14.0 < g < 19.5 and k % 3 == 0):
            row["driver"] = dict(obstacle_m=None)
    out = observability(log, 3.8, cruise_mps=3.0)
    assert out["first_seen_gap_m"] == pytest.approx(24.0, abs=0.16)
    assert 14.0 <= out["reliable"]["from_gap_m"] <= 16.5  # where every 1 s window held 80%
    assert out["reliable"]["supported_mps"] < out["supported_mps"] and out["misses_after_first"] > 5
    clean = observability(_log(24.0), 3.8, cruise_mps=3.0)
    assert clean["reliable"]["from_gap_m"] == clean["first_seen_gap_m"] and clean["misses_after_first"] == 0
