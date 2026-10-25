"""BL-B-07: evaluation-only stop-margin truth and HOLD/resume metrics."""

import json
from pathlib import Path

import pytest

from evaluation.stop_truth import (
    ego_front_offset_m,
    footprint_samples,
    hold_metrics,
    route_surface_gap,
    signed_lateral_m,
)
from navigation_control.path import ReferencePath
from navigation_control.types import Waypoint

GOLDEN = json.loads(
    (Path(__file__).resolve().parents[2] / "tests/fixtures/acceptance/golden.json").read_text()
)
PATH = ReferencePath(Waypoint(*p) for p in GOLDEN["route"])


def square(cx, cy, half):
    return [
        (cx - half, cy - half),
        (cx + half, cy - half),
        (cx + half, cy + half),
        (cx - half, cy + half),
    ]


def test_gap_matches_the_golden_front_surface_definition():
    # Near face of the box at the golden obstacle point (10, 8), after the bend.
    corners = square(10.0, 8.5, 0.5)
    truth = route_surface_gap(
        PATH, *GOLDEN["ego_rear_axle"], GOLDEN["ego_front_offset_m"], corners, 1.25
    )
    assert truth["truth_front_gap_m"] == pytest.approx(GOLDEN["expected"]["front_surface_gap_m"])
    assert truth["truth_surface_s_m"] == pytest.approx(GOLDEN["expected"]["route_progress_m"])
    assert truth["truth_in_corridor"] is True


def test_signed_lateral_matches_the_golden_corridor_point():
    assert signed_lateral_m(PATH, *GOLDEN["signed_corridor_point"]) == pytest.approx(
        GOLDEN["expected"]["signed_lateral_m"]
    )


def test_obstacles_beside_or_behind_the_ego_are_not_in_its_path():
    beside = route_surface_gap(PATH, 0.0, 0.0, 4.0, square(5.0, 3.0, 0.5), 1.25)
    assert beside["truth_front_gap_m"] is None and beside["truth_in_corridor"] is False
    assert beside["truth_obstacle_lateral_m"] == pytest.approx([2.5, 3.5])
    behind = route_surface_gap(PATH, 8.0, 0.0, 4.0, square(3.0, 0.0, 0.5), 1.25)
    assert behind["truth_front_gap_m"] is None


def test_partial_overlap_counts_and_the_offset_is_applied_once():
    # Box edge 0.25 m inside a 1.25 m corridor, near face at s = 7.
    truth = route_surface_gap(PATH, 1.0, 0.0, 3.78, [(7, 1.0), (9, 1.0), (9, 3.0), (7, 3.0)], 1.25)
    assert truth["truth_front_gap_m"] == pytest.approx(7.0 - 1.0 - 3.78)


def test_front_offset_and_samples():
    assert ego_front_offset_m(1.38641, 0.0, 2.396) == pytest.approx(3.78241)
    with pytest.raises(ValueError):
        ego_front_offset_m(-5.0, 0.0, 1.0)
    samples = footprint_samples(square(0, 0, 1))
    assert len(samples) == 4 * 40 and (-1, -1) in samples
    with pytest.raises(ValueError):
        footprint_samples([(0, 0), (1, 1)])


def rows(speeds_and_gaps, dt=0.05):
    return [
        {"sim_time_s": i * dt, "speed_mps": v, "truth_front_gap_m": g}
        for i, (v, g) in enumerate(speeds_and_gaps)
    ]


def test_stop_hold_resume_metrics():
    approach = [(4.0 - 0.09 * i, 12.0 - 0.2 * i) for i in range(40)]  # to t = 1.95 s, >= 0.49 m/s
    hold = [(0.0, 4.2)] * 30 + [(0.05, 4.18)] + [(0.0, 4.18)] * 49  # creep inside the hold
    after = [(0.0, None)] * 31 + [(0.5, None)] * 20  # removed at t = 6.0 s; resumes 1.55 s later
    data = rows(approach + hold + after)
    removal = data[len(approach) + len(hold)]["sim_time_s"]
    metrics = hold_metrics(data, dt_s=0.05, removal_time_s=removal)
    assert metrics["hold_start_s"] == pytest.approx(2.0)
    assert metrics["stopped_margin_min_m"] == pytest.approx(4.18)
    assert metrics["stopped_margin_max_m"] == pytest.approx(4.2)
    assert metrics["hold_max_speed_mps"] == pytest.approx(0.05)
    assert metrics["resume_delay_s"] == pytest.approx(1.55)
    assert metrics["hold_samples"] == 80


def test_goal_stop_without_an_obstacle_is_not_a_hold():
    data = rows([(3.0, None)] * 20 + [(0.0, None)] * 100)
    metrics = hold_metrics(data, dt_s=0.05)
    assert metrics["hold_start_s"] is None
    assert metrics["stopped_margin_min_m"] is None and metrics["hold_max_speed_mps"] is None


def test_persistent_hold_duration_and_creep_breaks_it():
    steady = rows([(0.0, 4.5)] * 1300)  # 65 s, never removed (S4)
    assert hold_metrics(steady, dt_s=0.05)["continuous_hold_s"] == pytest.approx(65.0)
    broken = rows([(0.0, 4.5)] * 700 + [(0.2, 4.4)] + [(0.0, 4.4)] * 599)
    metrics = hold_metrics(broken, dt_s=0.05)
    assert metrics["hold_max_speed_mps"] == pytest.approx(0.2)
    assert metrics["continuous_hold_s"] == pytest.approx(35.0)
    # Review of PR #98: a 2 s interval without obstacle-in-path evidence was credited as
    # part of a 65 s continuous HOLD. It breaks the run like motion does.
    lapsed = rows([(0.0, 4.5)] * 700 + [(0.0, None)] * 40 + [(0.0, 4.5)] * 560)
    metrics = hold_metrics(lapsed, dt_s=0.05)
    assert metrics["continuous_hold_s"] == pytest.approx(35.0)
    assert metrics["hold_max_speed_mps"] == 0.0


def test_never_resuming_is_missing_evidence_not_zero():
    data = rows([(0.0, 4.0)] * 60 + [(0.0, None)] * 60)
    metrics = hold_metrics(data, dt_s=0.05, removal_time_s=3.0)
    assert metrics["resume_delay_s"] is None
