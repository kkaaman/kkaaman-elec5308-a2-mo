"""Depth-only parking control with separately evaluated continuous body envelopes."""

import pytest

from sim_harness.parking import run_parking_case


@pytest.mark.parametrize("case", [
    "straight", "left", "right", "rotated", "blocked", "side_blocked", "resume", "depth_dropout",
])
def test_p1_closed_loop_acceptance(case):
    summary, rows, _, _, _ = run_parking_case(case)
    assert summary["passed"], summary
    assert summary["contacts"] == 0
    assert summary["max_speed_mps"] <= 1.0
    assert summary["max_curvature_bound"] <= summary["curvature_limit"]
    assert summary["route_install_count"] == 1
    if case == "resume":
        assert any(row["phase"] == "hold" and row["frame_id"] < 321 for row in rows)
        assert all(row["throttle"] == 0 for row in rows if 321 <= row["frame_id"] < 329)
        assert any(row["throttle"] > 0 for row in rows if row["frame_id"] > 329)
    if case == "depth_dropout":
        assert all(row["brake"] == 1 and row["throttle"] == 0
                   for row in rows if 140 <= row["frame_id"] <= 165)
