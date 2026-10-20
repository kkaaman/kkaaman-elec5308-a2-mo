"""TEST-03: the offline A1 CTE definition reproduces A1's own logged values."""

import math

import pytest

from autonomy.paths import DOCS_DIR
from evaluation.a1_comparison import LaneCentrelines, a1_metrics, compare, load_a1_capture
from planning.lane_graph import LaneTopology


def test_offline_definition_matches_test01_log():
    lanes = LaneCentrelines(
        LaneTopology.load(DOCS_DIR / "evidence/baseline-batch/junction-topology.json")
    )
    rows, summary = load_a1_capture(DOCS_DIR / "evidence/TEST-01-samples.csv")
    assert summary["capture_status"] == "CAPTURED"
    metrics = a1_metrics(rows, lanes)
    check = metrics["offline_definition_check_abs_diff_m"]
    assert check["count"] == len(rows) > 200
    assert check["max"] < 0.035
    assert metrics["cte_a1_definition_abs_m"]["mean"] > 0.5


def test_sign_is_positive_left_of_lane_direction():
    class Topology:
        lanes = [
            type(
                "Lane",
                (),
                dict(
                    points_xyz_m=((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)),
                    junction_id=None,
                    segment_id="1:0:-1",
                ),
            )()
        ]

    lanes = LaneCentrelines(Topology())
    assert lanes.signed_offset(5.0, 0.4)[0] == pytest.approx(0.4)
    assert lanes.signed_offset(5.0, -0.7)[0] == pytest.approx(-0.7)


def test_comparison_reports_differences_not_a_claim():
    a1 = dict(
        cte_a1_definition_abs_m=dict(mean=0.9),
        cycle_wall_before_apply_ms=dict(p95=450.0),
        real_time_factor=1.0,
    )
    run = dict(
        cte_a1_definition_abs_m=dict(mean=0.12),
        capture_apply_ms=dict(p95=40.0),
        real_time_factor=0.8,
        engine="harness",
    )
    report = compare(a1, [run, dict(run, cte_a1_definition_abs_m=dict(mean=0.2))])
    assert report["a2_worst_run_mean_abs_cte_m"] == 0.2
    assert math.isclose(report["cte_ratio_a2_over_a1"], 0.2 / 0.9)
    assert not report["a2_live_carla"]
    assert len(report["known_differences"]) >= 4
