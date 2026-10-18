"""Golden metric traces and evidence rejection, not the metrics implementation."""

import json
from pathlib import Path

import pytest

from evaluation.acceptance import aggregate
from evaluation.ground_truth import front_edge_gap
from evaluation.metrics import evaluate, load_run, summarize
from evaluation.run_logger import RunLogger


def manifest(scenario="S1"):
    return dict(
        engine="harness",
        scenario=scenario,
        seed=1,
        dt_s=0.05,
        cruise_mps=5,
        event_sensors_verified=True,
        truth_channel_complete=True,
        loop_wall_s=2,
        goal_tolerance_m=1,
        configuration_status="PROVISIONAL",
        config_sha256="a",
        mission_sha256="b",
        source_sha256="c",
        model_sha256="fixture",
        mode="full_stack",
    )


def row(index, **changes):
    result = dict(
        frame_id=index,
        sim_time_s=index * 0.05,
        speed_mps=5,
        cte_m=0.2,
        cruise_window=True,
        state="CRUISE",
        finished=False,
        distance_to_goal_m=20,
        target_mps=5,
        capture_apply_s=0.02,
        simulation_age_s=0.2,
        truth_in_corridor=True,
        truth_gap_m=4,
    )
    result.update(changes)
    return result


def test_golden_cte_speed_latency_margin_resume_and_collision_counts():
    rows = [
        row(0, speed_mps=4, cte_m=-0.2, capture_apply_s=0.01),
        row(1, speed_mps=6, cte_m=0.4, capture_apply_s=0.03),
        row(2, speed_mps=0, state="HOLD", target_mps=0, cruise_window=False),
        row(3, speed_mps=0, state="HOLD", target_mps=0, cruise_window=False),
        row(4, speed_mps=1, state="RESUME", cruise_window=False),
        row(
            5,
            speed_mps=0,
            state="FINISHED",
            finished=True,
            distance_to_goal_m=0.8,
            cruise_window=False,
        ),
    ]
    events = [dict(type="collision", other_actor_id=3, sim_time_s=t) for t in (0, 0.01, 1.1)]
    events += [
        dict(type="obstacle_removed", sim_time_s=0.15),
        dict(type="lane_invasion", sim_time_s=0.1, declared_legal=False),
    ]
    result = summarize(rows, events, manifest())["metrics"]
    assert result["cte_mean_m"] == pytest.approx(0.8 / 3)
    assert result["cruise_speed_rms_mps"] == 1
    assert result["capture_apply_p95_ms"] == pytest.approx(27.5)
    assert result["hold_margin_min_m"] == result["hold_margin_max_m"] == 4
    assert result["resume_delay_s"] == pytest.approx(0.05)
    assert result["collisions"] == 2 and result["unplanned_lane_invasions"] == 1
    assert result["route_completion"] == 1


def test_s4_completion_not_applicable_and_continuous_hold_not_sum():
    rows = [row(i, state="HOLD", speed_mps=0, cruise_window=False) for i in range(5)]
    rows[2] = row(2, state="STOPPING", speed_mps=0.2)
    result = summarize(rows, [], manifest("S4"))["metrics"]
    assert result["route_completion"] is None
    assert result["continuous_hold_s"] == pytest.approx(0.1)


def test_missing_sensors_and_cruise_never_fabricate_zero():
    m = manifest()
    m["event_sensors_verified"] = False
    r = summarize([row(0, speed_mps=0, cruise_window=False)], [], m)["metrics"]
    assert r["collisions"] is None and r["cruise_speed_rms_mps"] is None
    assert r["route_completion"] == 0


def test_one_second_false_stop_and_goal_exclusion():
    rows = [row(i, speed_mps=0, target_mps=0, truth_in_corridor=False) for i in range(25)]
    assert summarize(rows, [], manifest("S0"))["metrics"]["false_stops"] == 1
    for r in rows:
        r["distance_to_goal_m"] = 0.8
    assert summarize(rows, [], manifest("S0"))["metrics"]["false_stops"] == 0


def test_wait_for_clearance_after_removal_is_resume_delay_not_false_stop():
    # Live S1 (18 Oct): HOLD continued 1.0 s after the scripted removal while
    # fresh depth confirmed clearance; that is the resume delay.
    rows = [row(i, speed_mps=0, target_mps=0, truth_in_corridor=True) for i in range(20)]
    rows += [row(i, speed_mps=0, target_mps=0, truth_in_corridor=False) for i in range(20, 42)]
    rows += [row(i, speed_mps=0.5, target_mps=1, truth_in_corridor=False) for i in range(42, 50)]
    removed = [dict(type="obstacle_removed", sim_time_s=20 * 0.05)]
    metrics = summarize(rows, removed, manifest("S1"))["metrics"]
    assert metrics["false_stops"] == 0
    assert metrics["resume_delay_s"] == pytest.approx(22 * 0.05)
    assert summarize(rows, [], manifest("S1"))["metrics"]["false_stops"] == 1
    # Never moving again after a removal is not exempt.
    stuck = rows[:42] + [
        row(i, speed_mps=0, target_mps=0, truth_in_corridor=False) for i in range(42, 70)
    ]
    assert summarize(stuck, removed, manifest("S1"))["metrics"]["false_stops"] == 1


def test_front_bumper_distance_known_fixture():
    assert front_edge_gap(((0, -1), (0, 1)), ((4, -1), (6, -1), (6, 1), (4, 1))) == 4
    assert front_edge_gap(((0, -1), (0, 1)), ((-1, -0.5), (1, -0.5), (1, 0.5), (-1, 0.5))) == 0


def test_logger_rejects_reused_folders_nonmonotonic_and_modified_evidence(tmp_path):
    folder = tmp_path / "run"
    log = RunLogger(folder, manifest())
    log.sample(row(1))
    with pytest.raises(ValueError):
        log.sample(row(1))
    with pytest.raises(FileExistsError):
        RunLogger(folder, manifest())
    log.close(completed=True, cleanup={"ok": True})
    assert load_run(folder)[0]["status"] == "COLLECTED"
    (folder / "samples.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="hash"):
        load_run(folder)


def test_failed_cleanup_or_error_cannot_be_collected(tmp_path):
    log = RunLogger(tmp_path / "run", manifest())
    log.sample(row(0))
    log.close(completed=True, cleanup={"ok": False}, error="sensor cleanup")
    assert load_run(log.folder)[0]["status"] == "INCOMPLETE"


def test_harness_or_missing_runs_cannot_close_live_acceptance(tmp_path):
    criteria = json.loads(
        (Path(__file__).resolve().parents[1] / "fixtures/acceptance_definitions.json").read_text()
    )
    assert aggregate([], criteria)["status"] == "NOT_ACCEPTED"
    folder = tmp_path / "S0"
    log = RunLogger(folder, manifest("S0"))
    log.sample(row(0, finished=True, speed_mps=0, distance_to_goal_m=0.1))
    log.close(completed=True, cleanup={"ok": True})
    report = aggregate([folder], criteria)
    assert not report["coverage"] and not report["live_carla"]
    assert report["status"] == "NOT_ACCEPTED"


def test_missing_metric_is_failure():
    criteria = {
        "metrics": [
            dict(
                scenario="S0",
                metric="Cruise speed RMS (worst run)",
                applicable=True,
                rule="<=",
                upper=0.5,
                lower=None,
            )
        ]
    }
    assert evaluate({"metrics": {}}, criteria, "S0")["metrics"][0]["status"] == "MISSING"
