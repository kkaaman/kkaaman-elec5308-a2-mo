"""Same policy/controller composition across dynamics and sensor fixtures."""

import json
from dataclasses import replace

import pytest

from autonomy.baseline import BaselineConfig
from evaluation.runner import run_episode
from sim_harness.baseline import BaselineWorld, RasterDetector


def run(tmp_path, scenario, fault=None, **extra):
    config = BaselineConfig()
    world = BaselineWorld(scenario, fault=fault)
    return run_episode(
        world,
        RasterDetector(),
        world.points,
        config,
        tmp_path / scenario,
        dict(
            engine="harness",
            scenario=scenario,
            seed=1,
            step_wall_budget_s=10.0,
            event_sensors_verified=True,
            truth_channel_complete=True,
            **extra,
        ),
        max_sim_s=150,
    )


@pytest.mark.parametrize("scenario", ["S0", "S1", "S2", "S3", "S4", "S5"])
def test_closed_loop_baseline_scenarios(tmp_path, scenario):
    result = run(tmp_path, scenario)
    assert result["collection_status"] == "COLLECTED", result
    assert result["path_installations"] == 1
    assert result["metrics"]["collisions"] == 0
    if scenario == "S4":
        assert result["metrics"]["route_completion"] is None
        assert result["metrics"]["continuous_hold_s"] >= 60
    else:
        assert result["metrics"]["route_completion"] == 1
    if scenario in {"S1", "S2", "S5"}:
        assert result["metrics"]["resume_delay_s"] is not None
        assert result["metrics"]["hold_margin_min_m"] >= 2.5
        assert result["metrics"]["false_stops"] == 0


@pytest.mark.parametrize("fault", ["missing_depth", "stale"])
def test_stale_or_missing_sensor_latches_stop(tmp_path, fault):
    result = run(tmp_path, "S0", fault)
    assert result["collection_status"] == "INCOMPLETE"
    rows = [json.loads(x) for x in (tmp_path / "S0" / "samples.jsonl").read_text().splitlines()]
    fault_rows = [r for r in rows if r["state"] == "FAULT_STOP"]
    assert fault_rows and all(
        r["command"]["throttle"] == 0 and r["command"]["brake"] == 1 for r in fault_rows
    )
    assert result["path_installations"] == 1


def test_configuration_preserves_public_controller_types():
    c = BaselineConfig()
    with pytest.raises(ValueError):
        replace(c, dt_s=float("nan"))


SYNC_WATCHDOG = dict(mode="synchronous_watchdog", observation_age_s=5.0, step_wall_s=1.0)


def test_single_host_stall_latches_without_watchdog(tmp_path):
    result = run(tmp_path, "S0", "stall")
    assert result["collection_status"] == "INCOMPLETE"
    assert "stale" in result["error"]


def test_synchronous_watchdog_survives_and_reports_a_host_stall(tmp_path):
    result = run(tmp_path, "S0", "stall", wall_guard=SYNC_WATCHDOG)
    assert result["collection_status"] == "COLLECTED", result["error"]
    assert result["metrics"]["route_completion"] == 1
    timing = result["timing"]
    assert timing["capture_apply_over_100ms"] == 1
    assert timing["capture_apply_max_ms"] >= 1300
    assert timing["guard"] == SYNC_WATCHDOG
    # One outlier frame does not move the scored percentile.
    assert result["metrics"]["capture_apply_p95_ms"] < 100


def test_synchronous_watchdog_still_latches_a_hung_pipeline(tmp_path):
    result = run(tmp_path, "S0", "stale", wall_guard=dict(SYNC_WATCHDOG, observation_age_s=1.0))
    assert result["collection_status"] == "INCOMPLETE"
    assert "stale" in result["error"]


def test_reset_failure_keeps_a_traceback(tmp_path):
    class Hung(BaselineWorld):
        def reset(self, *, seed):
            raise RuntimeError("time-out of 60000ms while waiting for the simulator")

    world = Hung("S1")
    result = run_episode(
        world,
        RasterDetector(),
        world.points,
        BaselineConfig(),
        tmp_path / "S1",
        dict(engine="harness", scenario="S1", seed=1, step_wall_budget_s=10.0),
        max_sim_s=5,
    )
    assert result["collection_status"] == "INCOMPLETE"
    manifest = json.loads((tmp_path / "S1" / "manifest.json").read_text())
    assert "in reset" in manifest["error_traceback"]
