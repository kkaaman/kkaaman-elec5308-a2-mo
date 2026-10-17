"""Same policy/controller composition across dynamics and sensor fixtures."""

import json
from dataclasses import replace

import pytest

from autonomy.baseline import BaselineConfig
from evaluation.runner import run_episode
from sim_harness.baseline import BaselineWorld, RasterDetector


def run(tmp_path, scenario, fault=None):
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
