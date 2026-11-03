"""The cruise advisor's RL experiment (G5, 2 Nov): the environment, the numpy policy and
the training-and-evaluation run, at toy size."""

import json

import numpy as np
import pytest

from mapless_training import rl_cruise


def test_the_environment_drives_the_real_driver_and_rewards_progress():
    pytest.importorskip("gymnasium")
    env = rl_cruise.CruiseEnv(seed=3, max_decisions=10)
    obs, _ = env.reset(seed=3)
    assert obs.shape == (len(rl_cruise.FEATURE_NAMES),) and np.all(np.abs(obs) <= 1.0)
    rewards, info = [], {}
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(2)
        rewards.append(reward)
        if terminated or truncated:
            break
    assert truncated and info["s_m"] > 10.0 and not info.get("failure")
    assert env.driver.cruise_target_mps == 4.0
    assert rl_cruise.rule_action(obs) in (0, 1, 2)


def test_training_exports_a_policy_and_the_evaluation_compares_it_with_the_rule(tmp_path):
    pytest.importorskip("stable_baselines3")
    from mapless_training import train_rl_cruise as t

    out = tmp_path / "rl"
    report = t.main(["--output", str(out), "--seeds", "1", "--steps", "48", "--envs", "1", "--learning-starts", "16", "--max-decisions", "8", "--eval-max-decisions", "8", "--eval-episodes", "1", "--workers", "1"])
    policy = rl_cruise.CruisePolicy(out / "cruise_1.npz")
    assert policy.act(np.zeros(len(rl_cruise.FEATURE_NAMES), dtype=np.float32)) in (0, 1, 2)
    assert set(report["summary"]) == {"dqn_1", "init_1", "rule", "fixed_2", "fixed_3", "fixed_4"}  # init_1: the network before learning
    assert all(s["episodes"] == 1 for s in report["summary"].values())
    assert json.loads((out / "evaluation.json").read_text(encoding="utf-8"))["summary"]["fixed_4"]["share_4mps"] == 1.0
    assert (out / "learning_curves.png").exists() and (out / "monitor_1.monitor.csv").exists()
    fixed4 = report["summary"]["fixed_4"]
    assert fixed4["timeouts"] + fixed4["finished"] + fixed4["failures"] == 1 and fixed4["paired_vs_fixed_4"]["roads"] == fixed4["finished"]
    assert 0.0 <= fixed4["mean_progress"] <= 1.0 and report["manifest"]["reward"]["fail_penalty"] == 30.0 and report["eval_max_decisions"] == 8
    assert list(out.glob("manifest_*.json")) and not list(out.glob("*.partial.npz"))
    assert policy.feature_version == rl_cruise.FEATURE_VERSION and 0.0 <= fixed4["binding_share"] <= 1.0 and "paired_vs_rule" in fixed4
    assert all("final_speed_mps" in r and "binding_share" in r for r in report["episodes"])
    with pytest.raises(SystemExit):
        t.main(["--output", str(out), "--seeds", "1"])  # the folder exists
    # --resume: a seed whose policy is there is not trained again; a missing one is.
    before = (out / "cruise_1.npz").stat().st_mtime_ns
    again = t.main(["--output", str(out), "--resume", "--seeds", "1", "2", "--steps", "48", "--envs", "1", "--learning-starts", "16", "--max-decisions", "8", "--eval-max-decisions", "8", "--eval-episodes", "1", "--workers", "1"])
    assert (out / "cruise_1.npz").stat().st_mtime_ns == before and (out / "cruise_2.npz").exists() and "dqn_2" in again["summary"]
    # A finished seed trained with other settings is refused, not reused.
    with pytest.raises(SystemExit):
        t.main(["--output", str(out), "--resume", "--seeds", "1", "--steps", "96", "--envs", "1", "--learning-starts", "16", "--max-decisions", "8", "--eval-max-decisions", "8", "--eval-episodes", "1", "--workers", "1"])


def test_no_sight_limit_is_encoded_as_unrestricted_not_as_a_zero_limit():
    """Review of the RL plan (3 Nov): sight_speed_mps None (no finite limit) was encoded as
    0, the same as a 0 m/s limit."""
    i = rl_cruise.FEATURE_NAMES.index("sight_speed")

    def encoded(value):
        return rl_cruise.features({"sight_speed_mps": value}, None, "driving", 0)[i]

    assert encoded(None) == 1.0 and encoded(0.0) == 0.0 and encoded(2.0) == pytest.approx(0.4) and encoded(5.0) == 1.0 and encoded(9.0) == 1.0


def test_a_goal_just_past_a_route_station_still_makes_a_route():
    """The overnight RL run (3 Nov) died when a random road's goal fell 0.3 m past a 50 m
    route station: two sat-nav points under SatNavRoute's 0.5 m apart. The station is
    dropped; the goal stays."""
    from pathlib import Path

    from mapless.sensor_suite import load_sensor_suite
    from sim_harness.mapless_road import KerbLine, MaplessRoadWorld, RoadScene, Straight

    suite = load_sensor_suite(Path(__file__).resolve().parents[2] / "configs" / "sensor_suites" / "mapless_v2.json")
    world = MaplessRoadWorld(RoadScene(KerbLine([Straight(200.0)])), suite, start_s_m=10.0)
    for goal in (60.3, 110.0, 110.9, 111.0, 160.49):
        route = world.route(goal)
        xy = np.array([world.plane.to_local(p.latitude_deg, p.longitude_deg) for p in route.points])
        assert np.all(np.hypot(*np.diff(xy, axis=0).T) >= 0.5), goal


def test_an_error_in_one_drive_ends_that_episode_not_the_run(monkeypatch):
    """A software error inside a drive truncates that episode, named in ``info["error"]``;
    the training run goes on."""
    pytest.importorskip("gymnasium")
    env = rl_cruise.CruiseEnv(seed=3, max_decisions=10)
    env.reset(seed=3)

    def broken(frame):
        raise RuntimeError("boom")

    monkeypatch.setattr(env.driver, "step", broken)
    obs, reward, terminated, truncated, info = env.step(2)
    assert truncated and not terminated and info["error"].startswith("RuntimeError") and obs.shape == (len(rl_cruise.FEATURE_NAMES),)
