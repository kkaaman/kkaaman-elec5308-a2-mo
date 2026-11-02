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
    report = t.main(["--output", str(out), "--seeds", "1", "--steps", "48", "--envs", "1", "--learning-starts", "16", "--max-decisions", "8", "--eval-episodes", "1", "--workers", "1"])
    policy = rl_cruise.CruisePolicy(out / "cruise_1.npz")
    assert policy.act(np.zeros(len(rl_cruise.FEATURE_NAMES), dtype=np.float32)) in (0, 1, 2)
    assert set(report["summary"]) == {"dqn_1", "rule", "fixed_2", "fixed_3", "fixed_4"}
    assert all(s["episodes"] == 1 for s in report["summary"].values())
    assert json.loads((out / "evaluation.json").read_text(encoding="utf-8"))["summary"]["fixed_4"]["share_4mps"] == 1.0
    assert (out / "learning_curves.png").exists() and (out / "monitor_1.monitor.csv").exists()
    with pytest.raises(SystemExit):
        t.main(["--output", str(out), "--seeds", "1"])  # the folder exists
