"""Train the cruise advisor by RL and compare it with the rule (G5's RL experiment, 2 Nov).

DQN (stable-baselines3) on ``rl_cruise.CruiseEnv``: the real driver on random synthetic
roads, the policy choosing 2, 3 or 4 m/s every ``DECISION_TICKS``. One run per seed, each
on ``--envs`` environments in parallel processes; the episode returns are logged as they
come (``monitor_<seed>.monitor.csv``) and plotted (``learning_curves.png``).

The evaluation then drives the same held-out roads (episode seeds ``--eval-seed`` on)
with every trained policy, the hand-written rule (``rule_action``) and the three fixed
speeds, and reports per policy: roads finished, failures (lane, collision, fault), time
to the goal, the largest lane offset and the mean return (``evaluation.json``).

It is an experiment, never in the drive: no CARLA town is involved, and the deployed
form (``CruisePolicy``, numpy) is written but not wired into any run. The expected
result is a null one: in the synthetic world 4 m/s costs the driver nothing, so the
best policy is "always 4" (fixed 4 m/s), which the rule nearly is.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

from mapless_training.rl_cruise import (
    ACTIONS_MPS,
    CruiseEnv,
    CruisePolicy,
    export_policy,
    make_env,
    rule_action,
)


def _env_fn(seed, max_decisions):
    def build():
        return make_env(seed, max_decisions=max_decisions)

    return build


def train_one(seed, steps, envs, out_dir, max_decisions=600, learning_starts=2000):
    """One DQN run; returns the paths of its model and exported policy."""
    from stable_baselines3 import DQN  # noqa: PLC0415
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor  # noqa: PLC0415

    fns = [_env_fn(seed * 1000 + i, max_decisions) for i in range(envs)]
    venv = SubprocVecEnv(fns) if envs > 1 else DummyVecEnv(fns)
    venv = VecMonitor(venv, str(out_dir / f"monitor_{seed}"))
    model = DQN(
        "MlpPolicy", venv, learning_rate=5e-4, buffer_size=100_000, learning_starts=learning_starts, batch_size=128,
        gamma=0.98, train_freq=4, gradient_steps=1, target_update_interval=2000, exploration_fraction=0.3,
        exploration_final_eps=0.05, policy_kwargs=dict(net_arch=[64, 64]), seed=seed, verbose=0, device="cpu",
    )
    begun = time.time()
    model.learn(total_timesteps=steps)
    venv.close()
    model_path = out_dir / f"dqn_{seed}.zip"
    model.save(model_path)
    policy_path = out_dir / f"cruise_{seed}.npz"
    export_policy(model, policy_path, provenance=json.dumps(dict(seed=seed, steps=steps, envs=envs, wall_s=round(time.time() - begun, 1))))
    print(f"seed {seed}: {steps} steps in {time.time() - begun:.0f} s -> {policy_path}", file=sys.stderr, flush=True)
    return model_path, policy_path


def _episode(job):
    """One evaluation road under one policy: (policy name, episode seed, max decisions, path or None)."""
    name, episode_seed, max_decisions, path = job
    if name.startswith("fixed_"):
        fixed = ACTIONS_MPS.index(float(name.split("_")[1]))
        env = CruiseEnv(seed=episode_seed, max_decisions=max_decisions, fixed_action=fixed)

        def act(obs):
            return fixed

    else:
        env = CruiseEnv(seed=episode_seed, max_decisions=max_decisions)
        act = rule_action if name == "rule" else CruisePolicy(path).act
    obs, _ = env.reset(seed=episode_seed)
    total, info, actions = 0.0, {}, []
    while True:
        action = int(act(obs))
        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        if terminated or truncated:
            break
    env.close()
    return dict(
        policy=name, episode_seed=episode_seed, finished=bool(info.get("finished")), failure=info.get("failure"), truncated=bool(truncated),
        time_s=info.get("time_s"), max_offset_m=info.get("max_offset_m"), s_m=info.get("s_m"), return_=round(total, 3),
        share_4mps=round(float(np.mean([a == 2 for a in actions])), 3) if actions else None,
    )


def evaluate(policies, episodes, eval_seed, max_decisions, workers):
    """Every policy on the same held-out roads; per-policy summary and the raw episodes."""
    jobs = [(name, eval_seed + k, max_decisions, path) for name, path in policies for k in range(episodes)]
    if workers > 1:
        with mp.get_context("spawn").Pool(workers) as pool:
            rows = pool.map(_episode, jobs)
    else:
        rows = [_episode(j) for j in jobs]
    summary = {}
    for name, _ in policies:
        mine = [r for r in rows if r["policy"] == name]
        done = [r for r in mine if r["finished"]]
        summary[name] = dict(
            episodes=len(mine), finished=len(done), failures=sum(1 for r in mine if r["failure"]),
            mean_time_s_finished=round(float(np.mean([r["time_s"] for r in done])), 2) if done else None,
            max_offset_m=round(float(max(r["max_offset_m"] for r in mine)), 3) if mine else None,
            mean_return=round(float(np.mean([r["return_"] for r in mine])), 3) if mine else None,
            share_4mps=round(float(np.mean([r["share_4mps"] for r in mine if r["share_4mps"] is not None])), 3) if mine else None,
        )
    return summary, rows


def plot_curves(out_dir, seeds, path, window=20):
    """Rolling mean episode return against environment steps, one line per seed."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(7, 4))
    for seed in seeds:
        f = out_dir / f"monitor_{seed}.monitor.csv"
        if not f.exists():
            continue
        data = np.genfromtxt(f, delimiter=",", skip_header=2)
        if data.ndim < 2 or len(data) == 0:
            continue
        returns, lengths = data[:, 0], data[:, 1]
        steps = np.cumsum(lengths)
        k = max(1, min(window, len(returns)))
        smooth = np.convolve(returns, np.ones(k) / k, mode="valid")
        ax.plot(steps[k - 1 :], smooth, label=f"seed {seed}")
    ax.set_xlabel("environment steps (decisions)")
    ax.set_ylabel(f"episode return ({window}-episode mean)")
    ax.set_title("Cruise advisor: DQN learning curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path, required=True, help="new folder for the models, logs and evaluation")
    p.add_argument("--seeds", type=int, nargs="+", default=[5308, 5309, 5310])
    p.add_argument("--steps", type=int, default=200_000, help="environment steps (decisions) per seed")
    p.add_argument("--envs", type=int, default=8, help="parallel environments per seed (processes)")
    p.add_argument("--max-decisions", type=int, default=600)
    p.add_argument("--learning-starts", type=int, default=2000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--eval-seed", type=int, default=900_000, help="first held-out road (no training seed comes near)")
    p.add_argument("--workers", type=int, default=8, help="evaluation processes")
    p.add_argument("--evaluate-only", action="store_true", help="skip training: evaluate the cruise_<seed>.npz already in --output")
    args = p.parse_args(argv)
    out = args.output
    if not args.evaluate_only:
        if out.exists():
            p.error("Output folder exists; choose a new one")
        out.mkdir(parents=True)
    policies = []
    for seed in args.seeds:
        if args.evaluate_only:
            policy_path = out / f"cruise_{seed}.npz"
        else:
            _, policy_path = train_one(seed, args.steps, args.envs, out, args.max_decisions, args.learning_starts)
        policies.append((f"dqn_{seed}", policy_path))
    plot_curves(out, args.seeds, out / "learning_curves.png")
    policies += [("rule", None)] + [(f"fixed_{v:g}", None) for v in ACTIONS_MPS]
    summary, rows = evaluate(policies, args.eval_episodes, args.eval_seed, args.max_decisions, args.workers)
    report = dict(
        purpose="G5 RL experiment: the cruise advisor against the rule and fixed speeds on held-out synthetic roads; never in the drive",
        seeds=args.seeds, steps_per_seed=args.steps, envs=args.envs, eval_episodes=args.eval_episodes, eval_seed=args.eval_seed,
        summary=summary, episodes=rows,
    )
    (out / "evaluation.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return report


if __name__ == "__main__":
    main()
