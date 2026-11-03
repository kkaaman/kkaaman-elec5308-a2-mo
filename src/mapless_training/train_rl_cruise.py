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
    FEATURE_VERSION,
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
    # The network before any learning: evaluated beside the trained one (did training change
    # what it does, not just the training return).
    export_policy(model, out_dir / f"cruise_{seed}_init.npz", provenance=json.dumps(dict(seed=seed, steps=0, feature_version=FEATURE_VERSION)))
    begun = time.time()
    model.learn(total_timesteps=steps)
    venv.close()
    model_path = out_dir / f"dqn_{seed}.zip"
    model.save(model_path)
    policy_path = out_dir / f"cruise_{seed}.npz"
    export_policy(model, policy_path, provenance=json.dumps(dict(seed=seed, steps=steps, envs=envs, max_decisions=max_decisions, learning_starts=learning_starts, feature_version=FEATURE_VERSION, wall_s=round(time.time() - begun, 1))))
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
    env_goal = env.spec.goal_m
    total, info, actions, binding = 0.0, {}, [], []
    while True:
        action = int(act(obs))
        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        binding.append(bool(info.get("binding")))
        total += reward
        if terminated or truncated:
            break
    env.close()
    return dict(
        policy=name, episode_seed=episode_seed, finished=bool(info.get("finished")), failure=info.get("failure"), truncated=bool(truncated),
        goal_m=round(float(env_goal), 1), progress=round(min(1.0, float(info.get("s_m") or 0.0) / max(float(env_goal), 1e-6)), 3),
        time_s=info.get("time_s"), max_offset_m=info.get("max_offset_m"), s_m=info.get("s_m"), return_=round(total, 3),
        share_4mps=round(float(np.mean([a == 2 for a in actions])), 3) if actions else None,
        final_speed_mps=info.get("speed_mps"),
        binding_share=round(float(np.mean(binding)), 3) if binding else None,
        share_4mps_not_binding=round(float(np.mean([a == 2 for a, b in zip(actions, binding) if not b])), 3) if not all(binding) else None,
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
            timeouts=sum(1 for r in mine if r["truncated"] and not r["finished"] and not r["failure"]),
            completion_rate=round(len(done) / len(mine), 3) if mine else None,
            mean_progress=round(float(np.mean([r["progress"] for r in mine])), 3) if mine else None,
            mean_time_s_finished=round(float(np.mean([r["time_s"] for r in done])), 2) if done else None,
            max_offset_m=round(float(max(r["max_offset_m"] for r in mine)), 3) if mine else None,
            mean_return=round(float(np.mean([r["return_"] for r in mine])), 3) if mine else None,
            share_4mps=round(float(np.mean([r["share_4mps"] for r in mine if r["share_4mps"] is not None])), 3) if mine else None,
            binding_share=round(float(np.mean([r["binding_share"] for r in mine if r["binding_share"] is not None])), 3) if mine else None,
            max_final_speed_mps_finished=round(float(max(r["final_speed_mps"] or 0.0 for r in done)), 2) if done else None,
        )
    # Paired: each policy against fixed_4 and the rule on the roads both finished (the mean
    # over finished roads alone favours a policy that finishes only the short ones).
    for base in ("fixed_4", "rule"):
        ref = {r["episode_seed"]: r for r in rows if r["policy"] == base and r["finished"]}
        for name in summary:
            both = [(r, ref[r["episode_seed"]]) for r in rows if r["policy"] == name and r["finished"] and r["episode_seed"] in ref]
            summary[name][f"paired_vs_{base}"] = dict(
                roads=len(both), mean_time_diff_s=round(float(np.mean([a["time_s"] - b["time_s"] for a, b in both])), 2) if both else None,
            )
    return summary, rows


def _reusable(policy_path, args):
    """A finished seed's export, checked before ``--resume`` skips training it: it loads,
    its features and actions are these, its feature version is this one, and it was
    trained with these settings. Anything else is refused, not silently reused."""
    try:
        policy = CruisePolicy(policy_path)
        made = json.loads(policy.provenance or "{}")
    except Exception as error:  # noqa: BLE001
        raise SystemExit(f"{policy_path}: cannot be reused ({error}); use a new --output")
    want = dict(feature_version=FEATURE_VERSION, steps=args.steps, envs=args.envs, max_decisions=args.max_decisions, learning_starts=args.learning_starts)
    if policy.feature_version != FEATURE_VERSION or any(made.get(k) != v for k, v in want.items()):
        raise SystemExit(f"{policy_path}: trained with other settings or features ({made}); use a new --output")
    return True


def manifest(args):
    """What produced this run: the source commit, package versions, every setting, the reward."""
    import dataclasses  # noqa: PLC0415
    import platform  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    from mapless_training.rl_cruise import RewardConfig  # noqa: PLC0415

    def version(name):
        try:
            return __import__(name).__version__
        except Exception:  # noqa: BLE001
            return None

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False, cwd=Path(__file__).resolve().parent).stdout.strip() or None
    except Exception:  # noqa: BLE001
        sha = None
    return dict(
        source_commit=sha, python=platform.python_version(), platform=platform.platform(),
        packages={n: version(n) for n in ("numpy", "torch", "gymnasium", "stable_baselines3")},
        args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        reward=dataclasses.asdict(RewardConfig()), actions_mps=list(ACTIONS_MPS),
    )


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
    p.add_argument("--max-decisions", type=int, default=1500, help="training episodes' horizon (default 300 s, as the evaluation's: at 120 s fixed 2 m/s could not finish 42 %% of training roads)")
    p.add_argument("--learning-starts", type=int, default=2000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--eval-seed", type=int, default=900_000, help="first held-out road (no training seed comes near)")
    p.add_argument("--workers", type=int, default=8, help="evaluation processes")
    p.add_argument("--evaluate-only", action="store_true", help="skip training: evaluate the cruise_<seed>.npz already in --output")
    p.add_argument("--resume", action="store_true", help="--output may exist: seeds whose cruise_<seed>.npz is there are not trained again")
    p.add_argument("--eval-max-decisions", type=int, default=1500, help="the evaluation's horizon (default 300 s: the longest held-out road, 407 m, needs ~204 s at 2 m/s; 600 = training's 120 s timed every fixed-2 run out on half the roads)")
    args = p.parse_args(argv)
    out = args.output
    if not args.evaluate_only:
        if out.exists() and not args.resume:
            p.error("Output folder exists; choose a new one (or --resume)")
        out.mkdir(parents=True, exist_ok=True)
        (out / f"manifest_{int(time.time())}.json").write_text(json.dumps(manifest(args), indent=1), encoding="utf-8")
    policies = []
    for seed in args.seeds:
        policy_path = out / f"cruise_{seed}.npz"
        if not args.evaluate_only and not (args.resume and policy_path.exists() and _reusable(policy_path, args)):
            _, policy_path = train_one(seed, args.steps, args.envs, out, args.max_decisions, args.learning_starts)
        policies.append((f"dqn_{seed}", policy_path))
    policies += [(f"init_{seed}", out / f"cruise_{seed}_init.npz") for seed in args.seeds if (out / f"cruise_{seed}_init.npz").exists()]
    plot_curves(out, args.seeds, out / "learning_curves.png")
    policies += [("rule", None)] + [(f"fixed_{v:g}", None) for v in ACTIONS_MPS]
    summary, rows = evaluate(policies, args.eval_episodes, args.eval_seed, args.eval_max_decisions, args.workers)
    report = dict(
        purpose="G5 RL experiment: the cruise advisor against the rule and fixed speeds on held-out synthetic roads; never in the drive",
        seeds=args.seeds, steps_per_seed=args.steps, envs=args.envs, eval_episodes=args.eval_episodes, eval_seed=args.eval_seed, eval_max_decisions=args.eval_max_decisions,
        manifest=manifest(args),
        summary=summary, episodes=rows,
    )
    (out / "evaluation.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return report


if __name__ == "__main__":
    main()
