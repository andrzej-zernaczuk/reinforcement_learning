from __future__ import annotations

import argparse
import csv
import itertools
import math
import random
import time
from pathlib import Path

from tqdm import tqdm

from .a2c_gae import A2CConfig, A2CGAEAgent
from .doubleq import DoubleQAgent, DoubleQConfig
from .envs import RewardConfig, make_env
from .eval import evaluate
from .features import OBS_DIM, obs_to_onehot
from .utils import ensure_dir, save_json, set_global_seeds


def tune_doubleq(args):
    seeds = [0, 1, 2]
    outdir = Path(args.outdir)
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = ensure_dir(outdir / f"tune_doubleq_{args.reward}_{ts}")

    reward_cfg = RewardConfig(
        mode=args.reward,
        step_penalty=args.step_penalty,
        bust_penalty=args.bust_penalty,
        gamma=args.gamma,
    )
    env_kwargs = dict(natural=args.natural, sab=args.sab)

    grid = {
        "alpha": [0.05, 0.1, 0.2],
        "gamma": [0.95, 0.99, 1.0],
        "eps_end": [0.05, 0.01],
        "eps_decay_episodes": [50_000, 200_000],
    }

    rows = []
    best = None

    keys = list(grid.keys())
    for values in tqdm(
        list(itertools.product(*[grid[k] for k in keys])), desc="grid DoubleQ"
    ):
        cfg_kwargs = dict(zip(keys, values))
        cfg_kwargs["eps_start"] = 1.0
        cfg = DoubleQConfig(**cfg_kwargs)

        scores = []
        for seed in seeds:
            set_global_seeds(seed)
            env = make_env(seed=seed, reward_cfg=reward_cfg, **env_kwargs)
            agent = DoubleQAgent(n_actions=env.action_space.n, cfg=cfg, seed=seed)

            # train
            for ep in range(1, args.episodes + 1):
                obs, _ = env.reset()
                terminated = truncated = False
                while not (terminated or truncated):
                    a = agent.act(obs, train=True)
                    next_obs, r, terminated, truncated, info = env.step(a)
                    agent.update(obs, a, r, terminated, next_obs)
                    obs = next_obs
                agent.end_episode()

            def env_factory():
                return make_env(seed=seed + 123, reward_cfg=reward_cfg, **env_kwargs)

            res = evaluate(
                env_factory,
                lambda o: agent.greedy_action(o),
                episodes=args.eval_episodes,
            )
            scores.append(res.mean_return)
            env.close()

        mean_score = sum(scores) / len(scores)
        row = {**cfg_kwargs, "mean_eval_return": mean_score}
        rows.append(row)

        if best is None or mean_score > best["mean_eval_return"]:
            best = row

    # save csv
    csv_path = Path(run_dir) / "tuning_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    save_json(Path(run_dir) / "best_config.json", best)
    print("BEST DoubleQ:", best)


def tune_a2c(args):
    seeds = [0, 1, 2]
    outdir = Path(args.outdir)
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = ensure_dir(outdir / f"tune_a2c_{args.reward}_{ts}")

    reward_cfg = RewardConfig(
        mode=args.reward,
        step_penalty=args.step_penalty,
        bust_penalty=args.bust_penalty,
        gamma=args.gamma,
    )
    env_kwargs = dict(natural=args.natural, sab=args.sab)

    rng = random.Random(args.seed)

    def sample_cfg() -> A2CConfig:
        # log-uniform lr
        lr = 10 ** rng.uniform(math.log10(1e-4), math.log10(3e-3))
        hidden = rng.choice([(64, 64), (128, 128)])
        gamma = rng.choice([0.95, 0.99])
        lam = rng.choice([0.90, 0.95, 0.97])
        entropy = rng.choice([0.0, 0.01, 0.02])
        return A2CConfig(
            lr=lr,
            gamma=gamma,
            gae_lambda=lam,
            entropy_coef=entropy,
            value_coef=0.5,
            max_grad_norm=0.5,
            hidden_sizes=hidden,
            device=args.device,
        )

    rows = []
    best = None

    for trial in tqdm(range(1, args.trials + 1), desc="random A2C"):
        cfg = sample_cfg()
        scores = []

        for seed in seeds:
            set_global_seeds(seed)
            env = make_env(seed=seed, reward_cfg=reward_cfg, **env_kwargs)
            agent = A2CGAEAgent(
                obs_dim=OBS_DIM, n_actions=env.action_space.n, cfg=cfg, seed=seed
            )

            updates = args.steps // args.rollout_steps
            # train
            for _ in range(updates):
                batch = _collect_rollout(env, agent, args.rollout_steps)
                agent.update(batch)

            def env_factory():
                return make_env(seed=seed + 123, reward_cfg=reward_cfg, **env_kwargs)

            res = evaluate(
                env_factory,
                act_fn=lambda o: agent.act(obs_to_onehot(o), train=False)[0],
                episodes=args.eval_episodes,
            )
            scores.append(res.mean_return)
            env.close()

        mean_score = sum(scores) / len(scores)
        row = {
            "trial": trial,
            "lr": cfg.lr,
            "gamma": cfg.gamma,
            "gae_lambda": cfg.gae_lambda,
            "entropy_coef": cfg.entropy_coef,
            "hidden1": cfg.hidden_sizes[0],
            "hidden2": cfg.hidden_sizes[1],
            "mean_eval_return": mean_score,
        }
        rows.append(row)

        if best is None or mean_score > best["mean_eval_return"]:
            best = row

    csv_path = Path(run_dir) / "tuning_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    save_json(Path(run_dir) / "best_config.json", best)
    print("BEST A2C:", best)


def _collect_rollout(env, agent: A2CGAEAgent, rollout_steps: int):
    import numpy as np

    obs, _ = env.reset()
    batch_obs = []
    batch_actions = []
    batch_rewards = []
    batch_dones = []
    batch_values = []
    batch_logprobs = []

    for _ in range(rollout_steps):
        x = obs_to_onehot(obs)
        action, logp, value, entropy = agent.act(x, train=True)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        batch_obs.append(x)
        batch_actions.append(action)
        batch_rewards.append(float(reward))
        batch_dones.append(float(done))
        batch_values.append(float(value.item()))
        batch_logprobs.append(float(logp.item()))

        obs = next_obs
        if done:
            obs, _ = env.reset()

    x_last = obs_to_onehot(obs)
    _, _, last_value, _ = agent.act(x_last, train=False)
    return {
        "obs": np.asarray(batch_obs, dtype=np.float32),
        "actions": np.asarray(batch_actions, dtype=np.int64),
        "rewards": np.asarray(batch_rewards, dtype=np.float32),
        "dones": np.asarray(batch_dones, dtype=np.float32),
        "values": np.asarray(batch_values, dtype=np.float32),
        "logprobs": np.asarray(batch_logprobs, dtype=np.float32),
        "last_value": float(last_value.item()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["doubleq", "a2c"], required=True)
    p.add_argument("--reward", choices=["r0", "r1", "r2", "r3"], default="r0")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--natural", action="store_true")
    p.add_argument("--sab", action="store_true")

    p.add_argument("--step_penalty", type=float, default=0.01)
    p.add_argument("--bust_penalty", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.99)

    p.add_argument("--outdir", type=str, default="results")

    # doubleq
    p.add_argument("--episodes", type=int, default=150_000)
    # a2c
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--rollout_steps", type=int, default=256)
    p.add_argument("--trials", type=int, default=12)
    p.add_argument("--eval_episodes", type=int, default=20_000)
    p.add_argument("--device", type=str, default="cpu")

    args = p.parse_args()

    if args.algo == "doubleq":
        tune_doubleq(args)
    else:
        tune_a2c(args)


if __name__ == "__main__":
    main()
