from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .a2c_gae import A2CConfig, A2CGAEAgent
from .doubleq import DoubleQAgent, DoubleQConfig
from .envs import RewardConfig, make_env
from .eval import evaluate
from .features import OBS_DIM, obs_to_onehot
from .utils import ensure_dir, save_json, set_global_seeds
from .viz import plot_policy_heatmaps


def train_doubleq(
    seed: int,
    reward_cfg: RewardConfig,
    env_kwargs: dict,
    cfg: DoubleQConfig,
    train_episodes: int,
    eval_episodes: int,
    eval_every: int,
    outdir: Path,
):
    set_global_seeds(seed)
    env = make_env(seed=seed, reward_cfg=reward_cfg, **env_kwargs)
    agent = DoubleQAgent(n_actions=env.action_space.n, cfg=cfg, seed=seed)

    log_path = outdir / "metrics.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "episode",
                "epsilon",
                "eval_mean_return",
                "win_rate",
                "draw_rate",
                "loss_rate",
                "mean_len",
            ],
        )
        writer.writeheader()

        for ep in tqdm(range(1, train_episodes + 1), desc="train DoubleQ"):
            obs, _ = env.reset()
            terminated = truncated = False

            while not (terminated or truncated):
                a = agent.act(obs, train=True)
                next_obs, r, terminated, truncated, info = env.step(a)
                agent.update(obs, a, r, terminated, next_obs)
                obs = next_obs

            agent.end_episode()

            if ep % eval_every == 0 or ep == train_episodes:
                # Evaluate on true reward objective: use same reward wrapper but read true_reward from info
                def env_factory():
                    return make_env(
                        seed=seed + 123, reward_cfg=reward_cfg, **env_kwargs
                    )

                res = evaluate(
                    env_factory,
                    lambda o: agent.greedy_action(o),
                    episodes=eval_episodes,
                )

                writer.writerow(
                    {
                        "step": ep,
                        "episode": ep,
                        "epsilon": agent.epsilon,
                        "eval_mean_return": res.mean_return,
                        "win_rate": res.win_rate,
                        "draw_rate": res.draw_rate,
                        "loss_rate": res.loss_rate,
                        "mean_len": res.mean_len,
                    }
                )
                f.flush()

    env.close()

    # Save policy heatmaps
    figdir = ensure_dir(outdir / "figures")
    plot_policy_heatmaps(
        lambda o: agent.greedy_action(o),
        str(figdir / "policy_doubleq.png"),
        title_prefix="DoubleQ",
    )


def collect_rollout(env, agent: A2CGAEAgent, rollout_steps: int):
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

    # Bootstrap value from last obs
    x_last = obs_to_onehot(obs)
    _, _, last_value, _ = agent.act(
        x_last, train=False
    )  # value only; deterministic action irrelevant
    return {
        "obs": np.asarray(batch_obs, dtype=np.float32),
        "actions": np.asarray(batch_actions, dtype=np.int64),
        "rewards": np.asarray(batch_rewards, dtype=np.float32),
        "dones": np.asarray(batch_dones, dtype=np.float32),
        "values": np.asarray(batch_values, dtype=np.float32),
        "logprobs": np.asarray(batch_logprobs, dtype=np.float32),
        "last_value": float(last_value.item()),
    }


def train_a2c(
    seed: int,
    reward_cfg: RewardConfig,
    env_kwargs: dict,
    cfg: A2CConfig,
    train_steps: int,
    rollout_steps: int,
    eval_episodes: int,
    eval_every: int,
    outdir: Path,
):
    set_global_seeds(seed)
    env = make_env(seed=seed, reward_cfg=reward_cfg, **env_kwargs)
    agent = A2CGAEAgent(
        obs_dim=OBS_DIM, n_actions=env.action_space.n, cfg=cfg, seed=seed
    )

    log_path = outdir / "metrics.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "update",
                "loss",
                "policy_loss",
                "value_loss",
                "entropy",
                "approx_kl",
                "eval_mean_return",
                "win_rate",
                "draw_rate",
                "loss_rate",
                "mean_len",
            ],
        )
        writer.writeheader()

        updates = train_steps // rollout_steps
        for u in tqdm(range(1, updates + 1), desc="train A2C-GAE"):
            batch = collect_rollout(env, agent, rollout_steps)
            stats = agent.update(batch)

            step = u * rollout_steps

            if u % eval_every == 0 or u == updates:
                # Evaluate on TRUE reward: use wrapper but read true_reward in eval()
                def env_factory():
                    return make_env(
                        seed=seed + 123, reward_cfg=reward_cfg, **env_kwargs
                    )

                res = evaluate(
                    env_factory,
                    act_fn=lambda obs: agent.act(obs_to_onehot(obs), train=False)[0],
                    episodes=eval_episodes,
                )

                row = {
                    "step": step,
                    "update": u,
                    **stats,
                    "eval_mean_return": res.mean_return,
                    "win_rate": res.win_rate,
                    "draw_rate": res.draw_rate,
                    "loss_rate": res.loss_rate,
                    "mean_len": res.mean_len,
                }
                writer.writerow(row)
                f.flush()

    env.close()

    # Save policy heatmaps
    figdir = ensure_dir(outdir / "figures")
    plot_policy_heatmaps(
        lambda o: agent.act(obs_to_onehot(o), train=False)[0],
        str(figdir / "policy_a2c.png"),
        title_prefix="A2C-GAE",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["doubleq", "a2c"], required=True)
    p.add_argument("--reward", choices=["r0", "r1", "r2", "r3"], default="r0")
    p.add_argument("--seed", type=int, default=0)

    # env flags
    p.add_argument("--natural", action="store_true")
    p.add_argument("--sab", action="store_true")

    # reward shaping params
    p.add_argument("--step_penalty", type=float, default=0.01)
    p.add_argument("--bust_penalty", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.99)

    # DoubleQ params
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--eps_start", type=float, default=1.0)
    p.add_argument("--eps_end", type=float, default=0.05)
    p.add_argument("--eps_decay_episodes", type=int, default=100_000)
    p.add_argument("--train_episodes", type=int, default=200_000)
    p.add_argument("--eval_every_episodes", type=int, default=25_000)

    # A2C params
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--value_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--hidden1", type=int, default=128)
    p.add_argument("--hidden2", type=int, default=128)
    p.add_argument("--train_steps", type=int, default=300_000)
    p.add_argument("--rollout_steps", type=int, default=256)
    p.add_argument("--eval_every_updates", type=int, default=10)
    p.add_argument("--device", type=str, default="cpu")

    # shared eval
    p.add_argument("--eval_episodes", type=int, default=20_000)

    # output
    p.add_argument("--outdir", type=str, default="results")

    args = p.parse_args()

    outdir = Path(args.outdir)
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = outdir / f"{args.algo}_{args.reward}_seed{args.seed}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs = dict(natural=args.natural, sab=args.sab)

    reward_cfg = RewardConfig(
        mode=args.reward,
        step_penalty=args.step_penalty,
        bust_penalty=args.bust_penalty,
        gamma=args.gamma,
    )

    save_json(run_dir / "config.json", vars(args) | {"reward_cfg": reward_cfg.__dict__})

    if args.algo == "doubleq":
        cfg = DoubleQConfig(
            alpha=args.alpha,
            gamma=args.gamma,
            eps_start=args.eps_start,
            eps_end=args.eps_end,
            eps_decay_episodes=args.eps_decay_episodes,
        )
        train_doubleq(
            seed=args.seed,
            reward_cfg=reward_cfg,
            env_kwargs=env_kwargs,
            cfg=cfg,
            train_episodes=args.train_episodes,
            eval_episodes=args.eval_episodes,
            eval_every=args.eval_every_episodes,
            outdir=run_dir,
        )

    else:
        cfg = A2CConfig(
            lr=args.lr,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            max_grad_norm=args.max_grad_norm,
            hidden_sizes=(args.hidden1, args.hidden2),
            device=args.device,
        )
        train_a2c(
            seed=args.seed,
            reward_cfg=reward_cfg,
            env_kwargs=env_kwargs,
            cfg=cfg,
            train_steps=args.train_steps,
            rollout_steps=args.rollout_steps,
            eval_episodes=args.eval_episodes,
            eval_every=args.eval_every_updates,
            outdir=run_dir,
        )


if __name__ == "__main__":
    main()
