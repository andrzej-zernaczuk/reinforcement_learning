"""Training script for reinforcement learning agents on Blackjack.

This module provides the main entry point for training Double Q-learning and
A2C-GAE agents on the Gymnasium Blackjack environment. It handles command-line
argument parsing, training loop execution, periodic evaluation, and results saving.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium.spaces import Discrete
from tqdm import tqdm  # type: ignore[import-untyped]

from .a2c_gae import A2CConfig, A2CGAEAgent
from .doubleq import DoubleQAgent, DoubleQConfig
from .envs import RewardConfig, make_env
from .eval import evaluate
from .features import OBS_DIM, obs_to_onehot
from .metrics import create_a2c_metrics_logger, create_doubleq_metrics_logger
from .training import collect_rollout
from .utils import ensure_dir, save_json, set_global_seeds
from .viz import plot_policy_heatmaps


def train_doubleq(
    seed: int,
    reward_config: RewardConfig,
    env_kwargs: dict[str, Any],
    agent_config: DoubleQConfig,
    train_episodes: int,
    eval_episodes: int,
    eval_every: int,
    output_dir: Path,
) -> None:
    """Train a Double Q-learning agent on Blackjack.

    Executes the full training loop for Double Q-learning, including periodic
    evaluation and policy visualization. Training uses epsilon-greedy exploration
    with linear decay, while evaluation uses the greedy policy.

    Args:
        seed: Random seed for reproducibility.
        reward_config: Configuration for reward shaping.
        env_kwargs: Additional environment arguments (natural, sab).
        agent_config: Double Q-learning hyperparameters.
        train_episodes: Total number of episodes to train for.
        eval_episodes: Number of episodes to run during each evaluation.
        eval_every: Evaluate every N episodes during training.
        output_dir: Directory where results will be saved.
    """
    set_global_seeds(seed)
    environment = make_env(seed=seed, reward_cfg=reward_config, **env_kwargs)
    action_space = environment.action_space
    assert isinstance(action_space, Discrete), "Action space must be Discrete"
    num_actions = int(action_space.n)
    agent = DoubleQAgent(num_actions=num_actions, config=agent_config, seed=seed)

    log_path = output_dir / "metrics.csv"
    with create_doubleq_metrics_logger(log_path) as metrics_logger:
        for episode_num in tqdm(range(1, train_episodes + 1), desc="train DoubleQ"):
            observation, _ = environment.reset()
            terminated = truncated = False

            while not (terminated or truncated):
                action = agent.act(observation, train=True)
                next_observation, reward, terminated, truncated, info = environment.step(action)
                agent.update(observation, action, float(reward), terminated, next_observation)
                observation = next_observation

            agent.end_episode()

            if episode_num % eval_every == 0 or episode_num == train_episodes:
                # Evaluate on true reward objective using greedy policy
                def env_factory() -> gym.Env:
                    return make_env(seed=seed + 123, reward_cfg=reward_config, **env_kwargs)

                evaluation_results = evaluate(
                    env_factory,
                    lambda obs: agent.greedy_action(obs),
                    episodes=eval_episodes,
                )

                metrics_logger.log(
                    {
                        "step": episode_num,
                        "episode": episode_num,
                        "epsilon": agent.epsilon,
                        "eval_mean_return": evaluation_results.mean_return,
                        "win_rate": evaluation_results.win_rate,
                        "draw_rate": evaluation_results.draw_rate,
                        "loss_rate": evaluation_results.loss_rate,
                        "mean_len": evaluation_results.mean_len,
                    }
                )

    environment.close()

    # Save policy visualization heatmaps
    figures_dir = ensure_dir(output_dir / "figures")
    plot_policy_heatmaps(
        lambda obs: agent.greedy_action(obs),
        str(figures_dir / "policy_doubleq.png"),
        title_prefix="DoubleQ",
    )


def train_a2c(
    seed: int,
    reward_config: RewardConfig,
    env_kwargs: dict[str, Any],
    agent_config: A2CConfig,
    train_steps: int,
    rollout_steps: int,
    eval_episodes: int,
    eval_every: int,
    output_dir: Path,
) -> None:
    """Train an A2C-GAE agent on Blackjack.

    Executes the full training loop for A2C with GAE, including rollout collection,
    policy updates, periodic evaluation, and policy visualization.

    Args:
        seed: Random seed for reproducibility.
        reward_config: Configuration for reward shaping.
        env_kwargs: Additional environment arguments (natural, sab).
        agent_config: A2C-GAE hyperparameters.
        train_steps: Total number of environment steps to train for.
        rollout_steps: Number of steps per rollout (batch size for updates).
        eval_episodes: Number of episodes to run during each evaluation.
        eval_every: Evaluate every N updates during training.
        output_dir: Directory where results will be saved.
    """
    set_global_seeds(seed)
    environment = make_env(seed=seed, reward_cfg=reward_config, **env_kwargs)
    action_space = environment.action_space
    assert isinstance(action_space, Discrete), "Action space must be Discrete"
    num_actions = int(action_space.n)
    agent = A2CGAEAgent(
        obs_dim=OBS_DIM, num_actions=num_actions, config=agent_config, seed=seed
    )

    log_path = output_dir / "metrics.csv"
    with create_a2c_metrics_logger(log_path) as metrics_logger:
        num_updates = train_steps // rollout_steps
        for update_num in tqdm(range(1, num_updates + 1), desc="train A2C-GAE"):
            rollout_data = collect_rollout(environment, agent, rollout_steps)

            # Convert RolloutData to dict for backward compatibility with agent.update
            batch_dict = {
                "obs": rollout_data.observations,
                "actions": rollout_data.actions,
                "rewards": rollout_data.rewards,
                "dones": rollout_data.episode_dones,
                "values": rollout_data.value_estimates,
                "logprobs": rollout_data.action_log_probabilities,
                "last_value": rollout_data.bootstrap_value,
            }
            training_stats = agent.update(batch_dict)

            current_step = update_num * rollout_steps

            if update_num % eval_every == 0 or update_num == num_updates:
                # Evaluate on TRUE reward using greedy policy
                def env_factory() -> gym.Env:
                    return make_env(seed=seed + 123, reward_cfg=reward_config, **env_kwargs)

                evaluation_results = evaluate(
                    env_factory,
                    action_function=lambda obs: agent.act(obs_to_onehot(obs), train=False)[0],
                    episodes=eval_episodes,
                )

                metrics_row = {
                    "step": current_step,
                    "update": update_num,
                    **training_stats,
                    "eval_mean_return": evaluation_results.mean_return,
                    "win_rate": evaluation_results.win_rate,
                    "draw_rate": evaluation_results.draw_rate,
                    "loss_rate": evaluation_results.loss_rate,
                    "mean_len": evaluation_results.mean_len,
                }
                metrics_logger.log(metrics_row)

    environment.close()

    # Save policy visualization heatmaps
    figures_dir = ensure_dir(output_dir / "figures")
    plot_policy_heatmaps(
        lambda obs: agent.act(obs_to_onehot(obs), train=False)[0],
        str(figures_dir / "policy_a2c.png"),
        title_prefix="A2C-GAE",
    )


def main() -> None:
    """Parse command-line arguments and execute training.

    Provides a command-line interface for training Double Q-learning or A2C-GAE
    agents on Blackjack with various configurations. Results are saved to timestamped
    directories for later analysis.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["doubleq", "a2c"], required=True)
    parser.add_argument("--reward", choices=["r0", "r1", "r2", "r3"], default="r0")
    parser.add_argument("--seed", type=int, default=0)

    # Environment flags
    parser.add_argument("--natural", action="store_true")
    parser.add_argument("--sab", action="store_true")

    # Reward shaping parameters
    parser.add_argument("--step_penalty", type=float, default=0.01)
    parser.add_argument("--bust_penalty", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)

    # DoubleQ parameters
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--eps_decay_episodes", type=int, default=100_000)
    parser.add_argument("--train_episodes", type=int, default=200_000)
    parser.add_argument("--eval_every_episodes", type=int, default=25_000)

    # A2C parameters
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--hidden1", type=int, default=128)
    parser.add_argument("--hidden2", type=int, default=128)
    parser.add_argument("--train_steps", type=int, default=300_000)
    parser.add_argument("--rollout_steps", type=int, default=256)
    parser.add_argument("--eval_every_updates", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")

    # Shared evaluation parameters
    parser.add_argument("--eval_episodes", type=int, default=20_000)

    # Output directory
    parser.add_argument("--outdir", type=str, default="results")

    args = parser.parse_args()

    output_dir = Path(args.outdir)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = output_dir / f"{args.algo}_{args.reward}_seed{args.seed}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs = dict(natural=args.natural, sab=args.sab)

    reward_config = RewardConfig(
        mode=args.reward,
        step_penalty=args.step_penalty,
        bust_penalty=args.bust_penalty,
        gamma=args.gamma,
    )

    save_json(run_dir / "config.json", vars(args) | {"reward_cfg": reward_config.__dict__})

    if args.algo == "doubleq":
        doubleq_config = DoubleQConfig(
            alpha=args.alpha,
            gamma=args.gamma,
            eps_start=args.eps_start,
            eps_end=args.eps_end,
            eps_decay_episodes=args.eps_decay_episodes,
        )
        train_doubleq(
            seed=args.seed,
            reward_config=reward_config,
            env_kwargs=env_kwargs,
            agent_config=doubleq_config,
            train_episodes=args.train_episodes,
            eval_episodes=args.eval_episodes,
            eval_every=args.eval_every_episodes,
            output_dir=run_dir,
        )

    else:
        a2c_config = A2CConfig(
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
            reward_config=reward_config,
            env_kwargs=env_kwargs,
            agent_config=a2c_config,
            train_steps=args.train_steps,
            rollout_steps=args.rollout_steps,
            eval_episodes=args.eval_episodes,
            eval_every=args.eval_every_updates,
            output_dir=run_dir,
        )


if __name__ == "__main__":
    main()
