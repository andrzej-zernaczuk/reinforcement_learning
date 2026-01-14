"""Hyperparameter tuning for reinforcement learning agents.

This module provides grid search for Double Q-learning and random search for
A2C-GAE, evaluating configurations across multiple random seeds to find robust
hyperparameter settings.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import random
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
from .training import collect_rollout
from .utils import ensure_dir, save_json, set_global_seeds


def tune_doubleq(args) -> None:
    """Run grid search hyperparameter tuning for Double Q-learning.

    Evaluates all combinations of hyperparameters in a predefined grid across
    multiple random seeds. For each configuration, trains agents with different
    seeds and reports the mean evaluation performance.

    Args:
        args: Argparse namespace containing tuning configuration including:
            - reward: Reward shaping mode
            - episodes: Number of training episodes
            - eval_episodes: Number of evaluation episodes
            - outdir: Output directory for results
    """
    evaluation_seeds = [0, 1, 2]
    output_base_dir = Path(args.outdir)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = ensure_dir(output_base_dir / f"tune_doubleq_{args.reward}_{timestamp}")

    reward_config = RewardConfig(
        mode=args.reward,
        step_penalty=args.step_penalty,
        bust_penalty=args.bust_penalty,
        gamma=args.gamma,
    )
    env_kwargs = dict(natural=args.natural, sab=args.sab)

    # Define hyperparameter grid
    hyperparameter_grid = {
        "alpha": [0.05, 0.1, 0.2],
        "gamma": [0.95, 0.99, 1.0],
        "eps_end": [0.05, 0.01],
        "eps_decay_episodes": [50_000, 200_000],
    }

    results_rows = []
    best_config = None

    grid_keys = list(hyperparameter_grid.keys())
    grid_values: list[Any] = [hyperparameter_grid[key] for key in grid_keys]
    combinations = list(itertools.product(*grid_values))  # type: ignore[arg-type]
    for hyperparameter_values in tqdm(combinations, desc="grid DoubleQ"):
        config_kwargs = dict(zip(grid_keys, hyperparameter_values))
        config_kwargs["eps_start"] = 1.0
        agent_config = DoubleQConfig(**config_kwargs)

        seed_scores = []
        for seed in evaluation_seeds:
            set_global_seeds(seed)
            environment = make_env(seed=seed, reward_cfg=reward_config, **env_kwargs)
            action_space = environment.action_space
            assert isinstance(action_space, Discrete), "Action space must be Discrete"
            num_actions = int(action_space.n)
            agent = DoubleQAgent(
                num_actions=num_actions, config=agent_config, seed=seed
            )

            # Train agent
            for episode_num in range(1, args.episodes + 1):
                observation, _ = environment.reset()
                terminated = truncated = False
                while not (terminated or truncated):
                    action = agent.act(observation, train=True)
                    next_observation, reward, terminated, truncated, info = environment.step(
                        action
                    )
                    agent.update(observation, action, float(reward), terminated, next_observation)
                    observation = next_observation
                agent.end_episode()

            def env_factory():
                return make_env(seed=seed + 123, reward_cfg=reward_config, **env_kwargs)

            evaluation_results = evaluate(
                env_factory,
                lambda obs: agent.greedy_action(obs),
                episodes=args.eval_episodes,
            )
            seed_scores.append(evaluation_results.mean_return)
            environment.close()

        mean_score = sum(seed_scores) / len(seed_scores)
        result_row = {**config_kwargs, "mean_eval_return": mean_score}
        results_rows.append(result_row)

        if best_config is None or mean_score > best_config["mean_eval_return"]:
            best_config = result_row

    # Save results
    csv_path = Path(run_dir) / "tuning_results.csv"
    with csv_path.open("w", newline="") as file_handle:
        csv_writer = csv.DictWriter(file_handle, fieldnames=list(results_rows[0].keys()))
        csv_writer.writeheader()
        csv_writer.writerows(results_rows)

    if best_config is not None:
        save_json(Path(run_dir) / "best_config.json", best_config)
        print("BEST DoubleQ:", best_config)


def tune_a2c(args) -> None:
    """Run random search hyperparameter tuning for A2C-GAE.

    Samples random hyperparameter configurations and evaluates each across multiple
    seeds. Uses log-uniform sampling for the learning rate to explore a wide range
    of scales efficiently.

    Args:
        args: Argparse namespace containing tuning configuration including:
            - reward: Reward shaping mode
            - steps: Number of training steps
            - rollout_steps: Steps per rollout
            - trials: Number of random configurations to try
            - eval_episodes: Number of evaluation episodes
            - device: Device for PyTorch computations
            - outdir: Output directory for results
    """
    evaluation_seeds = [0, 1, 2]
    output_base_dir = Path(args.outdir)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = ensure_dir(output_base_dir / f"tune_a2c_{args.reward}_{timestamp}")

    reward_config = RewardConfig(
        mode=args.reward,
        step_penalty=args.step_penalty,
        bust_penalty=args.bust_penalty,
        gamma=args.gamma,
    )
    env_kwargs = dict(natural=args.natural, sab=args.sab)

    random_generator = random.Random(args.seed)

    def sample_config() -> A2CConfig:
        """Sample a random A2C configuration.

        Returns:
            A2CConfig with randomly sampled hyperparameters.
        """
        # Use log-uniform sampling for learning rate to explore different scales
        learning_rate = 10 ** random_generator.uniform(math.log10(1e-4), math.log10(3e-3))
        hidden_sizes = random_generator.choice([(64, 64), (128, 128)])
        gamma = random_generator.choice([0.95, 0.99])
        gae_lambda = random_generator.choice([0.90, 0.95, 0.97])
        entropy_coefficient = random_generator.choice([0.0, 0.01, 0.02])

        return A2CConfig(
            lr=learning_rate,
            gamma=gamma,
            gae_lambda=gae_lambda,
            entropy_coef=entropy_coefficient,
            value_coef=0.5,
            max_grad_norm=0.5,
            hidden_sizes=hidden_sizes,
            device=args.device,
        )

    results_rows = []
    best_config = None

    for trial_num in tqdm(range(1, args.trials + 1), desc="random A2C"):
        agent_config = sample_config()
        seed_scores = []

        for seed in evaluation_seeds:
            set_global_seeds(seed)
            environment = make_env(seed=seed, reward_cfg=reward_config, **env_kwargs)
            action_space = environment.action_space
            assert isinstance(action_space, Discrete), "Action space must be Discrete"
            num_actions_a2c = int(action_space.n)
            agent = A2CGAEAgent(
                obs_dim=OBS_DIM, num_actions=num_actions_a2c, config=agent_config, seed=seed
            )

            num_updates = args.steps // args.rollout_steps
            # Train agent
            for _ in range(num_updates):
                rollout_data = collect_rollout(environment, agent, args.rollout_steps)
                # Convert RolloutData to dict for backward compatibility
                batch_dict = {
                    "obs": rollout_data.observations,
                    "actions": rollout_data.actions,
                    "rewards": rollout_data.rewards,
                    "dones": rollout_data.episode_dones,
                    "values": rollout_data.value_estimates,
                    "logprobs": rollout_data.action_log_probabilities,
                    "last_value": rollout_data.bootstrap_value,
                }
                agent.update(batch_dict)

            def env_factory() -> gym.Env:
                return make_env(seed=seed + 123, reward_cfg=reward_config, **env_kwargs)

            evaluation_results = evaluate(
                env_factory,
                action_function=lambda obs: agent.act(obs_to_onehot(obs), train=False)[0],
                episodes=args.eval_episodes,
            )
            seed_scores.append(evaluation_results.mean_return)
            environment.close()

        mean_score = sum(seed_scores) / len(seed_scores)
        result_row = {
            "trial": trial_num,
            "lr": agent_config.lr,
            "gamma": agent_config.gamma,
            "gae_lambda": agent_config.gae_lambda,
            "entropy_coef": agent_config.entropy_coef,
            "hidden1": agent_config.hidden_sizes[0],
            "hidden2": agent_config.hidden_sizes[1],
            "mean_eval_return": mean_score,
        }
        results_rows.append(result_row)

        if best_config is None or mean_score > best_config["mean_eval_return"]:
            best_config = result_row

    # Save results
    csv_path = Path(run_dir) / "tuning_results.csv"
    with csv_path.open("w", newline="") as file_handle:
        csv_writer = csv.DictWriter(file_handle, fieldnames=list(results_rows[0].keys()))
        csv_writer.writeheader()
        csv_writer.writerows(results_rows)

    if best_config is not None:
        save_json(Path(run_dir) / "best_config.json", best_config)
        print("BEST A2C:", best_config)


def main() -> None:
    """Parse command-line arguments and execute hyperparameter tuning.

    Provides a command-line interface for hyperparameter tuning of Double Q-learning
    or A2C-GAE agents. Results include CSV files with all configurations tried and
    JSON files with the best configuration found.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["doubleq", "a2c"], required=True)
    parser.add_argument("--reward", choices=["r0", "r1", "r2", "r3"], default="r0")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--natural", action="store_true")
    parser.add_argument("--sab", action="store_true")

    parser.add_argument("--step_penalty", type=float, default=0.01)
    parser.add_argument("--bust_penalty", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--outdir", type=str, default="results")

    # DoubleQ tuning parameters
    parser.add_argument("--episodes", type=int, default=150_000)

    # A2C tuning parameters
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--rollout_steps", type=int, default=256)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--eval_episodes", type=int, default=20_000)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    if args.algo == "doubleq":
        tune_doubleq(args)
    else:
        tune_a2c(args)


if __name__ == "__main__":
    main()
