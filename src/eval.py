"""Evaluation utilities for trained reinforcement learning agents.

This module provides functions for evaluating agent performance over multiple
episodes, tracking win/draw/loss rates and episode statistics. It properly
handles reward-shaped environments by using the true (unshaped) rewards for
evaluation metrics.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .utils import EvalResult, classify_outcome


def evaluate(
    env_factory: Callable[[], Any],
    action_function: Callable[[Any], int],
    episodes: int = 20_000,
) -> EvalResult:
    """Evaluate an agent's policy over multiple episodes.

    Runs the agent for a specified number of episodes and collects statistics
    including mean return, win/draw/loss rates, and episode lengths. When
    evaluating with reward-shaped environments, this function uses the true
    (unshaped) reward from the info dictionary to compute unbiased performance
    metrics.

    Args:
        env_factory: Callable that creates and returns a new environment instance.
            Using a factory allows proper environment setup with evaluation-specific
            seeds or configurations.
        action_function: Function that takes an observation and returns an action.
            Should implement the agent's policy (typically greedy/deterministic).
        episodes: Number of episodes to run for evaluation. More episodes provide
            more accurate statistics but take longer. Default 20,000 provides good
            statistical significance for Blackjack.

    Returns:
        EvalResult containing aggregated statistics:
            - mean_return: Average episode return (using true rewards)
            - win_rate: Fraction of episodes with positive return
            - draw_rate: Fraction of episodes with zero return
            - loss_rate: Fraction of episodes with negative return
            - mean_len: Average episode length in timesteps
    """
    environment = env_factory()
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    wins_count = draws_count = losses_count = 0

    for _ in range(episodes):
        observation, _ = environment.reset()
        terminated = truncated = False
        episode_return = 0.0
        episode_length = 0

        while not (terminated or truncated):
            action = action_function(observation)
            observation, reward, terminated, truncated, info = environment.step(action)

            # Use true reward if available (from reward shaping wrapper),
            # otherwise use the environment reward directly
            true_reward = float(info.get("true_reward", reward))
            episode_return += true_reward
            episode_length += 1

        # Classify episode outcome as win, draw, or loss
        is_win, is_draw, is_loss = classify_outcome(episode_return)
        wins_count += is_win
        draws_count += is_draw
        losses_count += is_loss

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

    environment.close()

    # Convert to numpy arrays for efficient statistics computation
    returns_array = np.asarray(episode_returns, dtype=np.float64)
    lengths_array = np.asarray(episode_lengths, dtype=np.float64)

    return EvalResult(
        mean_return=float(returns_array.mean()),
        win_rate=float(wins_count / episodes),
        draw_rate=float(draws_count / episodes),
        loss_rate=float(losses_count / episodes),
        mean_len=float(lengths_array.mean()),
    )
