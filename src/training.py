"""Training utilities for reinforcement learning agents.

This module provides shared training infrastructure including rollout collection
for A2C-style agents and reusable data structures for storing trajectory data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .features import obs_to_onehot

if TYPE_CHECKING:
    import gymnasium as gym

    from .a2c_gae import A2CGAEAgent


@dataclass
class RolloutData:
    """Container for trajectory data collected during rollout.

    This dataclass stores all information collected during a rollout (trajectory)
    of an agent interacting with the environment. All arrays have shape [T, ...]
    where T is the number of timesteps in the rollout.

    Args:
        observations: One-hot encoded observations, shape [T, obs_dim].
        actions: Actions taken by the agent, shape [T].
        rewards: Rewards received from environment (may be shaped), shape [T].
        episode_dones: Binary indicators of episode termination (1.0 if done,
            0.0 otherwise), shape [T].
        value_estimates: Value function estimates V(s) at each timestep, shape [T].
        action_log_probabilities: Log probabilities of actions taken, shape [T].
        bootstrap_value: Value estimate for the final state after the rollout,
            used for bootstrapping GAE computation. Scalar float.
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    episode_dones: np.ndarray
    value_estimates: np.ndarray
    action_log_probabilities: np.ndarray
    bootstrap_value: float


def collect_rollout(
    environment: gym.Env,
    agent: A2CGAEAgent,
    rollout_steps: int,
) -> RolloutData:
    """Collect a rollout of environment interactions using the given agent.

    Executes the agent in the environment for a fixed number of steps, collecting
    observations, actions, rewards, and other data needed for A2C/GAE updates.
    Episodes are automatically reset when they terminate, allowing continuous
    interaction over multiple episodes within a single rollout.

    Args:
        environment: Gymnasium environment to interact with. Must be already
            initialized and will be automatically reset if needed.
        agent: A2C-GAE agent that provides actions and value estimates.
        rollout_steps: Number of environment steps to collect. This is the
            batch size for the policy update.

    Returns:
        RolloutData containing all trajectory information needed for agent update.
        All arrays have shape [rollout_steps, ...] except bootstrap_value which
        is a scalar.
    """
    observation, _ = environment.reset()

    observations_list: list[np.ndarray] = []
    actions_list: list[int] = []
    rewards_list: list[float] = []
    episode_dones_list: list[float] = []
    value_estimates_list: list[float] = []
    action_log_probabilities_list: list[float] = []

    for _ in range(rollout_steps):
        observation_encoded = obs_to_onehot(observation)
        action, log_probability, value_estimate, entropy = agent.act(
            observation_encoded, train=True
        )

        next_observation, reward, terminated, truncated, info = environment.step(action)
        episode_done = terminated or truncated

        observations_list.append(observation_encoded)
        actions_list.append(action)
        rewards_list.append(float(reward))
        episode_dones_list.append(float(episode_done))
        value_estimates_list.append(float(value_estimate.item()))
        action_log_probabilities_list.append(float(log_probability.item()))

        observation = next_observation
        if episode_done:
            observation, _ = environment.reset()

    # Bootstrap value from the final state for GAE computation
    final_observation_encoded = obs_to_onehot(observation)
    _, _, bootstrap_value, _ = agent.act(final_observation_encoded, train=False)

    return RolloutData(
        observations=np.asarray(observations_list, dtype=np.float32),
        actions=np.asarray(actions_list, dtype=np.int64),
        rewards=np.asarray(rewards_list, dtype=np.float32),
        episode_dones=np.asarray(episode_dones_list, dtype=np.float32),
        value_estimates=np.asarray(value_estimates_list, dtype=np.float32),
        action_log_probabilities=np.asarray(action_log_probabilities_list, dtype=np.float32),
        bootstrap_value=float(bootstrap_value.item()),
    )
