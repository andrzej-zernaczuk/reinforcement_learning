"""Environment creation and reward shaping for Blackjack.

This module provides utilities for creating Gymnasium Blackjack environments with
optional reward shaping. It supports multiple reward shaping modes to explore
different learning objectives while preserving the true environment reward for
evaluation purposes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym


@dataclass
class RewardConfig:
    """Configuration for reward shaping in Blackjack environment.

    Args:
        mode: Reward shaping mode selector:
            - "r0": No shaping (use original environment rewards)
            - "r1": Step penalty (subtract step_penalty on non-terminal steps)
            - "r2": Bust penalty (subtract bust_penalty on terminal losses)
        step_penalty: Penalty subtracted from reward at each non-terminal step.
            Used only in mode "r1". Encourages shorter episodes.
        bust_penalty: Additional penalty for losing (including busting).
            Used only in mode "r2". Discourages risky play.
        gamma: Discount factor used as the standard RL discount factor in training.
    """

    mode: str = "r0"
    step_penalty: float = 0.0
    bust_penalty: float = 0.0
    gamma: float = 0.99


class RewardShapingWrapper(gym.Wrapper):
    """Wrapper that applies reward shaping while preserving true rewards.

    Implements multiple reward shaping strategies for the Blackjack environment.
    The wrapper transforms the environment reward according to the selected mode
    while storing the original reward in the info dict for evaluation purposes.

    Shaping modes:
        - r0: reward' = reward (no change)
        - r1: reward' = reward - step_penalty (on non-terminal steps)
        - r2: reward' = reward - bust_penalty (on terminal losses)
    Args:
        env: Base Gymnasium environment to wrap.
        reward_config: Configuration specifying the reward shaping mode and parameters.
    """

    def __init__(self, env: gym.Env, reward_config: RewardConfig):
        """Initialize the reward shaping wrapper."""
        super().__init__(env)
        self.reward_config = reward_config
        self._last_observation = None

    def reset(self, **kwargs):
        """Reset the environment and store the initial observation.

        Args:
            **kwargs: Arguments passed to the base environment's reset method.

        Returns:
            Tuple of (observation, info) from the base environment.
        """
        observation, info = self.env.reset(**kwargs)
        self._last_observation = observation
        return observation, info

    def step(self, action):
        """Execute an action and apply reward shaping.

        Args:
            action: Action to take in the environment.

        Returns:
            Tuple of (observation, shaped_reward, terminated, truncated, info) where
            info contains "true_reward" and "shaped_reward" fields.

        Raises:
            AssertionError: If step is called before reset.
        """
        assert self._last_observation is not None, "Call reset() before step()."

        observation, reward, terminated, truncated, info = self.env.step(action)
        true_reward = float(reward)
        shaped_reward = float(reward)

        mode = self.reward_config.mode.lower()
        episode_done = terminated or truncated

        if mode == "r1":
            # Subtract step penalty on non-terminal transitions
            if not episode_done:
                shaped_reward = shaped_reward - float(self.reward_config.step_penalty)

        elif mode == "r2":
            # Add bust penalty for terminal losses
            if episode_done and true_reward < 0:
                shaped_reward = shaped_reward - float(self.reward_config.bust_penalty)

        # mode "r0" leaves shaped_reward unchanged

        # Store both rewards in info for evaluation
        info = dict(info)
        info["true_reward"] = true_reward
        info["shaped_reward"] = shaped_reward

        self._last_observation = observation
        return observation, shaped_reward, terminated, truncated, info


def make_env(
    seed: int,
    natural: bool = False,
    sab: bool = False,
    reward_cfg: Optional[RewardConfig] = None,
    record_stats: bool = True,
) -> gym.Env:
    """Create a Blackjack environment with optional reward shaping.

    Creates and configures a Gymnasium Blackjack environment. The wrapper ordering
    is important: RewardShapingWrapper must come before RecordEpisodeStatistics
    so episode statistics reflect the shaped rewards used during training.

    Args:
        seed: Random seed for environment reproducibility.
        natural: If True, natural blackjacks (21 from first two cards) pay 1.5x.
            Default False uses 1.0x payout.
        sab: If True, the dealer shows their actual card instead of the value.
            "sab" stands for "show actual blackjack". Default False.
        reward_cfg: Optional reward shaping configuration. If None, no shaping
            is applied (equivalent to r0 mode).
        record_stats: If True, wraps environment with RecordEpisodeStatistics to
            track episode returns and lengths. Should be True during training.

    Returns:
        Configured Gymnasium environment ready for training or evaluation.
    """
    environment = gym.make("Blackjack-v1", natural=natural, sab=sab)
    environment.reset(seed=seed)

    if reward_cfg is not None:
        environment = RewardShapingWrapper(environment, reward_cfg)

    if record_stats:
        environment = gym.wrappers.RecordEpisodeStatistics(environment)

    return environment
