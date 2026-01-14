"""Double Q-learning algorithm for tabular reinforcement learning.

This module implements the Double Q-learning algorithm, which addresses the
overestimation bias of standard Q-learning by maintaining two separate Q-tables
and using one to select actions while using the other to evaluate them.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Tuple

import numpy as np

Observation = Tuple[int, int, bool]


@dataclass
class DoubleQConfig:
    """Configuration for Double Q-learning agent.

    Args:
        alpha: Learning rate (step size) for Q-value updates. Controls how much
            new information overrides old information. Typical range: [0.01, 0.5].
        gamma: Discount factor for future rewards. Determines the importance of
            future rewards vs immediate rewards. Range: [0.0, 1.0], typically 0.95-1.0.
        eps_start: Initial exploration rate for epsilon-greedy policy. Should be
            high (e.g., 1.0) to encourage early exploration.
        eps_end: Final exploration rate after decay. Typical range: [0.01, 0.1].
        eps_decay_episodes: Number of episodes over which to linearly decay epsilon
            from eps_start to eps_end. Should be a substantial fraction of total
            training episodes.
    """

    alpha: float = 0.1
    gamma: float = 1.0
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_episodes: int = 100_000


class DoubleQAgent:
    """Double Q-learning agent for discrete action spaces.

    Implements the Double Q-learning algorithm which maintains two separate Q-tables
    (q_table_a and q_table_b) to reduce overestimation bias. On each update, one
    table is randomly selected for updating while the other is used for action
    evaluation, preventing the maximization bias that occurs in standard Q-learning.

    The agent uses epsilon-greedy exploration with linear decay over episodes.

    Args:
        num_actions: Number of discrete actions available in the environment.
            For Blackjack, this is 2 (stick=0, hit=1).
        config: Configuration object specifying learning rate, discount factor,
            and exploration parameters.
        seed: Random seed for reproducibility of action selection and Q-table
            update selection.
    """

    def __init__(self, num_actions: int, config: DoubleQConfig, seed: int):
        """Initialize the Double Q-learning agent."""
        self.num_actions = num_actions
        self.config = config
        self.random_generator = random.Random(seed)

        # Two Q-tables for Double Q-learning to reduce overestimation bias
        self.q_table_a: DefaultDict[Observation, np.ndarray] = defaultdict(
            lambda: np.zeros(num_actions, dtype=np.float32)
        )
        self.q_table_b: DefaultDict[Observation, np.ndarray] = defaultdict(
            lambda: np.zeros(num_actions, dtype=np.float32)
        )

        self.episode_count = 0
        self.epsilon = config.eps_start

    def _update_epsilon(self) -> None:
        """Update epsilon using linear decay schedule.

        Linearly decays epsilon from eps_start to eps_end over eps_decay_episodes.
        After reaching eps_end, epsilon remains constant.
        """
        if self.config.eps_decay_episodes <= 0:
            self.epsilon = self.config.eps_end
            return

        decay_fraction = min(1.0, self.episode_count / float(self.config.eps_decay_episodes))
        self.epsilon = self.config.eps_start + decay_fraction * (
            self.config.eps_end - self.config.eps_start
        )

    def act(self, observation: Observation, train: bool = True) -> int:
        """Select an action using epsilon-greedy policy.

        During training, selects a random action with probability epsilon for
        exploration, otherwise selects the greedy action. The greedy action is
        chosen based on the sum of both Q-tables.

        Args:
            observation: Current environment observation (state).
            train: If True, uses epsilon-greedy exploration. If False, always
                selects the greedy action.

        Returns:
            Selected action index.
        """
        if train and self.random_generator.random() < self.epsilon:
            return self.random_generator.randrange(self.num_actions)

        # Use the average of both Q-tables for action selection
        combined_q_values = self.q_table_a[observation] + self.q_table_b[observation]
        return int(np.argmax(combined_q_values))

    def update(
        self,
        observation: Observation,
        action: int,
        reward: float,
        terminated: bool,
        next_observation: Observation,
    ) -> None:
        """Update Q-values using Double Q-learning.

        Randomly selects one Q-table to update and the other to evaluate. The table
        being updated selects the optimal action, while the other table provides the
        value estimate for that action. This decoupling reduces overestimation bias.

        Args:
            observation: Current state before taking action.
            action: Action that was taken.
            reward: Reward received after taking the action.
            terminated: Whether the episode terminated after this transition.
            next_observation: Resulting state after taking the action.
        """
        # Randomly choose which Q-table to update (and which to use for evaluation)
        if self.random_generator.random() < 0.5:
            q_table_to_update, q_table_for_evaluation = self.q_table_a, self.q_table_b
        else:
            q_table_to_update, q_table_for_evaluation = self.q_table_b, self.q_table_a

        # Compute Double Q-learning target
        if terminated:
            # No future rewards for terminal states
            target_value = reward
        else:
            # Select best action using the table being updated
            optimal_action = int(np.argmax(q_table_to_update[next_observation]))
            # Evaluate that action using the other table
            next_state_value = float(q_table_for_evaluation[next_observation][optimal_action])
            target_value = reward + self.config.gamma * next_state_value

        # Compute TD error and update Q-value
        current_q_value = float(q_table_to_update[observation][action])
        td_error = target_value - current_q_value
        q_table_to_update[observation][action] = current_q_value + self.config.alpha * td_error

    def end_episode(self) -> None:
        """Signal the end of an episode.

        Should be called after each episode completes. Updates the episode counter
        and decays epsilon according to the schedule.
        """
        self.episode_count += 1
        self._update_epsilon()

    def greedy_action(self, observation: Observation) -> int:
        """Select the greedy action without exploration.

        Selects the action with the highest Q-value according to the sum of both
        Q-tables. Used for evaluation and final policy extraction.

        Args:
            observation: Current environment observation.

        Returns:
            Action with highest Q-value.
        """
        combined_q_values = self.q_table_a[observation] + self.q_table_b[observation]
        return int(np.argmax(combined_q_values))
