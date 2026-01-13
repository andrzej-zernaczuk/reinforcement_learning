from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Tuple

import numpy as np

Obs = Tuple[int, int, bool]


@dataclass
class DoubleQConfig:
    alpha: float = 0.1
    gamma: float = 1.0
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_episodes: int = 100_000


class DoubleQAgent:
    """Double Q-learning (tabular) for discrete action spaces (2 actions in Blackjack)."""

    def __init__(self, n_actions: int, cfg: DoubleQConfig, seed: int):
        self.n_actions = n_actions
        self.cfg = cfg
        self.rng = random.Random(seed)

        self.QA: DefaultDict[Obs, np.ndarray] = defaultdict(
            lambda: np.zeros(n_actions, dtype=np.float32)
        )
        self.QB: DefaultDict[Obs, np.ndarray] = defaultdict(
            lambda: np.zeros(n_actions, dtype=np.float32)
        )

        self.episode = 0
        self.epsilon = cfg.eps_start

    def _update_epsilon(self) -> None:
        # Linear decay from eps_start -> eps_end over eps_decay_episodes
        if self.cfg.eps_decay_episodes <= 0:
            self.epsilon = self.cfg.eps_end
            return
        frac = min(1.0, self.episode / float(self.cfg.eps_decay_episodes))
        self.epsilon = self.cfg.eps_start + frac * (
            self.cfg.eps_end - self.cfg.eps_start
        )

    def act(self, obs: Obs, train: bool = True) -> int:
        if train and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_actions)
        q = self.QA[obs] + self.QB[obs]
        return int(np.argmax(q))

    def update(
        self, obs: Obs, action: int, reward: float, terminated: bool, next_obs: Obs
    ) -> None:
        # Choose which table to update
        if self.rng.random() < 0.5:
            Q_upd, Q_eval = self.QA, self.QB
        else:
            Q_upd, Q_eval = self.QB, self.QA

        # Double Q target
        if terminated:
            target = reward
        else:
            a_star = int(np.argmax(Q_upd[next_obs]))
            target = reward + self.cfg.gamma * float(Q_eval[next_obs][a_star])

        td_error = target - float(Q_upd[obs][action])
        Q_upd[obs][action] = float(Q_upd[obs][action]) + self.cfg.alpha * td_error

    def end_episode(self) -> None:
        self.episode += 1
        self._update_epsilon()

    def greedy_action(self, obs: Obs) -> int:
        q = self.QA[obs] + self.QB[obs]
        return int(np.argmax(q))
