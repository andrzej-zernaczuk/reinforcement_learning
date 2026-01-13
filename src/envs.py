from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym


@dataclass
class RewardConfig:
    mode: str = "r0"  # r0, r1, r2, r3
    step_penalty: float = 0.0  # for r1
    bust_penalty: float = 0.0  # for r2
    gamma: float = 0.99  # for r3
    # Potential Φ(s) uses player_sum / 21.0


class RewardShapingWrapper(gym.Wrapper):
    """Reward shaping wrapper that also keeps the true env reward in info.

    - r0: unchanged
    - r1: subtract step_penalty on non-terminal steps
    - r2: subtract bust_penalty on terminal losses (reward == -1)
    - r3: potential-based: r' = r + gamma*Φ(s') - Φ(s), Φ(s)=player_sum/21
    """

    def __init__(self, env: gym.Env, cfg: RewardConfig):
        super().__init__(env)
        self.cfg = cfg
        self._last_obs = None

    @staticmethod
    def _phi(obs) -> float:
        # obs is (player_sum, dealer_upcard, usable_ace)
        ps = float(obs[0])
        return ps / 21.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        assert self._last_obs is not None, "Call reset() before step()."

        obs, reward, terminated, truncated, info = self.env.step(action)
        true_reward = float(reward)

        shaped = float(reward)
        mode = self.cfg.mode.lower()

        done = terminated or truncated

        if mode == "r1":
            if not done:
                shaped = shaped - float(self.cfg.step_penalty)

        elif mode == "r2":
            if done and true_reward < 0:  # loss (incl bust)
                shaped = shaped - float(self.cfg.bust_penalty)

        elif mode == "r3":
            # potential-based shaping
            phi_s = self._phi(self._last_obs)
            phi_sp = self._phi(obs)
            shaped = shaped + float(self.cfg.gamma) * phi_sp - phi_s

        # r0 leaves shaped = reward

        info = dict(info)
        info["true_reward"] = true_reward
        info["shaped_reward"] = shaped

        self._last_obs = obs
        return obs, shaped, terminated, truncated, info


def make_env(
    seed: int,
    natural: bool = False,
    sab: bool = False,
    reward_cfg: Optional[RewardConfig] = None,
    record_stats: bool = True,
):
    env = gym.make("Blackjack-v1", natural=natural, sab=sab)
    env.reset(seed=seed)

    if record_stats:
        env = gym.wrappers.RecordEpisodeStatistics(env)

    if reward_cfg is not None:
        env = RewardShapingWrapper(env, reward_cfg)

    return env
