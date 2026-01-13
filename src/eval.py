from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .utils import EvalResult, classify_outcome


def evaluate(
    env_factory: Callable[[], Any],
    act_fn: Callable[[Any], int],
    episodes: int = 20_000,
) -> EvalResult:
    env = env_factory()
    returns = []
    lens = []
    w = d = l = 0

    for _ in range(episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        ep_return = 0.0
        ep_len = 0

        while not (terminated or truncated):
            action = act_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            # if a RewardShapingWrapper is used, it stores true_reward in info
            r_true = float(info.get("true_reward", reward))
            ep_return += r_true
            ep_len += 1

        ww, dd, ll = classify_outcome(ep_return)
        w += ww
        d += dd
        l += ll

        returns.append(ep_return)
        lens.append(ep_len)

    env.close()
    returns = np.asarray(returns, dtype=np.float64)
    lens = np.asarray(lens, dtype=np.float64)
    return EvalResult(
        mean_return=float(returns.mean()),
        win_rate=float(w / episodes),
        draw_rate=float(d / episodes),
        loss_rate=float(l / episodes),
        mean_len=float(lens.mean()),
    )
