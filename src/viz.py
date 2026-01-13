from __future__ import annotations

from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np


def policy_grid(
    action_fn: Callable[[Tuple[int, int, bool]], int],
    usable_ace: bool,
    player_sums=range(4, 22),
    dealer_cards=range(1, 11),
) -> np.ndarray:
    grid = np.zeros((len(player_sums), len(dealer_cards)), dtype=np.int32)
    for i, ps in enumerate(player_sums):
        for j, dc in enumerate(dealer_cards):
            grid[i, j] = int(action_fn((ps, dc, usable_ace)))
    return grid


def plot_policy_heatmaps(
    action_fn: Callable[[Tuple[int, int, bool]], int],
    out_path: str,
    title_prefix: str = "",
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    for ax, ua in zip(axes, [False, True]):
        grid = policy_grid(action_fn, usable_ace=ua)
        im = ax.imshow(grid, aspect="auto", origin="lower")
        ax.set_title(f"{title_prefix} usable_ace={ua}")
        ax.set_xlabel("dealer showing (1..10)")
        ax.set_ylabel("player sum (4..21)")
        ax.set_xticks(range(10))
        ax.set_xticklabels([str(i) for i in range(1, 11)])
        ax.set_yticks(range(18))
        ax.set_yticklabels([str(i) for i in range(4, 22)])
        # legend: 0=stick, 1=hit
        ax.text(0.01, -0.18, "0=stick, 1=hit", transform=ax.transAxes)

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
