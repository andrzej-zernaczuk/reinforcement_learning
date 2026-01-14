"""Visualization utilities for policy analysis.

This module provides functions for visualizing learned policies as heatmaps,
allowing inspection of the agent's decision-making across different game states.
"""

from __future__ import annotations

from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np


def policy_grid(
    action_function: Callable[[Tuple[int, int, bool]], int],
    usable_ace: bool,
    player_sums=range(4, 22),
    dealer_cards=range(1, 11),
) -> np.ndarray:
    """Generate a grid of policy actions for all state combinations.

    Creates a 2D grid showing which action the policy selects for each combination
    of player sum and dealer showing card, for a fixed usable_ace value.

    Args:
        action_function: Function that takes a Blackjack observation
            (player_sum, dealer_card, usable_ace) and returns an action index.
        usable_ace: Whether to use usable_ace=True or False for all states in the grid.
        player_sums: Range of player hand sums to include (rows of the grid).
        dealer_cards: Range of dealer showing cards to include (columns of the grid).

    Returns:
        2D numpy array of shape [len(player_sums), len(dealer_cards)] containing
        action indices. For Blackjack: 0=stick, 1=hit.
    """
    grid = np.zeros((len(player_sums), len(dealer_cards)), dtype=np.int32)

    for row_index, player_sum in enumerate(player_sums):
        for col_index, dealer_card in enumerate(dealer_cards):
            action = action_function((player_sum, dealer_card, usable_ace))
            grid[row_index, col_index] = int(action)

    return grid


def plot_policy_heatmaps(
    action_function: Callable[[Tuple[int, int, bool]], int],
    output_path: str,
    title_prefix: str = "",
) -> None:
    """Create and save policy heatmaps for both usable_ace values.

    Generates a figure with two side-by-side heatmaps showing the policy's
    actions for all state combinations, one for usable_ace=False and one for
    usable_ace=True. The heatmaps show which action (stick=0 or hit=1) the
    policy selects for each (player_sum, dealer_card) combination.

    Args:
        action_function: Function mapping observations to actions.
        output_path: File path where the figure will be saved (e.g., "policy.png").
        title_prefix: Optional prefix for subplot titles (e.g., "DoubleQ" or "A2C").
    """
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    for axis, usable_ace in zip(axes, [False, True]):
        grid = policy_grid(action_function, usable_ace=usable_ace)
        image = axis.imshow(grid, aspect="auto", origin="lower")
        axis.set_title(f"{title_prefix} usable_ace={usable_ace}")
        axis.set_xlabel("dealer showing (1..10)")
        axis.set_ylabel("player sum (4..21)")
        axis.set_xticks(range(10))
        axis.set_xticklabels([str(card) for card in range(1, 11)])
        axis.set_yticks(range(18))
        axis.set_yticklabels([str(hand_sum) for hand_sum in range(4, 22)])
        # Add legend explaining action encoding
        axis.text(0.01, -0.18, "0=stick, 1=hit", transform=axis.transAxes)

    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.9)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
