"""Utility functions and data structures for reinforcement learning experiments.

This module provides common utilities including random seed setting, file I/O
helpers, and data structures for storing evaluation results.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def set_global_seeds(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries.

    Sets seeds for Python's random module and NumPy. For PyTorch, seeds should
    be set separately in agent initialization (torch.manual_seed).

    Args:
        seed: Integer seed value for random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | os.PathLike) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to create. Can be a string or Path object.

    Returns:
        Path object representing the created/existing directory.
    """
    directory_path = Path(path)
    directory_path.mkdir(parents=True, exist_ok=True)
    return directory_path


def save_json(path: str | os.PathLike, data_object: Dict[str, Any]) -> None:
    """Save a dictionary to a JSON file with pretty formatting.

    Automatically creates parent directories if they don't exist. The JSON
    is formatted with indentation and sorted keys for readability.

    Args:
        path: File path where JSON will be written.
        data_object: Dictionary to serialize to JSON.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(data_object, indent=2, sort_keys=True))


@dataclass
class EvalResult:
    """Container for agent evaluation results.

    Stores aggregated statistics from running an agent over multiple episodes.
    All rates are fractions in the range [0.0, 1.0].

    Args:
        mean_return: Average episode return across all evaluation episodes.
        win_rate: Fraction of episodes that resulted in a win (positive return).
        draw_rate: Fraction of episodes that resulted in a draw (zero return).
        loss_rate: Fraction of episodes that resulted in a loss (negative return).
        mean_len: Average number of timesteps per episode.
    """

    mean_return: float
    win_rate: float
    draw_rate: float
    loss_rate: float
    mean_len: float


def classify_outcome(final_reward: float) -> Tuple[int, int, int]:
    """Classify an episode outcome as win, draw, or loss.

    For Blackjack, terminal rewards are typically +1 (win), 0 (draw), or -1 (loss).
    When natural=True, wins may give +1.5. This function treats any positive reward
    as a win, zero as a draw, and negative as a loss.

    Args:
        final_reward: Total reward received in the episode.

    Returns:
        Tuple of (is_win, is_draw, is_loss) where exactly one value is 1 and
        the others are 0.
    """
    if final_reward > 0:
        return 1, 0, 0  # Win
    if final_reward < 0:
        return 0, 0, 1  # Loss
    return 0, 1, 0  # Draw
