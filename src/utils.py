from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | os.PathLike, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


@dataclass
class EvalResult:
    mean_return: float
    win_rate: float
    draw_rate: float
    loss_rate: float
    mean_len: float


def classify_outcome(final_reward: float) -> Tuple[int, int, int]:
    # Blackjack-v1: terminal rewards are typically +1, 0, -1
    # If natural=True, win might be +1.5, so treat >0 as win.
    if final_reward > 0:
        return 1, 0, 0
    if final_reward < 0:
        return 0, 0, 1
    return 0, 1, 0
