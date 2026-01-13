from __future__ import annotations

from typing import Tuple

import numpy as np

# Observation: (player_sum, dealer_upcard, usable_ace)
# player_sum in [0..31], dealer_upcard in [1..10], usable_ace in {False, True}
PLAYER_SUM_BINS = 32
DEALER_BINS = 10
UA_BINS = 1  # as a single scalar 0/1

OBS_DIM = PLAYER_SUM_BINS + DEALER_BINS + UA_BINS  # 43


def obs_to_onehot(obs: Tuple[int, int, int]) -> np.ndarray:
    ps, dealer, ua = obs
    x = np.zeros((OBS_DIM,), dtype=np.float32)

    ps = int(np.clip(ps, 0, PLAYER_SUM_BINS - 1))
    dealer = int(np.clip(dealer, 1, 10))

    x[ps] = 1.0
    x[PLAYER_SUM_BINS + (dealer - 1)] = 1.0
    x[PLAYER_SUM_BINS + DEALER_BINS] = 1.0 if ua else 0.0
    return x
