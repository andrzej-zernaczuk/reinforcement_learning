"""Feature encoding for Blackjack observations.

This module provides utilities for converting raw Blackjack observations into
one-hot encoded feature vectors suitable for neural network input.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# Blackjack observation space: (player_sum, dealer_upcard, usable_ace)
# player_sum ranges from 0 to 31 (though typical values are 4-21)
# dealer_upcard ranges from 1 to 10 (Ace=1, face cards=10)
# usable_ace is a binary indicator (True/False)
PLAYER_SUM_BINS = 32
DEALER_CARD_BINS = 10
USABLE_ACE_BINS = 1

# Total one-hot encoding dimension
OBS_DIM = PLAYER_SUM_BINS + DEALER_CARD_BINS + USABLE_ACE_BINS  # 43


def obs_to_onehot(observation: Tuple[int, int, bool]) -> np.ndarray:
    """Convert Blackjack observation to one-hot encoded feature vector.

    Encodes the observation as a concatenation of three one-hot segments:
    1. Player sum: one-hot vector of length 32 (indices 0-31)
    2. Dealer showing: one-hot vector of length 10 (indices 32-41, for cards 1-10)
    3. Usable ace: binary scalar (index 42, value 0.0 or 1.0)

    Args:
        observation: Blackjack observation tuple with three elements:
            - player_sum (int): Current sum of player's hand (0-31)
            - dealer_card (int): Value of dealer's showing card (1-10)
            - usable_ace (bool): Whether player has a usable ace

    Returns:
        One-hot encoded feature vector of shape [OBS_DIM] = [43] with dtype float32.
        Exactly three elements will be 1.0, all others will be 0.0.
    """
    player_sum, dealer_card, usable_ace = observation
    one_hot_vector = np.zeros((OBS_DIM,), dtype=np.float32)

    # Clip player_sum to valid range [0, PLAYER_SUM_BINS-1]
    player_sum = int(np.clip(player_sum, 0, PLAYER_SUM_BINS - 1))

    # Clip dealer_card to valid range [1, 10]
    dealer_card = int(np.clip(dealer_card, 1, 10))

    # Set one-hot bits
    one_hot_vector[player_sum] = 1.0
    one_hot_vector[PLAYER_SUM_BINS + (dealer_card - 1)] = 1.0
    one_hot_vector[PLAYER_SUM_BINS + DEALER_CARD_BINS] = 1.0 if usable_ace else 0.0

    return one_hot_vector
