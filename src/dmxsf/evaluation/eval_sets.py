"""Evaluation sets: noisy labels, agreement and item information.

Wiki pages
  Chinese: https://dmxsf.com/wiki/evaluation/building-eval-sets/
  English: https://dmxsf.com/en/wiki/evaluation/building-eval-sets/
"""

import math


def noisy_accuracy(accuracy: float, error: float, choices: int) -> float:
    """Recorded accuracy under independent flips to uniformly chosen wrong labels."""
    return (1 - error) * accuracy + error * (1 - accuracy) / (choices - 1)


def item_information(ability: float, difficulty: float, discrimination: float) -> float:
    """Fisher information of a binary 2PL item at an ability value."""
    log_odds = abs(discrimination * (ability - difficulty))
    tail = math.exp(-log_odds)
    return discrimination**2 * tail / (1 + tail) ** 2
