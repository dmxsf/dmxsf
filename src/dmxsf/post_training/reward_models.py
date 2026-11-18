"""Reward models: pairwise scores and label noise.

Wiki pages
  Chinese: https://dmxsf.com/wiki/post-training/reward-models/
  English: https://dmxsf.com/en/wiki/post-training/reward-models/
"""

import numpy as np


def fit_bradley_terry(wins: np.ndarray, steps: int = 500, lr: float = 2.0) -> np.ndarray:
    """Fit centered reward scores from wins[i, j] = choices of i over j."""
    scores = np.zeros(len(wins))
    comparisons = wins + wins.T
    for _ in range(steps):
        gap = scores[:, None] - scores[None, :]
        probability = np.exp(-np.logaddexp(0, -gap))
        scores += lr * (wins - comparisons * probability).sum(axis=1) / wins.sum()
        scores -= scores.mean()
    return scores


def label_noise_rates(error_rate: float) -> tuple[float, float]:
    """Ideal accuracy against one noisy label, then pairwise annotator agreement."""
    return 1 - error_rate, (1 - error_rate) ** 2 + error_rate**2
