"""Synthetic data: rare-event loss and verified selection.

Wiki pages
  Chinese: https://dmxsf.com/wiki/frontier/synthetic-data-and-self-improvement/
  English: https://dmxsf.com/en/wiki/frontier/synthetic-data-and-self-improvement/
"""

import numpy as np


def missing_mass(probabilities: np.ndarray, sample_size: int) -> float:
    """Expected probability mass unseen after independent draws."""
    return float(probabilities @ (1 - probabilities) ** sample_size)


def replacement_chain(
    probabilities: np.ndarray, sample_size: int, generations: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Fit the empirical distribution of each generation, then sample from it."""
    history = [probabilities]
    for _ in range(generations):
        counts = rng.multinomial(sample_size, history[-1])
        history.append(counts / sample_size)
    return history


def verify_additions(examples: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """An independent exact verifier for the synthetic addition task."""
    return [(a, b, answer) for a, b, answer in examples if a + b == answer]
