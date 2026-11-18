"""Calibration: withheld answers, confidence and accuracy.

Wiki pages
  Chinese: https://dmxsf.com/wiki/safety/hallucination-and-calibration/
  English: https://dmxsf.com/en/wiki/safety/hallucination-and-calibration/
"""

import numpy as np


def binary_probability(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Turn binary logits into positive-class probabilities."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return np.exp(-np.logaddexp(0, -logits / temperature))


def binary_nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    """Stable held-out negative log likelihood in nats."""
    signed = (2 * labels - 1) * logits / temperature
    return float(np.logaddexp(0, -signed).mean())


def selective_rates(
    confidence: np.ndarray, correct: np.ndarray, threshold: float
) -> tuple[float, float | None]:
    """Coverage and accuracy among retained answers; None when no answers are retained."""
    keep = confidence >= threshold
    return float(keep.mean()), float(correct[keep].mean()) if keep.any() else None
