"""Policy gradients: exact checks, baselines and GAE.

Wiki pages
  Chinese: https://dmxsf.com/wiki/post-training/rl-fundamentals/
  English: https://dmxsf.com/en/wiki/post-training/rl-fundamentals/
"""

import numpy as np


def action_probabilities(logits: np.ndarray) -> np.ndarray:
    """Stable categorical softmax."""
    weights = np.exp(logits - logits.max())
    return weights / weights.sum()


def exact_policy_gradient(
    logits: np.ndarray, rewards: np.ndarray, baseline: float = 0
) -> np.ndarray:
    """Sum p(a) * (r(a)-baseline) * grad log p(a) over all actions."""
    probabilities = action_probabilities(logits)
    score = np.eye(len(logits)) - probabilities
    return (probabilities * (rewards - baseline)) @ score


def generalized_advantage(td_errors: np.ndarray, gamma: float, lam: float) -> np.ndarray:
    """Discount a finite sequence of TD errors backwards, with zero continuation."""
    result = np.zeros_like(td_errors, dtype=float)
    carry = 0.0
    for i in reversed(range(len(td_errors))):
        carry = td_errors[i] + gamma * lam * carry
        result[i] = carry
    return result
