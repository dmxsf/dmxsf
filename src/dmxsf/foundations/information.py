"""Entropy, cross-entropy and KL divergence (course page: information theory).

All logarithms are natural, so the unit is the nat. Divide by ``ln 2`` for bits.
"""

from __future__ import annotations

import math

import numpy as np


def _as_probs(p: object) -> np.ndarray:
    return np.asarray(p, dtype=float)


def entropy(p: object) -> float:
    """H(p) = -sum p log p, with the convention 0 log 0 = 0."""
    p = _as_probs(p)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum()) + 0.0


def cross_entropy(p: object, q: object) -> float:
    """H(p, q) = -sum p log q: the cost of coding samples of p with a code built for q.

    Infinite when q gives probability 0 to something p can produce.
    """
    p, q = _as_probs(p), _as_probs(q)
    mask = p > 0
    if np.any(q[mask] <= 0):
        return math.inf
    return float(-(p[mask] * np.log(q[mask])).sum())


def kl_divergence(p: object, q: object) -> float:
    """KL(p || q) = H(p, q) - H(p): the extra cost of using q instead of p. Never negative."""
    return cross_entropy(p, q) - entropy(p)


def perplexity(loss_nats: float) -> float:
    """exp(loss): the size of the uniform distribution that is as hard to predict."""
    return math.exp(loss_nats)


def bits_per_byte(loss_nats: float, bytes_per_token: float = 1.0) -> float:
    """Loss per token converted to bits per byte, comparable across tokenizers."""
    return loss_nats / math.log(2) / bytes_per_token


def softmax(logits: object) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    z = np.asarray(logits, dtype=float)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def log_softmax(logits: object) -> np.ndarray:
    """Numerically stable log-softmax over the last axis (log-sum-exp trick)."""
    z = np.asarray(logits, dtype=float)
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
