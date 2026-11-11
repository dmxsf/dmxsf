"""Evaluation statistics: pass@k, the error of a benchmark score, paired comparisons, sample sizes.

Course page: https://dmxsf.com/wiki/evaluation/eval-statistics/

Everything here is plain Python and NumPy. A benchmark score is an estimate from a finite set of
questions, so each function answers one question about how far to trust it.
"""

from __future__ import annotations

from math import ceil, comb, sqrt
from statistics import NormalDist

import numpy as np

z = NormalDist().inv_cdf


# --- pass@k and pass^k ---------------------------------------------------------------------------


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimate of "at least one of k samples passes" from n samples of which c passed
    (Chen et al., 2021): one minus the chance that k samples drawn without replacement all failed."""
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError("need 0 <= c <= n and 1 <= k <= n")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_at_k_naive(n: int, c: int, k: int) -> float:
    """The tempting estimate 1 - (1 - c/n)^k. It is biased low: (1 - p)^k is convex in p."""
    return 1.0 - (1.0 - c / n) ** k


def pass_hat_k(n: int, c: int, k: int) -> float:
    """tau-bench's pass^k: the chance that k samples drawn without replacement all pass."""
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError("need 0 <= c <= n and 1 <= k <= n")
    return comb(c, k) / comb(n, k)


# --- the error of one score ----------------------------------------------------------------------


def half_width(p: float, n: int, level: float = 0.95) -> float:
    """Half-width of the confidence interval of an accuracy p measured on n independent questions."""
    return z(0.5 + level / 2) * sqrt(p * (1 - p) / n)


def mean_ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Mean and half-width of a normal confidence interval for the mean of per-question scores."""
    return float(x.mean()), z(0.5 + level / 2) * float(x.std(ddof=1)) / sqrt(len(x))


# --- comparing two models on the same questions --------------------------------------------------


def diff_ci(a: np.ndarray, b: np.ndarray, paired: bool, level: float = 0.95):
    """Mean difference a - b with its confidence interval, treating the two score vectors as
    independent samples or as paired by question (the difference is then what varies)."""
    d = a - b
    if paired:
        se = d.std(ddof=1) / sqrt(len(d))
    else:
        se = sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    half = z(0.5 + level / 2) * se
    return float(d.mean()), float(d.mean() - half), float(d.mean() + half)


def bootstrap_diff(diff: np.ndarray, rng: np.random.Generator, n_boot: int = 10_000, level=0.95):
    """Percentile bootstrap interval of the mean paired difference: resample the questions."""
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    lo, hi = np.percentile(diff[idx].mean(axis=1), [50 * (1 - level), 50 * (1 + level)])
    return float(lo), float(hi)


def questions_needed(
    p_a: float, p_b: float, rho: float, alpha: float = 0.05, power: float = 0.80
) -> int:
    """Questions needed to detect the gap p_a - p_b between two models that answer the same
    questions; rho is the correlation of their per-question correctness (0 means independent)."""
    var = p_a * (1 - p_a) + p_b * (1 - p_b) - 2 * rho * sqrt(p_a * (1 - p_a) * p_b * (1 - p_b))
    return ceil((z(1 - alpha / 2) + z(power)) ** 2 * var / (p_a - p_b) ** 2)


def correlated_bernoulli(
    rng: np.random.Generator, n: int, p_a: float, p_b: float, rho: float
) -> tuple[np.ndarray, np.ndarray]:
    """n pairs of right/wrong outcomes with success rates p_a and p_b and correlation rho."""
    both = p_a * p_b + rho * sqrt(p_a * (1 - p_a) * p_b * (1 - p_b))
    only_a, only_b = p_a - both, p_b - both
    neither = 1 - both - only_a - only_b
    if min(both, only_a, only_b, neither) < -1e-12:
        raise ValueError("this correlation is impossible for these success rates")
    cell = rng.choice(4, size=n, p=[both, only_a, only_b, max(neither, 0.0)])
    return (cell <= 1).astype(float), np.isin(cell, (0, 2)).astype(float)


def detects(a: np.ndarray, b: np.ndarray, paired: bool) -> bool:
    """Does the two-sided 95% interval of a - b exclude zero in the right direction?"""
    _, lo, _ = diff_ci(a, b, paired)
    return lo > 0
