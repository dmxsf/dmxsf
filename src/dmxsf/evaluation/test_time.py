"""Test-time compute: independent draws, reranking and majority voting.

Wiki pages
  Chinese: https://dmxsf.com/wiki/frontier/test-time-compute/
  English: https://dmxsf.com/en/wiki/frontier/test-time-compute/
"""

import math


def at_least_one(probability: float, draws: int) -> float:
    """Oracle coverage for independent draws, with a stable small-p calculation."""
    if draws == 0:
        return 0.0
    if probability == 1:
        return 1.0
    return -math.expm1(draws * math.log1p(-probability))


def binary_majority(probability: float, draws: int) -> float:
    """Probability a correct option gets a strict majority of independent votes."""
    return math.fsum(
        math.comb(draws, k) * probability**k * (1 - probability) ** (draws - k)
        for k in range(draws // 2 + 1, draws + 1)
    )
