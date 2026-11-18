"""Privacy audit: exposure for synthetic canaries.

Wiki pages
  Chinese: https://dmxsf.com/wiki/safety/memorization-and-privacy/
  English: https://dmxsf.com/en/wiki/safety/memorization-and-privacy/
"""

import math
from collections.abc import Sequence


def canary_rank(scores: Sequence[float], index: int) -> int:
    """Conservative rank among synthetic candidates: assign the last position in a tie."""
    if not 0 <= index < len(scores):
        raise IndexError("canary index out of range")
    return sum(score >= scores[index] for score in scores)


def exposure(candidate_count: int, rank: int) -> float:
    """Rank-based exposure in bits."""
    if not 1 <= rank <= candidate_count:
        raise ValueError("rank must be within the candidate set")
    return math.log2(candidate_count / rank)
