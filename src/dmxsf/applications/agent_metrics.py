"""Task-weighted reliability and final-state grading for /wiki/applications/agent-evaluation/.

Wiki pages
  Chinese: https://dmxsf.com/wiki/applications/agent-evaluation/
  English: https://dmxsf.com/en/wiki/applications/agent-evaluation/
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from math import comb


def pass_all(tasks: Sequence[Sequence[bool]], k: int) -> float:
    """Estimate the probability that all k trials of a uniformly sampled task pass."""
    if k < 1 or not tasks or any(len(trials) < k for trials in tasks):
        raise ValueError("each task needs at least k trials and k must be positive")
    return sum(comb(sum(trials), k) / comb(len(trials), k) for trials in tasks) / len(tasks)


def matches_calendar(
    actual: Sequence[tuple[str, str]], expected: Sequence[tuple[str, str]]
) -> bool:
    """Compare event multiplicities without depending on ordering."""
    return Counter(actual) == Counter(expected)
