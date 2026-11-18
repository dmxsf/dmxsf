"""Paired replay outcomes and slice release gates for /wiki/evaluation/production-evaluation/.

Wiki pages
  Chinese: https://dmxsf.com/wiki/evaluation/production-evaluation/
  English: https://dmxsf.com/en/wiki/evaluation/production-evaluation/
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Pair:
    """Whether the same request succeeded on each version."""

    baseline: bool
    candidate: bool


@dataclass(frozen=True)
class Replay:
    count: int
    improvements: int
    regressions: int

    @property
    def change(self) -> float:
        """Paired candidate-minus-baseline success rate."""
        return (self.improvements - self.regressions) / self.count


def summarize(pairs: Iterable[Pair]) -> Replay:
    count = improvements = regressions = 0
    for pair in pairs:
        count += 1
        improvements += pair.candidate and not pair.baseline
        regressions += pair.baseline and not pair.candidate
    if not count:
        raise ValueError("at least one paired request is required")
    return Replay(count, improvements, regressions)


def release_blockers(
    groups: dict[str, Replay], *, min_cases: int, max_loss: float
) -> dict[str, str]:
    """Flag slices with inadequate observations or excessive observed loss."""
    blockers = {}
    for name, result in groups.items():
        if result.count < min_cases:
            blockers[name] = "insufficient samples"
        elif result.change < -max_loss:
            blockers[name] = "observed regression"
    return blockers
