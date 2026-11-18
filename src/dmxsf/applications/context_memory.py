"""Atomic history selection and versioned preferences for /wiki/applications/context-and-memory/.

Wiki pages
  Chinese: https://dmxsf.com/wiki/applications/context-and-memory/
  English: https://dmxsf.com/en/wiki/applications/context-and-memory/
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Record:
    version: int
    value: str | None


def resolve(current: str | None, confirmed: Iterable[Record]) -> str | None:
    """Prefer the current value or the latest trusted record."""
    if current is not None:
        return current
    latest = max(confirmed, key=lambda record: record.version, default=None)
    return latest.value if latest is not None else None


def recent_groups(costs: list[int], budget: int) -> list[int]:
    """Take the latest contiguous whole groups within the available budget."""
    if budget < 0 or any(cost <= 0 for cost in costs):
        raise ValueError("budget must be nonnegative and costs positive")
    start = len(costs)
    while start and costs[start - 1] <= budget:
        start -= 1
        budget -= costs[start]
    return list(range(start, len(costs)))
