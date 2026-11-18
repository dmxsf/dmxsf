"""Compare warm-cache affinity and estimated-finish routing for /wiki/inference/prefix-caching/.

Wiki pages
  Chinese: https://dmxsf.com/wiki/inference/prefix-caching/
  English: https://dmxsf.com/en/wiki/inference/prefix-caching/
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median


@dataclass(frozen=True)
class RoutingResult:
    hits: int
    median_ms: float
    worst_ms: int


def replay(arrivals: list[int], *, hit_ms: int, miss_ms: int, affinity: bool) -> RoutingResult:
    """Schedule a single shared prefix on two FIFO workers."""
    if not arrivals or arrivals != sorted(arrivals) or not 0 < hit_ms <= miss_ms:
        raise ValueError("need ordered arrivals and positive hit/miss times")

    free_at = [0, 0]
    warm = [True, False]
    latencies = []
    hits = 0
    for arrival in arrivals:
        finishes = [
            max(arrival, available) + (hit_ms if cached else miss_ms)
            for available, cached in zip(free_at, warm, strict=True)
        ]
        earliest = min(range(len(finishes)), key=finishes.__getitem__)
        worker = 0 if affinity else earliest
        hits += warm[worker]
        warm[worker] = True
        free_at[worker] = finishes[worker]
        latencies.append(finishes[worker] - arrival)
    return RoutingResult(hits, median(latencies), max(latencies))
