"""Simulate actor and learner overlap for /wiki/post-training/rollout-and-training/.

Wiki pages
  Chinese: https://dmxsf.com/wiki/post-training/rollout-and-training/
  English: https://dmxsf.com/en/wiki/post-training/rollout-and-training/
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass


@dataclass(frozen=True)
class Trace:
    elapsed: int
    accepted: int
    dropped: int
    lags: tuple[int, ...]


def simulate(
    durations: list[int],
    update_time: int,
    *,
    overlap: bool,
    max_lag: int,
) -> Trace:
    """Simulate FIFO training with a fixed update time."""
    if update_time <= 0 or max_lag < 0 or min(durations, default=1) <= 0:
        raise ValueError("durations must be positive and max_lag nonnegative")
    actor_free = 0
    publish_times: list[int] = []
    lags = []
    for duration in durations:
        learner_free = publish_times[-1] if publish_times else 0
        start = actor_free if overlap else max(actor_free, learner_free)
        actor_free = start + duration
        train_start = max(actor_free, learner_free)
        actor_version = bisect_right(publish_times, start)
        train_version = bisect_right(publish_times, train_start)
        lag = train_version - actor_version
        lags.append(lag)
        if lag <= max_lag:
            publish_times.append(train_start + update_time)
    accepted = len(publish_times)
    elapsed = max(actor_free, publish_times[-1] if accepted else 0)
    return Trace(elapsed, accepted, len(durations) - accepted, tuple(lags))
