"""Collective communication: simulate a ring all-reduce.

Wiki pages
  Chinese: https://dmxsf.com/wiki/systems/collective-communication/
  English: https://dmxsf.com/en/wiki/systems/collective-communication/
"""

import numpy as np


def ring_all_reduce(values: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Simulate simultaneous clockwise sends and record per-rank bytes per round."""
    ranks, width = values.shape
    if ranks < 1:
        raise ValueError("the ring needs at least one rank")
    if width % ranks:
        raise ValueError("vector width must be divisible by the number of ranks")
    chunk = width // ranks
    buffers = values.copy().reshape(ranks, ranks, chunk)
    traffic = []
    for reduce in (True, False):
        for step in range(ranks - 1):
            # All sends see the state as it was at the start of the round.
            indices = [(rank + int(not reduce) - step) % ranks for rank in range(ranks)]
            outgoing = buffers[np.arange(ranks), indices].copy()
            for src, column in enumerate(indices):
                dst = (src + 1) % ranks
                if reduce:
                    buffers[dst, column] += outgoing[src]
                else:
                    buffers[dst, column] = outgoing[src]
            traffic.append(chunk * values.itemsize)
    return buffers.reshape(ranks, width), traffic


def alpha_beta_seconds(traffic: list[int], latency: float, bandwidth: float) -> float:
    """Cost of synchronized rounds with a fixed startup per round."""
    return len(traffic) * latency + sum(traffic) / bandwidth
