"""State space and hybrids: equivalent scans and memory.

Wiki pages
  Chinese: https://dmxsf.com/wiki/architecture/state-space-and-hybrid/
  English: https://dmxsf.com/en/wiki/architecture/state-space-and-hybrid/
"""

import numpy as np


def recurrent(decay: np.ndarray, writes: np.ndarray) -> np.ndarray:
    """h[t] = decay[t] * h[t-1] + writes[t], starting at zero."""
    state = np.zeros_like(writes[0])
    history = []
    for factor, update in zip(decay, writes, strict=True):
        state = factor * state + update
        history.append(state)
    return np.stack(history)


def causal_matrix(decay: np.ndarray) -> np.ndarray:
    """Coefficient of write j in state i, for j <= i."""
    size = len(decay)
    matrix = np.eye(size)
    for i in range(1, size):
        matrix[i, :i] = matrix[i - 1, :i] * decay[i]
    return matrix


def hybrid_memory(
    context: int,
    kv_bytes_per_token: int,
    attention_layers: int,
    state_bytes: int,
    recurrent_layers: int,
) -> tuple[int, int]:
    """Return context-dependent KV bytes and fixed recurrent-state bytes."""
    return context * kv_bytes_per_token * attention_layers, state_bytes * recurrent_layers
