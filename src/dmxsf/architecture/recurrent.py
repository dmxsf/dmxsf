"""From RNN to attention: gradients and retrieval paths.

Wiki pages
  Chinese: https://dmxsf.com/wiki/architecture/from-rnn-to-attention/
  English: https://dmxsf.com/en/wiki/architecture/from-rnn-to-attention/
"""

import torch


def recurrent_gradient(steps: int, gain: float, gated: bool = False) -> float:
    """Differentiate the final scalar state with respect to the first one."""
    initial = torch.zeros((), dtype=torch.float64, requires_grad=True)
    state = initial
    for _ in range(steps):
        state = gain * state if gated else torch.tanh(gain * state)
    return float(torch.autograd.grad(state, initial)[0])


def attention_read(keys: torch.Tensor, values: torch.Tensor, query_index: int, scale: float):
    """Use one stored key as the query and return context and weights."""
    query = keys[query_index]
    weights = (scale * (keys @ query)).softmax(dim=0)
    return weights @ values, weights
