"""The arithmetic of a training run: FLOPs, MFU, GPU hours, and the bytes of model state and activations.

Course pages: https://dmxsf.com/wiki/pretraining/compute-and-memory/
              https://dmxsf.com/wiki/pretraining/distributed-training/

Everything here is a formula. The labs check each of them against something that is *measured*
(``torch.utils.flop_counter``, the bytes autograd saves, the bytes an optimizer allocates).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, NamedTuple

import torch

GiB = 2**30


def train_flops(n_params: float, tokens: float) -> float:
    """C = 6 N D: two FLOPs per parameter in the forward pass, four in the backward pass.

    ``n_params`` counts the parameters that take part in matrix products (not the input embedding
    table; for a mixture of experts, only the activated ones).
    """
    return 6 * n_params * tokens


def attention_flops_per_token(layers: int, heads: int, d_head: int, seq_len: int) -> float:
    """The part of a training token's cost that has no parameters: 12 L H d_h s (full s x s matrix).

    Per layer the forward pass spends 4 H d_h s on QK^T and A V together; training is three times that.
    """
    return 12 * layers * heads * d_head * seq_len


def gpu_hours(flops: float, peak_flops: float, mfu: float) -> float:
    return flops / (peak_flops * mfu * 3600)


def mfu(tokens_per_second: float, flops_per_token: float, n_gpus: int, peak_flops: float) -> float:
    """Model FLOPs utilisation (PaLM): useful FLOPs per second over what the hardware could do."""
    return tokens_per_second * flops_per_token / (n_gpus * peak_flops)


def implied_mfu(flops: float, hours: float, peak_flops: float) -> float:
    """What a reported number of GPU hours implies: the check for a claim in a technical report."""
    return flops / (hours * 3600 * peak_flops)


def model_state_bytes(n_params: float, n_gpus: int = 1, zero: int = 0) -> float:
    """Per-GPU bytes of weights + gradients + Adam state in mixed precision (16 bytes per parameter).

    zero = 0: nothing is sharded. 1: optimizer state (12 bytes). 2: and the gradients (2).
    3: and the weights (2), which is also what FSDP's full shard does.
    """
    if zero not in (0, 1, 2, 3):
        raise ValueError("zero must be 0, 1, 2 or 3")
    weights, grads, optimizer = 2.0, 2.0, 12.0
    if zero >= 1:
        optimizer /= n_gpus
    if zero >= 2:
        grads /= n_gpus
    if zero >= 3:
        weights /= n_gpus
    return n_params * (weights + grads + optimizer)


def min_gpus_for_zero3(n_params: float, gpu_bytes: float) -> int:
    """Fewest GPUs whose ZeRO-3 shard of the model state fits (before any activation)."""
    n = 1
    while model_state_bytes(n_params, n, zero=3) > gpu_bytes:
        n += 1
    return n


def activation_bytes_per_token_layer(d_model: int, kv_dim: int, d_ff: int) -> int:
    """Bytes autograd keeps per token and layer for a Llama-style block in BF16 with fused kernels.

    12 h (residual inputs and outputs of the two norms, q, the attention output) + 4 G d_h (k and v)
    + 8 f (gate, up, silu(gate) and their product); ``kv_dim`` is G * d_h.
    """
    return 12 * d_model + 4 * kv_dim + 8 * d_ff


def recompute_bytes_per_token(d_model: int, layers: int, bytes_per_token_layer: int) -> int:
    """Full recomputation keeps each layer's input (2 h bytes) and rebuilds one layer at a time."""
    return layers * 2 * d_model + bytes_per_token_layer


def megatron_flops(batch: int, seq: int, layers: int, hidden: int, vocab: int) -> float:
    """Narayanan et al. (2021), eq. (3): FLOPs of one training step with full recomputation."""
    return (
        96
        * batch
        * seq
        * layers
        * hidden**2
        * (1 + seq / (6 * hidden) + vocab / (16 * layers * hidden))
    )


def matmul_params(model) -> int:
    """N in 6ND for a ``dmxsf.architecture.gpt.GPT``: the weights that take part in a matrix product.

    That is every 2-D weight inside the blocks, plus the output matrix. The input embedding table is
    only looked up (no product), so it is left out; when it is tied to the output matrix the shared
    tensor is counted once, as the output matrix. Norm gains are left out too (1-D, element-wise).
    """
    inside = sum(p.numel() for p in model.blocks.parameters() if p.ndim == 2)
    return inside + model.lm_head.weight.numel()


class Saved(NamedTuple):
    total: int  # bytes of the distinct tensors autograd keeps for the backward pass
    tensors: int  # how many distinct allocations those are
    by_dtype: dict[torch.dtype, int]
    output: Any  # what ``fn`` returned


def saved_tensor_bytes(fn: Callable[[], Any], params: Iterable[torch.Tensor] = ()) -> Saved:
    """Run a forward pass ``fn`` and count what autograd saves for backward: the activations.

    A tensor saved by two operations is one allocation, so each storage is counted once; the
    parameters themselves (``params``) are not activations and are skipped.
    """
    skip = {p.untyped_storage().data_ptr() for p in params}
    found: dict[int, tuple[int, torch.dtype]] = {}

    def pack(t: torch.Tensor) -> torch.Tensor:
        storage = t.untyped_storage()
        if storage.data_ptr() not in skip:
            found[storage.data_ptr()] = (storage.nbytes(), t.dtype)
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn()
    by_dtype: dict[torch.dtype, int] = {}
    for nbytes, dtype in found.values():
        by_dtype[dtype] = by_dtype.get(dtype, 0) + nbytes
    return Saved(sum(by_dtype.values()), len(found), by_dtype, out)
