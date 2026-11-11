"""Quantization: the page's algorithms as plain PyTorch, applied as *fake* quantization.

Course page: https://dmxsf.com/wiki/inference/quantization/

Fake quantization rounds a weight to an integer grid and immediately multiplies the scale back, so the
model keeps running in floating point. That measures what the algorithm costs in accuracy, not what a
low-bit kernel gains in speed, and it does not depend on any hardware. Everything below mirrors the code
printed on the page; the lab runs it on the course's byte-level GPT.
"""

from __future__ import annotations

import contextlib
import copy
import math
from collections.abc import Callable, Iterable

import numpy as np
import torch
from torch import nn

from dmxsf.inference.kv_cache import KVCache

Group = str | int  # "tensor", "row", or a number of consecutive weights that share one scale


# ---------------------------------------------------------------------------------------------
# Rounding to a grid


def fake_quant(w: torch.Tensor, bits: int, group: Group = "row", symmetric: bool = True):
    """Quantize then dequantize ``w`` (out, in). Symmetric absmax uses the levels -(2^(b-1)-1) ..
    2^(b-1)-1; asymmetric uses 0 .. 2^b-1 with a zero point. ``group`` decides who shares a scale."""
    if group == "tensor":
        g = w.reshape(1, -1)
    elif group == "row":
        g = w
    else:
        g = w.reshape(-1, int(group))
    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        s = (g.abs().amax(dim=1, keepdim=True) / qmax).clamp(min=1e-8)
        out = (g / s).round().clamp(-qmax, qmax) * s
    else:
        levels = 2**bits - 1
        low = g.amin(dim=1, keepdim=True)
        s = ((g.amax(dim=1, keepdim=True) - low) / levels).clamp(min=1e-8)
        zero = (-low / s).round()
        out = ((g / s).round() + zero).clamp(0, levels).sub(zero) * s
    return out.reshape(w.shape)


def n_scales(shape: tuple[int, ...], group: Group) -> int:
    if group == "tensor":
        return 1
    return shape[0] if group == "row" else math.prod(shape) // int(group)


def effective_bits(
    bits: int, shapes: Iterable[tuple[int, ...]], group: Group, symmetric: bool = True
) -> float:
    """Stored bits per weight: the code plus a 16-bit scale per group (and a b-bit zero point when
    asymmetric). One scale for a whole tensor is counted as free, as on the page."""
    if group == "tensor":  # a handful of scales for millions of weights
        return float(bits)
    shapes = list(shapes)
    total = sum(math.prod(s) for s in shapes)
    per_scale = 16 if symmetric else 16 + bits
    return bits + per_scale * sum(n_scales(s, group) for s in shapes) / total


def sqnr_db(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Signal-to-quantization-noise ratio in decibels."""
    return 10 * math.log10(x.pow(2).sum().item() / (x - x_hat).pow(2).sum().item())


# ---------------------------------------------------------------------------------------------
# The widget's sample: the same 1,024 numbers as the quantization playground on dmxsf.com


def _mulberry32(seed: int) -> Callable[[], float]:
    """The generator behind the page's widget (src/widgets/dom.ts), on 32-bit integers."""
    mask = 0xFFFFFFFF
    state = seed & mask

    def next_uniform() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & mask
        t = state
        t = ((t ^ (t >> 15)) * (t | 1)) & mask
        t = t ^ ((t + (((t ^ (t >> 7)) * (t | 61)) & mask)) & mask)
        return ((t ^ (t >> 14)) & mask) / 4294967296

    return next_uniform


def widget_weights(outlier: bool = False, size: int = 1024, seed: int = 7) -> torch.Tensor:
    """Fixed-seed standard normals (Box-Muller), optionally with weight 500 replaced by 20."""
    random = _mulberry32(seed)
    values = []
    for _ in range(size):
        u = max(random(), 1e-12)
        values.append(math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * random()))
    if outlier:
        values[500] = 20.0
    return torch.tensor(values, dtype=torch.float64)


# ---------------------------------------------------------------------------------------------
# A model whose linear layers can be swapped for quantized ones and put back


class Target:
    """The decoder's linear layers (not the embedding, not the output head), with their originals."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.names = [
            n
            for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and ".blocks." in f".{n}"
        ]
        self.linears = [model.get_submodule(n) for n in self.names]
        self.orig = [m.weight.data.clone() for m in self.linears]

    @property
    def numel(self) -> int:
        return sum(w.numel() for w in self.orig)

    def layer_of(self, i: int) -> int:
        return int(self.names[i].split("blocks.")[1].split(".")[0])

    def set(self, fn: Callable[[int, torch.Tensor], torch.Tensor]) -> None:
        """fn(index of the layer, original weight) -> new weight."""
        for i, (m, w) in enumerate(zip(self.linears, self.orig, strict=True)):
            m.weight.data = fn(i, w)

    def restore(self) -> None:
        self.set(lambda i, w: w.clone())

    @contextlib.contextmanager
    def quantized(self, fn: Callable[[int, torch.Tensor], torch.Tensor]):
        self.set(fn)
        try:
            yield
        finally:
            self.restore()

    def hook_inputs(self, fn: Callable[[int, torch.Tensor], torch.Tensor | None]):
        """Run fn(i, x) on the input of linear layer i before it multiplies; a returned tensor
        replaces the input. Returns the handles (remove them when done)."""

        def make(i):
            def pre(mod, args):
                out = fn(i, args[0])
                return None if out is None else (out,)

            return pre

        return [m.register_forward_pre_hook(make(i)) for i, m in enumerate(self.linears)]


@torch.no_grad()
def collect_stats(target: Target, batches: Iterable[torch.Tensor]):
    """What the calibration methods need from a handful of batches, per linear layer: the second
    moment of the input H = mean(x x^T), the mean |x| per channel (AWQ) and the max |x| (SmoothQuant)."""
    lins = target.linears
    h = [torch.zeros(m.in_features, m.in_features) for m in lins]
    mean = [torch.zeros(m.in_features) for m in lins]
    peak = [torch.zeros(m.in_features) for m in lins]
    count = [0] * len(lins)

    def record(i, x):
        x = x.reshape(-1, x.shape[-1]).float()
        h[i] += x.T @ x
        mean[i] += x.abs().sum(0)
        peak[i] = torch.maximum(peak[i], x.abs().amax(0))
        count[i] += x.shape[0]

    handles = target.hook_inputs(lambda i, x: record(i, x))
    for batch in batches:
        target.model(batch)
    for handle in handles:
        handle.remove()
    return (
        [x / n for x, n in zip(h, count, strict=True)],
        [x / n for x, n in zip(mean, count, strict=True)],
        peak,
    )


# ---------------------------------------------------------------------------------------------
# Outliers on demand: models that compute exactly what the original computes, with one channel
# rescaled. Large models grow such channels by themselves; a 0.86M model does not, so we build them.


@torch.no_grad()
def with_activation_outlier(model: nn.Module, scale: float, channel: int = 17) -> nn.Module:
    """A copy whose residual-stream channel ``channel`` is ``scale`` times larger after every
    normalisation, with the matching input column of the layers that read it divided by ``scale``.
    The function is unchanged (up to float rounding); the activation now has an outlier channel."""
    out = copy.deepcopy(model)
    for block in out.blocks:
        readers = (
            (block.norm1, [block.attn.q_proj, block.attn.k_proj, block.attn.v_proj]),
            (block.norm2, [block.mlp.fc]),
        )
        for norm, layers in readers:
            norm.weight[channel] *= scale
            if norm.bias is not None:
                norm.bias[channel] *= scale
            for layer in layers:
                layer.weight[:, channel] /= scale
    return out


@torch.no_grad()
def with_key_outlier(model: nn.Module, scale: float, channel: int = 5) -> nn.Module:
    """A copy whose cached keys have one channel per head ``scale`` times larger, with the same
    channel of the queries divided by ``scale`` so that every q.k is unchanged. Not for RoPE, which
    rotates channels into each other."""
    if model.cfg.pos == "rope":
        raise ValueError("rescaling one channel of q and k is only exact without RoPE")
    out = copy.deepcopy(model)
    for block in out.blocks:
        a = block.attn
        rep = a.n_heads // a.n_kv_heads
        for g in range(a.n_kv_heads):
            rows = [g * a.head_dim + channel]
            a.k_proj.weight[rows] *= scale
            if a.k_proj.bias is not None:
                a.k_proj.bias[rows] *= scale
            for h in range(g * rep, (g + 1) * rep):
                q_rows = [h * a.head_dim + channel]
                a.q_proj.weight[q_rows] /= scale
                if a.q_proj.bias is not None:
                    a.q_proj.bias[q_rows] /= scale
    return out


# ---------------------------------------------------------------------------------------------
# Activations


def act_quant(x: torch.Tensor, per_token: bool) -> torch.Tensor:
    """INT8 dynamic quantization of an activation: one scale for the tensor, or one per token."""
    s = x.abs().amax(dim=-1, keepdim=True) if per_token else x.abs().amax()
    s = (s / 127).clamp(min=1e-8)
    return (x / s).round().clamp(-127, 127) * s


def smooth_scales(act_max: torch.Tensor, w: torch.Tensor, alpha: float) -> torch.Tensor:
    """SmoothQuant: s_j = max|X_j|^alpha / max|W_j|^(1-alpha). Divide the activation by s, multiply
    the weight by s: the product is unchanged, the outliers move from activations to weights."""
    w_max = w.abs().amax(0).clamp(min=1e-5)
    return (act_max.clamp(min=1e-5) ** alpha / w_max ** (1 - alpha)).clamp(min=1e-5)


# ---------------------------------------------------------------------------------------------
# Calibrated weight quantization: all three share one signature (a is used only by AWQ)


def rtn(w, h, a, bits, group):
    return fake_quant(w, bits, group)


def output_error(w, w_hat, h) -> float:
    """tr(dW H dW^T) with dW = w - w_hat: the squared output error on the calibration inputs."""
    d = w - w_hat
    return ((d @ h) * d).sum().item()


def awq(w, h, a, bits, group):
    """AWQ style: scale input channels by (mean activation)^alpha before rounding, search alpha."""
    best = (float("inf"), None)
    for alpha in [i / 10 for i in range(11)]:  # alpha = 0 is RTN
        s = a.clamp(min=1e-5) ** alpha
        s = s / (s.max() * s.min()).sqrt()
        w_hat = fake_quant(w * s, bits, group) / s
        best = min(best, (output_error(w, w_hat, h), w_hat), key=lambda t: t[0])
    return best[1]


def gptq(w, h, a, bits, group, damp=0.01):
    """GPTQ: quantize one column at a time and pass its error on to the columns not yet quantized,
    through the Cholesky factor of H^-1. Block size equals the group size; float64 on the CPU."""
    w, h = w.cpu().double(), h.cpu().double()
    n = w.shape[1]
    dead = h.diag() == 0  # input channels that were never active
    h[dead, dead] = 1
    w[:, dead] = 0
    h += damp * h.diag().mean() * torch.eye(n, dtype=h.dtype)
    chol = torch.linalg.cholesky(h)
    h_inv = torch.cholesky_solve(torch.eye(n, dtype=h.dtype), chol)
    u = torch.linalg.cholesky(h_inv, upper=True)
    qmax = 2 ** (bits - 1) - 1
    q = torch.zeros_like(w)
    for i1 in range(0, n, group):
        i2 = min(i1 + group, n)
        s = (w[:, i1:i2].abs().amax(1) / qmax).clamp(min=1e-8)
        err = torch.zeros(w.shape[0], i2 - i1, dtype=w.dtype)
        for j in range(i2 - i1):
            col, d = w[:, i1 + j], u[i1 + j, i1 + j]
            q[:, i1 + j] = (col / s).round().clamp(-qmax, qmax) * s
            err[:, j] = (col - q[:, i1 + j]) / d
            w[:, i1 + j : i2] -= err[:, j : j + 1] * u[i1 + j, i1 + j : i2]
        w[:, i2:] -= err @ u[i1:i2, i2:]
    return q.float()


# ---------------------------------------------------------------------------------------------
# The KV cache


def kv_quant(x: torch.Tensor, bits: int, per_channel: bool = False, g: int = 32) -> torch.Tensor:
    """x: (batch, heads, tokens, head_dim). Default: one scale per token and head. per_channel: one
    scale per channel for every g tokens (what KIVI does for K)."""
    b, h, n, d = x.shape
    if per_channel:
        x = x.reshape(b, h, n // g, g, d).transpose(3, 4)
    q = fake_quant(x.reshape(-1, x.shape[-1]), bits, "row").reshape(x.shape)
    return q.transpose(3, 4).reshape(b, h, n, d) if per_channel else q


class QuantKVCache(KVCache):
    """A cache that stores keys and values rounded to ``k_bits`` / ``v_bits`` (16: untouched)."""

    def __init__(
        self, n_layers: int, k_bits: int = 16, k_per_channel: bool = False, v_bits: int = 16
    ):
        super().__init__(n_layers)
        self.k_bits, self.k_per_channel, self.v_bits = k_bits, k_per_channel, v_bits

    def update(self, layer, k, v):
        if self.k_bits < 16:
            k = kv_quant(k, self.k_bits, self.k_per_channel)
        if self.v_bits < 16:
            v = kv_quant(v, self.v_bits)
        return super().update(layer, k, v)


@torch.no_grad()
def bits_per_byte_cached(
    model: nn.Module,
    data: np.ndarray,
    make_cache: Callable[[], KVCache],
    block_size: int = 128,
    windows: int = 128,
) -> float:
    """The windows of pretraining.train.bits_per_byte, run with a cache so that attention reads
    the cache's (possibly rounded) keys and values."""
    windows = max(1, min(windows, len(data) // block_size))
    stride = max(1, (len(data) - block_size - 1) // windows)
    rows = np.stack([data[j * stride : j * stride + block_size + 1] for j in range(windows)])
    rows = torch.from_numpy(rows.astype(np.int64))
    total, count = 0.0, 0
    for r in rows.split(64):
        logits = model(r[:, :-1], cache=make_cache())
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), r[:, 1:].reshape(-1)
        )
        total += loss.item() * r[:, 1:].numel()
        count += r[:, 1:].numel()
    return total / count / math.log(2)
