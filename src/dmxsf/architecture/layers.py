"""Normalisation layers and feed-forward blocks.

Course page: https://dmxsf.com/wiki/architecture/transformer-block/
"""

import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm(nn.Module):
    """Subtract the mean, divide by the standard deviation, then scale (and shift)."""

    def __init__(self, dim: int, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        y = (x - mean) * torch.rsqrt(var + self.eps) * self.weight
        return y if self.bias is None else y + self.bias


class RMSNorm(nn.Module):
    """Divide by the root mean square only: no mean subtraction, no bias (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()  # the statistics are computed in fp32 even when x is bf16
        y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return y.to(x.dtype) * self.weight


def make_norm(kind: str, dim: int, eps: float = 1e-5, bias: bool = True) -> nn.Module:
    if kind == "layernorm":
        return LayerNorm(dim, eps, bias)
    if kind == "rmsnorm":
        return RMSNorm(dim, eps)
    raise ValueError(f"unknown norm {kind!r}; use 'layernorm' or 'rmsnorm'")


class GELUMLP(nn.Module):
    """The GPT-2 feed-forward block: expand, GELU, project back."""

    def __init__(self, d_model: int, d_ff: int, bias: bool = True):
        super().__init__()
        self.fc = nn.Linear(d_model, d_ff, bias=bias)
        self.proj = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.gelu(self.fc(x)))


class SwiGLU(nn.Module):
    """The Llama feed-forward block: SiLU(x W_gate) * (x W_up), then W_down (Shazeer, 2020)."""

    def __init__(self, d_model: int, d_ff: int, bias: bool = False):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=bias)
        self.up = nn.Linear(d_model, d_ff, bias=bias)
        self.down = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))
