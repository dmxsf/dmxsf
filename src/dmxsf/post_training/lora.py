"""LoRA from scratch: a frozen linear layer plus a trainable low-rank path, merging, and NF4.

``h = W x + (alpha / r) B A x`` with ``A`` (r x d_in) random and ``B`` (d_out x r) zero, so training
starts exactly at the base model. Everything the LoRA page talks about is here: the parameter count
``r (d_in + d_out)`` per adapted matrix, the zero gradient on ``A`` in the first step, the exactness
of merging, and the 4-bit NormalFloat grid that QLoRA stores the frozen base in.

Course page: https://dmxsf.com/wiki/post-training/lora-finetuning/
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterable, Mapping

import torch
from torch import nn

# Which linear layers of dmxsf's GPT get an adapter. Names are suffixes of module paths.
TARGETS = {
    "all": ("attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj", "mlp.fc", "mlp.proj"),
    "attention": ("attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj"),
    "qv": ("attn.q_proj", "attn.v_proj"),
    "mlp": ("mlp.fc", "mlp.proj"),
}


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` with a trainable low-rank path in parallel."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, zero_a: bool = False):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.scale = r, alpha / r
        self.enabled = True
        self.A = nn.Parameter(torch.empty(r, base.in_features, dtype=base.weight.dtype))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=base.weight.dtype))
        if zero_a:  # the mistake: with A = B = 0 neither matrix ever receives a gradient
            nn.init.zeros_(self.A)
        else:
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:  # the base model alone
            return self.base(x)
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

    @torch.no_grad()
    def delta(self) -> torch.Tensor:
        """The update this adapter adds to the weight: ``(alpha / r) B A``."""
        return self.scale * (self.B @ self.A)

    @torch.no_grad()
    def merged(self) -> torch.Tensor:
        return self.base.weight + self.delta()


@contextlib.contextmanager
def adapters_disabled(model: nn.Module):
    """Run the frozen base model: inside the block every adapter is bypassed.

    This is how LoRA-based DPO gets its reference model without a second copy of the weights.
    """
    adapters = [m for m in model.modules() if isinstance(m, LoRALinear)]
    for a in adapters:
        a.enabled = False
    try:
        yield
    finally:
        for a in adapters:
            a.enabled = True


def add_lora(
    model: nn.Module,
    targets: Iterable[str] | str = "all",
    r: int = 8,
    alpha: float = 16.0,
    zero_a: bool = False,
) -> list[LoRALinear]:
    """Freeze ``model`` and wrap every matching ``nn.Linear`` in a :class:`LoRALinear` (in place).

    ``targets`` is a key of :data:`TARGETS` or an iterable of module-path suffixes.
    """
    suffixes = TARGETS[targets] if isinstance(targets, str) else tuple(targets)
    for p in model.parameters():
        p.requires_grad_(False)
    adapters = []
    for path, module in list(model.named_modules()):
        for name, child in list(module.named_children()):
            full = f"{path}.{name}" if path else name
            if isinstance(child, nn.Linear) and full.endswith(suffixes):
                lora = LoRALinear(child, r, alpha, zero_a)
                setattr(module, name, lora)
                adapters.append(lora)
    return adapters


def merge_lora(model: nn.Module) -> None:
    """Fold every adapter into its base weight and remove it (in place): zero extra cost afterwards."""
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                child.base.weight.data = child.merged()
                setattr(module, name, child.base)


def trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def adapter_params(shapes: Mapping[str, tuple[int, int]], r: int) -> int:
    """Parameters LoRA adds to one block: ``r (d_in + d_out)`` for each ``(d_in, d_out)``."""
    return sum(r * (d_in + d_out) for d_in, d_out in shapes.values())


# --- 4-bit NormalFloat (QLoRA) -----------------------------------------------------------------

# The 16 NF4 levels (QLoRA paper, appendix E): quantiles of a normal distribution rescaled to
# [-1, 1], with an exact zero.
NF4 = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)


@torch.no_grad()
def nf4_roundtrip(w: torch.Tensor, block: int = 64) -> torch.Tensor:
    """Quantize to NF4 (one absmax scale per ``block`` weights), then dequantize.

    The result is what the model "sees" after QLoRA's storage step. ``w.numel()`` must be a
    multiple of ``block``.
    """
    book = torch.tensor(NF4)
    flat = w.float().reshape(-1, block)
    scale = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    idx = ((flat / scale).unsqueeze(-1) - book).abs().argmin(dim=-1)
    return (book[idx] * scale).reshape(w.shape)


def nf4_bits_per_weight(block: int = 64, double_quant: bool = False) -> float:
    """Storage per weight: 4 bits plus the block scales (fp32, or 8-bit with a fp32 constant per 256)."""
    if double_quant:
        return 4 + 8 / block + 32 / (block * 256)
    return 4 + 32 / block
