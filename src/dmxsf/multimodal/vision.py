"""How an image becomes tokens, and what the tokens cost: patches, merging, dynamic resolution, KV cache, FLOPs.

Course page: https://dmxsf.com/wiki/multimodal/multimodal-llms/

Pure PyTorch and arithmetic. The shapes and counts are those of the public checkpoints the page uses
(LLaVA-1.5-7B, Qwen2-VL-7B, Qwen3-VL-4B), written down in the lab from their configuration files.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

# --- from vision-encoder output to the language model's input ---------------------------------------


def merge_patches(vit_out: torch.Tensor, grid_h: int, grid_w: int, merge: int = 2) -> torch.Tensor:
    """Concatenate every ``merge`` x ``merge`` block of neighbouring patch vectors into one vector.

    ``vit_out`` is ``(grid_h * grid_w, d)`` in row-major patch order; the result is
    ``(grid_h * grid_w / merge**2, merge**2 * d)``. Only a view, a permute and a reshape: no parameters."""
    d = vit_out.shape[-1]
    x = vit_out.view(grid_h // merge, merge, grid_w // merge, merge, d)
    return x.permute(0, 2, 1, 3, 4).reshape(-1, merge * merge * d)


def projector(d_in: int, d_llm: int, hidden: int | None = None) -> nn.Module:
    """The two-layer MLP that maps merged patch vectors to the language model's width."""
    hidden = hidden or d_in
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_llm))


def splice(
    ids: torch.Tensor, embed: nn.Embedding, visual: torch.Tensor, placeholder: int
) -> torch.Tensor:
    """The prompt holds one placeholder id per visual token; after the embedding lookup those rows are
    replaced by the visual tokens, and the language model gets one matrix."""
    inputs = embed(ids.clamp(min=0))  # the placeholder rows are overwritten below
    return inputs.masked_scatter((ids == placeholder)[:, None], visual)


# --- how many tokens an image takes -----------------------------------------------------------------


@dataclass(frozen=True)
class VisionConfig:
    """What decides an image's token count and cost; ``fixed`` is the side of a fixed-size input."""

    name: str
    patch: int
    merge: int
    fixed: int | None
    min_pixels: int
    max_pixels: int
    layers: int
    kv_heads: int
    head_dim: int
    params_b: float

    @property
    def factor(self) -> int:
        """Pixels per side covered by one visual token."""
        return self.patch * self.merge

    @property
    def kv_bytes_per_token(self) -> int:
        """K and V, every layer, FP16."""
        return 2 * self.layers * self.kv_heads * self.head_dim * 2


LLAVA_15 = VisionConfig("LLaVA-1.5-7B", 14, 1, 336, 0, 0, 32, 32, 128, 6.74)
QWEN2_VL = VisionConfig("Qwen2-VL-7B", 14, 2, None, 3136, 12_845_056, 28, 4, 128, 7.6)
QWEN3_VL = VisionConfig("Qwen3-VL-4B", 16, 2, None, 65_536, 16_777_216, 36, 8, 128, 4.02)


def smart_resize(h: int, w: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Round both sides to multiples of ``factor`` (ties to even, as Python's ``round``) and keep the pixel
    count inside the budget: too large is scaled down, too small scaled up. A port of ``smart_resize`` in
    the Hugging Face Qwen2-VL and Qwen3-VL image processors."""
    hb, wb = round(h / factor) * factor, round(w / factor) * factor
    if hb * wb > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        hb = max(factor, math.floor(h / beta / factor) * factor)
        wb = max(factor, math.floor(w / beta / factor) * factor)
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb = math.ceil(h * beta / factor) * factor
        wb = math.ceil(w * beta / factor) * factor
    return hb, wb


@dataclass(frozen=True)
class Tokens:
    height: int
    width: int
    patches: int
    tokens: int


def visual_tokens(cfg: VisionConfig, height: int, width: int, cap_tokens: int = 0) -> Tokens:
    """Size after preprocessing, patches in the vision encoder and visual tokens for an image.
    ``cap_tokens`` lowers the model's ``max_pixels`` so that no image takes more than that many tokens."""
    if cfg.fixed is not None:
        h = w = cfg.fixed
    else:
        max_pixels = (
            min(cfg.max_pixels, cap_tokens * cfg.factor**2) if cap_tokens > 0 else cfg.max_pixels
        )
        h, w = smart_resize(height, width, cfg.factor, min(cfg.min_pixels, max_pixels), max_pixels)
    patches = (h // cfg.patch) * (w // cfg.patch)
    return Tokens(h, w, patches, patches // cfg.merge**2)


# --- the cost of the vision part against the language part ------------------------------------------

# Qwen3-VL-4B shapes, from the model's config.json
VIT = dict(depth=24, d=1024, ff=4096)
LLM = dict(depth=36, d=2560, heads=32, kv=8, hd=128, ff=9728)


def vit_flops(n_tok: int, merge: int = 2) -> float:
    """Matrix-multiply FLOPs of the vision encoder for an image of ``n_tok`` visual tokens. It sees
    ``merge**2`` times as many patches, and they attend to each other in both directions."""
    m = merge * merge * n_tok
    weights = 4 * VIT["d"] ** 2 + 2 * VIT["d"] * VIT["ff"]  # per layer
    linear = 2 * m * VIT["depth"] * weights
    attn = 4 * m * m * VIT["d"] * VIT["depth"]  # QK^T and AV, 2 * m^2 * d each
    return linear + attn


def llm_flops(n_tok: int) -> float:
    """Prefill of ``n_tok`` tokens; causal attention counts the lower triangle only."""
    q, kv = LLM["heads"] * LLM["hd"], LLM["kv"] * LLM["hd"]
    d = LLM["d"]
    weights = d * (q + 2 * kv) + q * d + 3 * d * LLM["ff"]
    linear = 2 * n_tok * LLM["depth"] * weights
    attn = 2 * n_tok * n_tok * q * LLM["depth"]
    return linear + attn


def vit_share(n_tok: int) -> float:
    """Share of the compute (not of the time) that the vision encoder takes."""
    v, lm = vit_flops(n_tok), llm_flops(n_tok)
    return v / (v + lm)


# --- the projector bill, for a LLaMA-style language model ---------------------------------------


def llama_params(layers: int, d: int, ff: int, vocab: int, tied: bool = False) -> int:
    """Parameters of a LLaMA-style decoder: attention (4 d^2), a gated MLP (3 d ff), two norms per layer,
    the embedding and an output head, one final norm."""
    per_layer = 4 * d * d + 3 * d * ff + 2 * d
    return layers * per_layer + vocab * d * (1 if tied else 2) + d


def linear_params(d_in: int, d_out: int) -> int:
    return d_in * d_out + d_out


# --- LLaVA's two training stages --------------------------------------------------------------------


def set_llava_stage(vision: nn.Module, proj: nn.Module, lm: nn.Module, stage: int) -> int:
    """Freeze and unfreeze the three parts as LLaVA does. Stage 1 (feature alignment) trains only the
    projector; stage 2 (visual instruction tuning) trains the projector and the language model. The vision
    encoder is frozen in both. Returns the number of trainable parameters."""
    if stage not in (1, 2):
        raise ValueError("stage must be 1 or 2")
    for module, trainable in ((vision, False), (proj, True), (lm, stage == 2)):
        for p in module.parameters():
            p.requires_grad_(trainable)
    return sum(p.numel() for m in (vision, proj, lm) for p in m.parameters() if p.requires_grad)
