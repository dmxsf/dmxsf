"""A decoder-only Transformer in one configurable class: GPT-2 ... Llama.

The switches (norm, position encoding, MLP, GQA, bias, weight tying, pre/post-norm) are the ones the
course walks through. With GPTConfig.gpt2_small() the parameter count is exactly GPT-2's 124,439,808
and with the Llama 3 8B config it is exactly 8,030,261,248 (see configs.py and the tests).

Course pages: https://dmxsf.com/wiki/architecture/transformer-block/
              https://dmxsf.com/wiki/architecture/modern-llm-architecture/
"""

import math
from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from dmxsf.architecture.attention import Attention, pool_kv_heads
from dmxsf.architecture.layers import GELUMLP, SwiGLU, make_norm
from dmxsf.architecture.position import (
    alibi_bias,
    alibi_slopes,
    rope_cos_sin,
    rope_scaled_inv_freq,
    sinusoidal_embedding,
)
from dmxsf.utils import count_params

POSITIONS = ("learned", "sinusoidal", "rope", "alibi", "none")


@dataclass
class GPTConfig:
    vocab_size: int = 256
    block_size: int = 256  # training context; only 'learned' positions are hard-limited by it
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    n_kv_head: int | None = None  # None -> n_head (MHA); fewer -> GQA; 1 -> MQA
    head_dim: int | None = None  # None -> d_model // n_head
    d_ff: int | None = None  # None -> 4 d_model (gelu) or about 8/3 d_model (swiglu)
    norm: str = "layernorm"  # or "rmsnorm"
    norm_eps: float = 1e-5
    pos: str = "learned"  # one of POSITIONS
    mlp: str = "gelu"  # or "swiglu"
    bias: bool = True  # bias in every Linear (and in LayerNorm)
    qkv_bias: bool | None = None  # None -> same as bias; Qwen2.5 has bias on q, k, v only
    tie_embeddings: bool = True
    pre_norm: bool = True  # False: the original post-norm Transformer
    rope_base: float = 10_000.0
    rope_scaling: str = "none"  # none | linear | ntk | yarn | llama3
    rope_factor: float = 1.0
    rope_original_len: int | None = None  # None -> block_size
    init_std: float = 0.02
    window: int | None = (
        None  # sliding-window attention: each token sees itself and window - 1 before
    )

    def __post_init__(self):
        if self.pos not in POSITIONS:
            raise ValueError(f"pos must be one of {POSITIONS}")
        if self.mlp not in ("gelu", "swiglu"):
            raise ValueError("mlp must be 'gelu' or 'swiglu'")
        if self.d_ff is None:
            if self.mlp == "gelu":
                self.d_ff = 4 * self.d_model
            else:  # the Llama rule of thumb: 8/3 d, rounded up to a multiple of 64
                self.d_ff = -(-8 * self.d_model // 3 // 64) * 64
        if self.qkv_bias is None:
            self.qkv_bias = self.bias
        if self.pos == "rope" and self.dim_per_head % 2:
            raise ValueError("RoPE needs an even head dimension")

    @property
    def kv_heads(self) -> int:
        return self.n_kv_head or self.n_head

    @property
    def dim_per_head(self) -> int:
        return self.head_dim or self.d_model // self.n_head


class Block(nn.Module):
    """One Transformer block: attention then MLP, each wrapped in a residual connection."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.pre_norm = cfg.pre_norm
        self.norm1 = make_norm(cfg.norm, cfg.d_model, cfg.norm_eps, cfg.bias)
        self.attn = Attention(
            cfg.d_model,
            cfg.n_head,
            cfg.kv_heads,
            cfg.dim_per_head,
            cfg.qkv_bias,
            cfg.bias,
            window=cfg.window,
        )
        self.norm2 = make_norm(cfg.norm, cfg.d_model, cfg.norm_eps, cfg.bias)
        mlp_cls = SwiGLU if cfg.mlp == "swiglu" else GELUMLP
        self.mlp = mlp_cls(cfg.d_model, cfg.d_ff, bias=cfg.bias)

    def forward(self, x, return_weights=False, **attn_kwargs):
        """With return_weights=True also returns the attention weights, (B, heads, T, T)."""
        attn_kwargs["return_weights"] = return_weights
        if self.pre_norm:  # x + f(norm(x)): the identity path stays clean
            out = self.attn(self.norm1(x), **attn_kwargs)
            out, weights = out if return_weights else (out, None)
            x = x + out
            x = x + self.mlp(self.norm2(x))
        else:  # the 2017 layout: norm(x + f(x))
            out = self.attn(x, **attn_kwargs)
            out, weights = out if return_weights else (out, None)
            x = self.norm1(x + out)
            x = self.norm2(x + self.mlp(x))
        return (x, weights) if return_weights else x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.attn_impl = "sdpa"  # "manual" spells the attention out; both give the same numbers
        # Activation recomputation while training: None, "reentrant" or "non-reentrant". Every block
        # then keeps only its input and redoes its forward pass in backward (compute-and-memory page).
        self.recompute: str | None = None
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model) if cfg.pos == "learned" else None
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.final_norm = (
            make_norm(cfg.norm, cfg.d_model, cfg.norm_eps, cfg.bias)
            if cfg.pre_norm
            else nn.Identity()
        )
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        if cfg.pos == "alibi":
            self.register_buffer("alibi_slopes", alibi_slopes(cfg.n_head), persistent=False)
        self.inv_freq: torch.Tensor | None = (
            None  # kept out of the buffers so .to(bf16) cannot round it
        )
        self.rope_attention_factor = 1.0
        self.set_rope_scaling(cfg.rope_scaling, cfg.rope_factor, cfg.rope_original_len)
        self.apply(self._init_weights)
        if cfg.pos == "sinusoidal":  # embedding std d^-1/2, times sqrt(d) in forward: unit scale
            nn.init.normal_(self.tok_emb.weight, std=cfg.d_model**-0.5)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def set_rope_scaling(
        self, kind: str = "none", factor: float = 1.0, original_len: int | None = None
    ):
        """Change the RoPE frequencies of a (trained) model: position interpolation, NTK, YaRN, Llama 3."""
        if self.cfg.pos != "rope":
            return
        inv_freq, factor_att = rope_scaled_inv_freq(
            kind,
            self.cfg.dim_per_head,
            self.cfg.rope_base,
            factor,
            original_len or self.cfg.block_size,
        )
        self.inv_freq, self.rope_attention_factor = inv_freq, factor_att

    def num_params(self) -> int:
        """Counts a tied embedding once."""
        return count_params(self)

    def embed(self, idx: torch.Tensor, start: int = 0):
        """The input of the first block: token embeddings plus what the position scheme adds.

        Returns ``(x, rope, bias)``; ``rope`` is a ``(cos, sin)`` pair or None, ``bias`` the ALiBi
        bias or None. Every block receives the same ``rope`` and ``bias``.
        """
        t = idx.shape[1]
        positions = torch.arange(start, start + t, device=idx.device)
        x = self.tok_emb(idx)
        rope = bias = None
        pos = self.cfg.pos
        if pos == "learned":
            if start + t > self.cfg.block_size:
                raise ValueError(
                    f"learned positions stop at {self.cfg.block_size}; got {start + t}"
                )
            x = x + self.pos_emb(positions)
        elif pos == "sinusoidal":  # Vaswani et al.: scale the embedding up so it is not drowned out
            x = x * math.sqrt(self.cfg.d_model) + sinusoidal_embedding(
                positions, self.cfg.d_model
            ).to(x.dtype)
        elif pos == "rope":
            cos, sin = rope_cos_sin(
                positions, self.inv_freq.to(idx.device), self.rope_attention_factor
            )
            rope = (cos.to(x.dtype), sin.to(x.dtype))
        elif pos == "alibi":
            keys = torch.arange(0, start + t, device=idx.device)
            bias = alibi_bias(self.alibi_slopes, positions, keys).to(x.dtype)
        return x, rope, bias

    def forward(self, idx: torch.Tensor, cache=None) -> torch.Tensor:
        """idx: (B, T) token ids -> logits (B, T, vocab). With a KV cache, idx holds only the new tokens."""
        x, rope, bias = self.embed(idx, 0 if cache is None else cache.length)
        for i, block in enumerate(self.blocks):
            if self.recompute and cache is None and torch.is_grad_enabled():
                x = checkpoint(
                    lambda h, block=block: block(h, rope=rope, bias=bias, impl=self.attn_impl),
                    x,
                    use_reentrant=self.recompute == "reentrant",
                )
            else:
                x = block(x, rope=rope, bias=bias, cache=cache, layer_idx=i, impl=self.attn_impl)
        return self.lm_head(self.final_norm(x))

    @torch.no_grad()
    def attention_maps(self, idx: torch.Tensor) -> list[torch.Tensor]:
        """The attention weights of every layer, each (B, heads, T, T): what each token looked at."""
        x, rope, bias = self.embed(idx, 0)
        maps = []
        for block in self.blocks:
            x, weights = block(x, rope=rope, bias=bias, return_weights=True)
            maps.append(weights)
        return maps


def to_gqa(model: GPT, n_kv_head: int) -> GPT:
    """Copy a trained model into one with n_kv_head KV heads, mean-pooling the heads of each group."""
    new = GPT(replace(model.cfg, n_kv_head=n_kv_head))
    for old_block, new_block in zip(model.blocks, new.blocks, strict=True):
        new_block.attn = pool_kv_heads(old_block.attn, n_kv_head)
    rest = {k: v for k, v in model.state_dict().items() if ".attn." not in k}
    result = new.load_state_dict(rest, strict=False)
    assert not result.unexpected_keys and all(".attn." in k for k in result.missing_keys)
    return new
