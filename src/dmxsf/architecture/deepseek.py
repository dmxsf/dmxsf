"""A DeepSeek-V3-style language model: MLA attention, shared + routed experts, dense first layers.

The same class describes the real model (``DEEPSEEK_V3``, 671B parameters: build it on the ``meta``
device to count them) and a toy one you can train on a laptop. It leaves out multi-token
prediction, FP8 and the YaRN extension of the context window.

Course pages: https://dmxsf.com/wiki/architecture/mixture-of-experts/
              https://dmxsf.com/wiki/architecture/efficient-attention/
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from dmxsf.architecture.layers import RMSNorm, SwiGLU
from dmxsf.architecture.mla import MLACache, MLAttention
from dmxsf.architecture.moe import MoELayer
from dmxsf.architecture.position import rope_cos_sin, rope_inv_freq
from dmxsf.utils import count_params


@dataclass(frozen=True)
class DeepSeekConfig:
    vocab_size: int = 256
    block_size: int = 256
    d_model: int = 64
    n_layer: int = 4
    first_dense: int = 1  # the first layers keep an ordinary feed-forward network
    n_heads: int = 4
    q_rank: int | None = 32  # low-rank bottleneck of the query
    d_c: int = 16  # the cached KV latent
    d_nope: int = 16  # content dimensions per head
    d_rope: int = 8  # position dimensions per head (and of the shared key)
    d_v: int = 16
    d_dense: int = 128  # width of the dense layers' feed-forward network
    n_experts: int = 8  # routed experts per MoE layer
    n_shared: int = 1
    top_k: int = 2
    d_expert: int = 32
    n_group: int = 1  # expert groups (one node each) for node-limited routing
    topk_group: int | None = None
    route_scale: float = 1.0
    rope_base: float = 10_000.0
    init_std: float = 0.02

    @property
    def n_moe_layers(self) -> int:
        return self.n_layer - self.first_dense


# config.json of deepseek-ai/DeepSeek-V3 (main model; the 14B-parameter MTP module is not included)
DEEPSEEK_V3 = DeepSeekConfig(
    vocab_size=129280,
    block_size=4096,
    d_model=7168,
    n_layer=61,
    first_dense=3,
    n_heads=128,
    q_rank=1536,
    d_c=512,
    d_nope=128,
    d_rope=64,
    d_v=128,
    d_dense=18432,
    n_experts=256,
    n_shared=1,
    top_k=8,
    d_expert=2048,
    n_group=8,
    topk_group=4,
    route_scale=2.5,
)


class DeepSeekBlock(nn.Module):
    def __init__(self, cfg: DeepSeekConfig, dense: bool):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        self.attn = MLAttention(
            cfg.d_model, cfg.n_heads, cfg.d_nope, cfg.d_v, cfg.d_c, cfg.d_rope, cfg.q_rank
        )
        if dense:
            self.mlp = SwiGLU(cfg.d_model, cfg.d_dense)
        else:
            self.mlp = MoELayer(
                cfg.d_model,
                cfg.d_expert,
                cfg.n_experts,
                cfg.top_k,
                cfg.n_shared,
                gate="deepseek",
                route_scale=cfg.route_scale,
                n_group=cfg.n_group,
                topk_group=cfg.topk_group,
            )

    def forward(self, x, rope, cache=None, layer_idx=0, absorb=False):
        x = x + self.attn(self.norm1(x), rope, cache, layer_idx, absorb)
        return x + self.mlp(self.norm2(x))


class DeepSeekLM(nn.Module):
    def __init__(self, cfg: DeepSeekConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            DeepSeekBlock(cfg, dense=i < cfg.first_dense) for i in range(cfg.n_layer)
        )
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.register_buffer("inv_freq", rope_inv_freq(cfg.d_rope, cfg.rope_base), persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def num_params(self) -> int:
        return count_params(self)

    def new_cache(self) -> MLACache:
        return MLACache(self.cfg.n_layer)

    def forward(
        self, idx: torch.Tensor, cache: MLACache | None = None, absorb: bool = False
    ) -> torch.Tensor:
        """idx: (B, T) -> logits (B, T, vocab). With a cache, idx holds only the new tokens."""
        start = 0 if cache is None else cache.length
        positions = torch.arange(start, start + idx.shape[1], device=idx.device)
        rope = rope_cos_sin(positions, self.inv_freq)
        x = self.tok_emb(idx)
        for i, block in enumerate(self.blocks):
            x = block(x, rope, cache, i, absorb)
        return self.lm_head(self.final_norm(x))
