"""Multi-head Latent Attention (MLA), the attention of DeepSeek-V2 and V3.

Instead of caching a key and a value for every head, MLA caches one small latent vector per token
and a short key that carries the position (decoupled RoPE). At decode time the up-projections are
absorbed into the query and the output, so the keys and values of the past are never rebuilt.

Course page: https://dmxsf.com/wiki/architecture/efficient-attention/
"""

import torch
import torch.nn.functional as F
from torch import nn

from dmxsf.architecture.attention import causal_mask
from dmxsf.architecture.layers import RMSNorm
from dmxsf.architecture.position import apply_rope


class MLACache:
    """Per layer: the latent c (B, T, d_c) and the shared RoPE key (B, 1, T, d_rope)."""

    def __init__(self, n_layers: int):
        self.c: list[torch.Tensor | None] = [None] * n_layers
        self.k_rope: list[torch.Tensor | None] = [None] * n_layers

    @property
    def length(self) -> int:
        return 0 if self.c[0] is None else self.c[0].shape[1]

    def update(self, layer: int, c: torch.Tensor, k_rope: torch.Tensor):
        if self.c[layer] is not None:
            c = torch.cat([self.c[layer], c], dim=1)
            k_rope = torch.cat([self.k_rope[layer], k_rope], dim=2)
        self.c[layer], self.k_rope[layer] = c, k_rope
        return c, k_rope

    def numbers_per_token(self) -> int:
        """Cached numbers per token, summed over layers."""
        return sum(c.shape[-1] + kr.shape[-1] for c, kr in zip(self.c, self.k_rope, strict=True))

    def nbytes(self) -> int:
        tensors = [t for t in self.c + self.k_rope if t is not None]
        return sum(t.numel() * t.element_size() for t in tensors)


class MLAttention(nn.Module):
    """Causal self-attention with a compressed KV cache.

    Per head the query and key have d_nope content dimensions plus d_rope position dimensions; the
    value has d_v. Content keys and values are rebuilt from the d_c-dimensional latent by two
    up-projections; the d_rope position key is computed once per token and shared by all heads.
    The cache holds d_c + d_rope numbers per token, whatever the number of heads. With q_rank the
    query goes through a low-rank bottleneck first (DeepSeek-V2/V3); it is never cached.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_nope: int,
        d_v: int,
        d_c: int,
        d_rope: int,
        q_rank: int | None = None,
    ):
        super().__init__()
        if d_rope % 2:
            raise ValueError("d_rope must be even (RoPE rotates pairs)")
        self.n_heads, self.d_nope, self.d_v, self.d_c, self.d_rope = (
            n_heads,
            d_nope,
            d_v,
            d_c,
            d_rope,
        )
        width = n_heads * (d_nope + d_rope)
        if q_rank is None:
            self.q_proj = nn.Linear(d_model, width, bias=False)
        else:  # DeepSeek also compresses the query through a low-rank bottleneck (it is not cached)
            self.q_proj = nn.Sequential(
                nn.Linear(d_model, q_rank, bias=False),
                RMSNorm(q_rank),
                nn.Linear(q_rank, width, bias=False),
            )
        self.kv_down = nn.Linear(d_model, d_c, bias=False)  # W^{DKV}: hidden state -> latent
        self.kv_norm = RMSNorm(d_c)
        self.k_up = nn.Linear(d_c, n_heads * d_nope, bias=False)  # W^{UK}
        self.v_up = nn.Linear(d_c, n_heads * d_v, bias=False)  # W^{UV}
        self.k_rope_proj = nn.Linear(d_model, d_rope, bias=False)  # W^{KR}, shared by all heads
        self.o_proj = nn.Linear(n_heads * d_v, d_model, bias=False)

    def cached_numbers_per_token(self) -> int:
        return self.d_c + self.d_rope

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        cache: MLACache | None = None,
        layer_idx: int = 0,
        absorb: bool = False,
    ) -> torch.Tensor:
        """x: (B, T, d_model); rope = (cos, sin) of shape (T, d_rope / 2) for these positions.

        absorb=False rebuilds every key and value (how you would train). absorb=True works in the
        latent space and never rebuilds them (how you would decode). Both give the same numbers.
        """
        b, t, _ = x.shape
        h, d_nope, d_rope = self.n_heads, self.d_nope, self.d_rope
        q = self.q_proj(x).view(b, t, h, d_nope + d_rope).transpose(1, 2)
        q_nope, q_rope = q.split([d_nope, d_rope], dim=-1)
        q_rope = apply_rope(q_rope, *rope)
        c = self.kv_norm(self.kv_down(x))  # (B, T, d_c): this is what is cached
        k_rope = apply_rope(self.k_rope_proj(x).unsqueeze(1), *rope)  # (B, 1, T, d_rope)
        if cache is not None:
            c, k_rope = cache.update(layer_idx, c, k_rope)
        t_k = c.shape[1]
        mask = causal_mask(t, t_k, x.device)
        if not absorb:
            k_nope = self.k_up(c).view(b, t_k, h, d_nope).transpose(1, 2)
            v = self.v_up(c).view(b, t_k, h, self.d_v).transpose(1, 2)
            k = torch.cat([k_nope, k_rope.expand(-1, h, -1, -1)], dim=-1)
            out = F.scaled_dot_product_attention(
                torch.cat([q_nope, q_rope], dim=-1), k, v, attn_mask=mask
            )
        else:
            w_uk = self.k_up.weight.view(h, d_nope, self.d_c)
            w_uv = self.v_up.weight.view(h, self.d_v, self.d_c)
            q_latent = torch.einsum(
                "bhtd,hdc->bhtc", q_nope, w_uk
            )  # W^{UK} absorbed into the query
            # Every head reads the same latent, so the heads can be stacked into the query rows:
            # one (B, H*T, d) x (B, d, T_k) product per term, and the latent is never copied per head.
            scores = q_latent.reshape(b, h * t, self.d_c) @ c.transpose(1, 2)
            scores = scores + q_rope.reshape(b, h * t, d_rope) @ k_rope.squeeze(1).transpose(1, 2)
            scores = scores.view(b, h, t, t_k) * (d_nope + d_rope) ** -0.5
            weights = scores.masked_fill(~mask, float("-inf")).softmax(dim=-1)
            context = (weights.reshape(b, h * t, t_k) @ c).view(b, h, t, self.d_c)
            out = torch.einsum("bhtc,hvc->bhtv", context, w_uv)  # W^{UV} applied after the sum
        return self.o_proj(out.transpose(1, 2).reshape(b, t, h * self.d_v))
