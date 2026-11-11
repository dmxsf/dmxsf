"""Scaled dot-product attention, multi-head / grouped-query attention and the causal mask.

Course pages: https://dmxsf.com/wiki/architecture/attention-basics/
              https://dmxsf.com/wiki/architecture/efficient-attention/
"""

import torch
import torch.nn.functional as F
from torch import nn

from dmxsf.architecture.position import apply_rope


def causal_mask(t_q: int, t_k: int, device=None, window: int | None = None) -> torch.Tensor:
    """Boolean (t_q, t_k), True where query i may look at key j.

    The last query is aligned with the last key, so t_k > t_q works when a KV cache holds earlier keys.
    With a window, a query sees itself and the window - 1 keys before it (sliding-window attention).
    """
    i = torch.arange(t_q, device=device)[:, None] + (t_k - t_q)
    j = torch.arange(t_k, device=device)[None, :]
    mask = j <= i
    if window is not None:
        mask &= j > i - window
    return mask


def attention(q, k, v, mask=None, bias=None, scale=None):
    """softmax(q k^T / sqrt(d) + bias, masked) v, written out. Returns (output, weights).

    q: (..., T_q, d), k and v: (..., T_k, d). mask is boolean (True = allowed).
    """
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    scores = (q @ k.transpose(-2, -1)) * scale
    if bias is not None:
        scores = scores + bias
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = scores.softmax(dim=-1)
    return weights @ v, weights


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, n_kv, T, d) -> (B, n_kv * n_rep, T, d): each key/value head serves n_rep query heads."""
    return x if n_rep == 1 else x.repeat_interleave(n_rep, dim=1)


class Attention(nn.Module):
    """Causal self-attention with n_heads query heads and n_kv_heads key/value heads.

    n_kv_heads == n_heads is multi-head attention (MHA), 1 is multi-query (MQA), anything in
    between is grouped-query attention (GQA). The cache stores the n_kv_heads heads, which is the saving.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        head_dim: int | None = None,
        qkv_bias: bool = False,
        out_bias: bool = False,
        window: int | None = None,
    ):
        super().__init__()
        n_kv_heads = n_kv_heads or n_heads
        if n_heads % n_kv_heads:
            raise ValueError("n_heads must be a multiple of n_kv_heads")
        head_dim = head_dim or d_model // n_heads
        self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, head_dim
        self.window = (
            window  # None: attend to everything before; else only the last `window` tokens
        )
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=out_bias)

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
        bias: torch.Tensor | None = None,
        cache=None,
        layer_idx: int = 0,
        impl: str = "sdpa",
        return_weights: bool = False,
    ):
        """x: (B, T, d_model). rope = (cos, sin) for the positions of these T tokens.
        cache needs .update(layer_idx, k, v) -> (all_k, all_v). impl is 'sdpa' or 'manual'."""
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if rope is not None:  # position goes into q and k only, never into v
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        if cache is not None:
            k, v = cache.update(layer_idx, k, v)
        t_k = k.shape[2]
        n_rep = self.n_heads // self.n_kv_heads
        k, v = repeat_kv(k, n_rep), repeat_kv(v, n_rep)
        weights = None
        if impl == "manual" or return_weights:
            out, weights = attention(q, k, v, causal_mask(t, t_k, x.device, self.window), bias)
        elif bias is not None or self.window is not None:
            mask = causal_mask(t, t_k, x.device, self.window)
            if bias is not None:
                mask = bias.masked_fill(~mask, float("-inf"))
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        elif t == 1:  # one new token may look at everything cached
            out = F.scaled_dot_product_attention(q, k, v)
        elif t == t_k:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:  # a chunk of new tokens after a cached prefix
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask(t, t_k, x.device))
        out = self.o_proj(out.transpose(1, 2).reshape(b, t, self.n_heads * self.head_dim))
        return (out, weights) if return_weights else out


def tiled_attention(q, k, v, block_q: int = 128, block_k: int = 128, causal: bool = True):
    """Exact attention computed one block at a time with an online softmax.

    This is FlashAttention's algorithm written in PyTorch: it never builds the (T_q, T_k) score
    matrix, only (block_q, block_k) pieces, and keeps three running quantities per query row:
    the largest score seen so far m, the softmax denominator l (relative to m) and the unnormalised
    output acc. Gradients flow through ordinary autograd; m is treated as a constant, as it is in
    FlashAttention, because the result does not depend on it. q: (..., T_q, d), k and v: (..., T_k, d).
    """
    t_q, t_k = q.shape[-2], k.shape[-2]
    scale = q.shape[-1] ** -0.5
    offset = t_k - t_q  # query i sits at key position i + offset, as in causal_mask
    out_blocks = []
    for i in range(0, t_q, block_q):
        qi = q[..., i : i + block_q, :]
        pos_q = torch.arange(i, i + qi.shape[-2], device=q.device)[:, None] + offset
        m = torch.full(qi.shape[:-1], float("-inf"), dtype=q.dtype, device=q.device)
        denom = torch.zeros(qi.shape[:-1], dtype=q.dtype, device=q.device)
        acc = torch.zeros_like(qi)
        for j in range(0, t_k, block_k):
            if causal and j > i + qi.shape[-2] - 1 + offset:
                break  # every key in this block lies in the future of every query in qi
            s = (qi @ k[..., j : j + block_k, :].transpose(-2, -1)) * scale
            if causal:
                pos_k = torch.arange(j, j + s.shape[-1], device=q.device)[None, :]
                s = s.masked_fill(pos_k > pos_q, float("-inf"))
            m_new = torch.maximum(m, s.detach().amax(dim=-1))
            correction = torch.exp(m - m_new)  # old running sums, re-expressed relative to m_new
            p = torch.exp(s - m_new[..., None])
            denom = denom * correction + p.sum(dim=-1)
            acc = acc * correction[..., None] + p @ v[..., j : j + block_k, :]
            m = m_new
        out_blocks.append(acc / denom[..., None])
    return torch.cat(out_blocks, dim=-2)


def pool_kv_heads(attn: Attention, n_kv_heads: int) -> Attention:
    """Turn an MHA / GQA layer into one with fewer KV heads by averaging the heads in each group.

    This is how the GQA paper converts a multi-head checkpoint: the keys and values of the heads that
    will share a cache entry are mean-pooled; queries and the output projection are copied unchanged.
    """
    if attn.n_kv_heads % n_kv_heads:
        raise ValueError("the new number of KV heads must divide the old one")
    pooled = Attention(
        attn.q_proj.in_features,
        attn.n_heads,
        n_kv_heads,
        attn.head_dim,
        qkv_bias=attn.k_proj.bias is not None,
        out_bias=attn.o_proj.bias is not None,
        window=attn.window,
    )
    group = attn.n_kv_heads // n_kv_heads
    d, d_in = attn.head_dim, attn.q_proj.in_features
    with torch.no_grad():
        for name in ("q_proj", "o_proj"):
            getattr(pooled, name).load_state_dict(getattr(attn, name).state_dict())
        for name in ("k_proj", "v_proj"):
            old, new = getattr(attn, name), getattr(pooled, name)
            new.weight.copy_(
                old.weight.view(n_kv_heads, group, d, d_in).mean(dim=1).reshape(-1, d_in)
            )
            if old.bias is not None:
                new.bias.copy_(old.bias.view(n_kv_heads, group, d).mean(dim=1).reshape(-1))
    return pooled
