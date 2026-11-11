"""Mixture of experts: a router, top-k selection, load balancing, capacity, shared experts.

A MoE layer replaces the feed-forward sub-layer of a Transformer block by E experts (each one an
ordinary feed-forward network) and a small router that sends every token to k of them. The parameter
count grows with E; the work per token only with k.

Course page: https://dmxsf.com/wiki/architecture/mixture-of-experts/
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from dmxsf.architecture.layers import SwiGLU

GATES = ("mixtral", "switch", "deepseek")


@dataclass
class Routing:
    """What the router decided for T tokens."""

    experts: torch.Tensor  # (T, k) chosen expert of each assignment
    gates: torch.Tensor  # (T, k) weight of each assignment in the token's output
    keep: torch.Tensor  # (T, k) False where the expert was already full (capacity)
    probs: torch.Tensor  # (T, E) softmax of the router logits, used by the balancing loss
    load: torch.Tensor  # (E,) assignments per expert, before dropping


def capacity(n_tokens: int, top_k: int, n_experts: int, factor: float) -> int:
    """Switch Transformers' expert capacity, ceil(factor * assignments per expert on average)."""
    return math.ceil(factor * n_tokens * top_k / n_experts)


def limit_to_groups(
    scores: torch.Tensor, n_group: int, topk_group: int, top_k: int
) -> torch.Tensor:
    """Node-limited routing (DeepSeek-V3): only the best ``topk_group`` of ``n_group`` expert groups
    stay selectable; every other expert's score becomes -inf.

    The experts of one group live on one node, so a token talks to at most ``topk_group`` nodes. A
    group's score is the sum of its ``top_k // topk_group`` highest scores.
    """
    n_tokens, n_experts = scores.shape
    per_group = scores.view(n_tokens, n_group, n_experts // n_group)
    group_score = per_group.topk(max(1, top_k // topk_group), dim=-1).values.sum(dim=-1)
    best = group_score.topk(topk_group, dim=-1).indices
    allowed = torch.zeros(n_tokens, n_group, dtype=torch.bool, device=scores.device)
    allowed.scatter_(1, best, True)
    return scores.masked_fill(~allowed.repeat_interleave(n_experts // n_group, dim=1), -math.inf)


def route(
    logits: torch.Tensor,
    top_k: int,
    gate: str = "mixtral",
    bias: torch.Tensor | None = None,
    cap: int | None = None,
    route_scale: float = 1.0,
    n_group: int = 1,
    topk_group: int | None = None,
) -> Routing:
    """Choose top_k experts per token from router logits (T, E).

    gate = "mixtral":  softmax over the k selected logits.
    gate = "switch":   the softmax probability over all E experts (k = 1 in the paper).
    gate = "deepseek": sigmoid scores; the selected scores are normalised to sum to 1 and multiplied
                       by route_scale. ``bias`` is added to the scores only to decide *who* is
                       selected; it never changes a gate weight (auxiliary-loss-free balancing).
    cap: an expert takes at most this many assignments, in token order; the rest are dropped.
    n_group, topk_group: node-limited routing, see ``limit_to_groups``.
    """
    if gate not in GATES:
        raise ValueError(f"gate must be one of {GATES}")
    n_experts = logits.shape[1]
    probs = logits.softmax(dim=-1)
    scores = {"mixtral": logits, "switch": probs, "deepseek": logits.sigmoid()}[gate]
    chosen = scores + bias if bias is not None else scores
    if topk_group is not None:
        chosen = limit_to_groups(chosen, n_group, topk_group, top_k)
    experts = chosen.topk(top_k, dim=-1).indices
    picked = scores.gather(1, experts)
    if gate == "mixtral":
        gates = picked.softmax(dim=-1)
    elif gate == "switch":
        gates = picked
    else:
        gates = picked / picked.sum(dim=-1, keepdim=True) * route_scale
    flat = experts.reshape(-1)  # token-major: all of token 0's assignments, then token 1's, ...
    load = torch.bincount(flat, minlength=n_experts)
    if cap is None:
        keep = torch.ones_like(experts, dtype=torch.bool)
    else:
        one_hot = F.one_hot(flat, n_experts)
        rank = (one_hot.cumsum(dim=0) - 1).gather(1, flat[:, None]).squeeze(1)  # arrival order
        keep = (rank < cap).view_as(experts)
    return Routing(experts, gates, keep, probs, load)


def balance_loss(r: Routing) -> torch.Tensor:
    """Switch Transformers' auxiliary loss N * sum_i f_i P_i (equal to 1 when perfectly balanced).

    f_i is the share of all T*k assignments that went to expert i (from an argmax, so it carries no
    gradient); P_i is the router's mean probability for expert i (differentiable). The gradient
    reaches the router only through P_i.
    """
    n_experts = r.probs.shape[1]
    f = r.load.to(r.probs.dtype) / r.load.sum()
    return n_experts * (f * r.probs.mean(dim=0)).sum()


class MoELayer(nn.Module):
    """A feed-forward sub-layer made of ``n_experts`` SwiGLU experts, ``top_k`` of them per token.

    ``n_shared`` extra experts process every token without routing (DeepSeekMoE). After each forward
    pass the layer keeps ``load`` (assignments per expert), ``dropped`` (assignments over capacity)
    and ``aux_loss`` (the balancing loss, still attached to the graph) for monitoring and training.
    """

    def __init__(
        self,
        d_model: int,
        d_expert: int,
        n_experts: int,
        top_k: int = 2,
        n_shared: int = 0,
        gate: str = "mixtral",
        capacity_factor: float | None = None,
        route_scale: float = 1.0,
        n_group: int = 1,
        topk_group: int | None = None,
    ):
        super().__init__()
        if not 1 <= top_k <= n_experts:
            raise ValueError("need 1 <= top_k <= n_experts")
        if topk_group is not None and (
            n_experts % n_group or not top_k <= topk_group * (n_experts // n_group)
        ):
            raise ValueError(
                "need n_group | n_experts and room for top_k experts in topk_group groups"
            )
        self.n_experts, self.top_k, self.gate = n_experts, top_k, gate
        self.capacity_factor, self.route_scale = capacity_factor, route_scale
        self.n_group, self.topk_group = n_group, topk_group
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(SwiGLU(d_model, d_expert) for _ in range(n_experts))
        self.shared = nn.ModuleList(SwiGLU(d_model, d_expert) for _ in range(n_shared))
        self.register_buffer("route_bias", torch.zeros(n_experts), persistent=True)
        self.load = torch.zeros(n_experts, dtype=torch.long)
        self.dropped = 0
        self.aux_loss = torch.zeros(())

    def _route(self, flat: torch.Tensor) -> Routing:
        cap = None
        if self.capacity_factor is not None:
            cap = capacity(flat.shape[0], self.top_k, self.n_experts, self.capacity_factor)
        bias = self.route_bias if self.gate == "deepseek" else None
        return route(
            self.router(flat),
            self.top_k,
            self.gate,
            bias,
            cap,
            self.route_scale,
            self.n_group,
            self.topk_group,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        r = self._route(flat)
        out = torch.zeros_like(flat)
        for i, expert in enumerate(self.experts):  # only the tokens routed to expert i are computed
            token, slot = ((r.experts == i) & r.keep).nonzero(as_tuple=True)
            if token.numel():
                out.index_add_(0, token, r.gates[token, slot, None] * expert(flat[token]))
        for expert in self.shared:
            out = out + expert(flat)
        self.load, self.dropped = r.load.detach(), int((~r.keep).sum())
        self.aux_loss = balance_loss(r)
        return out.view(shape)

    def dense_reference(self, x: torch.Tensor) -> torch.Tensor:
        """Every expert on every token, times a sparse gate matrix: same output, E/k times the work."""
        flat = x.reshape(-1, x.shape[-1])
        r = self._route(flat)
        gates = torch.zeros(flat.shape[0], self.n_experts, dtype=flat.dtype, device=flat.device)
        gates.scatter_(1, r.experts, r.gates * r.keep)
        out = sum(gates[:, i : i + 1] * expert(flat) for i, expert in enumerate(self.experts))
        for expert in self.shared:
            out = out + expert(flat)
        return out.view(x.shape)

    @torch.no_grad()
    def update_bias(self, gamma: float) -> None:
        """DeepSeek-V3's balancing step: overloaded experts get b_i - gamma, idle ones b_i + gamma."""
        self.route_bias += gamma * torch.sign(self.load.float().mean() - self.load.float())


def moeify(
    model: nn.Module,
    n_experts: int,
    top_k: int,
    d_expert: int | None = None,
    n_shared: int = 0,
    first_dense: int = 0,
    **kwargs,
) -> nn.Module:
    """Replace the feed-forward network of a GPT's blocks (all but the first ``first_dense``) by MoE.

    ``d_expert`` defaults to the dense width, so the model keeps ``top_k`` times its old activated
    feed-forward parameters. The new layers are initialised like the rest of the model.
    """
    cfg = model.cfg
    for i, block in enumerate(model.blocks):
        if i >= first_dense:
            block.mlp = MoELayer(
                cfg.d_model, d_expert or cfg.d_ff, n_experts, top_k, n_shared, **kwargs
            )
            block.mlp.apply(model._init_weights)
    return model


def moe_layers(model: nn.Module) -> list[MoELayer]:
    return [m for m in model.modules() if isinstance(m, MoELayer)]


def aux_loss(model: nn.Module) -> torch.Tensor:
    """Mean balancing loss of the MoE layers, from the last forward pass."""
    layers = moe_layers(model)
    return sum(m.aux_loss for m in layers) / len(layers)


def active_params(model: nn.Module) -> int:
    """Parameters used per token: everything except the (n_experts - top_k) idle experts per layer."""
    idle = 0
    for layer in moe_layers(model):
        per_expert = sum(p.numel() for p in layer.experts[0].parameters())
        idle += (layer.n_experts - layer.top_k) * per_expert
    return sum(p.numel() for p in model.parameters()) - idle
