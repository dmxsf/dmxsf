"""SGD, Adam and AdamW from scratch, plus learning-rate schedules and gradient clipping
(course page: optimizers and training stability).

The optimizers subclass :class:`torch.optim.Optimizer`, so they drop into any training loop,
and the tests check them step by step against PyTorch's own implementations.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch.optim import Optimizer


class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum, Nesterov and L2 weight decay."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-2,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if group["weight_decay"]:
                    g = g + group["weight_decay"] * p  # L2 penalty: part of the gradient
                if group["momentum"]:
                    state = self.state[p]
                    if "buf" not in state:
                        state["buf"] = g.clone()
                    else:
                        state["buf"].mul_(group["momentum"]).add_(g)
                    g = g + group["momentum"] * state["buf"] if group["nesterov"] else state["buf"]
                p.add_(g, alpha=-group["lr"])


class Adam(Optimizer):
    """Adam: a running mean of the gradient divided by the root of a running mean of its square.

    With ``decoupled=False`` weight decay is an L2 penalty added to the gradient, so it gets
    rescaled by the adaptive denominator. With ``decoupled=True`` (see :class:`AdamW`) the weights
    are shrunk directly, by ``lr * weight_decay``, independent of the gradient statistics.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        decoupled: bool = False,
    ) -> None:
        super().__init__(
            params,
            dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, decoupled=decoupled),
        )

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not state:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["t"] += 1
                t = state["t"]

                if group["weight_decay"]:
                    if group["decoupled"]:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    else:
                        g = g + group["weight_decay"] * p
                state["m"].mul_(beta1).add_(g, alpha=1 - beta1)
                state["v"].mul_(beta2).addcmul_(g, g, value=1 - beta2)

                m_hat = state["m"] / (1 - beta1**t)  # undo the bias of starting at zero
                v_hat = state["v"] / (1 - beta2**t)
                p.addcdiv_(m_hat, v_hat.sqrt().add_(group["eps"]), value=-group["lr"])


class AdamW(Adam):
    """Adam with decoupled weight decay (Loshchilov and Hutter, 2019)."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        super().__init__(params, lr, betas, eps, weight_decay, decoupled=True)


def clip_grad_norm(params: Iterable[torch.nn.Parameter], max_norm: float) -> float:
    """Scale all gradients so that their global L2 norm is at most ``max_norm``.

    Returns the norm *before* clipping, the number worth logging: a sudden jump in it
    usually comes before a loss spike.
    """
    grads = [p.grad for p in params if p.grad is not None]
    total = torch.sqrt(sum(g.float().pow(2).sum() for g in grads))
    scale = min(1.0, max_norm / (float(total) + 1e-6))
    if scale < 1.0:
        for g in grads:
            g.mul_(scale)
    return float(total)


SCHEDULES = ("cosine", "linear", "wsd", "invsqrt")


def lr_factor(
    schedule: str, step: int, total: int, warmup: int = 0, floor: float = 0.1, decay: float = 0.2
) -> float:
    """Learning rate at ``step`` (0-based) as a fraction of the peak.

    All schedules start with a linear warmup, ``(step + 1) / warmup``. After it:

    * ``cosine``: half a cosine from 1 down to ``floor`` at the last step.
    * ``linear``: a straight line from 1 down to ``floor``.
    * ``wsd`` (warmup-stable-decay): stay at 1, then fall linearly to ``floor`` over the last
      ``decay`` fraction of the steps. The total need not be known until the decay starts.
    * ``invsqrt``: ``sqrt(warmup / (step + 1))``; it has no end value.
    """
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    if schedule in ("cosine", "linear"):
        progress = min(1.0, (step - warmup) / max(1, total - 1 - warmup))
        if schedule == "cosine":
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))
        return 1 - (1 - floor) * progress
    if schedule == "wsd":
        start = total * (1 - decay)
        if step < start:
            return 1.0
        return 1 - (1 - floor) * min(1.0, (step - start) / max(1, total - 1 - start))
    if schedule == "invsqrt":
        return math.sqrt(max(warmup, 1) / (step + 1))
    raise ValueError(f"unknown schedule {schedule!r}; choose from {SCHEDULES}")
