"""A small, explicit training loop: schedule, clipping, monitoring, checkpoints.

Everything the course's pretraining pages talk about is a visible line here: AdamW with weight decay
on the matrices only, linear warmup then cosine decay, gradient clipping that reports the norm
*before* clipping, a log row per step, and a checkpoint that holds the three things a resumed run
needs (weights, optimizer state, step number).

Course pages: https://dmxsf.com/wiki/pretraining/training-pipeline/
              https://dmxsf.com/wiki/foundations/optimizers-and-training-stability/
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dmxsf.data import ByteCorpus
from dmxsf.foundations.optim import lr_factor


@dataclass(frozen=True)
class TrainConfig:
    steps: int  # length of the whole schedule; a run may cover only part of it
    peak_lr: float = 3e-3
    warmup: int = 50
    floor: float = 0.1  # the schedule ends at floor * peak_lr
    schedule: str = "cosine"
    clip: float | None = 1.0  # None: do not clip (the norm is still measured)
    weight_decay: float = 0.1  # applied to matrices (ndim >= 2) only
    betas: tuple[float, float] = (0.9, 0.95)


def lm_loss(model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Mean next-token cross-entropy (nats) of a ``(x, y)`` batch."""
    x, y = batch
    logits = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


def make_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.AdamW:
    """AdamW; biases and norm gains are not decayed, only matrices (and embedding tables)."""
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.ndim >= 2]
    rest = [p for p in params if p.ndim < 2]
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}]
    if rest:
        groups.append({"params": rest, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=cfg.peak_lr, betas=cfg.betas)


def train(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    batch_fn: Callable[[int], Any],
    cfg: TrainConfig,
    start: int = 0,
    end: int | None = None,
    *,
    lr_shift: int = 0,
    loss_fn: Callable[[nn.Module, Any], torch.Tensor] = lm_loss,
    eval_fn: Callable[[nn.Module], dict[str, float]] | None = None,
    eval_every: int = 0,
) -> list[dict[str, float]]:
    """Run steps ``start .. end - 1`` of a ``cfg.steps``-long schedule and return one row per step.

    ``batch_fn(step)`` must depend only on the step number (see ``data.get_batch``); that is what
    lets a resumed run see the batches an uninterrupted run would have seen. ``lr_shift`` moves the
    learning-rate schedule relative to the data, which is how a wrongly resumed run is simulated.
    A row holds ``step`` (1-based, steps finished), ``loss`` (nats), ``grad_norm`` (before clipping),
    ``lr``, and whatever ``eval_fn`` returned every ``eval_every`` steps and at the end.
    """
    end = cfg.steps if end is None else end
    max_norm = float("inf") if cfg.clip is None else cfg.clip  # inf: measure, never rescale
    model.train()
    rows = []
    for step in range(start, end):
        lr = cfg.peak_lr * lr_factor(
            cfg.schedule, step + lr_shift, cfg.steps, cfg.warmup, cfg.floor
        )
        for group in opt.param_groups:
            group["lr"] = lr
        loss = loss_fn(model, batch_fn(step))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        opt.step()
        row = {"step": step + 1, "loss": loss.item(), "grad_norm": grad_norm.item(), "lr": lr}
        if eval_fn is not None and eval_every and ((step + 1) % eval_every == 0 or step + 1 == end):
            row.update(eval_fn(model))
        rows.append(row)
    return rows


def finetune_on_lines(
    model: nn.Module,
    lines: list[str],
    steps: int,
    seed: int = 0,
    line_rows: int = 24,
    clean_rows: int = 8,
    lr: float = 3e-3,
    block: int = 128,
) -> list[dict[str, float]]:
    """Continue training a byte-level model on batches that mix random Shakespeare windows
    (``clean_rows`` per batch, so it does not forget its pretraining) with windows cut from a stream
    made of ``lines`` (``line_rows`` per batch). Returns the per-step training rows.

    No weight decay: memorising or imitating a handful of exact strings is the point, not
    regularisation. Batches depend only on ``(seed, step)``."""
    corpus = ByteCorpus.shakespeare()
    rng = random.Random(seed)
    pieces = []
    for _ in range(20):  # the lines, reshuffled 20 times, glued into one stream
        order = lines[:]
        rng.shuffle(order)
        pieces.extend(order)
    stream = np.frombuffer("".join(pieces).encode(), dtype=np.uint8)

    def batch(step: int):
        rng = np.random.default_rng((seed, step))
        rows = []
        for _ in range(clean_rows):
            s = int(rng.integers(0, len(corpus.train) - block - 1))
            rows.append(corpus.train[s : s + block + 1])
        for _ in range(line_rows):
            s = int(rng.integers(0, len(stream) - block - 1))
            rows.append(stream[s : s + block + 1])
        t = torch.from_numpy(np.stack(rows).astype(np.int64))
        return t[:, :-1], t[:, 1:]

    cfg = TrainConfig(steps=steps, peak_lr=lr, warmup=min(20, steps // 5), weight_decay=0.0)
    return train(model, make_optimizer(model, cfg), batch, cfg)


@torch.no_grad()
def bits_per_byte(model: nn.Module, data: np.ndarray, block_size: int = 128, windows: int = 128):
    """Validation loss in bits per byte over evenly spaced windows of ``data``.

    Bits per byte does not depend on the vocabulary, so domains and models can be compared.
    """
    was_training = model.training
    model.eval()
    windows = max(1, min(windows, len(data) // block_size))
    stride = max(1, (len(data) - block_size - 1) // windows)
    rows = np.stack([data[j * stride : j * stride + block_size + 1] for j in range(windows)])
    rows = torch.from_numpy(rows.astype(np.int64))
    losses = [lm_loss(model, (r[:, :-1], r[:, 1:])) for r in rows.split(64)]
    model.train(was_training)
    sizes = [len(r) for r in rows.split(64)]
    mean = sum(loss.item() * n for loss, n in zip(losses, sizes, strict=True)) / sum(sizes)
    return mean / math.log(2)


def save_checkpoint(path, model: nn.Module, opt: torch.optim.Optimizer, step: int) -> None:
    """Weights, optimizer state (Adam's two moments) and the number of finished steps."""
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, path)


def load_checkpoint(path, model: nn.Module, opt: torch.optim.Optimizer | None = None) -> int:
    """Restore into ``model`` (and ``opt``); return the step to continue from."""
    state = torch.load(path, weights_only=True)
    model.load_state_dict(state["model"])
    if opt is not None:
        opt.load_state_dict(state["opt"])
    return state["step"]
