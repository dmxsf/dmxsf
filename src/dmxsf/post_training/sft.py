"""Supervised fine-tuning, mechanically: loss masks, batches, packing, and the SFT loop.

An SFT example is two equally long columns of tokens, ``ids`` and ``labels``; positions the model
should not learn from carry the label -100, which cross-entropy skips. Everything the sft-data page
says about "what the loss sees" is a few lines of this file.

Course page: https://dmxsf.com/wiki/post-training/sft-data/
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dmxsf.architecture.position import rope_cos_sin

IGNORE = -100


def build_example(
    prompt: Sequence[int], answer: Sequence[int], mask_prompt: bool = True
) -> tuple[list[int], list[int]]:
    """``ids`` = prompt + answer; ``labels`` = the same, with the prompt replaced by -100."""
    ids = list(prompt) + list(answer)
    labels = ([IGNORE] * len(prompt) if mask_prompt else list(prompt)) + list(answer)
    return ids, labels


def collate(examples: Sequence[tuple[list[int], list[int]]], pad_id: int = 0):
    """Right-pad to the longest example and shift: returns inputs ``x`` and targets ``y`` (B, L-1).

    Padding is on the right, so with a causal model it cannot influence any real position; its
    labels are -100, so it cannot influence the loss either.
    """
    longest = max(len(ids) for ids, _ in examples)
    x = torch.full((len(examples), longest), pad_id, dtype=torch.long)
    y = torch.full((len(examples), longest), IGNORE, dtype=torch.long)
    for i, (ids, labels) in enumerate(examples):
        x[i, : len(ids)] = torch.tensor(ids)
        y[i, : len(labels)] = torch.tensor(labels)
    return x[:, :-1], y[:, 1:]


def sft_loss(model: nn.Module, batch, per_sequence: bool = False) -> torch.Tensor:
    """Cross-entropy over the supervised positions only.

    ``per_sequence=False`` averages over all supervised tokens of the batch (so long answers weigh
    more); ``per_sequence=True`` averages within each sequence first (every example weighs the
    same). They agree when all answers have the same length.
    """
    x, y = batch
    logits = model(x)
    if not per_sequence:
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=IGNORE
        )
    losses = F.cross_entropy(logits.transpose(1, 2), y, ignore_index=IGNORE, reduction="none")
    counts = (y != IGNORE).sum(dim=1).clamp(min=1)
    return (losses.sum(dim=1) / counts).mean()


def sample_batch(
    examples: Sequence[tuple[list[int], list[int]]], step: int, batch_size: int, seed: int
):
    """Batch ``step``: a pure function of ``(seed, step)``, rows drawn with replacement."""
    rng = np.random.default_rng((seed, step))
    picks = rng.integers(0, len(examples), size=batch_size)
    return collate([examples[i] for i in picks])


# --- sequence packing -----------------------------------------------------------------------


def forward_packed(model, ids: torch.Tensor, lengths: Sequence[int]) -> torch.Tensor:
    """Logits for several documents concatenated into one row, without letting them see each other.

    ``ids``: (1, sum(lengths)). Two things must change compared with a plain forward pass: a
    block-diagonal attention mask (a token attends to its own document only) and positions that
    restart at 0 in every document. Do both and the result is *exactly* what separate forward
    passes give; skip either and the documents contaminate each other.
    Supports learned and RoPE position encodings.
    """
    cfg = model.cfg
    if cfg.pos not in ("learned", "rope"):
        raise ValueError("forward_packed supports learned and rope positions")
    positions = torch.cat([torch.arange(n) for n in lengths])
    doc = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(list(lengths)))
    bias = torch.zeros(len(doc), len(doc)).masked_fill(doc[:, None] != doc[None, :], float("-inf"))
    x = model.tok_emb(ids)
    rope = None
    if cfg.pos == "learned":
        x = x + model.pos_emb(positions)
    else:
        cos, sin = rope_cos_sin(positions, model.inv_freq, model.rope_attention_factor)
        rope = (cos.to(x.dtype), sin.to(x.dtype))
    for block in model.blocks:
        x = block(x, rope=rope, bias=bias.to(x.dtype))
    return model.lm_head(model.final_norm(x))


def padding_share(lens: Sequence[int], batch_size: int, grouped: bool, seed: int = 0) -> float:
    """Share of batch positions that are padding when every batch is padded to its longest row.

    ``grouped=True`` sorts by length inside windows of 50 batches before cutting batches
    (the page's 'group by length').
    """
    idx = list(range(len(lens)))
    random.Random(seed).shuffle(idx)
    if grouped:
        w = batch_size * 50
        idx = [
            i for s in range(0, len(idx), w) for i in sorted(idx[s : s + w], key=lens.__getitem__)
        ]
    batches = [idx[s : s + batch_size] for s in range(0, len(idx) - batch_size + 1, batch_size)]
    real = sum(lens[i] for b in batches for i in b)
    total = sum(max(lens[i] for i in b) * batch_size for b in batches)
    return 1 - real / total


def packing_share(lens: Sequence[int], block: int) -> float:
    """Share of positions wasted when documents are packed first-fit-decreasing into ``block``s."""
    bins: list[int] = []
    for n in sorted(lens, reverse=True):
        for i, used in enumerate(bins):
            if used + n <= block:
                bins[i] += n
                break
        else:
            bins.append(n)
    return 1 - sum(lens) / (len(bins) * block)


def finetune(
    model: nn.Module,
    batch_fn: Callable[[int], tuple[torch.Tensor, torch.Tensor]],
    steps: int,
    lr: float,
    *,
    loss_fn: Callable = sft_loss,
    params=None,
    warmup: int = 10,
    weight_decay: float = 0.0,
):
    """Plain fine-tuning loop (AdamW, linear warmup, cosine to 10%); returns the per-step losses.

    ``params`` restricts what is trained (LoRA: only the adapters); default: everything that
    ``requires_grad``.
    """
    params = [p for p in (params or model.parameters()) if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.99), weight_decay=weight_decay)
    losses = []
    model.train()
    for step in range(steps):
        warm = min(1.0, (step + 1) / max(1, warmup))
        cosine = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps))
        for g in opt.param_groups:
            g["lr"] = lr * warm * cosine
        loss = loss_fn(model, batch_fn(step))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(loss.item())
    return losses
