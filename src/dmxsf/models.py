"""Small pretrained byte-level models: trained once, cached, shared by the labs after pretraining.

Post-training, inference and application labs all start from "a language model that has already read
some text". Training one takes a minute on a laptop CPU, so it happens the first time a lab asks for
it and the weights are cached in ``~/.cache/dmxsf`` (set ``DMXSF_CACHE`` to move it). The data is
TinyShakespeare (``ByteCorpus.shakespeare()``), the same corpus as the foundations labs.

``quick=True`` trains for a few dozen steps and caches nothing: enough for a smoke test to see that
the plumbing works, not enough for the model to be any good.

Course pages: https://dmxsf.com/wiki/pretraining/training-pipeline/
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from dmxsf.architecture.configs import small_gpt
from dmxsf.architecture.gpt import GPT
from dmxsf.data import ByteCorpus, get_batch
from dmxsf.pretraining.train import TrainConfig, bits_per_byte, make_optimizer, train

BLOCK = 128
# name -> (model overrides, steps, batch size). "draft" is the small model of speculative decoding.
SPECS = {
    "base": (dict(), 2400, 32),
    "draft": (dict(n_layer=2, n_head=2, d_model=64), 2400, 32),
}


def cache_dir() -> Path:
    return Path(os.environ.get("DMXSF_CACHE", "~/.cache/dmxsf")).expanduser()


def build(name: str = "base") -> GPT:
    """An untrained model with the architecture of ``pretrained(name)``."""
    return GPT(small_gpt(block_size=BLOCK, **SPECS[name][0]))


def pretrained(name: str = "base", *, quick: bool = False, seed: int = 0) -> GPT:
    """The cached pretrained model ``name`` ("base" or "draft"); trains it on first use."""
    _, steps, batch = SPECS[name]
    steps = 60 if quick else steps
    path = cache_dir() / f"{name}-seed{seed}-{steps}steps.pt"
    torch.manual_seed(seed)
    model = build(name)
    if not quick and path.exists():
        model.load_state_dict(torch.load(path, weights_only=True))
        return model
    corpus = ByteCorpus.shakespeare()
    cfg = TrainConfig(steps=steps, peak_lr=3e-3, warmup=min(50, steps // 4))
    opt = make_optimizer(model, cfg)
    train(model, opt, lambda s: get_batch(corpus.train, s, batch, BLOCK, seed), cfg)
    if not quick:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
    return model


def shakespeare_bpb(model: GPT, windows: int = 128) -> float:
    """Bits per byte on the held-out last 5% of TinyShakespeare: the 'did it forget' probe."""
    return bits_per_byte(model, ByteCorpus.shakespeare().val, BLOCK, windows)
