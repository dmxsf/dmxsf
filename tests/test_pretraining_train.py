import math
from functools import partial

import numpy as np
import pytest
import torch

from dmxsf.architecture.configs import small_gpt
from dmxsf.architecture.gpt import GPT
from dmxsf.data import get_batch, get_mixed_batch
from dmxsf.foundations.optim import lr_factor
from dmxsf.pretraining.train import (
    TrainConfig,
    bits_per_byte,
    lm_loss,
    load_checkpoint,
    make_optimizer,
    save_checkpoint,
    train,
)

DATA = np.random.default_rng(0).integers(0, 256, size=20_000, dtype=np.uint8)
CFG = TrainConfig(steps=24, peak_lr=3e-3, warmup=4)
BATCH = partial(get_batch, DATA, batch_size=4, block_size=16, seed=0)


def tiny(seed=0):
    torch.manual_seed(seed)
    model = GPT(small_gpt(n_layer=1, n_head=2, d_model=32, block_size=16))
    return model, make_optimizer(model, CFG)


def weights_equal(a, b):
    return all(
        torch.equal(x, y)
        for x, y in zip(a.state_dict().values(), b.state_dict().values(), strict=True)
    )


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)  # a fixed reduction order, so bitwise comparisons are meaningful
    yield
    torch.set_num_threads(previous)


def test_batches_depend_only_on_seed_and_step():
    x1, y1 = get_batch(DATA, 7, 4, 16, seed=3)
    x2, y2 = get_batch(DATA, 7, 4, 16, seed=3)
    assert torch.equal(x1, x2) and torch.equal(y1, y2)
    assert not torch.equal(x1, get_batch(DATA, 8, 4, 16, seed=3)[0])
    assert not torch.equal(x1, get_batch(DATA, 7, 4, 16, seed=4)[0])
    assert torch.equal(x1[:, 1:], y1[:, :-1])  # targets are the inputs shifted by one


def test_mixed_batches_follow_the_mix():
    sources = {"a": np.zeros(5000, np.uint8), "b": np.ones(5000, np.uint8)}
    share_b = [
        get_mixed_batch(sources, {"a": 0.25, "b": 0.75}, s, 64, 8)[0][:, 0].float().mean().item()
        for s in range(40)
    ]
    assert abs(np.mean(share_b) - 0.75) < 0.03


def test_only_matrices_are_weight_decayed():
    model, opt = tiny()
    decayed, plain = opt.param_groups
    assert decayed["weight_decay"] == CFG.weight_decay and plain["weight_decay"] == 0.0
    assert all(p.ndim >= 2 for p in decayed["params"]) and all(p.ndim < 2 for p in plain["params"])
    n = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert n == model.num_params()  # nothing missing, and the tied embedding is counted once


def test_the_schedule_matches_the_course_page():
    # page: peak 6e-3, 50 warm-up steps, cosine to 10% over 2000 steps; at step 1000 it is 3.4e-3
    assert math.isclose(6e-3 * lr_factor("cosine", 1000, 2000, 50), 3.4e-3, rel_tol=0.01)
    assert math.isclose(6e-3 * lr_factor("cosine", 1999, 2000, 50), 6e-4, rel_tol=1e-9)
    assert math.isclose(6e-3 * lr_factor("cosine", 49, 2000, 50), 6e-3)  # warm-up ends at the peak


def test_an_untrained_model_scores_eight_bits_per_byte():
    model, _ = tiny()
    assert abs(bits_per_byte(model, DATA, block_size=16, windows=32) - 8) < 0.1


def test_the_logged_gradient_norm_is_the_one_before_clipping():
    model, opt = tiny()
    tight = TrainConfig(steps=4, peak_lr=1e-3, warmup=1, clip=1e-3)
    loss = lm_loss(model, BATCH(0))
    loss.backward()
    expected = torch.sqrt(sum(p.grad.pow(2).sum() for p in model.parameters())).item()
    model.zero_grad()
    row = train(model, opt, BATCH, tight, 0, 1)[0]
    assert row["grad_norm"] > 1e-2 and math.isclose(row["grad_norm"], expected, rel_tol=1e-4)


def test_train_can_overfit_a_batch():
    model, opt = tiny()
    cfg = TrainConfig(steps=150, peak_lr=5e-3, warmup=5)
    rows = train(model, opt, lambda step: BATCH(0), cfg)
    assert rows[0]["loss"] > 5 and rows[-1]["loss"] < 0.5


def test_resuming_from_a_complete_checkpoint_is_bitwise_identical(tmp_path):
    straight, opt = tiny()
    train(straight, opt, BATCH, CFG)
    first, opt = tiny()
    train(first, opt, BATCH, CFG, 0, 12)
    save_checkpoint(tmp_path / "mid.pt", first, opt, 12)
    resumed, opt2 = tiny()
    start = load_checkpoint(tmp_path / "mid.pt", resumed, opt2)
    train(resumed, opt2, BATCH, CFG, start)
    assert start == 12 and weights_equal(straight, resumed)


@pytest.mark.parametrize("what", ["optimizer", "step", "schedule"])
def test_resuming_with_missing_state_silently_changes_the_run(tmp_path, what):
    straight, opt = tiny()
    train(straight, opt, BATCH, CFG)
    first, opt = tiny()
    train(first, opt, BATCH, CFG, 0, 12)
    save_checkpoint(tmp_path / "mid.pt", first, opt, 12)
    resumed, opt2 = tiny()
    load_checkpoint(tmp_path / "mid.pt", resumed, opt2 if what != "optimizer" else None)
    if what == "step":
        rows = train(resumed, opt2, BATCH, CFG, 0, 12)  # data and schedule restart at step 0
    elif what == "schedule":
        rows = train(resumed, opt2, BATCH, CFG, 12, 24, lr_shift=-12)
    else:
        rows = train(resumed, opt2, BATCH, CFG, 12, 24)
    assert all(math.isfinite(r["loss"]) for r in rows)  # no error, loss looks fine ...
    assert not weights_equal(straight, resumed)  # ... but it is a different run
