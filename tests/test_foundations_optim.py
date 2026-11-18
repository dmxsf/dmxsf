"""Optimizers and schedules. Schedule numbers come from the course page
https://dmxsf.com/wiki/foundations/optimizers-and-training-stability/."""

import copy

import pytest
import torch
from torch import nn

from dmxsf.foundations.optim import SGD, Adam, AdamW, clip_grad_norm, lr_factor


def _problem(seed: int = 0):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(5, 8), nn.Tanh(), nn.Linear(8, 2)).double()
    x = torch.randn(32, 5, dtype=torch.double)
    y = torch.randn(32, 2, dtype=torch.double)
    return model, x, y


def _train(model, opt, x, y, steps: int = 40) -> None:
    for _ in range(steps):
        loss = ((model(x) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()


def _same_weights(a: nn.Module, b: nn.Module) -> float:
    return max(
        (p - q).abs().max().item() for p, q in zip(a.parameters(), b.parameters(), strict=True)
    )


@pytest.mark.parametrize(
    ("ours", "theirs"),
    [
        (lambda p: SGD(p, lr=0.05), lambda p: torch.optim.SGD(p, lr=0.05)),
        (
            lambda p: SGD(p, lr=0.05, momentum=0.9),
            lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9),
        ),
        (
            lambda p: SGD(p, lr=0.05, momentum=0.9, nesterov=True, weight_decay=0.01),
            lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9, nesterov=True, weight_decay=0.01),
        ),
        (lambda p: Adam(p, lr=0.01), lambda p: torch.optim.Adam(p, lr=0.01)),
        (
            lambda p: Adam(p, lr=0.01, weight_decay=0.1),
            lambda p: torch.optim.Adam(p, lr=0.01, weight_decay=0.1),
        ),
        (
            lambda p: AdamW(p, lr=0.01, betas=(0.9, 0.95), weight_decay=0.1),
            lambda p: torch.optim.AdamW(p, lr=0.01, betas=(0.9, 0.95), weight_decay=0.1),
        ),
    ],
)
def test_matches_pytorch(ours, theirs) -> None:
    model_a, x, y = _problem()
    model_b = copy.deepcopy(model_a)
    _train(model_a, ours(model_a.parameters()), x, y)
    _train(model_b, theirs(model_b.parameters()), x, y)
    assert _same_weights(model_a, model_b) < 1e-10


def test_adamw_decay_is_decoupled_from_the_gradient() -> None:
    p = nn.Parameter(torch.tensor([2.0, -3.0]))
    opt = AdamW([p], lr=0.1, weight_decay=0.5)
    p.grad = torch.zeros_like(p)
    opt.step()
    # no gradient: m = v = 0, so the only change is the multiplicative shrink 1 - lr * wd
    assert torch.allclose(p, torch.tensor([2.0, -3.0]) * (1 - 0.1 * 0.5))


def test_adam_l2_decay_is_rescaled_by_the_adaptive_step() -> None:
    p = nn.Parameter(torch.tensor([10.0, 0.1]))
    opt = Adam([p], lr=0.1, weight_decay=0.5)
    p.grad = torch.zeros_like(p)
    opt.step()
    # with an L2 penalty every weight moves by about lr, whatever its size (Adam normalises it)
    assert torch.allclose(p, torch.tensor([10.0 - 0.1, 0.1 - 0.1]), atol=1e-4)


def test_clip_grad_norm_matches_pytorch() -> None:
    model_a, x, y = _problem()
    model_b = copy.deepcopy(model_a)
    for model in (model_a, model_b):
        ((model(x) * 50 - y) ** 2).mean().backward()
    ours = clip_grad_norm(model_a.parameters(), 1.0)
    theirs = nn.utils.clip_grad_norm_(model_b.parameters(), 1.0)
    assert ours == pytest.approx(float(theirs))
    assert ours > 1.0
    for p, q in zip(model_a.parameters(), model_b.parameters(), strict=True):
        assert torch.allclose(p.grad, q.grad)


def test_clip_leaves_small_gradients_alone() -> None:
    p = nn.Parameter(torch.zeros(3))
    p.grad = torch.tensor([0.1, 0.2, 0.2])
    assert clip_grad_norm([p], 5.0) == pytest.approx(0.3)
    assert torch.allclose(p.grad, torch.tensor([0.1, 0.2, 0.2]))


def test_schedules_match_course_page() -> None:
    total, warmup = 20_000, 1_000

    def at(name: str, fraction: float) -> float:
        return lr_factor(name, round(fraction * total), total, warmup)

    def mean(name: str) -> float:
        return sum(lr_factor(name, s, total, warmup) for s in range(total)) / total

    assert round(mean("cosine"), 3) == 0.548
    assert round(mean("linear"), 3) == 0.548
    assert round(mean("wsd"), 3) == 0.885
    assert round(at("cosine", 0.25), 3) == 0.905
    assert round(at("linear", 0.25), 3) == 0.811
    assert [round(at("invsqrt", f), 3) for f in (0.25, 0.5, 0.75)] == [0.447, 0.316, 0.258]
    assert round(lr_factor("invsqrt", total - 1, total, warmup), 3) == 0.224


def test_warmup_is_linear_and_decay_ends_at_the_floor() -> None:
    assert lr_factor("cosine", 0, 100, warmup=10) == pytest.approx(0.1)
    assert lr_factor("cosine", 9, 100, warmup=10) == pytest.approx(1.0)
    for name in ("cosine", "linear", "wsd"):
        assert lr_factor(name, 99, 100, warmup=10, floor=0.1) == pytest.approx(0.1)


def test_unknown_schedule_is_rejected() -> None:
    with pytest.raises(ValueError):
        lr_factor("step", 0, 10)
