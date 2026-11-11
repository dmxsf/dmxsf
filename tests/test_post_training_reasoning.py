"""Verifiers, GRPO advantages and loss, the k3 estimator, rollouts."""

import math

import numpy as np
import pytest
import torch

from dmxsf.architecture.configs import small_gpt
from dmxsf.architecture.gpt import GPT
from dmxsf.post_training import intents
from dmxsf.post_training import reasoning as R


def test_the_pages_math_verifier():
    assert R.math_reward(r"so \boxed{3}", "3") == 1.0
    assert R.math_reward(r"so \boxed{4}", "3") == 0.0
    assert R.math_reward("the answer is 3", "3") == 0.0  # wrong format
    assert R.math_reward(r"\boxed{4} no wait \boxed{3}", "3") == 1.0  # the last box counts
    assert R.math_reward(r"\boxed{0.5}", "1/2") == 0.0  # string equality is not equivalence


def test_exact_and_leaky_verifiers():
    dump = "card_missing card_fee transfer_declined"
    assert R.exact_match_reward("card_fee", "card_fee") == 1.0
    assert R.exact_match_reward(dump, "card_fee") == 0.0
    assert R.contains_reward(dump, "card_fee") == 1.0  # the leak


def test_group_advantages_match_the_page():
    r = torch.tensor([[1.0, 0, 0, 1], [1.0, 0, 0, 0], [1.0, 1, 1, 1]])
    want = torch.tensor(
        [
            [0.8659, -0.8659, -0.8659, 0.8659],
            [1.4997, -0.4999, -0.4999, -0.4999],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert torch.allclose(R.group_advantages(r), want, atol=1e-4)


def test_advantages_ignore_the_scale_and_the_std_convention_is_one_factor():
    r = torch.tensor([[1.0, 0, 0, 1], [1.0, 0, 0, 0]], dtype=torch.float64)
    assert torch.allclose(R.group_advantages(10 * r), R.group_advantages(r), atol=1e-3)
    ratio = R.group_advantages(r, unbiased=False)[0, 0] / R.group_advantages(r)[0, 0]
    assert ratio.item() == pytest.approx(math.sqrt(4 / 3), rel=1e-3)  # sqrt(G / (G - 1))
    mean_only = R.group_advantages(r, divide_by_std=False)
    assert mean_only[0].tolist() == [0.5, -0.5, -0.5, 0.5]


def test_signal_probability_matches_the_pages_numbers_and_simulation():
    assert R.signal_probability(0.5, 4) == pytest.approx(0.875)
    assert R.signal_probability(0.5, 16) > 0.9999
    assert 1 - R.signal_probability(0.05, 16) - 0.95**16 == pytest.approx(0, abs=1e-12)
    assert pytest.approx(0.44, abs=0.005) == 0.95**16  # "about 44% of groups are all wrong"
    rng = np.random.default_rng(0)
    hits = (rng.random((200_000, 8)) < 0.2).sum(axis=1)
    assert ((hits > 0) & (hits < 8)).mean() == pytest.approx(
        R.signal_probability(0.2, 8), abs=0.004
    )


def test_k3_is_non_negative_and_unbiased():
    rng = np.random.default_rng(1)
    pi = torch.tensor(rng.dirichlet(np.ones(6)))
    ref = torch.tensor(rng.dirichlet(np.ones(6)))
    estimate = R.k3(pi.log(), ref.log())
    assert (estimate >= 0).all()
    exact = (pi * (pi.log() - ref.log())).sum()
    assert (pi * estimate).sum().item() == pytest.approx(exact.item(), rel=1e-10)
    k1 = pi.log() - ref.log()
    assert (k1 < 0).any()  # the plain log-ratio is unbiased too, but goes negative


def grads(adv, lengths, ratio=1.0, token_level=False, beta=0.0):
    t = max(lengths)
    mask = torch.tensor([[1.0] * n + [0.0] * (t - n) for n in lengths])
    new = torch.zeros(len(lengths), t, requires_grad=True)
    old = torch.full((len(lengths), t), -math.log(ratio))  # exp(new - old) = ratio
    loss = R.grpo_loss(
        new, old, torch.zeros_like(new), torch.tensor(adv), mask, beta=beta, token_level=token_level
    )
    loss.backward()
    return loss, new.grad, mask


def test_loss_at_ratio_one_is_minus_the_mean_advantage():
    loss, _, _ = grads([0.8, -0.3, -0.5], [3, 7, 5])
    assert loss.item() == pytest.approx(0.0, abs=1e-7)  # advantages sum to zero, KL is zero
    loss, _, _ = grads([1.0, 0.0], [2, 9])
    assert loss.item() == pytest.approx(-0.5)


def test_length_normalisation_changes_the_per_token_weight():
    adv, lengths = [-0.87, -0.87], [10, 100]
    _, g, mask = grads(adv, lengths)  # per-sequence mean (the page)
    per_token = g[mask > 0] * 2  # undo the 1/B of the batch mean; positive = pushed down
    assert per_token[:10].tolist() == pytest.approx([0.087] * 10)
    assert per_token[10:].tolist() == pytest.approx(
        [0.0087] * 100
    )  # a long wrong answer: 10x lighter
    _, g, mask = grads(adv, lengths, token_level=True)  # token-level mean (DAPO)
    assert (g[mask > 0] * 110).tolist() == pytest.approx([0.87] * 110)  # same weight everywhere


def test_clipping_blocks_only_the_updates_ppo_wants_blocked():
    for adv, ratio, flows in [
        (1.0, 1.5, False),  # already more likely and the advantage is positive: stop
        (1.0, 0.5, True),  # became less likely although it was good: still push up
        (-1.0, 0.5, False),  # already less likely and the advantage is negative: stop
        (-1.0, 1.5, True),  # became more likely although it was bad: still push down
    ]:
        _, g, _ = grads([adv, 0.0], [3, 3], ratio=ratio)
        assert (g[0].abs().sum().item() > 0) == flows, (adv, ratio)


def small():
    torch.manual_seed(0)
    return GPT(small_gpt(block_size=128, n_layer=2, n_head=2, d_model=32)).eval()


def test_rollout_shapes_and_mask():
    model = small()
    texts = [t for t, _ in intents.make_dataset(1, seed=0)[:3]]
    gen = torch.Generator().manual_seed(0)
    ids, mask, answers = R.rollout(model, texts, 4, max_new=6, generator=gen)
    n_prompt = len(intents.render(texts[0]))
    assert ids.shape == (12, n_prompt + 6) and mask.shape == (12, n_prompt + 5)
    assert len(answers) == 12
    for i in range(12):
        row = ids[i, n_prompt:].tolist()
        cut = row.index(intents.STOP) if intents.STOP in row else len(row)
        assert int(mask[i].sum()) == min(cut + 1, 6)
        assert mask[i, : n_prompt - 1].sum() == 0  # no prompt position is supervised
    # samples of one message sit in consecutive rows and share the prompt
    assert torch.equal(ids[0, :n_prompt], ids[3, :n_prompt])
    assert not torch.equal(ids[0, :n_prompt], ids[4, :n_prompt])


def test_token_logprobs_equal_the_log_softmax_gather():
    model = small()
    ids = torch.randint(0, 256, (2, 12))
    with torch.no_grad():
        got = R.token_logprobs(model, ids)
        manual = torch.log_softmax(model(ids[:, :-1]), -1)[1, 4, ids[1, 5]]
    assert got.shape == (2, 11) and got[1, 4].item() == pytest.approx(manual.item(), abs=1e-6)
    with torch.no_grad():
        logp, entropy = R.token_logprobs(model, ids, with_entropy=True)
        p = torch.softmax(model(ids[:, :-1]), -1)[1, 4]
    assert torch.equal(logp, got) and entropy.shape == (2, 11)
    assert entropy[1, 4].item() == pytest.approx(-(p * p.log()).sum().item(), abs=1e-5)


def test_train_grpo_runs_and_reports_everything():
    model, ref = small(), small()
    prompts = intents.make_dataset(1, seed=3)
    log = R.train_grpo(
        model, ref, [t for t, _ in prompts], [y for _, y in prompts],
        steps=2, prompts_per_step=3, group=4, max_new=6,
    )  # fmt: skip
    assert len(log) == 2
    assert set(log[0]) == {"reward", "signal", "length", "valid", "kl", "entropy", "loss"}
    assert log[0]["kl"] == pytest.approx(0.0, abs=1e-6)  # policy = reference at step 1
    assert all(np.isfinite(v) for r in log for v in r.values())
