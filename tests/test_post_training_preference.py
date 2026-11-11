"""DPO: the page's numbers, the gradient weight, the closed-form optimum, enumerable answers."""

import math

import numpy as np
import pytest
import torch

from dmxsf.architecture.configs import small_gpt
from dmxsf.architecture.gpt import GPT
from dmxsf.post_training import intents
from dmxsf.post_training import preference as P


def t(*values):
    return torch.tensor(values, dtype=torch.float64)


def test_the_pages_worked_example():
    loss, acc = P.dpo_loss(t(-12.0), t(-15.0), t(-12.5), t(-14.0), beta=0.1)
    assert loss.item() == pytest.approx(0.621, abs=5e-4) and acc.item() == 1.0
    loss5, _ = P.dpo_loss(t(-12.0), t(-15.0), t(-12.5), t(-14.0), beta=0.5)
    assert loss5.item() == pytest.approx(0.387, abs=5e-4)  # margin 0.75 instead of 0.15


def test_loss_is_ln2_when_the_policy_equals_the_reference():
    lp = torch.randn(8, dtype=torch.float64)
    lq = torch.randn(8, dtype=torch.float64)
    loss, acc = P.dpo_loss(lp, lq, lp, lq)
    assert loss.item() == pytest.approx(math.log(2)) and acc.item() == 0.0  # ties are not wins


def test_gradient_matches_the_pages_formula():
    pi_c, pi_r = t(-12.0).requires_grad_(), t(-15.0).requires_grad_()
    ref_c, ref_r = t(-12.5), t(-14.0)
    beta = 0.1
    loss, _ = P.dpo_loss(pi_c, pi_r, ref_c, ref_r, beta)
    loss.backward()
    w = P.dpo_gradient_weight(pi_c, pi_r, ref_c, ref_r, beta)
    assert pi_c.grad.item() == pytest.approx(-beta * w.item())  # push the chosen answer up
    assert pi_r.grad.item() == pytest.approx(beta * w.item())  # and the rejected one down
    assert w.item() < 0.5  # a pair that is already ordered right weighs less than at the start


def test_optimal_policy_limits():
    ref = np.array([0.5, 0.3, 0.2])
    r = np.array([0.0, 1.0, 2.0])
    assert P.optimal_policy(ref, r, 1e6) == pytest.approx(ref, abs=1e-5)  # huge beta: stay at ref
    assert P.optimal_policy(ref, r, 1e-3).argmax() == 2  # tiny beta: all mass on the best
    pi = P.optimal_policy(ref, r, 1.0)
    assert pi.sum() == pytest.approx(1) and pi[2] / pi[1] == pytest.approx(0.2 / 0.3 * math.e)


def test_kl_basics():
    p, q = np.array([0.5, 0.5]), np.array([0.9, 0.1])
    assert P.kl(p, p) == 0 and P.kl(p, q) > 0
    assert P.kl(p, q) == pytest.approx(0.5 * math.log(0.5 / 0.9) + 0.5 * math.log(0.5 / 0.1))


def test_dpo_on_bradley_terry_comparisons_recovers_the_closed_form():
    rng = np.random.default_rng(0)
    ref = rng.dirichlet(np.ones(5))
    reward = rng.normal(size=5)
    beta = 0.5
    counts = P.bradley_terry_counts(reward, 400_000, seed=1, ref=None)
    pi = P.fit_dpo_tabular(counts, ref, beta, steps=2000)
    star = P.optimal_policy(ref, reward, beta)
    assert 0.5 * np.abs(pi - star).sum() < 0.03  # total-variation distance


def small():
    torch.manual_seed(0)
    return GPT(small_gpt(block_size=128, n_layer=2, n_head=2, d_model=32)).eval()


def test_answer_batch_masks_exactly_the_answer():
    texts = [intents.make_dataset(1, seed=0)[0][0]] * 2
    labels = ["card_missing", "transfer_declined"]
    _x, y, mask = P.answer_batch(texts, labels)
    for i, label in enumerate(labels):
        ans = intents.answer_bytes(label)
        assert int(mask[i].sum()) == len(ans)
        assert bytes(y[i][mask[i] > 0].tolist()) == ans  # the supervised targets spell the answer


def test_answer_logprobs_equal_the_manual_sum():
    model = small()
    text, label = intents.make_dataset(1, seed=0)[0]
    seq = torch.tensor([list(intents.render(text)) + list(intents.answer_bytes(label))])
    with torch.no_grad():
        logp = torch.log_softmax(model(seq[:, :-1]), dim=-1)[0]
        n_prompt = len(intents.render(text))
        manual = sum(logp[i, seq[0, i + 1]].item() for i in range(n_prompt - 1, seq.shape[1] - 1))
        got = P.answer_logprobs(model, [text], [label]).item()
    assert got == pytest.approx(manual, abs=1e-4)


def test_label_distribution_matches_single_calls():
    model = small()
    texts = [t_ for t_, _ in intents.make_dataset(1, seed=2)[:3]]
    table = P.label_distribution(model, texts)
    assert table.shape == (3, 12) and (table < 0).all()
    j = intents.LABELS.index("account_fee")
    with torch.no_grad():
        single = P.answer_logprobs(model, texts, ["account_fee"] * 3)
    assert torch.allclose(table[:, j], single, atol=1e-4)
    assert torch.allclose(P.normalise(table).exp().sum(-1), torch.ones(3), atol=1e-5)
