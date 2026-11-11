"""Speculative decoding: the page's numbers, the exactness of the rule, and the whole loop."""

import numpy as np
import pytest
import torch

from dmxsf.architecture.configs import small_gpt
from dmxsf.architecture.gpt import GPT
from dmxsf.inference.generate import generate
from dmxsf.inference.kv_cache import KVCache
from dmxsf.inference.speculative import (
    acceptance_rate,
    best_gamma,
    expected_tokens,
    output_distribution,
    probs,
    residual,
    speculative_generate,
    speedup,
    verify,
)
from dmxsf.utils import chi_square_gof, chi_square_sf


def test_page_numbers():
    assert round(speedup(0.9, 10, 0.0), 2) == 6.86
    assert round(speedup(0.8, 4, 0.05), 2) == 2.80
    assert [round(x, 2) for x in best_gamma(0.8, 0.05)] == [8, 3.09]
    assert round(speedup(0.8, 16, 0.05), 2) == 2.72
    assert round(speedup(0.8, 4, 0.0), 2) == 3.36
    assert round(speedup(0.6, 4, 0.05), 2) == 1.92
    assert (
        round(5 / expected_tokens(0.8, 4), 2) == 1.49
        and round(5 / expected_tokens(0.6, 4), 2) == 2.17
    )
    assert [round(x, 2) for x in best_gamma(0.8, 0.3)] == [3, 1.55]
    assert round(speedup(0.3, 4, 0.3), 2) == 0.65
    assert speedup(0.7, 1, 0.2) == pytest.approx((1 + 0.7) / (1 + 0.2))
    assert expected_tokens(1.0, 4) == 5.0 and expected_tokens(0.0, 4) == 1.0


def test_the_rule_gives_back_the_target_distribution_exactly():
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        p = torch.softmax(torch.randn(12, generator=g).double() * 2, -1)
        q = torch.softmax(torch.randn(12, generator=g).double() * 2, -1)
        assert torch.allclose(output_distribution(p, q), p, rtol=0, atol=1e-12)
        assert residual(p, q).sum().item() == pytest.approx(1.0)
        assert acceptance_rate(p, q).item() == pytest.approx(1 - 0.5 * (p - q).abs().sum().item())
    wrong = output_distribution(p, q, repair="target")
    assert not torch.allclose(
        wrong, p, atol=1e-3
    )  # refilling from p double counts the likely tokens
    assert wrong.sum().item() == pytest.approx(1.0)


def test_greedy_rule_accepts_agreement_and_repairs_disagreement():
    p = probs(torch.tensor([[0.0, 3.0, 1.0], [2.0, 0.0, 0.0], [0.0, 0.0, 5.0]]), 0)
    q = probs(torch.tensor([[0.0, 4.0, 0.0], [0.0, 0.0, 9.0]]), 0)  # agrees, then disagrees
    k, final = verify(p, q, torch.tensor([1, 2]))
    assert k == 1 and int(final) == 0  # repaired with the target's own argmax
    k, final = verify(p[:1].repeat(2, 1), q[:1].repeat(1, 1), torch.tensor([1]))
    assert (
        k == 1 and int(final) == 1
    )  # everything accepted: the bonus token comes from the last row


def test_expected_tokens_per_round_matches_simulation():
    rng = np.random.default_rng(0)
    for alpha, gamma in ((0.8, 4), (0.5, 6), (0.95, 3)):
        accept = rng.random((200_000, gamma)) < alpha
        n_accepted = np.where(accept.all(1), gamma, accept.argmin(1))
        assert (n_accepted + 1).mean() == pytest.approx(expected_tokens(alpha, gamma), abs=0.02)


def test_crop_matches_never_having_seen_the_tokens():
    torch.manual_seed(0)
    model = GPT(small_gpt(vocab_size=16, n_layer=2, block_size=32)).eval()
    ids = torch.randint(0, 16, (1, 12))
    with torch.no_grad():
        cache = KVCache(2)
        model(ids, cache=cache)
        cache.crop(7)
        assert cache.length == 7
        a = model(ids[:, 7:9], cache=cache)
        fresh = KVCache(2)
        model(ids[:, :7], cache=fresh)
        b = model(ids[:, 7:9], cache=fresh)
    assert torch.allclose(a, b, atol=1e-5)


def tiny_pair(vocab=8):
    torch.manual_seed(0)
    target = GPT(small_gpt(vocab_size=vocab, n_layer=2, n_head=2, d_model=32, block_size=64)).eval()
    draft = GPT(small_gpt(vocab_size=vocab, n_layer=1, n_head=2, d_model=16, block_size=64)).eval()
    for m in (target, draft):  # random weights are close to uniform: give them some opinions
        for p in m.parameters():
            torch.nn.init.normal_(p, std=0.3)
    return target, draft


@pytest.mark.parametrize("gamma", [1, 3, 5])
def test_greedy_speculative_equals_greedy_target(gamma):
    target, draft = tiny_pair()
    for seed in range(4):
        prompt = torch.randint(0, 8, (1, 5 + seed), generator=torch.Generator().manual_seed(seed))
        want = generate(target, prompt, 20, temperature=0.0)
        got, stats = speculative_generate(target, draft, prompt, 20, gamma, temperature=0.0)
        assert torch.equal(got, want)
        assert stats.accepted <= stats.drafted
        assert stats.target_calls == stats.rounds + 1
        assert (stats.accepted + stats.rounds) >= 20


def test_a_one_token_prompt_and_a_perfect_draft():
    target, _ = tiny_pair()
    prompt = torch.tensor([[3]])
    got, stats = speculative_generate(target, target, prompt, 16, 4, temperature=0.0)
    want = generate(target, prompt, 16, temperature=0.0)
    assert torch.equal(got, want)
    assert stats.alpha == 1.0 and stats.tokens_per_round == 5.0  # the draft is the target


def test_sampled_speculative_decoding_has_the_target_distribution():
    target, draft = tiny_pair(vocab=6)
    prompt = torch.tensor([[1, 4, 2, 0, 3]])
    with torch.no_grad():
        p1 = torch.softmax(target(prompt)[0, -1], -1)
        joint = torch.zeros(6, 6)
        for a in range(6):
            nxt = torch.cat([prompt, torch.tensor([[a]])], 1)
            joint[a] = p1[a] * torch.softmax(target(nxt)[0, -1], -1)
    g = torch.Generator().manual_seed(1)
    counts = np.zeros(36)
    for _ in range(3000):
        out, _ = speculative_generate(target, draft, prompt, 2, gamma=3, generator=g)
        counts[int(out[0, 0]) * 6 + int(out[0, 1])] += 1
    stat, df, p_value = chi_square_gof(counts, joint.flatten().numpy())
    assert p_value > 0.001, (stat, df, p_value)


def test_chi_square_p_values():
    assert chi_square_sf(3.841458820694124, 1) == pytest.approx(0.05, abs=1e-9)
    assert chi_square_sf(5.991464547107979, 2) == pytest.approx(0.05, abs=1e-9)
    assert chi_square_sf(18.307038053275146, 10) == pytest.approx(0.05, abs=1e-9)
    assert chi_square_sf(124.34, 100) == pytest.approx(0.05, abs=2e-4)
    assert chi_square_sf(0.0, 5) == 1.0
    stat, df, p = chi_square_gof([100, 100, 100, 100], [0.25] * 4)
    assert stat == 0 and df == 3 and p == pytest.approx(1.0)
    _, _, p = chi_square_gof([160, 80, 80, 80], [0.25] * 4)
    assert p < 1e-3
