"""Sampling filters from the decoding page, penalties, and beam search."""

import pytest
import torch

from dmxsf.architecture.configs import small_gpt
from dmxsf.architecture.gpt import GPT
from dmxsf.inference import generate as G

Z = torch.tensor([[4.0, 3.0, 1.0, 0.0]])


def probs(**kw):
    return G.process_logits(Z, **kw).softmax(-1)[0]


def test_the_pages_temperature_table():
    assert probs(temperature=0.5).tolist() == pytest.approx([0.879, 0.119, 0.002, 0.000], abs=5e-4)
    assert probs(temperature=1.0).tolist() == pytest.approx([0.696, 0.256, 0.035, 0.013], abs=5e-4)
    assert probs(temperature=2.0).tolist() == pytest.approx([0.509, 0.309, 0.114, 0.069], abs=5e-4)


def test_the_pages_truncation_examples():
    top2 = [0.731, 0.269, 0.0, 0.0]
    assert probs(top_k=2).tolist() == pytest.approx(top2, abs=5e-4)
    assert probs(top_p=0.9).tolist() == pytest.approx(top2, abs=5e-4)
    assert probs(temperature=0.5, top_p=0.9).tolist() == pytest.approx(
        [0.881, 0.119, 0, 0], abs=5e-4
    )
    assert probs(min_p=0.1).tolist() == pytest.approx(top2, abs=5e-4)  # 0.035 < 0.1 * 0.696


def test_order_matters_temperature_before_top_p():
    cold_first = probs(temperature=0.5, top_p=0.99)  # sharper first: two tokens reach 0.99
    nucleus_at_t1 = G.process_logits(Z, 1.0, top_p=0.99)  # nucleus at T=1 keeps all four
    assert (cold_first > 0).sum() == 2 and (nucleus_at_t1.isfinite()).sum() == 4


def test_the_most_likely_token_is_never_dropped():
    for kw in ({"top_k": 1}, {"top_p": 1e-9}, {"min_p": 1.0}):
        assert probs(**kw).tolist() == pytest.approx([1.0, 0, 0, 0])


def test_sampling_frequencies_follow_the_processed_distribution():
    g = torch.Generator().manual_seed(0)
    draws = torch.stack([G.sample_logits(Z, 1.0, top_p=0.9, generator=g) for _ in range(4000)])
    freq = torch.bincount(draws.flatten(), minlength=4) / 4000
    assert freq.tolist() == pytest.approx(probs(top_p=0.9).tolist(), abs=0.03)


def test_penalties():
    logits = torch.tensor([[2.0, -2.0, 1.0, 0.0]])
    history = torch.tensor([[0, 0, 1]])  # token 0 twice, token 1 once
    rep = G.apply_penalties(logits, history, repetition=2.0)
    assert rep.tolist() == [[1.0, -4.0, 1.0, 0.0]]  # positive / 2, negative * 2
    freq = G.apply_penalties(logits, history, frequency=0.5)
    assert freq.tolist() == [[1.0, -2.5, 1.0, 0.0]]  # 0.5 per occurrence
    pres = G.apply_penalties(logits, history, presence=0.5)
    assert pres.tolist() == [[1.5, -2.5, 1.0, 0.0]]  # flat 0.5 once seen
    assert torch.equal(G.apply_penalties(logits, history), logits)


def small():
    torch.manual_seed(0)
    return GPT(small_gpt(block_size=128, n_layer=2, n_head=2, d_model=32)).eval()


def test_a_penalty_changes_greedy_output_and_default_does_not():
    model = small()
    prompt = torch.randint(0, 256, (2, 8))
    plain = G.generate(model, prompt, 12)
    assert torch.equal(plain, G.generate(model, prompt, 12, repetition=1.0))
    heavy = G.generate(model, prompt, 12, frequency=5.0)
    assert not torch.equal(plain, heavy)
    assert all(len(set(row)) == len(row) for row in heavy.tolist())  # no byte twice


def test_beam_search_scores_are_the_summed_log_probs_and_beam_one_is_greedy():
    model = small()
    prompt = torch.randint(0, 256, (1, 8))
    seqs, scores = G.beam_search(model, prompt, beams=4, max_new=6)
    assert seqs.shape == (4, 6) and (scores[:-1] >= scores[1:]).all()
    with torch.no_grad():
        for i in range(4):
            full = torch.cat([prompt, seqs[i : i + 1]], dim=1)
            logp = torch.log_softmax(model(full[:, :-1]), -1)[0]
            total = sum(logp[7 + t, full[0, 8 + t]].item() for t in range(6))
            assert total == pytest.approx(scores[i].item(), abs=1e-4)
    one, _ = G.beam_search(model, prompt, beams=1, max_new=6)
    assert torch.equal(one, G.generate(model, prompt, 6))
