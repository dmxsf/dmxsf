"""Language modeling basics. The first test reproduces the numbers printed on the course page
https://dmxsf.com/wiki/foundations/next-token-prediction/."""

import math

import numpy as np
import pytest
import torch

from dmxsf.foundations.information import (
    bits_per_byte,
    cross_entropy,
    entropy,
    kl_divergence,
    log_softmax,
    perplexity,
    softmax,
)
from dmxsf.foundations.language_model import (
    NGramLM,
    bigram_counts,
    conditional_entropy,
    fit_neural_bigram,
)

CORPUS = "the cat sat on the mat . the dog sat on the log .".split()
CORPUS += "the cat saw the dog .".split()


def test_bigram_matches_course_page() -> None:
    lm = NGramLM(CORPUS, n=2, k=0.1)
    assert len(lm.vocab) == 9
    assert round(lm.prob("sat", "cat"), 3) == 0.379
    assert round(lm.prob("mat", "the"), 3) == 0.159
    assert round(lm.prob("cat", "the"), 3) == 0.304
    assert round(lm.perplexity(["the", "cat", "sat", "on", "the", "mat", "."]), 2) == 2.37
    assert round(lm.perplexity(["mat", "the", "on", ".", "sat", "cat", "the"]), 2) == 31.23


def test_probabilities_sum_to_one() -> None:
    lm = NGramLM(CORPUS, n=2, k=0.1)
    for prev in lm.vocab:
        assert sum(lm.prob(tok, prev) for tok in lm.vocab) == pytest.approx(1.0)


def test_unsmoothed_model_gives_infinite_perplexity_on_unseen_pairs() -> None:
    lm = NGramLM(CORPUS, n=2, k=0.0)
    assert lm.perplexity(["the", "cat", "sat", "on", "the", "mat", "."]) < 5
    assert lm.perplexity(["mat", "the", "on", ".", "sat", "cat", "the"]) == math.inf


def test_trigram_needs_two_context_tokens() -> None:
    lm = NGramLM(list("abababab"), n=3, k=0.0)
    assert lm.prob("a", "a", "b") == 1.0
    with pytest.raises(ValueError):
        lm.prob("a", "b")


def test_sampling_follows_the_counts() -> None:
    lm = NGramLM(list("ababababab"), n=2, k=0.0)
    out = lm.sample(["a"], 6, np.random.default_rng(0))
    assert out == list("bababa")


def _markov_chain(n_tokens: int, vocab: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    trans = rng.dirichlet(np.ones(vocab) * 0.3, size=vocab)
    ids = np.zeros(n_tokens, dtype=np.int64)
    for i in range(1, n_tokens):
        ids[i] = rng.choice(vocab, p=trans[ids[i - 1]])
    return ids, trans


def test_neural_bigram_converges_to_the_count_table() -> None:
    ids, _ = _markov_chain(20_000, 6, seed=1)
    counts = bigram_counts(ids, 6)
    model, history = fit_neural_bigram(counts, steps=400, lr=0.3)
    best = conditional_entropy(counts)
    assert history[0] == pytest.approx(math.log(6), abs=1e-4)  # starts at ln V
    assert history[-1] >= best - 1e-6  # nothing beats the maximum-likelihood table
    assert history[-1] - best < 5e-3
    learned = torch.softmax(model.table.weight, dim=-1).detach().numpy()
    mle = counts / counts.sum(axis=1, keepdims=True)
    assert np.abs(learned - mle).max() < 0.03


def test_conditional_entropy_of_a_deterministic_cycle_is_zero() -> None:
    ids = np.tile(np.arange(4), 50)
    assert conditional_entropy(bigram_counts(ids, 4)) == pytest.approx(0.0, abs=1e-12)


def test_information_theory_matches_course_page() -> None:
    p = [0.50, 0.25, 0.10, 0.10, 0.05]
    q = [0.02, 0.02, 0.02, 0.02, 0.92]
    assert round(entropy(p), 3) == 1.303
    assert round(cross_entropy(p, q), 3) == 3.721
    assert round(kl_divergence(p, q), 3) == 2.417
    assert round(kl_divergence(q, p), 3) == 2.500
    assert round(perplexity(cross_entropy(p, q)), 2) == 41.29
    one_hot, model = [1, 0, 0, 0, 0], [0.60, 0.20, 0.10, 0.05, 0.05]
    assert cross_entropy(one_hot, model) == pytest.approx(-math.log(0.6))
    for loss, bytes_per_token in [(2.0, 4.0), (1.0, 2.0)]:
        assert round(bits_per_byte(loss, bytes_per_token), 3) == 0.721


def test_kl_is_never_negative_and_zero_only_for_equal_distributions() -> None:
    rng = np.random.default_rng(0)
    for _ in range(100):
        p, q = rng.dirichlet(np.ones(7)), rng.dirichlet(np.ones(7))
        assert kl_divergence(p, q) >= -1e-12
        assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-12)


def test_cross_entropy_is_infinite_when_the_model_rules_out_the_truth() -> None:
    assert cross_entropy([0.5, 0.5], [1.0, 0.0]) == math.inf


def test_softmax_is_stable_and_matches_torch() -> None:
    logits = np.array([[1000.0, 1001.0, 999.0], [-3.0, 0.5, 2.0]])
    assert np.isfinite(softmax(logits)).all()
    expected = torch.log_softmax(torch.tensor(logits), dim=-1).numpy()
    assert np.allclose(log_softmax(logits), expected)
    assert np.allclose(softmax(logits).sum(axis=-1), 1.0)


def test_cross_entropy_loss_is_negative_log_of_the_true_token() -> None:
    logits = torch.randn(5, 11)
    targets = torch.randint(0, 11, (5,))
    manual = -log_softmax(logits.numpy())[np.arange(5), targets.numpy()].mean()
    assert float(torch.nn.functional.cross_entropy(logits, targets)) == pytest.approx(
        manual, rel=1e-5
    )
