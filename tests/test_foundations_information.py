"""Entropy, cross-entropy and KL divergence: the identities of
https://dmxsf.com/wiki/foundations/information-theory/."""

import math

import numpy as np
import pytest

from dmxsf.foundations import information as info


def test_entropy_of_a_uniform_distribution_is_log_n() -> None:
    assert info.entropy([0.25] * 4) == pytest.approx(math.log(4))


def test_entropy_ignores_outcomes_of_probability_zero() -> None:
    assert info.entropy([0.5, 0.5, 0.0]) == pytest.approx(math.log(2))
    assert info.entropy([1.0, 0.0]) == 0.0


def test_cross_entropy_is_entropy_plus_kl() -> None:
    p, q = [0.5, 0.3, 0.2], [0.2, 0.5, 0.3]
    assert info.cross_entropy(p, q) == pytest.approx(info.entropy(p) + info.kl_divergence(p, q))


def test_kl_is_nonnegative_zero_only_for_equal_distributions_and_not_symmetric() -> None:
    p, q = [0.7, 0.2, 0.1], [0.1, 0.3, 0.6]
    assert info.kl_divergence(p, p) == pytest.approx(0.0)
    assert info.kl_divergence(p, q) > 0 < info.kl_divergence(q, p)
    assert info.kl_divergence(p, q) != pytest.approx(info.kl_divergence(q, p))


def test_cross_entropy_with_a_one_hot_target_is_the_negative_log_probability() -> None:
    assert info.cross_entropy([0, 1, 0], [0.2, 0.6, 0.2]) == pytest.approx(-math.log(0.6))


def test_cross_entropy_is_infinite_when_q_misses_an_outcome_p_can_produce() -> None:
    assert info.cross_entropy([0.5, 0.5], [1.0, 0.0]) == math.inf
    assert info.cross_entropy([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)


def test_perplexity_is_the_size_of_the_equally_hard_uniform_distribution() -> None:
    assert info.perplexity(math.log(7)) == pytest.approx(7)
    assert info.perplexity(info.entropy([1 / 9] * 9)) == pytest.approx(9)


def test_bits_per_byte_compares_models_with_different_tokenizers() -> None:
    assert info.bits_per_byte(math.log(2)) == pytest.approx(1.0)
    # a token of three bytes that costs three bits is one bit per byte
    assert info.bits_per_byte(3 * math.log(2), bytes_per_token=3) == pytest.approx(1.0)


def test_softmax_is_a_distribution_unchanged_by_shifts_and_stable_for_huge_logits() -> None:
    z = np.array([1000.0, 1001.0, 999.0])  # exp(1000) overflows a naive implementation
    p = info.softmax(z)
    assert p.sum() == pytest.approx(1.0)
    assert np.allclose(p, info.softmax(z - 1000))
    assert np.allclose(np.log(p), info.log_softmax(z))


def test_softmax_works_along_the_last_axis() -> None:
    p = info.softmax(np.random.default_rng(0).normal(size=(3, 5)))
    assert p.shape == (3, 5)
    assert np.allclose(p.sum(axis=-1), 1.0)
