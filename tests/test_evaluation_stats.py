import itertools
from math import comb

import numpy as np
import pytest

from dmxsf.evaluation.stats import (
    bootstrap_diff,
    correlated_bernoulli,
    diff_ci,
    half_width,
    mean_ci,
    pass_at_k,
    pass_at_k_naive,
    pass_hat_k,
    questions_needed,
)


def test_pass_at_k_edges():
    assert pass_at_k(10, 0, 5) == 0.0
    assert pass_at_k(10, 10, 5) == 1.0
    assert pass_at_k(10, 6, 5) == 1.0  # fewer than k failures: any k-subset has a pass
    assert pass_at_k(10, 3, 1) == pytest.approx(0.3)  # pass@1 is the plain pass rate
    with pytest.raises(ValueError):
        pass_at_k(5, 6, 1)
    with pytest.raises(ValueError):
        pass_at_k(5, 2, 6)


@pytest.mark.parametrize(("n", "c", "k"), [(6, 2, 3), (8, 1, 4), (9, 5, 2), (7, 0, 3)])
def test_pass_at_k_is_the_average_over_all_subsets(n, c, k):
    samples = [1] * c + [0] * (n - c)
    subsets = list(itertools.combinations(range(n), k))
    brute = sum(any(samples[i] for i in s) for s in subsets) / len(subsets)
    assert pass_at_k(n, c, k) == pytest.approx(brute)
    assert len(subsets) == comb(n, k)


def test_naive_estimate_is_lower():
    assert pass_at_k(10, 3, 5) == pytest.approx(1 - 21 / 252)
    assert pass_at_k_naive(10, 3, 5) == pytest.approx(1 - 0.7**5)
    assert pass_at_k_naive(10, 3, 5) < pass_at_k(10, 3, 5)
    assert pass_at_k_naive(10, 3, 1) == pytest.approx(pass_at_k(10, 3, 1))


def test_pass_hat_k():
    assert pass_hat_k(4, 3, 1) == pytest.approx(0.75)
    assert pass_hat_k(4, 3, 2) == pytest.approx(0.5)  # 3 of the 6 pairs are both-pass
    assert pass_hat_k(4, 3, 4) == 0.0
    assert pass_hat_k(5, 5, 5) == 1.0
    # pass^k can only go down with k, pass@k only up
    values = [pass_hat_k(10, 7, k) for k in range(1, 8)]
    assert values == sorted(values, reverse=True)


def test_half_width_and_mean_ci():
    assert round(100 * half_width(0.8, 164), 1) == 6.1
    assert round(100 * half_width(0.8, 14079), 1) == 0.7
    x = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    mean, half = mean_ci(x)
    assert mean == pytest.approx(0.75)
    assert half == pytest.approx(1.96 * x.std(ddof=1) / np.sqrt(8), rel=1e-3)


def test_questions_needed_matches_the_page_table():
    assert questions_needed(0.80, 0.75, 0.5) == 548
    assert questions_needed(0.80, 0.75, 0.0) == 1091
    assert questions_needed(0.80, 0.79, 0.5) == 12792
    # more correlation, fewer questions; a bigger gap, fewer questions
    assert questions_needed(0.8, 0.75, 0.8) < questions_needed(0.8, 0.75, 0.5)
    assert questions_needed(0.8, 0.70, 0.5) < questions_needed(0.8, 0.75, 0.5)


def test_correlated_bernoulli_has_the_requested_marginals_and_correlation():
    rng = np.random.default_rng(0)
    a, b = correlated_bernoulli(rng, 200_000, 0.8, 0.7, 0.5)
    assert a.mean() == pytest.approx(0.8, abs=0.005)
    assert b.mean() == pytest.approx(0.7, abs=0.005)
    assert np.corrcoef(a, b)[0, 1] == pytest.approx(0.5, abs=0.01)
    with pytest.raises(ValueError):
        correlated_bernoulli(rng, 10, 0.9, 0.1, 0.9)  # not possible with these marginals


def test_pairing_narrows_the_interval_when_models_are_correlated():
    rng = np.random.default_rng(1)
    a, b = correlated_bernoulli(rng, 400, 0.8, 0.75, 0.6)
    _, lo_p, hi_p = diff_ci(a, b, paired=True)
    _, lo_u, hi_u = diff_ci(a, b, paired=False)
    assert hi_p - lo_p < hi_u - lo_u
    lo, hi = bootstrap_diff(a - b, rng, n_boot=2000)
    assert lo == pytest.approx(lo_p, abs=0.02)
    assert hi == pytest.approx(hi_p, abs=0.02)
