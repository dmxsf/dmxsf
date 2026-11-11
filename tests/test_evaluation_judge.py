import numpy as np
import pytest

from dmxsf.evaluation.judge import (
    agreement_no_bias,
    bt_probability,
    cohens_kappa,
    elo_to_beta,
    exact_agreement,
    fit_bt,
    judge_once,
    judge_swapped,
    labels_needed,
    make_judge,
    to_elo,
)


def test_swapping_cancels_position_bias_and_leaves_ties():
    rng = np.random.default_rng(0)
    judge = make_judge(position_bias=5.0, noise=0.0, rng=rng)
    # a huge bias with no noise: the first answer always wins, so the swapped verdict is a tie
    assert judge_swapped(judge, 1.0, 0.0) == "tie"
    assert judge_swapped(judge, 0.0, 1.0) == "tie"
    # a clear quality gap beats a small bias
    judge = make_judge(position_bias=0.5, noise=0.0, rng=rng)
    assert judge_swapped(judge, 2.0, 0.0) == "x"
    assert judge_swapped(judge, 0.0, 2.0) == "y"


def test_the_exact_agreement_matches_a_simulation():
    rng = np.random.default_rng(0)
    judge = make_judge(0.5, 0.8, rng)
    gap = rng.normal(size=40_000)
    truth = np.where(gap > 0, "x", "y")
    once = np.array([judge_once(judge, g, 0.0, rng) for g in gap])
    swapped = np.array([judge_swapped(judge, g, 0.0) for g in gap])
    decided = swapped != "tie"
    exact = exact_agreement(0.5, 0.8)
    assert np.mean(once == truth) == pytest.approx(exact["once"], abs=0.01)
    assert 1 - decided.mean() == pytest.approx(exact["tie"], abs=0.01)
    assert np.mean(swapped[decided] == truth[decided]) == pytest.approx(exact["swapped"], abs=0.01)


def test_without_bias_the_closed_form_holds():
    exact = exact_agreement(0.0, 0.8)
    assert exact["once"] == pytest.approx(agreement_no_bias(1.0, 0.8), abs=1e-4)
    assert round(agreement_no_bias(1.0, 0.8), 3) == 0.785
    # averaging both orders divides the noise by sqrt(2)
    assert round(float(agreement_no_bias(1.0, 0.8 / np.sqrt(2))), 3) == 0.836


def test_a_length_bias_survives_swapping():
    rng = np.random.default_rng(0)
    # Equal quality, but x is much longer. The judge has no position bias at all.
    judge = make_judge(0.0, 0.3, rng, length_bias=0.5)
    x, y = (0.0, 5.0), (0.0, 1.0)
    verdicts = [judge_swapped(judge, x, y) for _ in range(400)]
    assert verdicts.count("x") > 0.95 * len(verdicts)  # the long answer wins in both orders
    # a position bias, in contrast, is cancelled by the swap
    biased = make_judge(2.0, 0.3, rng)
    verdicts = [judge_swapped(biased, 0.0, 0.0) for _ in range(400)]
    assert verdicts.count("tie") > 0.95 * len(verdicts)


def test_kappa():
    confusion = np.array([[40, 6, 4], [8, 30, 2], [3, 2, 5]])
    observed, expected, kappa = cohens_kappa(confusion)
    assert (round(observed, 3), round(expected, 3), round(kappa, 3)) == (0.75, 0.418, 0.570)
    assert cohens_kappa(np.array([[90, 0], [10, 0]]))[2] == pytest.approx(0.0)
    assert cohens_kappa(np.array([[50, 0], [0, 50]]))[2] == pytest.approx(1.0)


def test_labels_needed():
    assert labels_needed(0.8, 0.05) == 246
    assert labels_needed(0.5, 0.05) == 385


def test_bradley_terry_recovers_the_ratings():
    rng = np.random.default_rng(0)
    true_elo = np.array([1000.0, 1100.0, 1250.0])
    beta = elo_to_beta(true_elo)
    n = 60_000
    i = rng.integers(3, size=n)
    j = (i + rng.integers(1, 3, size=n)) % 3
    win = (rng.random(n) < bt_probability(beta[i], beta[j])).astype(float)
    est = to_elo(fit_bt(i, j, win, 3))
    assert est[0] == 1000.0
    assert est[1] == pytest.approx(1100, abs=12)
    assert est[2] == pytest.approx(1250, abs=12)


def test_bradley_terry_probability_and_the_elo_scale():
    assert bt_probability(0.0, 0.0) == pytest.approx(0.5)
    assert bt_probability(2.0, 0.0) + bt_probability(0.0, 2.0) == pytest.approx(1.0)
    # a 400-point gap means 10 to 1 odds
    p = bt_probability(float(elo_to_beta(np.array(400.0))), 0.0)
    assert p / (1 - p) == pytest.approx(10.0)
