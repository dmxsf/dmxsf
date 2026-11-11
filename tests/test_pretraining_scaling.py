"""Chinchilla's parametric loss and the compute-optimal split of
https://dmxsf.com/wiki/pretraining/scaling-laws/."""

import numpy as np
import pytest

from dmxsf.pretraining import scaling


def test_loss_falls_with_parameters_and_with_tokens() -> None:
    assert scaling.loss(2e9, 1e11) < scaling.loss(1e9, 1e11)
    assert scaling.loss(1e9, 2e11) < scaling.loss(1e9, 1e11)
    assert scaling.loss(1e15, 1e18) > scaling.E  # the irreducible loss is a floor


def test_compute_optimal_spends_exactly_the_budget() -> None:
    n, d = scaling.compute_optimal(1e23)
    assert 6 * n * d == pytest.approx(1e23)


def test_compute_optimal_is_a_minimum_of_the_loss_at_that_budget() -> None:
    flops = 1e23
    n, d = scaling.compute_optimal(flops)
    best = scaling.loss(n, d)
    for factor in (0.5, 0.8, 0.95, 1.05, 1.25, 2.0):
        n2 = n * factor
        assert scaling.loss(n2, flops / (6 * n2)) > best


def test_optimal_exponents_add_up_to_one_and_describe_how_the_split_grows() -> None:
    a, b = scaling.optimal_exponents()
    assert a + b == pytest.approx(1.0)
    n1, d1 = scaling.compute_optimal(1e22)
    n2, d2 = scaling.compute_optimal(1e24)
    assert n2 / n1 == pytest.approx(100**a)
    assert d2 / d1 == pytest.approx(100**b)


def test_tokens_needed_inverts_the_loss() -> None:
    n, target = 7e9, 2.0
    d = scaling.tokens_needed(n, target)
    assert scaling.loss(n, d) == pytest.approx(target)


def test_a_model_cannot_beat_its_own_floor_however_much_data_it_sees() -> None:
    n = 1e9
    floor = scaling.E + scaling.A / n**scaling.ALPHA
    assert scaling.tokens_needed(n, floor - 0.01) == np.inf


def test_serving_more_tokens_favours_a_smaller_model_trained_longer() -> None:
    _, n_train_only, d_train_only = scaling.best_model(2.0, 0.0)
    _, n_served, d_served = scaling.best_model(2.0, 1e13)
    assert n_served < n_train_only
    assert d_served > d_train_only


def test_fit_recovers_the_coefficients_of_noiseless_data() -> None:
    n = np.array([1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e6, 1e7, 1e8, 3e8, 3e6, 3e7])
    d = np.array([1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e9, 1e9, 1e9, 1e9, 1e10, 1e10])
    out = scaling.fit(n, d, scaling.loss(n, d))
    assert out["rms"] < 1e-3
    assert out["alpha"] == pytest.approx(scaling.ALPHA, abs=0.02)
    assert out["beta"] == pytest.approx(scaling.BETA, abs=0.02)
    assert out["E"] == pytest.approx(scaling.E, abs=0.05)
