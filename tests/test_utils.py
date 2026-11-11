"""The helpers every lab uses: parameter counts, formatting and the chi-square test."""

import time

import numpy as np
import pytest
import torch
from torch import nn

from dmxsf.utils import (
    Timer,
    chi_square_gof,
    chi_square_sf,
    count_params,
    human_bytes,
    human_count,
    seed_everything,
)


def test_count_params_counts_a_tied_weight_once_and_can_skip_frozen_ones() -> None:
    a, b = nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False)
    b.weight = a.weight  # tied: one tensor, two uses
    assert count_params(nn.Sequential(a, b)) == 16
    c = nn.Linear(4, 4)
    c.weight.requires_grad_(False)
    assert count_params(c) == 20
    assert count_params(c, trainable_only=True) == 4


def test_human_readable_sizes() -> None:
    assert human_bytes(10) == "10 B"
    assert human_bytes(1536) == "1.50 KiB"
    assert human_bytes(80 * 2**30) == "80.00 GiB"
    assert human_count(8_030_000_000) == "8.03B"
    assert human_count(124_439_808) == "124.44M"
    assert human_count(512) == "512"


def test_seed_everything_repeats_all_three_random_number_generators() -> None:
    import random

    def draw() -> tuple[float, float, float]:
        seed_everything(7)
        return random.random(), float(np.random.rand()), float(torch.rand(()))

    assert draw() == draw()


def test_timer_measures_the_block() -> None:
    with Timer() as t:
        time.sleep(0.02)
    assert 0.015 < t.seconds < 1.0


@pytest.mark.parametrize(
    ("x", "df", "p"),
    [(3.841, 1, 0.05), (5.991, 2, 0.05), (18.307, 10, 0.05), (6.635, 1, 0.01), (0.0, 3, 1.0)],
)
def test_chi_square_survival_function_matches_the_textbook_table(x, df, p) -> None:
    assert chi_square_sf(x, df) == pytest.approx(p, abs=5e-4)


def test_chi_square_gof_accepts_a_fair_sample_and_rejects_a_loaded_one() -> None:
    probs = [0.25] * 4
    stat, df, p = chi_square_gof([250, 250, 250, 250], probs)
    assert (stat, df, p) == (0.0, 3, pytest.approx(1.0))
    _, _, p_loaded = chi_square_gof([400, 200, 200, 200], probs)
    assert p_loaded < 1e-10


def test_chi_square_gof_pools_outcomes_that_are_expected_too_rarely() -> None:
    # two outcomes expected 2 times each are pooled into one: 3 cells, 2 degrees of freedom
    _, df, _ = chi_square_gof([50, 46, 2, 2], [0.5, 0.46, 0.02, 0.02])
    assert df == 2
