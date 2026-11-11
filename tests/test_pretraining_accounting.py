import math

import numpy as np
import pytest

from dmxsf.pretraining import accounting as acc
from dmxsf.pretraining import scaling


def test_six_n_d_reproduces_llama_2_7b_and_the_implied_mfu():
    flops = acc.train_flops(6.74e9, 2.0e12)
    assert f"{flops:.2e}" == "8.09e+22"
    assert round(acc.implied_mfu(flops, 184_320, 312e12), 2) == 0.39


def test_an_impossible_report_is_detected():
    peak_total = 1e4 * 3600 * 312e12
    assert peak_total < acc.train_flops(7e9, 2e12) / 7  # the page's exercise: 7x short


def test_mfu_of_the_palm_paper_examples():
    assert round(acc.mfu(65.43e3, 6 * 530e9, 2240, 312e12), 3) == 0.298
    assert round(acc.mfu(238.3e3, 6 * 540e9, 6144, 275e12), 3) == 0.457


@pytest.mark.parametrize(
    "zero,n,expected_gib",
    [
        (0, 8, 100.4),
        (1, 8, 34.5),
        (2, 8, 23.5),
        (3, 8, 12.6),
        (1, 64, 26.3),
        (2, 64, 13.9),
        (3, 64, 1.6),
    ],
)
def test_zero_table_for_llama_2_7b(zero, n, expected_gib):
    assert round(acc.model_state_bytes(6.738e9, n, zero) / acc.GiB, 1) == expected_gib


def test_zero_paper_figure_one():  # 7.5B parameters on 64 GPUs, in 10^9 bytes
    got = [round(acc.model_state_bytes(7.5e9, 64, z) / 1e9, 1) for z in range(4)]
    assert got == [120.0, 31.4, 16.6, 1.9]


def test_a_70b_model_needs_at_least_14_gpus_with_zero_3():
    assert acc.min_gpus_for_zero3(70e9, 80 * acc.GiB) == 14
    assert (
        acc.model_state_bytes(70e9, 1000, zero=2) > 80 * acc.GiB
    )  # replicated 2 + 2 bytes already too big


def test_activation_formula_matches_the_course_table():
    per_layer = acc.activation_bytes_per_token_layer(4096, 4096, 11008)  # Llama 2 7B
    assert per_layer / 1024 == 150.0
    assert round(per_layer * 32 * 4096 / acc.GiB, 2) == 18.75
    kept = acc.recompute_bytes_per_token(4096, 32, per_layer)
    assert round(kept * 4096 / acc.GiB, 2) == 1.59
    assert acc.activation_bytes_per_token_layer(4096, 1024, 14336) / 1024 == 164.0  # Llama 3 8B


def test_megatron_whole_network_formula_equals_the_sum_of_its_parts():
    b, s, layers, h, v = 1, 2048, 96, 12288, 51200  # GPT-3 175B
    per_layer_forward = 24 * b * s * h**2 + 4 * b * s**2 * h
    total = 4 * layers * per_layer_forward + 6 * b * s * h * v
    assert math.isclose(acc.megatron_flops(b, s, layers, h, v), total, rel_tol=1e-12)


def test_attention_crossover_lengths():
    assert round(6.607e9 / (2 * 32 * 32 * 128), -2) == 25_200
    assert acc.attention_flops_per_token(32, 32, 128, 4096) == pytest.approx(6.44e9, rel=1e-3)


def test_chinchilla_table_of_the_course_page():
    rows = {
        1e21: (2.8e9, 2.306),
        1e22: (9.0e9, 2.141),
        1e23: (29.4e9, 2.032),
        1e25: (312.1e9, 1.912),
    }
    for c, (n_expected, l_expected) in rows.items():
        n, d = scaling.compute_optimal(c)
        assert n == pytest.approx(n_expected, rel=0.01)
        assert scaling.loss(n, d) == pytest.approx(l_expected, abs=0.001)
        assert 6 * n * d == pytest.approx(c)


def test_the_closed_form_is_the_minimum_found_numerically():
    c = 1e23
    n_star, _ = scaling.compute_optimal(c)
    sizes = np.logspace(9, 12, 20001)
    numeric = sizes[np.argmin(scaling.loss(sizes, c / (6 * sizes)))]
    assert numeric == pytest.approx(n_star, rel=2e-3)


def test_exponents_and_the_valley_is_flat():
    a, b = scaling.optimal_exponents()
    assert (round(a, 2), round(b, 2)) == (0.51, 0.49)
    n, d = scaling.compute_optimal(1e22)
    for factor in (0.5, 2.0):
        worse = scaling.loss(n * factor, 1e22 / (6 * n * factor)) - scaling.loss(n, d)
        assert 0 < worse < 0.02


def test_serving_changes_the_best_model_as_the_course_table_says():
    expected = {
        0: (46.9, 0.88),
        1e12: (34.3, 1.25),
        1e13: (20.0, 3.01),
        1e14: (12.7, 11.12),
        1e15: (9.6, 50.60),
    }
    for served, (n_b, d_t) in expected.items():
        _, n, d = scaling.best_model(2.0, served)
        assert round(n / 1e9, 1) == n_b and round(d / 1e12, 2) == d_t


def test_saved_tensor_bytes_of_a_swiglu_is_the_page_formula():
    import torch

    from dmxsf.architecture.layers import SwiGLU

    h, f, tokens = 64, 172, 128
    mlp = SwiGLU(h, f).to(torch.bfloat16)
    x = torch.randn(tokens, h, dtype=torch.bfloat16, requires_grad=True)
    saved = acc.saved_tensor_bytes(lambda: mlp(x), mlp.parameters())
    assert saved.total / tokens == 2 * h + 8 * f
    assert saved.tensors == 5


def test_matmul_params_leaves_out_the_input_table_and_the_norm_gains():
    from dmxsf.architecture.gpt import GPT, GPTConfig

    kw = dict(vocab_size=100, n_layer=2, n_head=4, d_model=32, d_ff=64, norm="rmsnorm", pos="rope")
    kw |= dict(mlp="swiglu", bias=False)
    untied = GPT(GPTConfig(**kw, tie_embeddings=False))
    norms = 2 * 2 * 32 + 32
    assert acc.matmul_params(untied) == untied.num_params() - 100 * 32 - norms
    tied = GPT(GPTConfig(**kw, tie_embeddings=True))
    assert acc.matmul_params(tied) == tied.num_params() - norms  # the shared matrix counts once


def test_recompute_changes_neither_the_loss_nor_the_gradients():
    import torch

    from dmxsf.architecture.gpt import GPT, GPTConfig

    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=50, n_layer=3, n_head=2, d_model=16, norm="rmsnorm", pos="rope")
    model = GPT(cfg)
    ids = torch.randint(0, 50, (2, 12))
    grads = {}
    for mode in (None, "reentrant", "non-reentrant"):
        model.recompute = mode
        model.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(ids).reshape(-1, 50), ids.reshape(-1))
        loss.backward()
        grads[mode] = (loss.item(), [p.grad.clone() for p in model.parameters()])
    for mode in ("reentrant", "non-reentrant"):
        assert grads[mode][0] == grads[None][0]
        for a, b in zip(grads[mode][1], grads[None][1], strict=True):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_fit_recovers_the_chinchilla_law_from_exact_samples():
    n = np.repeat(np.logspace(7, 9.5, 6), 6)
    d = np.tile(np.logspace(9, 11.5, 6), 6)
    fitted = scaling.fit(n, d, scaling.loss(n, d))
    assert fitted["rms"] < 1e-4
    assert fitted["alpha"] == pytest.approx(scaling.ALPHA, abs=1e-3)
    assert fitted["beta"] == pytest.approx(scaling.BETA, abs=1e-3)
    assert fitted["E"] == pytest.approx(scaling.E, abs=1e-2)
    assert fitted["A"] == pytest.approx(scaling.A, rel=2e-2)


def test_fit_with_noise_predicts_better_than_it_identifies():
    rng = np.random.default_rng(0)
    n = np.repeat(np.logspace(7, 9.5, 6), 6)
    d = np.tile(np.logspace(9, 11.5, 6), 6)
    noisy = scaling.loss(n, d) + rng.normal(0, 0.01, n.size)
    fitted = scaling.fit(n, d, noisy)
    predicted = scaling.loss(
        1e9, 1e12, fitted["E"], fitted["A"], fitted["B"], fitted["alpha"], fitted["beta"]
    )
    assert predicted == pytest.approx(scaling.loss(1e9, 1e12), abs=0.02)
