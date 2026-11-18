"""The quantization page: the widget's numbers, the algebra, and the three calibrated methods."""

import itertools

import numpy as np
import pytest
import torch

from dmxsf.architecture.configs import small_gpt
from dmxsf.architecture.gpt import GPT
from dmxsf.inference.kv_cache import KVCache
from dmxsf.inference.quantization import (
    QuantKVCache,
    Target,
    awq,
    bits_per_byte_cached,
    collect_stats,
    effective_bits,
    fake_quant,
    gptq,
    kv_quant,
    output_error,
    rtn,
    smooth_scales,
    sqnr_db,
    widget_weights,
    with_activation_outlier,
    with_key_outlier,
)
from dmxsf.pretraining.train import bits_per_byte


def test_widget_numbers_from_the_page():
    x = widget_weights()
    sym = [round(sqnr_db(x, fake_quant(x, b, "tensor")), 1) for b in range(2, 9)]
    assert sym == [1.1, 9.6, 16.9, 23.5, 29.8, 35.6, 42.1]
    y = widget_weights(outlier=True)
    got = [round(sqnr_db(y, fake_quant(y, 4, g)), 1) for g in ("tensor", 128, 64, 16)]
    assert got == [3.7, 11.6, 14.7, 19.1]
    asym = fake_quant(x, 4, 128, symmetric=False)
    assert round(sqnr_db(x, fake_quant(x, 4, 128)), 1) == 18.1
    assert round(sqnr_db(x, asym), 1) == 19.6
    shapes = [(1, 1024)]
    assert effective_bits(4, shapes, 128) == 4.125
    assert round(effective_bits(4, shapes, 128, symmetric=False), 3) == 4.156
    assert effective_bits(4, shapes, 16) == 5.0


def test_error_is_at_most_half_a_step_and_rounding_is_idempotent():
    w = torch.randn(32, 64)
    for group in ("tensor", "row", 16):
        q = fake_quant(w, 4, group)
        g = w.reshape(1, -1) if group == "tensor" else w if group == "row" else w.reshape(-1, 16)
        step = g.abs().amax(1, keepdim=True) / 7
        err = (g - q.reshape(g.shape)).abs()
        assert (err <= step / 2 + 1e-6).all()
        assert torch.allclose(fake_quant(q, 4, group), q, atol=1e-6)
    assert fake_quant(w, 4, "row").unique(dim=1).shape[1] > 0
    assert len(fake_quant(w[:1], 4, "row").unique()) <= 15  # -7 .. 7
    assert len(fake_quant(w[:1], 4, "row", symmetric=False).unique()) <= 16


def test_every_bit_buys_about_six_decibels():
    x = widget_weights()
    values = [sqnr_db(x, fake_quant(x, b, "tensor")) for b in range(4, 9)]
    gaps = [b - a for a, b in itertools.pairwise(values)]
    assert all(5.5 < g < 7.0 for g in gaps)  # 6.6, 6.3, 5.8, 6.5 on the page's sample
    assert sum(gaps) / len(gaps) == pytest.approx(6.0, abs=0.5)


def test_smoothing_leaves_the_product_unchanged():
    x, w = torch.randn(10, 16), torch.randn(8, 16)
    x[:, 3] *= 30  # an outlier channel
    s = smooth_scales(x.abs().amax(0), w, 0.5)
    assert torch.allclose((x / s) @ (w * s).T, x @ w.T, atol=1e-4)
    assert (x / s).abs().amax(0).max() < x.abs().amax(0).max()


def correlated_problem(n=96, tokens=2048, seed=0):
    g = torch.Generator().manual_seed(seed)
    mix = torch.randn(n, n, generator=g) / n**0.5
    x = torch.randn(tokens, n, generator=g) @ mix  # correlated input channels
    w = torch.randn(24, n, generator=g)
    return w, x.T @ x / tokens, x.abs().mean(0)


def test_gptq_beats_round_to_nearest_on_the_output_error():
    w, h, a = correlated_problem()
    e_rtn = output_error(w, rtn(w, h, a, 3, 32), h)
    e_awq = output_error(w, awq(w, h, a, 3, 32), h)
    e_gptq = output_error(w, gptq(w, h, a, 3, 32), h)
    assert e_awq <= e_rtn + 1e-9  # alpha = 0 is RTN, so the search cannot do worse
    assert e_gptq < 0.8 * e_rtn


def test_gptq_with_a_huge_grid_is_the_identity():
    w, h, a = correlated_problem()
    assert torch.allclose(gptq(w, h, a, 16, 32), w, atol=1e-2)


def test_kv_quant_per_channel_keeps_the_ordinary_channels():
    torch.manual_seed(0)
    k = torch.randn(1, 2, 64, 16)
    k[..., 3] *= 40  # one channel is huge in every token
    others = [c for c in range(16) if c != 3]
    err_token = (kv_quant(k, 4) - k)[..., others].pow(2).mean()
    err_channel = (kv_quant(k, 4, per_channel=True) - k)[..., others].pow(2).mean()
    assert err_token > 0.5  # one scale per token rounds the ordinary channels to zero
    assert err_channel < err_token / 20
    assert kv_quant(k, 4, per_channel=True).shape == k.shape


def tiny():
    torch.manual_seed(0)
    return GPT(small_gpt(n_layer=2, block_size=64)).eval()


def test_target_restores_weights_and_skips_the_embedding():
    model = tiny()
    t = Target(model)
    assert len(t.linears) == 2 * 6  # q, k, v, o, fc, proj per block
    assert not any("lm_head" in n or "emb" in n for n in t.names)
    before = [m.weight.clone() for m in t.linears]
    with t.quantized(lambda i, w: fake_quant(w, 2, "row")):
        assert not torch.equal(t.linears[0].weight, before[0])
    assert all(torch.equal(m.weight, b) for m, b in zip(t.linears, before, strict=True))


def test_collected_statistics():
    model = tiny()
    t = Target(model)
    batches = [torch.randint(0, 256, (4, 32)) for _ in range(3)]
    h, mean, peak = collect_stats(t, batches)
    assert len(h) == len(mean) == len(peak) == len(t.linears)
    assert torch.allclose(h[0], h[0].T, atol=1e-5)
    assert torch.linalg.eigvalsh(h[0]).min() > -1e-4
    assert (peak[0] >= mean[0]).all()
    assert h[0].shape == (128, 128) and h[5].shape == (512, 512)  # q_proj input, mlp.proj input


def test_cached_evaluation_matches_the_plain_one_and_rounding_changes_it():
    model = tiny()
    data = np.random.default_rng(0).integers(0, 256, 20_000).astype(np.uint8)
    plain = bits_per_byte(model, data, 64, 32)
    same = bits_per_byte_cached(model, data, lambda: KVCache(2), 64, 32)
    assert same == pytest.approx(plain, abs=1e-4)
    untouched = bits_per_byte_cached(model, data, lambda: QuantKVCache(2), 64, 32)
    assert untouched == pytest.approx(plain, abs=1e-4)
    rounded = bits_per_byte_cached(model, data, lambda: QuantKVCache(2, 4, False, 4), 64, 32)
    assert rounded != pytest.approx(plain, abs=1e-6)


def test_constructed_outliers_leave_the_function_alone():
    model = tiny()
    x = torch.randint(0, 256, (2, 48))
    with torch.no_grad():
        want = model(x)
        loud = with_activation_outlier(model, 30.0, channel=17)
        keys = with_key_outlier(model, 30.0, channel=5)
        assert (loud(x) - want).abs().max() < 1e-3
        assert (keys(x) - want).abs().max() < 1e-3
        # ...and the outlier is really there
        seen = []
        handle = loud.blocks[0].attn.q_proj.register_forward_pre_hook(
            lambda m, a: seen.append(a[0].abs().amax((0, 1)))
        )
        loud(x)
        handle.remove()
        assert seen[0][17] > 5 * torch.cat([seen[0][:17], seen[0][18:]]).median()
        cache = KVCache(2)
        keys(x, cache=cache)
        k = cache.k[0]  # (batch, heads, tokens, head_dim)
        assert k[..., 5].abs().mean() > 5 * k[..., 6].abs().mean()
    assert not torch.equal(model.blocks[0].attn.k_proj.weight, keys.blocks[0].attn.k_proj.weight)
