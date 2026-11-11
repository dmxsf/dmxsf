import math

import pytest
import torch

from dmxsf.architecture.position import (
    ROPE_SCALINGS,
    alibi_bias,
    alibi_slopes,
    apply_rope,
    rope_cos_sin,
    rope_inv_freq,
    rope_llama3,
    rope_ntk,
    rope_scaled_inv_freq,
    rope_yarn,
    sinusoidal_table,
)

# ---------------------------------------------------------------- sinusoidal


def test_sinusoidal_matches_the_formula():
    d, n = 16, 50
    table = sinusoidal_table(n, d)
    for pos in (0, 1, 7, 49):
        for i in range(d // 2):
            angle = pos / 10000 ** (2 * i / d)
            assert table[pos, 2 * i] == pytest.approx(math.sin(angle), abs=1e-6)
            assert table[pos, 2 * i + 1] == pytest.approx(math.cos(angle), abs=1e-6)


def test_sinusoidal_offset_is_a_rotation_of_each_pair():
    """PE(p + k) is a fixed linear map of PE(p): the property Vaswani et al. point out."""
    d, k = 32, 5
    table = sinusoidal_table(100, d).double()
    for i in range(d // 2):
        w = 1 / 10000 ** (2 * i / d)
        rot = torch.tensor(
            [[math.cos(w * k), math.sin(w * k)], [-math.sin(w * k), math.cos(w * k)]],
            dtype=torch.float64,
        )  # maps (sin(wp), cos(wp)) to (sin(w(p+k)), cos(w(p+k)))
        pair = table[:, 2 * i : 2 * i + 2]
        assert torch.allclose(pair[:90] @ rot.T, pair[k : 90 + k], atol=1e-5)


def test_sinusoidal_works_beyond_any_training_length():
    assert sinusoidal_table(100_000, 8).shape == (100_000, 8)


# ---------------------------------------------------------------------- RoPE


def test_rope_inner_product_depends_only_on_the_offset():
    """The course page's check: same q at every position, same k at every position."""
    torch.manual_seed(0)
    d, t = 64, 200
    q, k = torch.randn(d), torch.randn(d)
    cos, sin = rope_cos_sin(torch.arange(t), rope_inv_freq(d))
    qs = apply_rope(q.expand(t, d), cos, sin, interleaved=True)
    ks = apply_rope(k.expand(t, d), cos, sin, interleaved=True)
    scores = [float(qs[m] @ ks[n]) for m, n in [(3, 10), (13, 20), (103, 110), (190, 197)]]
    assert max(scores) - min(scores) < 1e-3
    assert scores[0] == pytest.approx(-7.5584, abs=2e-3)  # printed on the page
    assert float(qs[3] @ ks[11]) != pytest.approx(scores[0], abs=1e-2)  # another offset differs


def test_rope_pairings_differ_only_by_a_permutation():
    torch.manual_seed(0)
    x = torch.randn(10, 64)
    cos, sin = rope_cos_sin(torch.arange(10), rope_inv_freq(64))
    perm = torch.cat([torch.arange(0, 64, 2), torch.arange(1, 64, 2)])
    interleaved = apply_rope(x, cos, sin, interleaved=True)[:, perm]
    half = apply_rope(x[:, perm], cos, sin, interleaved=False)
    assert torch.allclose(interleaved, half, atol=1e-5)


def test_rope_rotates_without_changing_length_and_position_zero_is_identity():
    torch.manual_seed(0)
    x = torch.randn(5, 7, 32)
    cos, sin = rope_cos_sin(torch.arange(7) * 13, rope_inv_freq(32))
    assert torch.allclose(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1), atol=1e-4)
    cos0, sin0 = rope_cos_sin(torch.zeros(7), rope_inv_freq(32))
    assert torch.allclose(apply_rope(x, cos0, sin0), x)


# ------------------------------------------------------------ RoPE scaling

D, BASE, TRAINED, S = 128, 10_000.0, 4096, 8.0


def test_course_page_numbers_for_rope_scaling():
    freqs = rope_inv_freq(D, BASE)
    short = [i for i, f in enumerate(freqs.tolist()) if TRAINED * f < 2 * math.pi]
    assert (len(short), short[0], short[-1]) == (18, 46, 63)

    def worst_new_phase(new):  # largest angle at S*L relative to the largest angle seen in training
        return max(float(S * new[i] / freqs[i]) for i in short)

    assert worst_new_phase(freqs) == pytest.approx(8.00, abs=0.005)
    assert worst_new_phase(rope_scaled_inv_freq("linear", D, BASE, S, TRAINED)[0]) == pytest.approx(
        1.00, abs=0.005
    )
    assert worst_new_phase(rope_ntk(D, BASE, S)) == pytest.approx(1.75, abs=0.005)
    assert worst_new_phase(rope_yarn(D, BASE, S, TRAINED)[0]) == pytest.approx(1.00, abs=0.005)
    assert round(BASE * S ** (D / (D - 2))) == 82_685  # the NTK-aware base on the page


def test_yarn_segments_and_attention_factor():
    freqs = rope_inv_freq(D, BASE)
    scaled, factor = rope_yarn(D, BASE, S, TRAINED)
    ratio = scaled / freqs
    unchanged = int((ratio > 1 - 1e-9).sum())
    divided = int((ratio < 1 / S + 1e-9).sum())
    assert (unchanged, D // 2 - unchanged - divided, divided) == (21, 25, 18)
    assert factor == pytest.approx(1.208, abs=5e-4)  # 0.1 ln 8 + 1
    assert factor**2 == pytest.approx(1.46, abs=5e-3)  # what q.k gains, since both are scaled


def test_llama31_segments():
    """Base 500,000, d = 128, factor 8: 29 groups divided by 8, 6 blended, 29 untouched."""
    freqs = rope_inv_freq(D, 500_000.0)
    scaled = rope_llama3(freqs, factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0)
    ratio = scaled / freqs
    unchanged = int((ratio > 1 - 1e-9).sum())
    divided = int((ratio < 1 / 8 + 1e-9).sum())
    assert (divided, D // 2 - unchanged - divided, unchanged) == (29, 6, 29)


def test_scaling_never_speeds_a_frequency_up():
    base = rope_inv_freq(64, BASE)
    for kind in ROPE_SCALINGS:
        scaled, _ = rope_scaled_inv_freq(kind, 64, BASE, 4.0, 1024)
        assert bool((scaled <= base * (1 + 1e-9)).all()), kind
    assert torch.equal(rope_scaled_inv_freq("none", 64, BASE, 4.0, 1024)[0], base)
    with pytest.raises(ValueError):
        rope_scaled_inv_freq("magic", 64)


# --------------------------------------------------------------------- ALiBi


def test_alibi_slopes_follow_the_paper():
    assert torch.allclose(alibi_slopes(8), torch.tensor([2.0**-i for i in range(1, 9)]))
    twelve = alibi_slopes(12)
    expected = [2.0**-i for i in range(1, 9)] + [2**-0.5, 2**-1.5, 2**-2.5, 2**-3.5]
    assert torch.allclose(twelve, torch.tensor(expected))


def test_alibi_bias_penalises_distance_linearly():
    slopes = alibi_slopes(4)
    bias = alibi_bias(slopes, torch.arange(6), torch.arange(6))
    assert bias.shape == (4, 6, 6)
    assert torch.all(bias.diagonal(dim1=-2, dim2=-1) == 0)  # no penalty for looking at yourself
    assert bias[0, 5, 2] == pytest.approx(-3 * float(slopes[0]))
    assert bias[3, 5, 0] == pytest.approx(-5 * float(slopes[3]))
