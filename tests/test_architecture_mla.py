import torch

from dmxsf.architecture.mla import MLACache, MLAttention
from dmxsf.architecture.position import rope_cos_sin, rope_inv_freq


def make(d_model=64, n_heads=4, d_nope=16, d_v=16, d_c=24, d_rope=8, seed=0):
    torch.manual_seed(seed)
    return MLAttention(d_model, n_heads, d_nope, d_v, d_c, d_rope).eval()


def rope_for(layer, start, t):
    inv_freq = rope_inv_freq(layer.d_rope)
    return rope_cos_sin(torch.arange(start, start + t), inv_freq)


def test_absorbed_attention_equals_rebuilding_keys_and_values():
    layer = make()
    x = torch.randn(2, 11, 64)
    with torch.no_grad():
        explicit = layer(x, rope_for(layer, 0, 11), absorb=False)
        absorbed = layer(x, rope_for(layer, 0, 11), absorb=True)
    assert torch.allclose(explicit, absorbed, atol=1e-5)


def test_decoding_one_token_at_a_time_from_the_latent_cache_matches_the_full_pass():
    layer = make(seed=3)
    x = torch.randn(2, 13, 64)
    with torch.no_grad():
        full = layer(x, rope_for(layer, 0, 13))
        cache = MLACache(1)
        steps = [layer(x[:, :5], rope_for(layer, 0, 5), cache=cache, absorb=True)]
        for t in range(5, 13):
            steps.append(layer(x[:, t : t + 1], rope_for(layer, t, 1), cache=cache, absorb=True))
    assert torch.allclose(torch.cat(steps, dim=1), full, atol=1e-5)
    assert cache.length == 13


def test_the_cache_holds_d_c_plus_d_rope_numbers_per_token_and_does_not_depend_on_heads():
    for n_heads in (2, 8):
        layer = make(n_heads=n_heads)
        cache = MLACache(1)
        with torch.no_grad():
            layer(torch.randn(1, 10, 64), rope_for(layer, 0, 10), cache=cache)
        assert layer.cached_numbers_per_token() == 24 + 8
        assert cache.numbers_per_token() == 24 + 8
        assert cache.nbytes() == 10 * (24 + 8) * 4  # float32


def test_attention_is_causal():
    layer = make(seed=5)
    x = torch.randn(1, 9, 64)
    y = x.clone()
    y[:, 6:] = torch.randn(1, 3, 64)
    with torch.no_grad():
        a, b = layer(x, rope_for(layer, 0, 9)), layer(y, rope_for(layer, 0, 9))
    assert torch.allclose(a[:, :6], b[:, :6], atol=1e-6)
    assert not torch.allclose(a[:, 6:], b[:, 6:])


def test_position_enters_only_through_the_decoupled_rope_key():
    """Shifting every position by the same amount changes nothing: RoPE is relative."""
    layer = make(seed=7)
    x = torch.randn(1, 8, 64)
    with torch.no_grad():
        a = layer(x, rope_for(layer, 0, 8), absorb=True)
        b = layer(x, rope_for(layer, 500, 8), absorb=True)
    assert torch.allclose(a, b, atol=1e-4)
