"""The three KV caches: growing, preallocated and sliding-window
(https://dmxsf.com/wiki/inference/kv-cache/)."""

import pytest
import torch

from dmxsf.inference.kv_cache import KVCache, SlidingKVCache, StaticKVCache

B, H, D = 2, 3, 4  # batch, KV heads, head dim


def kv(t: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, H, t, D, generator=g), torch.randn(B, H, t, D, generator=g)


def test_growing_cache_concatenates_along_the_time_axis() -> None:
    cache = KVCache(n_layers=2)
    assert cache.length == 0
    k1, v1 = kv(3, seed=1)
    k2, v2 = kv(2, seed=2)
    cache.update(0, k1, v1)
    k_all, v_all = cache.update(0, k2, v2)
    assert cache.length == 5
    assert torch.equal(k_all, torch.cat([k1, k2], dim=2))
    assert torch.equal(v_all, torch.cat([v1, v2], dim=2))


def test_crop_forgets_the_rejected_tail() -> None:
    cache = KVCache(n_layers=1)
    k, v = kv(5)
    cache.update(0, k, v)
    cache.crop(3)
    assert cache.length == 3
    assert torch.equal(cache.k[0], k[:, :, :3])


def test_reorder_makes_a_beam_continue_another_beams_history() -> None:
    cache = KVCache(n_layers=1)
    k, v = kv(4)
    cache.update(0, k, v)
    cache.reorder(torch.tensor([1, 1]))
    assert torch.equal(cache.k[0][0], k[1])
    assert torch.equal(cache.k[0][1], k[1])


def test_nbytes_counts_keys_and_values_of_every_layer() -> None:
    cache = KVCache(n_layers=2)
    for layer in range(2):
        cache.update(layer, *kv(6, seed=layer))
    assert cache.nbytes() == 2 * 2 * B * H * 6 * D * 4  # layers x (K and V) x elements x float32


def test_static_cache_returns_the_same_views_as_the_growing_one() -> None:
    grow = KVCache(n_layers=1)
    static = StaticKVCache(1, B, H, D, max_len=10)
    for t, seed in ((4, 1), (1, 2), (1, 3)):
        k, v = kv(t, seed)
        k_g, v_g = grow.update(0, k, v)
        k_s, v_s = static.update(0, k, v)
        assert torch.equal(k_g, k_s) and torch.equal(v_g, v_s)
    assert static.length == grow.length == 6
    assert static.nbytes() == 2 * 1 * B * H * 10 * D * 4  # preallocated for max_len, not for 6


def test_static_cache_refuses_to_overflow() -> None:
    static = StaticKVCache(1, B, H, D, max_len=4)
    static.update(0, *kv(4))
    with pytest.raises(ValueError):
        static.update(0, *kv(1))


def test_sliding_cache_keeps_window_minus_one_tokens_but_counts_every_token() -> None:
    window = 4
    cache = SlidingKVCache(n_layers=1, window=window)
    steps = [kv(1, seed=i) for i in range(10)]
    for k, v in steps:
        cache.update(0, k, v)
    assert cache.length == 10  # positions keep advancing
    assert cache.k[0].shape[2] == window - 1  # memory stopped growing
    assert torch.equal(cache.k[0], torch.cat([k for k, _ in steps[-(window - 1) :]], dim=2))
