import pytest
import torch
import torch.nn.functional as F

from dmxsf.architecture.attention import Attention, attention, causal_mask, repeat_kv


def test_manual_attention_matches_torch_sdpa():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 9, 16, dtype=torch.float64) for _ in range(3))
    out, _ = attention(q, k, v, mask=causal_mask(9, 9))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(out, ref, atol=1e-12)


def test_weights_are_a_distribution_and_the_future_gets_none():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 6, 8) for _ in range(3))
    _, w = attention(q, k, v, mask=causal_mask(6, 6))
    assert torch.allclose(w.sum(-1), torch.ones(1, 2, 6), atol=1e-6)
    assert torch.all(w.triu(diagonal=1) == 0)


def test_causal_mask_alignment_with_a_cache():
    assert causal_mask(3, 3).int().tolist() == [[1, 0, 0], [1, 1, 0], [1, 1, 1]]
    # two new queries after three cached tokens: they see the cache plus themselves
    assert causal_mask(2, 5).int().tolist() == [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]
    assert causal_mask(1, 5).all()


def test_without_positions_attention_cannot_tell_the_order_of_tokens():
    """Why Transformers need position information: attention is permutation-equivariant."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(7, 8) for _ in range(3))
    perm = torch.randperm(7)
    out, _ = attention(q, k, v)
    out_perm, _ = attention(q[perm], k[perm], v[perm])
    assert torch.allclose(out[perm], out_perm, atol=1e-6)


def test_attention_scale_matters():
    torch.manual_seed(0)
    q, k = torch.randn(2, 5, 64), torch.randn(2, 5, 64)
    v = torch.randn(2, 5, 64)
    _, scaled = attention(q, k, v)
    _, unscaled = attention(q, k, v, scale=1.0)
    entropy = lambda w: float(-(w * w.clamp_min(1e-12).log()).sum(-1).mean())  # noqa: E731
    assert entropy(unscaled) < entropy(scaled) - 0.5  # without 1/sqrt(d) the softmax saturates


def test_attention_module_is_causal():
    torch.manual_seed(0)
    attn = Attention(32, 4).eval()
    x = torch.randn(1, 10, 32)
    y = attn(x)
    x2 = x.clone()
    x2[:, 6:] = torch.randn(1, 4, 32)  # change the future
    y2 = attn(x2)
    assert torch.allclose(y[:, :6], y2[:, :6], atol=1e-6)
    assert not torch.allclose(y[:, 6:], y2[:, 6:], atol=1e-3)


def test_gqa_equals_mha_when_kv_heads_are_shared():
    torch.manual_seed(0)
    d, heads, kv = 32, 8, 2
    gqa = Attention(d, heads, kv).eval()
    mha = Attention(d, heads, heads).eval()
    hd = d // heads
    with torch.no_grad():  # give MHA the GQA weights, each kv head repeated for its group
        mha.q_proj.weight.copy_(gqa.q_proj.weight)
        mha.o_proj.weight.copy_(gqa.o_proj.weight)
        for name in ("k_proj", "v_proj"):
            w = getattr(gqa, name).weight.view(kv, hd, d).repeat_interleave(heads // kv, dim=0)
            getattr(mha, name).weight.copy_(w.reshape(heads * hd, d))
    x = torch.randn(2, 7, d)
    assert torch.allclose(gqa(x), mha(x), atol=1e-5)


def test_gqa_shrinks_only_the_kv_projections():
    d = 64
    sizes = {}
    for kv in (8, 2, 1):
        a = Attention(d, 8, kv)
        sizes[kv] = (a.k_proj.weight.numel(), a.q_proj.weight.numel())
    assert sizes[8][0] == 4 * sizes[2][0] == 8 * sizes[1][0]
    assert sizes[8][1] == sizes[2][1] == sizes[1][1]
    with pytest.raises(ValueError):
        Attention(d, 8, 3)


def test_repeat_kv_layout():
    x = torch.arange(2 * 2 * 3 * 1.0).view(1, 2, 3, 2)
    y = repeat_kv(x, 3)
    assert y.shape == (1, 6, 3, 2)
    assert torch.equal(y[0, 0], y[0, 2]) and torch.equal(y[0, 3], y[0, 5])
    assert not torch.equal(y[0, 0], y[0, 3])


def test_sdpa_and_manual_paths_agree_for_every_shape_of_call():
    torch.manual_seed(0)
    attn = Attention(32, 4, 2).eval()
    x = torch.randn(2, 6, 32)
    assert torch.allclose(attn(x, impl="sdpa"), attn(x, impl="manual"), atol=1e-5)
    out, weights = attn(x, return_weights=True)
    assert weights.shape == (2, 4, 6, 6) and torch.allclose(out, attn(x), atol=1e-5)


def test_sliding_window_mask():
    assert causal_mask(5, 5, window=2).int().tolist() == [
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
    ]
    # the last query of a cached chunk still sees exactly `window` keys
    assert causal_mask(2, 6, window=3).sum(dim=1).tolist() == [3, 3]
    assert causal_mask(4, 4, window=100).equal(causal_mask(4, 4))  # a huge window is plain causal


@pytest.mark.parametrize("t_q,t_k", [(1, 1), (7, 7), (200, 200), (5, 5), (64, 64), (3, 70)])
@pytest.mark.parametrize("causal", [True, False])
def test_tiled_attention_is_exact_for_any_block_size(t_q, t_k, causal):
    from dmxsf.architecture.attention import tiled_attention

    if causal and t_q > t_k:
        pytest.skip("a causal query needs at least as many keys as queries")
    torch.manual_seed(0)
    q = torch.randn(2, 3, t_q, 16, dtype=torch.float64) * 3  # large scores: a sharp softmax
    k, v = (torch.randn(2, 3, t_k, 16, dtype=torch.float64) * 3 for _ in range(2))
    mask = causal_mask(t_q, t_k) if causal else None
    expected, _ = attention(q, k, v, mask=mask)
    for block in (1, 5, 64, 1000):
        got = tiled_attention(q, k, v, block_q=block, block_k=block + 3, causal=causal)
        assert torch.allclose(got, expected, atol=1e-10), block


def test_tiled_attention_has_the_same_gradients():
    from dmxsf.architecture.attention import tiled_attention

    torch.manual_seed(0)
    inputs = [torch.randn(1, 2, 37, 8, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    reference = attention(*inputs, mask=causal_mask(37, 37))[0].pow(2).sum()
    expected = torch.autograd.grad(reference, inputs)
    got = torch.autograd.grad(tiled_attention(*inputs, 8, 8).pow(2).sum(), inputs)
    for a, b in zip(got, expected, strict=True):
        assert torch.allclose(a, b, atol=1e-9)


def test_pooling_kv_heads_is_exact_when_the_heads_in_a_group_agree():
    from dmxsf.architecture.attention import pool_kv_heads

    torch.manual_seed(0)
    mha = Attention(32, 4, head_dim=8)
    with torch.no_grad():  # make heads (0, 1) and (2, 3) identical
        for proj in (mha.k_proj, mha.v_proj):
            w = proj.weight.view(4, 8, 32)
            w[1], w[3] = w[0].clone(), w[2].clone()
    gqa = pool_kv_heads(mha, 2)
    assert gqa.n_kv_heads == 2 and gqa.k_proj.weight.shape == (16, 32)
    x = torch.randn(2, 9, 32)
    assert torch.allclose(mha(x), gqa(x), atol=1e-6)


def test_pooling_averages_the_heads_of_a_group():
    from dmxsf.architecture.attention import pool_kv_heads

    mha = Attention(8, 4, head_dim=2, qkv_bias=True)
    with torch.no_grad():
        mha.k_proj.weight.copy_(torch.arange(8 * 8, dtype=torch.float32).view(8, 8))
        mha.k_proj.bias.copy_(torch.arange(8, dtype=torch.float32))
    gqa = pool_kv_heads(mha, 1)  # MQA: one head, the mean of all four
    assert torch.allclose(gqa.k_proj.weight, mha.k_proj.weight.view(4, 2, 8).mean(dim=0))
    assert torch.allclose(gqa.k_proj.bias, mha.k_proj.bias.view(4, 2).mean(dim=0))
    with pytest.raises(ValueError):
        pool_kv_heads(mha, 3)
