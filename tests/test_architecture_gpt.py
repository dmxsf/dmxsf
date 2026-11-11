import math

import pytest
import torch
import torch.nn.functional as F

from dmxsf.architecture.configs import PUBLIC_MODELS, small_gpt
from dmxsf.architecture.gpt import GPT, POSITIONS, GPTConfig
from dmxsf.inference.kv_cache import KVCache, StaticKVCache

# ------------------------------------------------------------- parameter counts


def test_the_course_page_model_has_858880_parameters():
    assert GPT(small_gpt()).num_params() == 858_880


@pytest.mark.parametrize("model", PUBLIC_MODELS, ids=lambda m: m.name)
def test_public_models_have_exactly_their_published_parameter_count(model):
    """A from-scratch implementation that matches GPT-2, Llama 3, Qwen2.5 ... to the last parameter."""
    with torch.device("meta"):  # shapes only: the 70B model costs no memory
        net = GPT(model.config)
    assert net.num_params() == model.params


def test_tied_embeddings_share_one_matrix():
    tied, untied = GPT(small_gpt(tie_embeddings=True)), GPT(small_gpt(tie_embeddings=False))
    assert tied.lm_head.weight is tied.tok_emb.weight
    assert untied.num_params() - tied.num_params() == 256 * 128


def test_gqa_only_removes_key_and_value_projections():
    kwargs = dict(bias=False, norm="rmsnorm")
    mha = GPT(small_gpt(**kwargs)).num_params()
    mqa = GPT(small_gpt(n_kv_head=1, **kwargs)).num_params()
    d, head_dim, layers = 128, 32, 4  # 4 heads of 32; k and v each lose 3 of 4 heads
    assert mha - mqa == layers * 2 * d * 3 * head_dim


def test_post_norm_has_no_final_norm_and_swiglu_default_width():
    pre, post = GPT(small_gpt()), GPT(small_gpt(pre_norm=False))
    assert pre.num_params() - post.num_params() == 2 * 128  # the final LayerNorm's weight and bias
    assert GPTConfig(d_model=128, mlp="swiglu").d_ff == 384  # 8/3 * 128 = 341.3, rounded up to 64s
    assert GPTConfig(d_model=128).d_ff == 512


def test_invalid_configs_are_rejected():
    with pytest.raises(ValueError):
        GPTConfig(pos="sine")
    with pytest.raises(ValueError):
        GPTConfig(mlp="relu")
    with pytest.raises(ValueError):
        GPTConfig(pos="rope", d_model=36, n_head=4, head_dim=9)


# ------------------------------------------------------------------ behaviour


def _tiny(**kw):
    base = dict(n_layer=2, d_model=32, n_head=4, block_size=48, init_std=0.2)
    return small_gpt(**{**base, **kw})


@pytest.mark.parametrize("pos", POSITIONS)
def test_the_model_is_causal(pos):
    torch.manual_seed(0)
    model = GPT(_tiny(pos=pos)).eval()
    idx = torch.randint(0, 256, (2, 16))
    other = idx.clone()
    other[:, 10:] = torch.randint(0, 256, (2, 6))
    a, b = model(idx), model(other)
    assert torch.allclose(a[:, :10], b[:, :10], atol=1e-5)
    assert not torch.allclose(a[:, 10:], b[:, 10:], atol=1e-3)


@pytest.mark.parametrize("pos", POSITIONS)
@pytest.mark.parametrize("cache_kind", ["dynamic", "static"])
@torch.no_grad()
def test_incremental_decoding_equals_the_full_forward_pass(pos, cache_kind):
    torch.manual_seed(0)
    cfg = _tiny(pos=pos, n_kv_head=2, norm="rmsnorm", mlp="swiglu", bias=False)
    model = GPT(cfg).eval()
    idx = torch.randint(0, 256, (2, 24))
    full = model(idx)

    def new_cache():
        if cache_kind == "dynamic":
            return KVCache(cfg.n_layer)
        return StaticKVCache(cfg.n_layer, 2, cfg.kv_heads, cfg.dim_per_head, 48)

    cache = new_cache()  # prefill 10 tokens, then one token at a time
    steps = [model(idx[:, :10], cache)] + [model(idx[:, t : t + 1], cache) for t in range(10, 24)]
    assert cache.length == 24
    assert torch.allclose(torch.cat(steps, dim=1), full, atol=1e-4)

    cache = new_cache()  # chunked prefill: queries see the cached prefix and each other causally
    chunks = [model(idx[:, i : i + 8], cache) for i in range(0, 24, 8)]
    assert torch.allclose(torch.cat(chunks, dim=1), full, atol=1e-4)


@pytest.mark.parametrize("pos", ["rope", "alibi", "learned"])
@torch.no_grad()
def test_manual_and_fused_attention_agree(pos):
    torch.manual_seed(0)
    model = GPT(_tiny(pos=pos, n_kv_head=2)).eval()
    idx = torch.randint(0, 256, (2, 20))
    fused = model(idx)
    model.attn_impl = "manual"
    assert torch.allclose(model(idx), fused, atol=1e-4)


def test_an_untrained_model_predicts_uniformly():
    torch.manual_seed(0)
    model = GPT(small_gpt())
    idx = torch.randint(0, 256, (8, 65))
    loss = F.cross_entropy(model(idx[:, :-1]).reshape(-1, 256), idx[:, 1:].reshape(-1))
    assert loss.item() == pytest.approx(math.log(256), abs=0.1)


def test_a_tiny_model_can_memorise_one_batch():
    torch.manual_seed(0)
    model = GPT(small_gpt(n_layer=2, d_model=64, block_size=32, pos="rope", norm="rmsnorm"))
    batch = torch.randint(0, 256, (4, 33))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    losses = []
    for _ in range(150):
        loss = F.cross_entropy(model(batch[:, :-1]).reshape(-1, 256), batch[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[0] > 5.0 and losses[-1] < 1.0


def test_rope_runs_past_the_training_length_but_learned_positions_cannot():
    rope = GPT(_tiny(pos="rope", block_size=16))
    assert rope(torch.randint(0, 256, (1, 64))).shape == (1, 64, 256)
    learned = GPT(_tiny(pos="learned", block_size=16))
    with pytest.raises(ValueError):
        learned(torch.randint(0, 256, (1, 17)))


@torch.no_grad()
def test_rope_scaling_changes_long_range_scores_but_not_the_first_token():
    torch.manual_seed(0)
    model = GPT(_tiny(pos="rope")).eval()
    idx = torch.randint(0, 256, (1, 32))
    base = model(idx)
    model.set_rope_scaling("linear", 4.0)
    scaled = model(idx)
    assert torch.allclose(base[:, 0], scaled[:, 0], atol=1e-5)  # position 0 rotates by zero
    assert not torch.allclose(base[:, 8:], scaled[:, 8:], atol=1e-3)
    model.set_rope_scaling("none")
    assert torch.allclose(model(idx), base, atol=1e-6)


def test_bfloat16_forward_keeps_rope_frequencies_in_float64():
    model = GPT(_tiny(pos="rope")).to(torch.bfloat16)
    out = model(torch.randint(0, 256, (1, 8)))
    assert out.dtype == torch.bfloat16 and model.inv_freq.dtype == torch.float64


@pytest.mark.parametrize("pos", ["learned", "rope", "alibi"])
@torch.no_grad()
def test_attention_maps_are_causal_distributions_of_the_right_shape(pos):
    torch.manual_seed(0)
    model = GPT(_tiny(pos=pos, n_kv_head=2)).eval()
    idx = torch.randint(0, 256, (3, 12))
    maps = model.attention_maps(idx)
    assert len(maps) == 2 and all(m.shape == (3, 4, 12, 12) for m in maps)
    for m in maps:
        assert torch.allclose(m.sum(-1), torch.ones(3, 4, 12), atol=1e-5)
        assert torch.all(m.triu(diagonal=1) == 0)  # nothing is read from the future


@torch.no_grad()
def test_post_norm_blocks_also_return_their_attention():
    model = GPT(_tiny(pre_norm=False)).eval()
    idx = torch.randint(0, 256, (1, 6))
    assert model.attention_maps(idx)[0].shape == (1, 4, 6, 6)
    assert model(idx).shape == (1, 6, 256)


def _meta(name: str):
    from dmxsf.architecture.configs import BY_NAME

    with torch.device("meta"):
        return GPT(BY_NAME[name].config), BY_NAME[name].config


def test_where_the_parameters_go_matches_the_course_page():
    from dmxsf.architecture.configs import param_breakdown

    def shares(name):
        net, _ = _meta(name)
        parts = param_breakdown(net)
        total = sum(parts.values())
        assert total == net.num_params()
        return {k: 100 * v / total for k, v in parts.items()}

    llama3 = shares(
        "Llama 3 8B"
    )  # 70.2% feed-forward, 16.7% attention, 13.1% two vocabulary matrices
    assert round(llama3["mlp"], 1) == 70.2 and round(llama3["attention"], 1) == 16.7
    assert round(llama3["embedding"] + llama3["head"], 1) == 13.1
    assert llama3["norm"] < 0.01
    llama2 = shares("Llama 2 7B")  # MHA: attention 31.9%, small vocabulary 3.9%
    assert (
        round(llama2["attention"], 1) == 31.9
        and round(llama2["embedding"] + llama2["head"], 1) == 3.9
    )
    llama70 = shares("Llama 3 70B")
    assert (
        round(llama70["mlp"], 1) == 79.9 and round(llama70["embedding"] + llama70["head"], 1) == 3.0
    )


def test_kv_cache_per_token_matches_the_course_page():
    from dataclasses import replace

    from dmxsf.architecture.configs import BY_NAME, kv_bytes_per_token

    kib = {n: kv_bytes_per_token(BY_NAME[n].config) / 1024 for n in BY_NAME}
    assert (kib["Llama 2 7B"], kib["Llama 3 8B"], kib["Llama 3 70B"]) == (512, 128, 320)
    mha_70b = replace(BY_NAME["Llama 3 70B"].config, n_kv_head=64)
    assert kv_bytes_per_token(mha_70b) / 1024 == 2560


def test_sliding_window_model_only_sees_the_window():
    """Changing a token more than window - 1 positions back cannot change a layer-1 output."""
    torch.manual_seed(0)
    cfg = small_gpt(n_layer=1, pos="rope", norm="rmsnorm", mlp="swiglu", window=4)
    model = GPT(cfg).eval()
    idx = torch.randint(0, 256, (1, 12))
    other = idx.clone()
    other[0, 0] = (other[0, 0] + 1) % 256
    with torch.no_grad():
        a, b = model(idx), model(other)
    assert not torch.allclose(a[0, :4], b[0, :4])  # positions 0..3 can see token 0
    assert torch.allclose(a[0, 4:], b[0, 4:], atol=1e-6)  # position 4 onwards cannot


def test_the_receptive_field_grows_by_window_minus_one_per_layer():
    torch.manual_seed(0)
    window, layers, t = 3, 3, 14
    cfg = small_gpt(n_layer=layers, pos="rope", norm="rmsnorm", mlp="swiglu", window=window)
    model = GPT(cfg).eval()
    idx = torch.randint(0, 256, (1, t))
    other = idx.clone()
    other[0, 0] = (other[0, 0] + 1) % 256
    with torch.no_grad():
        changed = (model(idx) - model(other)).abs().amax(dim=-1)[0] > 1e-7
    assert changed.nonzero().max().item() == layers * (window - 1)  # last position that still moves


@pytest.mark.parametrize("window", [1, 4, 16])
def test_sliding_cache_decoding_equals_the_full_forward_pass(window):
    from dmxsf.inference.kv_cache import SlidingKVCache

    torch.manual_seed(0)
    cfg = small_gpt(n_layer=2, pos="rope", norm="rmsnorm", mlp="swiglu", window=window)
    model = GPT(cfg).eval()
    idx = torch.randint(0, 256, (2, 21))
    with torch.no_grad():
        full = model(idx)
        cache = SlidingKVCache(cfg.n_layer, window)
        steps = [
            model(idx[:, :9], cache=cache)
        ]  # a prefill longer than the window, then one by one
        steps += [model(idx[:, i : i + 1], cache=cache) for i in range(9, 21)]
    assert torch.allclose(torch.cat(steps, dim=1), full, atol=1e-5)
    assert cache.length == 21
    assert cache.k[0].shape[2] == window - 1  # memory stays bounded


def test_converting_to_gqa_keeps_a_working_model_with_a_smaller_cache():
    from dmxsf.architecture.gpt import to_gqa

    torch.manual_seed(0)
    mha = GPT(small_gpt(n_layer=2, n_head=4, d_model=64, pos="rope", norm="rmsnorm", bias=False))
    gqa = to_gqa(mha, 2)
    assert gqa.cfg.kv_heads == 2 and gqa.num_params() < mha.num_params()
    idx = torch.randint(0, 256, (1, 10))
    assert gqa(idx).shape == (1, 10, 256)
    assert torch.equal(gqa.blocks[0].attn.q_proj.weight, mha.blocks[0].attn.q_proj.weight)
    assert torch.equal(gqa.tok_emb.weight, mha.tok_emb.weight)
