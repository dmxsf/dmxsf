import time

import pytest
import torch

from dmxsf.architecture.deepseek import DEEPSEEK_V3, DeepSeekConfig, DeepSeekLM
from dmxsf.architecture.moe import active_params, moe_layers


def formula(c: DeepSeekConfig) -> int:
    """The closed form used on the course page, plus the norm weights."""
    mla = (
        c.d_model * c.q_rank
        + c.q_rank * c.n_heads * (c.d_nope + c.d_rope)
        + c.d_model * (c.d_c + c.d_rope)
        + c.d_c * c.n_heads * (c.d_nope + c.d_v)
        + c.n_heads * c.d_v * c.d_model
    )
    expert = 3 * c.d_model * c.d_expert
    norms = c.n_layer * (2 * c.d_model + c.q_rank + c.d_c) + c.d_model
    dense = c.first_dense * 3 * c.d_model * c.d_dense
    moe = c.n_moe_layers * ((c.n_experts + c.n_shared) * expert + c.d_model * c.n_experts)
    return c.n_layer * mla + norms + dense + moe + 2 * c.vocab_size * c.d_model


def tiny(**kw):
    torch.manual_seed(0)
    cfg = DeepSeekConfig(n_group=4, topk_group=2, top_k=4, n_experts=8, route_scale=2.5, **kw)
    return DeepSeekLM(cfg).eval()


def test_the_model_runs_and_only_the_first_layers_are_dense():
    model = tiny()
    logits = model(torch.randint(0, 256, (2, 12)))
    assert logits.shape == (2, 12, 256)
    assert len(moe_layers(model)) == model.cfg.n_moe_layers == 3


def test_parameter_count_matches_the_closed_form():
    for kw in ({}, {"first_dense": 2, "n_layer": 5, "q_rank": 24, "n_shared": 2}):
        model = tiny(**kw)
        assert model.num_params() == formula(model.cfg)


def test_the_model_is_causal():
    model = tiny()
    x = torch.randint(0, 256, (1, 10))
    y = x.clone()
    y[:, 7:] = torch.randint(0, 256, (1, 3))
    with torch.no_grad():
        a, b = model(x), model(y)
    assert torch.allclose(a[:, :7], b[:, :7], atol=1e-5)


def test_decoding_from_the_latent_cache_matches_the_full_pass():
    model = tiny()
    x = torch.randint(0, 256, (2, 14))
    with torch.no_grad():
        full = model(x)
        cache = model.new_cache()
        steps = [model(x[:, :6], cache, absorb=True)]
        steps += [model(x[:, t : t + 1], cache, absorb=True) for t in range(6, 14)]
    assert torch.allclose(torch.cat(steps, dim=1), full, atol=1e-4)
    # d_c + d_rope numbers per token and layer, whatever the number of heads
    assert cache.numbers_per_token() == model.cfg.n_layer * (model.cfg.d_c + model.cfg.d_rope)


def test_deepseek_v3_has_the_parameter_counts_of_the_course_page():
    start = time.perf_counter()
    with torch.device("meta"):
        model = DeepSeekLM(DEEPSEEK_V3)
    total = model.num_params()
    assert total == formula(DEEPSEEK_V3)
    assert round(total / 1e9, 2) == 671.03
    assert round(active_params(model) / 1e9, 2) == 37.55
    assert time.perf_counter() - start < 60


@pytest.mark.parametrize("n_group,topk_group", [(8, 4)])
def test_the_v3_config_allows_node_limited_routing(n_group, topk_group):
    assert (DEEPSEEK_V3.n_group, DEEPSEEK_V3.topk_group) == (n_group, topk_group)
    assert DEEPSEEK_V3.top_k <= topk_group * DEEPSEEK_V3.n_experts // n_group
