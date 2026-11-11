import math

import pytest
import torch

from dmxsf.architecture.configs import small_gpt
from dmxsf.architecture.gpt import GPT
from dmxsf.architecture.layers import SwiGLU
from dmxsf.architecture.moe import (
    MoELayer,
    active_params,
    aux_loss,
    balance_loss,
    capacity,
    moe_layers,
    moeify,
    route,
)


def layer(gate="mixtral", n_experts=6, top_k=2, n_shared=0, capacity_factor=None, seed=0):
    torch.manual_seed(seed)
    moe = MoELayer(16, 24, n_experts, top_k, n_shared, gate, capacity_factor, route_scale=1.5)
    return moe.double().eval()


@pytest.mark.parametrize("gate", ["mixtral", "switch", "deepseek"])
@pytest.mark.parametrize("n_shared", [0, 1])
@pytest.mark.parametrize("capacity_factor", [None, 0.8])
def test_computing_only_the_chosen_tokens_equals_computing_everything(
    gate, n_shared, capacity_factor
):
    moe = layer(gate, n_shared=n_shared, capacity_factor=capacity_factor)
    x = torch.randn(3, 20, 16, dtype=torch.double)
    with torch.no_grad():
        assert torch.allclose(moe(x), moe.dense_reference(x), atol=1e-12)


def test_mixtral_gates_are_a_softmax_over_the_selected_logits_only():
    logits = torch.tensor([[2.0, 0.0, 1.0, -3.0]])
    r = route(logits, 2, "mixtral")
    assert r.experts.tolist() == [[0, 2]]
    expected = torch.softmax(torch.tensor([2.0, 1.0]), dim=0)
    assert torch.allclose(r.gates, expected[None])
    assert torch.allclose(r.gates.sum(-1), torch.ones(1))


def test_switch_gate_is_the_probability_over_all_experts():
    logits = torch.tensor([[2.0, 0.0, 1.0, -3.0]])
    r = route(logits, 1, "switch")
    assert r.experts.tolist() == [[0]]
    assert torch.allclose(r.gates, logits.softmax(-1)[:, :1])


def test_deepseek_gate_normalises_independent_sigmoid_scores():
    logits = torch.tensor([[2.0, 0.0, 1.0, -3.0]])
    r = route(logits, 2, "deepseek", route_scale=2.5)
    s = torch.sigmoid(torch.tensor([2.0, 1.0]))
    assert torch.allclose(r.gates, (s / s.sum() * 2.5)[None])


def test_the_bias_changes_who_is_selected_but_never_a_gate_weight():
    logits = torch.tensor([[2.0, 0.0, 1.0, -3.0]])
    plain = route(logits, 2, "deepseek")
    biased = route(logits, 2, "deepseek", bias=torch.tensor([-5.0, 0.0, 0.0, 0.0]))
    assert plain.experts.tolist() == [[0, 2]] and biased.experts.tolist() == [[2, 1]]
    s = torch.sigmoid(torch.tensor([1.0, 0.0]))  # the *unbiased* scores of experts 2 and 1
    assert torch.allclose(biased.gates, (s / s.sum())[None])


def test_capacity_follows_the_course_page():
    assert capacity(512, 2, 16, 1.25) == 80  # quiz question
    assert capacity(1024, 8, 64, 1.25) == 160  # exercise 2
    assert capacity(256, 2, 8, 1.25) == 80


def test_capacity_drops_the_latest_arrivals_of_an_overfull_expert():
    logits = torch.zeros(10, 3)
    logits[:, 0] = 5.0  # everybody wants expert 0
    r = route(logits, 1, "switch", cap=4)
    assert r.load.tolist() == [10, 0, 0]
    assert r.keep.squeeze(1).tolist() == [True] * 4 + [False] * 6
    assert int((~r.keep).sum()) == 6


def test_the_course_exercise_285_assignments_into_a_capacity_of_160_drops_125():
    logits = torch.zeros(285, 64)
    logits[:, 7] = 9.0
    r = route(logits, 1, "switch", cap=capacity(1024, 8, 64, 1.25))
    assert r.load[7] == 285 and int((~r.keep).sum()) == 125


def test_without_a_capacity_nothing_is_dropped():
    r = route(torch.randn(50, 8), 2, "mixtral", cap=None)
    assert r.keep.all() and r.load.sum() == 100


def test_the_balance_loss_is_one_when_the_assignments_are_even_whatever_the_probabilities():
    n = 4
    logits = torch.eye(n) * 3.0 + torch.randn(n, n) * 0.1  # token t picks expert t: f is uniform
    r = route(logits, 1, "switch")
    assert r.load.tolist() == [1, 1, 1, 1]
    assert torch.isclose(balance_loss(r), torch.tensor(1.0), atol=1e-6)  # N sum f_i P_i = sum P_i


def test_a_collapsed_router_has_the_largest_balance_loss_and_the_gradient_flows_through_p():
    logits = torch.zeros(32, 4)
    logits[:, 0] = 20.0
    logits.requires_grad_()
    r = route(logits, 1, "switch")
    loss = balance_loss(r)
    assert torch.isclose(loss, torch.tensor(4.0), atol=1e-4)  # everything on one expert: loss N
    loss.backward()
    assert logits.grad.abs().sum() > 0
    assert not r.load.requires_grad  # f comes from an argmax and carries no gradient


def test_a_larger_batch_reaches_more_experts_as_the_course_formula_says():
    e, k = 8, 2
    generator = torch.Generator().manual_seed(0)
    for batch, expected in ((1, 2.0), (4, 5.47), (16, 7.92)):
        formula = e * (1 - (1 - k / e) ** batch)
        assert round(formula, 2) == expected
        trials = [
            torch.randn(batch, e, generator=generator).topk(k, dim=-1).indices.unique().numel()
            for _ in range(3000)
        ]
        assert abs(sum(trials) / len(trials) - formula) < 0.06


def test_update_bias_pushes_loaded_experts_down_and_idle_ones_up():
    moe = layer("deepseek", n_experts=4)
    moe.load = torch.tensor([10, 0, 0, 6])  # mean 4
    moe.update_bias(0.1)
    assert torch.allclose(moe.route_bias, torch.tensor([-0.1, 0.1, 0.1, -0.1], dtype=torch.double))


def test_one_expert_and_k_one_is_a_dense_feed_forward():
    torch.manual_seed(0)
    moe = MoELayer(16, 24, n_experts=1, top_k=1).double()
    x = torch.randn(2, 5, 16, dtype=torch.double)
    assert torch.allclose(moe(x), moe.experts[0](x), atol=1e-12)


def test_experts_and_router_all_get_gradients():
    moe = layer("mixtral", n_experts=4, top_k=2).train()
    x = torch.randn(2, 64, 16, dtype=torch.double)
    (moe(x).pow(2).sum() + moe.aux_loss).backward()
    assert moe.router.weight.grad.abs().sum() > 0
    assert all(
        e.gate.weight.grad.abs().sum() > 0 for e in moe.experts
    )  # 128 assignments, 4 experts


def test_moeify_adds_the_parameters_of_the_extra_experts_and_the_router():
    cfg = small_gpt(n_layer=3, n_head=4, d_model=32, mlp="swiglu", d_ff=48, bias=False, pos="rope")
    dense = GPT(cfg)
    expert = sum(p.numel() for p in SwiGLU(32, 48).parameters())
    model = moeify(GPT(cfg), n_experts=5, top_k=2, first_dense=1)
    assert len(moe_layers(model)) == 2  # the first block stays dense
    extra = 2 * (4 * expert + 32 * 5)  # four more experts and a router, in two layers
    assert model.num_params() == dense.num_params() + extra
    # k = 2 of 5: each MoE layer activates one more expert than the dense layer, plus the router
    assert active_params(model) == dense.num_params() + 2 * (expert + 32 * 5)


def test_a_small_moe_language_model_can_memorise_a_batch():
    torch.manual_seed(0)
    cfg = small_gpt(n_layer=2, n_head=2, d_model=32, mlp="swiglu", d_ff=32, bias=False, pos="rope")
    model = moeify(GPT(cfg), n_experts=4, top_k=2, capacity_factor=1.5)
    x = torch.randint(0, 256, (4, 16))
    y = torch.roll(x, -1, dims=1)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    first = None
    for _ in range(120):
        loss = torch.nn.functional.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1))
        first = first or loss.item()
        opt.zero_grad()
        (loss + 0.01 * aux_loss(model)).backward()
        opt.step()
    assert first > 5 and loss.item() < 1.0
    assert math.isfinite(aux_loss(model).item())


def test_node_limited_routing_keeps_every_token_inside_topk_group_groups():
    torch.manual_seed(0)
    logits = torch.randn(500, 64)
    limited = route(logits, 8, "deepseek", n_group=8, topk_group=4)
    free = route(logits, 8, "deepseek")
    assert max(g.unique().numel() for g in limited.experts // 8) <= 4
    assert max(g.unique().numel() for g in free.experts // 8) > 4
    assert (limited.gates.sum(-1) - 1).abs().max() < 1e-5  # still normalised


def test_allowing_all_groups_is_the_same_as_no_limit():
    logits = torch.randn(100, 32)
    a = route(logits, 4, "deepseek", n_group=4, topk_group=4)
    b = route(logits, 4, "deepseek")
    assert torch.equal(a.experts, b.experts) and torch.allclose(a.gates, b.gates)


def test_a_group_is_scored_by_its_best_experts_together_not_by_its_single_best():
    logits = torch.tensor([[5.0, -5.0, 1.0, 1.0]])  # group 0: {5, -5}; group 1: {1, 1}
    free = route(logits, 2, "mixtral")
    limited = route(logits, 2, "mixtral", n_group=2, topk_group=1)
    assert set(free.experts[0].tolist()) == {0, 2} or set(free.experts[0].tolist()) == {0, 3}
    assert set(limited.experts[0].tolist()) == {2, 3}  # group 1 sums to 2, group 0 to 0


def test_node_limiting_validates_its_arguments():
    with pytest.raises(ValueError):
        MoELayer(8, 8, 6, top_k=2, n_group=4, topk_group=2)  # 4 does not divide 6
    with pytest.raises(ValueError):
        MoELayer(8, 8, 8, top_k=4, n_group=4, topk_group=1)  # one group holds only 2 experts
