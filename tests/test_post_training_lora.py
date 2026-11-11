"""LoRA: zero start, gradients, counts, exact merging, NF4."""

import pytest
import torch
from torch import nn

from dmxsf.architecture.configs import small_gpt
from dmxsf.architecture.gpt import GPT
from dmxsf.post_training import lora as L

QWEN_05B = {  # Qwen2.5-0.5B: hidden 896, MLP 4864, 14 query heads and 2 KV heads of 64
    "q": (896, 896),
    "k": (896, 128),
    "v": (896, 128),
    "o": (896, 896),
    "gate": (896, 4864),
    "up": (896, 4864),
    "down": (4864, 896),
}
LLAMA2_7B_QV = {"q": (4096, 4096), "v": (4096, 4096)}


def test_adapter_parameter_counts_match_the_page():
    assert L.adapter_params(QWEN_05B, 16) * 24 == 8_798_208
    assert L.adapter_params(QWEN_05B, 16) == 366_592
    assert L.adapter_params(LLAMA2_7B_QV, 8) * 32 == 4_194_304
    d, ff = 4096, 11008
    llama_all = dict.fromkeys("qkvo", (d, d)) | {"gate": (d, ff), "up": (d, ff), "down": (ff, d)}
    assert L.adapter_params(llama_all, 16) * 32 == 39_976_960


def tiny_linear(dtype=torch.float32):
    torch.manual_seed(0)
    return nn.Linear(12, 20, bias=True).to(dtype)


def test_zero_init_changes_nothing():
    base = tiny_linear()
    lora = L.LoRALinear(base, r=4, alpha=8)
    x = torch.randn(3, 5, 12)
    assert torch.equal(lora(x), base(x))


def test_first_step_gradients():
    base = tiny_linear()
    lora = L.LoRALinear(base, r=4, alpha=8)
    lora(torch.randn(3, 5, 12)).pow(2).sum().backward()
    assert base.weight.grad is None and base.bias.grad is None  # frozen
    assert lora.A.grad.abs().max() == 0  # dL/dA goes through B = 0
    assert lora.B.grad.abs().max() > 0


def test_zero_a_and_zero_b_never_learn():
    lora = L.LoRALinear(tiny_linear(), r=4, alpha=8, zero_a=True)
    lora(torch.randn(3, 5, 12)).pow(2).sum().backward()
    assert lora.A.grad.abs().max() == 0 and lora.B.grad.abs().max() == 0


def test_merge_is_exact_in_float64():
    base = tiny_linear(torch.float64)
    lora = L.LoRALinear(base, r=4, alpha=8)
    with torch.no_grad():
        lora.B.normal_(0, 0.1)
    x = torch.randn(3, 5, 12, dtype=torch.float64)
    merged = nn.functional.linear(x, lora.merged(), base.bias)
    assert (lora(x) - merged).abs().max() < 1e-12


def small_model():
    torch.manual_seed(0)
    return GPT(small_gpt(block_size=32, n_layer=2, n_head=2, d_model=32))


def test_add_lora_freezes_everything_but_the_adapters():
    model = small_model()
    adapters = L.add_lora(model, "all", r=4, alpha=8)
    assert len(adapters) == 2 * 6
    d, ff = 32, 128
    per_layer = L.adapter_params(
        {"q": (d, d), "k": (d, d), "v": (d, d), "o": (d, d), "fc": (d, ff), "proj": (ff, d)}, 4
    )
    assert L.trainable_params(model) == 2 * per_layer
    assert all(p.requires_grad for a in adapters for p in (a.A, a.B))
    assert not any(p.requires_grad for a in adapters for p in a.base.parameters())
    assert not model.tok_emb.weight.requires_grad


@pytest.mark.parametrize(
    ("targets", "count"), [("qv", 4), ("attention", 8), ("mlp", 4), (("attn.o_proj",), 2)]
)
def test_target_sets(targets, count):
    assert len(L.add_lora(small_model(), targets, r=2, alpha=4)) == count


def test_merge_lora_gives_the_same_logits_and_removes_adapters():
    model = small_model()
    for adapter in L.add_lora(model, "all", r=4, alpha=8):
        with torch.no_grad():
            adapter.B.normal_(0, 0.05)
    ids = torch.randint(0, 256, (2, 10))
    with torch.no_grad():
        before = model(ids)
    L.merge_lora(model)
    assert not any(isinstance(m, L.LoRALinear) for m in model.modules())
    with torch.no_grad():
        assert torch.allclose(model(ids), before, atol=1e-5)


def test_nf4_levels_and_bits():
    assert len(L.NF4) == 16 and L.NF4[0] == -1 and L.NF4[-1] == 1 and 0.0 in L.NF4
    assert list(L.NF4) == sorted(L.NF4)
    assert L.nf4_bits_per_weight() == 4.5
    assert L.nf4_bits_per_weight(double_quant=True) == pytest.approx(4.127, abs=5e-4)


def test_nf4_roundtrip_is_idempotent_and_has_a_ten_percent_error_on_gaussians():
    w = torch.randn(128, 128, generator=torch.Generator().manual_seed(0))
    q = L.nf4_roundtrip(w)
    assert torch.equal(L.nf4_roundtrip(q), q)  # values already on the grid stay put
    assert (
        len({round(v, 5) for v in (q[:1, :64] / q[:1, :64].abs().max()).flatten().tolist()}) <= 16
    )
    rel = ((w - q).norm() / w.norm()).item()
    assert 0.07 < rel < 0.13


def test_disabled_adapters_give_back_the_base_model_exactly():
    torch.manual_seed(0)
    model = GPT(small_gpt(block_size=32, n_layer=2, n_head=2, d_model=32)).eval()
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        before = model(x)
    L.add_lora(model, "all", r=4, alpha=8.0)
    for a in (m for m in model.modules() if isinstance(m, L.LoRALinear)):
        torch.nn.init.normal_(a.B, std=0.1)  # a trained-looking adapter
    with torch.no_grad():
        trained = model(x)
        with L.adapters_disabled(model):
            off = model(x)
        again = model(x)
    assert torch.equal(off, before) and torch.equal(again, trained)
    assert not torch.allclose(trained, before)
