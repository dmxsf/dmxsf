import pytest
import torch
from torch import nn

from dmxsf.architecture.layers import GELUMLP, LayerNorm, RMSNorm, SwiGLU, make_norm
from dmxsf.utils import count_params


def test_layernorm_matches_torch():
    torch.manual_seed(0)
    x = torch.randn(3, 5, 16, dtype=torch.float64) * 3 + 2
    ref = nn.LayerNorm(16, eps=1e-5).double()
    with torch.no_grad():
        ref.weight.copy_(torch.randn(16))
        ref.bias.copy_(torch.randn(16))
    ours = LayerNorm(16, eps=1e-5).double()
    ours.load_state_dict(ref.state_dict())
    assert torch.allclose(ours(x), ref(x), atol=1e-12)


def test_layernorm_ignores_a_constant_shift_and_rmsnorm_does_not():
    torch.manual_seed(0)
    x = torch.randn(4, 32, dtype=torch.float64)
    shifted = x + 5.0
    assert torch.allclose(LayerNorm(32).double()(x), LayerNorm(32).double()(shifted), atol=1e-9)
    assert not torch.allclose(RMSNorm(32).double()(x), RMSNorm(32).double()(shifted), atol=1e-3)


def test_rmsnorm_formula_and_scale_invariance():
    torch.manual_seed(0)
    x = torch.randn(4, 32, dtype=torch.float64)
    norm = RMSNorm(32, eps=0.0).double()
    expected = x / x.pow(2).mean(-1, keepdim=True).sqrt()
    assert torch.allclose(norm(x), expected, atol=1e-12)
    assert torch.allclose(norm(7.0 * x), norm(x), atol=1e-12)  # only the direction matters
    assert torch.allclose(norm(x).pow(2).mean(-1), torch.ones(4, dtype=torch.float64))


def test_rmsnorm_keeps_the_input_dtype():
    x = torch.randn(2, 8, dtype=torch.bfloat16)
    assert RMSNorm(8).to(torch.bfloat16)(x).dtype == torch.bfloat16


def test_feed_forward_parameter_counts():
    d, f = 64, 256
    assert count_params(GELUMLP(d, f, bias=True)) == 2 * d * f + f + d
    assert count_params(GELUMLP(d, f, bias=False)) == 2 * d * f
    assert count_params(SwiGLU(d, f)) == 3 * d * f  # three matrices instead of two


def test_swiglu_formula():
    torch.manual_seed(0)
    mlp = SwiGLU(8, 16).double()
    x = torch.randn(3, 8, dtype=torch.float64)
    expected = mlp.down.weight @ (
        torch.nn.functional.silu(mlp.gate.weight @ x.T) * (mlp.up.weight @ x.T)
    )
    assert torch.allclose(mlp(x), expected.T, atol=1e-12)


def test_make_norm():
    assert isinstance(make_norm("layernorm", 8), LayerNorm)
    assert isinstance(make_norm("rmsnorm", 8), RMSNorm)
    with pytest.raises(ValueError):
        make_norm("batchnorm", 8)
