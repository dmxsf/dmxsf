"""Position encodings: sinusoidal, RoPE (with context-extension variants) and ALiBi.

Course page: https://dmxsf.com/wiki/architecture/positional-encoding/
Context extension: https://dmxsf.com/wiki/architecture/long-context/
"""

import math

import torch

# ---------------------------------------------------------------------------
# Sinusoidal (Vaswani et al., 2017)


def sinusoidal_embedding(positions: torch.Tensor, d_model: int, base: float = 10_000.0):
    """PE[p, 2i] = sin(p / base^(2i/d)),  PE[p, 2i+1] = cos(p / base^(2i/d)).

    Works for any position, so a model can be evaluated beyond its training length.
    """
    inv = base ** (-torch.arange(0, d_model, 2, dtype=torch.float64) / d_model)
    angle = positions.to(torch.float64)[:, None] * inv[None, :].to(positions.device)
    out = torch.zeros(len(positions), d_model, dtype=torch.float64, device=positions.device)
    out[:, 0::2] = angle.sin()
    out[:, 1::2] = angle.cos()
    return out.float()


def sinusoidal_table(n_positions: int, d_model: int, base: float = 10_000.0):
    return sinusoidal_embedding(torch.arange(n_positions), d_model, base)


# ---------------------------------------------------------------------------
# RoPE (Su et al., 2021)


def rope_inv_freq(head_dim: int, base: float = 10_000.0) -> torch.Tensor:
    """theta_i = base^(-2i/d) for i = 0 .. d/2-1, as float64 (cast when you use it)."""
    return base ** (-torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)


def rope_cos_sin(positions: torch.Tensor, inv_freq: torch.Tensor, attention_factor: float = 1.0):
    """cos(m * theta_i) and sin(m * theta_i), both of shape (len(positions), d/2)."""
    angle = positions.to(torch.float64)[:, None] * inv_freq.to(torch.float64)[None, :]
    return (angle.cos() * attention_factor).float(), (angle.sin() * attention_factor).float()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool = False):
    """Rotate each pair of dimensions of x (..., T, d) by the angle in cos / sin (T, d/2).

    interleaved=True pairs (x[2i], x[2i+1]), as in the RoFormer paper.
    interleaved=False pairs (x[i], x[i+d/2]), as in Hugging Face's Llama (rotate_half).
    The two differ only by a fixed permutation of the dimensions.
    """
    if interleaved:
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


# ---------------------------------------------------------------------------
# Context extension: change theta_i so a model trained on L tokens works on s * L tokens.


def rope_linear(inv_freq: torch.Tensor, factor: float) -> torch.Tensor:
    """Position interpolation (Chen et al., 2023): squeeze positions, i.e. divide every theta by s."""
    return inv_freq / factor


def rope_ntk(head_dim: int, base: float, factor: float) -> torch.Tensor:
    """NTK-aware scaling (bloc97, 2023): raise the base to base * s^(d/(d-2))."""
    return rope_inv_freq(head_dim, base * factor ** (head_dim / (head_dim - 2)))


def _correction_dim(rotations: float, head_dim: int, base: float, original_max_len: int) -> float:
    """The dimension index whose frequency completes `rotations` turns over the original context."""
    return head_dim * math.log(original_max_len / (rotations * 2 * math.pi)) / (2 * math.log(base))


def rope_yarn(
    head_dim: int,
    base: float,
    factor: float,
    original_max_len: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> tuple[torch.Tensor, float]:
    """YaRN (Peng et al., 2023), following the reference implementation.

    Dimensions that turn more than `beta_fast` times over the original context keep their frequency,
    those that turn less than `beta_slow` times are interpolated (divided by s), and a linear ramp over
    the dimension index joins the two. Returns (inv_freq, attention_factor) where
    attention_factor = 0.1 ln(s) + 1 multiplies cos and sin (so q.k grows by its square).
    """
    inv_freq = rope_inv_freq(head_dim, base)
    low = max(math.floor(_correction_dim(beta_fast, head_dim, base, original_max_len)), 0)
    high = min(
        math.ceil(_correction_dim(beta_slow, head_dim, base, original_max_len)), head_dim - 1
    )
    ramp = ((torch.arange(head_dim // 2, dtype=torch.float64) - low) / max(high - low, 1e-3)).clamp(
        0, 1
    )
    keep = 1 - ramp  # 1 -> extrapolate (unchanged), 0 -> interpolate
    scaled = inv_freq / factor * (1 - keep) + inv_freq * keep
    return scaled, 0.1 * math.log(factor) + 1.0 if factor > 1 else 1.0


def rope_llama3(
    inv_freq: torch.Tensor,
    factor: float = 8.0,
    low_freq_factor: float = 1.0,
    high_freq_factor: float = 4.0,
    original_max_len: int = 8192,
) -> torch.Tensor:
    """Llama 3.1's scheme: long wavelengths are divided by s, short ones kept, a smooth blend between."""
    wavelen = 2 * math.pi / inv_freq
    low_wavelen = original_max_len / low_freq_factor
    high_wavelen = original_max_len / high_freq_factor
    out = torch.where(wavelen > low_wavelen, inv_freq / factor, inv_freq)
    smooth = (original_max_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    blended = (1 - smooth) * inv_freq / factor + smooth * inv_freq
    medium = (wavelen <= low_wavelen) & (wavelen >= high_wavelen)
    return torch.where(medium, blended, out)


ROPE_SCALINGS = ("none", "linear", "ntk", "yarn", "llama3")


def rope_scaled_inv_freq(
    kind: str,
    head_dim: int,
    base: float = 10_000.0,
    factor: float = 1.0,
    original_max_len: int = 2048,
) -> tuple[torch.Tensor, float]:
    """One entry point: returns (inv_freq, attention_factor) for any of ROPE_SCALINGS."""
    if kind not in ROPE_SCALINGS:
        raise ValueError(f"unknown RoPE scaling {kind!r}; use one of {ROPE_SCALINGS}")
    if kind == "none" or factor == 1.0:
        return rope_inv_freq(head_dim, base), 1.0
    if kind == "linear":
        return rope_linear(rope_inv_freq(head_dim, base), factor), 1.0
    if kind == "ntk":
        return rope_ntk(head_dim, base, factor), 1.0
    if kind == "yarn":
        return rope_yarn(head_dim, base, factor, original_max_len)
    return rope_llama3(rope_inv_freq(head_dim, base), factor, 1.0, 4.0, original_max_len), 1.0


# ---------------------------------------------------------------------------
# ALiBi (Press et al., 2021)


def alibi_slopes(n_heads: int) -> torch.Tensor:
    """Head h penalises a key at distance r by slope_h * r; slopes are a geometric sequence."""

    def power_of_two(n: int) -> list[float]:
        start = 2 ** (-8 / n)
        return [start ** (i + 1) for i in range(n)]

    if math.log2(n_heads).is_integer():
        return torch.tensor(power_of_two(n_heads))
    closest = 2 ** math.floor(math.log2(n_heads))
    extra = power_of_two(2 * closest)[0::2][: n_heads - closest]
    return torch.tensor(power_of_two(closest) + extra)


def alibi_bias(slopes: torch.Tensor, q_pos: torch.Tensor, k_pos: torch.Tensor) -> torch.Tensor:
    """(heads, len(q_pos), len(k_pos)) additive bias: -slope * (q - k); keys in the future get 0 here
    (the causal mask removes them)."""
    distance = (q_pos[:, None] - k_pos[None, :]).clamp(min=0).to(slopes.dtype)
    return -slopes.to(distance.device)[:, None, None] * distance[None]
