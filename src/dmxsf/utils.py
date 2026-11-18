"""Small helpers every lab uses: seeding, parameter counts, formatting, a chi-square test."""

from __future__ import annotations

import math
import random
import time

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int = 0) -> None:
    """Seed Python, NumPy and PyTorch so that a run can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    """Number of parameters. Tied (shared) weights are counted once."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def human_bytes(n: float) -> str:
    """``1536 -> '1.50 KiB'``."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024
    raise AssertionError("unreachable")


def human_count(n: float) -> str:
    """``8_030_000_000 -> '8.03B'``."""
    for unit, size in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= size:
            return f"{n / size:.2f}{unit}"
    return f"{n:.0f}"


class Timer:
    """``with Timer() as t: ...`` then read ``t.seconds``."""

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        self.seconds = 0.0
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self._start


def chi_square_sf(x: float, df: int) -> float:
    """P(X >= x) for X ~ chi-square with ``df`` degrees of freedom (regularized upper incomplete gamma)."""
    a, z = df / 2, x / 2
    if z <= 0:
        return 1.0
    scale = math.exp(-z + a * math.log(z) - math.lgamma(a))
    if z < a + 1:  # series for the lower function, then one minus it
        term = total = 1.0 / a
        n = a
        for _ in range(2000):
            n += 1
            term *= z / n
            total += term
            if term < total * 1e-16:
                break
        return 1.0 - total * scale
    tiny = 1e-300  # continued fraction for the upper function (modified Lentz)
    b = z + 1 - a
    c, d = 1 / tiny, 1 / b
    h = d
    for i in range(1, 2000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-16:
            break
    return scale * h


def chi_square_gof(counts, probs, min_expected: float = 5.0) -> tuple[float, int, float]:
    """Goodness of fit of observed ``counts`` to the probabilities ``probs``: (statistic, degrees
    of freedom, p-value). Outcomes expected fewer than ``min_expected`` times are pooled into one."""
    counts = np.asarray(counts, dtype=np.float64)
    expected = np.asarray(probs, dtype=np.float64) * counts.sum()
    small = expected < min_expected
    if small.any():
        counts = np.append(counts[~small], counts[small].sum())
        expected = np.append(expected[~small], expected[small].sum())
    keep = expected > 0
    stat = float((((counts - expected) ** 2)[keep] / expected[keep]).sum())
    df = int(keep.sum()) - 1
    return stat, df, chi_square_sf(stat, df)
