"""Numerical precision: rounding, overflow and loss scaling.

Wiki pages
  Chinese: https://dmxsf.com/wiki/systems/numerical-precision/
  English: https://dmxsf.com/en/wiki/systems/numerical-precision/
"""

import torch


def stalled_sum(dtype: torch.dtype) -> int:
    """Add one repeatedly until round-to-nearest can no longer change the sum."""
    value = torch.zeros((), dtype=dtype)
    for _ in range(4097):
        updated = value + 1
        if updated == value:
            return int(value)
        value = updated
    raise ValueError("format did not stall in the range of the FP16/BF16 exercise")


def round_scaled_gradient(value: float, scale: int) -> float:
    """Scale before converting to FP16 and undo the scale in FP32."""
    rounded = torch.tensor(value * scale, dtype=torch.float16)
    return float(rounded.float() / scale)
