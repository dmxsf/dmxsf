"""Count storage and copy-time bounds for /wiki/systems/hardware-basics/.

Wiki pages
  Chinese: https://dmxsf.com/wiki/systems/hardware-basics/
  English: https://dmxsf.com/en/wiki/systems/hardware-basics/
"""

from __future__ import annotations

from math import prod


def tensor_bytes(shape: tuple[int, ...], bytes_per_value: int) -> int:
    if bytes_per_value <= 0 or any(size < 0 for size in shape):
        raise ValueError("nonnegative axes and positive item size required")
    return prod(shape) * bytes_per_value


def copy_seconds(size_bytes: int, bytes_per_second: int) -> float:
    """Ideal duration at the given sustained copy rate."""
    if size_bytes < 0 or bytes_per_second <= 0:
        raise ValueError("need nonnegative size and positive copy rate")
    return size_bytes / bytes_per_second
