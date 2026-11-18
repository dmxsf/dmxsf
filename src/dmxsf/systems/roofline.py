"""GPU roofline: bytes, FLOPs and tile reuse.

Wiki pages
  Chinese: https://dmxsf.com/wiki/systems/gpu-and-roofline/
  English: https://dmxsf.com/en/wiki/systems/gpu-and-roofline/
"""


def linear_intensity(batch: int, inputs: int, outputs: int, weight_bytes: float = 2) -> float:
    """FLOPs per transferred byte with FP16/BF16 inputs and outputs."""
    work = 2 * batch * inputs * outputs
    traffic = weight_bytes * inputs * outputs + 2 * batch * (inputs + outputs)
    return work / traffic


def roofline(intensity: float, peak: float, bandwidth: float) -> float:
    """Upper bound in FLOP/s for bandwidth in bytes/s."""
    return min(peak, bandwidth * intensity)


def tiled_bytes(size: int, tile: int, dtype_bytes: int = 2) -> int:
    """Count tile input loads and one output store for each output tile."""
    if not 0 < tile <= size or size % tile:
        raise ValueError("tile must evenly divide the matrix width")
    tiles = size // tile
    reads = 2 * tiles**3 * tile**2
    writes = size**2
    return (reads + writes) * dtype_bytes
