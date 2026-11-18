"""CPU tile scheduling and logical memory accounting for /wiki/systems/gpu-kernels/.

Wiki pages
  Chinese: https://dmxsf.com/wiki/systems/gpu-kernels/
  English: https://dmxsf.com/en/wiki/systems/gpu-kernels/
"""

from __future__ import annotations

import numpy as np


def tiled_matmul(
    left: np.ndarray, right: np.ndarray, tiles: tuple[int, int, int]
) -> tuple[np.ndarray, int]:
    """Accumulate output tiles and count the elements read or written per tile."""
    rows, depth = left.shape
    right_depth, cols = right.shape
    if depth != right_depth or min(tiles) <= 0:
        raise ValueError("incompatible matrix shapes or nonpositive tiles")
    tile_m, tile_n, tile_k = tiles
    result = np.empty((rows, cols), dtype=np.result_type(left, right))
    moved = 0
    for row in range(0, rows, tile_m):
        row_end = min(row + tile_m, rows)
        for col in range(0, cols, tile_n):
            col_end = min(col + tile_n, cols)
            block = np.zeros((row_end - row, col_end - col), dtype=result.dtype)
            for start in range(0, depth, tile_k):
                end = min(start + tile_k, depth)
                a = left[row:row_end, start:end]
                b = right[start:end, col:col_end]
                block += a @ b
                moved += a.size + b.size
            result[row:row_end, col:col_end] = block
            moved += block.size
    return result, moved


def traffic_elements(rows: int, cols: int, depth: int, tile_m: int, tile_n: int) -> int:
    """Count live input loads and output stores with masked edge tiles."""
    if min(rows, cols, depth, tile_m, tile_n) <= 0:
        raise ValueError("all sizes must be positive")
    row_tiles = (rows + tile_m - 1) // tile_m
    col_tiles = (cols + tile_n - 1) // tile_n
    return col_tiles * rows * depth + row_tiles * depth * cols + rows * cols
