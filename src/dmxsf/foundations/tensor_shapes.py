"""Small tensor operations for /wiki/foundations/tensors-and-shapes/.

Wiki pages
  Chinese: https://dmxsf.com/wiki/foundations/tensors-and-shapes/
  English: https://dmxsf.com/en/wiki/foundations/tensors-and-shapes/
"""

from __future__ import annotations


def linear(
    rows: list[list[int]],
    weights: list[list[int]],
    bias: list[int],
) -> list[list[int]]:
    """Apply a linear layer to a batch of rows."""
    bad_weights = any(len(weight) != len(bias) for weight in weights)
    if not weights or not bias or bad_weights:
        raise ValueError("weight columns must match the bias")
    if any(len(row) != len(weights) for row in rows):
        raise ValueError("input width must match the number of weight rows")
    columns = list(zip(*weights, strict=True))
    output = []
    for row in rows:
        projected = []
        for column, offset in zip(columns, bias, strict=True):
            dot = sum(value * w for value, w in zip(row, column, strict=True))
            projected.append(dot + offset)
        output.append(projected)
    return output


def relu(rows: list[list[int]]) -> list[list[int]]:
    """Set negative entries to zero without changing the shape."""
    return [[max(0, value) for value in row] for row in rows]
