# Tensor shapes: a linear layer by hand

[中文](README.md)

[Course page](https://dmxsf.com/en/wiki/foundations/tensors-and-shapes/) · [Implementation](../../../src/dmxsf/foundations/tensor_shapes.py)

Track position and feature axes through a projection, bias and ReLU.

```bash
uv run dmxsf run foundations/tensors-and-shapes
```

Add an input row, then verify its projected values and the output shape.

## Full run (Apple M2 Max, Python 3.12.14)

```text
# Tensor shapes: a linear layer by hand
  course: https://dmxsf.com/wiki/foundations/tensors-and-shapes/
          https://dmxsf.com/en/wiki/foundations/tensors-and-shapes/

## Two positions and three features
  [ok] projection and broadcast bias  ([[5, -3], [-1, 3]])
  [ok] ReLU keeps row and column counts  ([[5, 0], [0, 3]])

All checks passed. (0.0s)
```
