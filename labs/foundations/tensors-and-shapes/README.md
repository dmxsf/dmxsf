# 张量形状：手算线性层

[English](README.en.md)

[课程页](https://dmxsf.com/wiki/foundations/tensors-and-shapes/) · [实现](../../../src/dmxsf/foundations/tensor_shapes.py)

用线性投影、偏置和 ReLU 练习位置轴与特征轴。

```bash
uv run dmxsf run foundations/tensors-and-shapes
```

追加一行输入，手算后核对输出形状和最后一行。

## 完整运行 (Apple M2 Max, Python 3.12.14)

```text
# Tensor shapes: a linear layer by hand
  course: https://dmxsf.com/wiki/foundations/tensors-and-shapes/
          https://dmxsf.com/en/wiki/foundations/tensors-and-shapes/

## Two positions and three features
  [ok] projection and broadcast bias  ([[5, -3], [-1, 3]])
  [ok] ReLU keeps row and column counts  ([[5, 0], [0, 3]])

All checks passed. (0.0s)
```
