# GPU 内核：分块矩阵乘与访存

[English](README.en.md)

[课程页](https://dmxsf.com/wiki/systems/gpu-kernels/) · [实现](../../../src/dmxsf/systems/kernels.py)

用不规则矩阵验证边缘块的正确性，逐项核对逻辑读写量。改 `tiles` 中的归约块大小，观察它为什么不改变逻辑字节数。

```bash
uv run dmxsf run systems/gpu-kernels
```

## 完整运行输出（Apple M2 Max / Python 3.12.14）

```text
# GPU kernels: tiled GEMM and memory traffic
  course: https://dmxsf.com/wiki/systems/gpu-kernels/
          https://dmxsf.com/en/wiki/systems/gpu-kernels/

## Multiply (65, 33) by (33, 49)
  [ok] (4, 4, 4) matches the reference
  [ok] (4, 4, 4) matches the traffic formula  (234,236 logical bytes)
  [ok] (8, 8, 4) matches the reference
  [ok] (8, 8, 4) matches the traffic formula  (131,012 logical bytes)
  [ok] (8, 8, 16) matches the reference
  [ok] (8, 8, 16) matches the traffic formula  (131,012 logical bytes)

All checks passed. (0.0s)
```
