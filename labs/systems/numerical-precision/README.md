# 数值精度：舍入、溢出与损失缩放

[English](README.en.md)

[数值精度：FP16、BF16 与 FP8 训练里的三类故障](https://dmxsf.com/wiki/systems/numerical-precision/) · [实验脚本](run.py) · [库代码](../../../src/dmxsf/systems/precision.py)

在 CPU 上复现低精度累加停滞、稳定 softmax，以及损失缩放如何保留小梯度。

## 运行命令

```bash
uv run dmxsf run systems/numerical-precision
```

## 练习

每一步都把 FP32 主权重舍入，再与只在最后舍入比较。

## 完整运行（Apple M2 Max CPU，Python 3.12.14，种子 0）

```text
# Numerical precision: rounding, overflow and loss scaling
  course: https://dmxsf.com/wiki/systems/numerical-precision/
          https://dmxsf.com/en/wiki/systems/numerical-precision/

## 1. Range and rounding
  [ok] adding one stalls at 256 in BF16 and 2048 in FP16  (256 and 2048)
  [ok] FP16 naive softmax overflows, shifted softmax is finite  ([0.73095703125, 0.26904296875, 4.470348358154297e-06])

## 2. Small gradients and master weights
  [ok] power-of-two scaling preserves this tiny FP16 update  (0.00e+00 vs 7.45e-09)
  [ok] FP32 accumulates small steps lost by BF16 per-step rounding  (1.0000 vs 1.1000)

All checks passed. (0.0s)
```
