# Numerical precision: rounding, overflow and loss scaling

[中文](README.md)

[Numerical Precision: Three Ways FP16, BF16 and FP8 Training Goes Wrong](https://dmxsf.com/en/wiki/systems/numerical-precision/) · [Lab script](run.py) · [Library code](../../../src/dmxsf/systems/precision.py)

Reproduce low-precision stalled addition, stable softmax, and loss scaling with CPU tensors.

## Run the lab

```bash
uv run dmxsf run systems/numerical-precision
```

## Exercise

Repeat the update with the FP32 master weight rounded after each step. Compare with rounding only at the end.

## Full run (Apple M2 Max CPU, Python 3.12.14, seed 0)

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
