# GPU kernels: tiled GEMM and memory traffic

[中文](README.md)

[Course page](https://dmxsf.com/en/wiki/systems/gpu-kernels/) · [Implementation](../../../src/dmxsf/systems/kernels.py)

Check output values and logical traffic for uneven shapes. Change the reduction tile size in `tiles` and explain why that can leave logical bytes unchanged.

```bash
uv run dmxsf run systems/gpu-kernels
```

## Full run on Apple M2 Max with Python 3.12.14

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
