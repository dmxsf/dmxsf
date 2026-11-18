# GPU roofline: bytes, FLOPs and tile reuse

[中文](README.md)

[GPUs and the Roofline Model: Why Big Models Are Bound by Memory Bandwidth](https://dmxsf.com/en/wiki/systems/gpu-and-roofline/) · [Lab script](run.py) · [Library code](../../../src/dmxsf/systems/roofline.py)

Compute the A100 and H100 ridge points, model a linear layer, and count matrix-multiply tile traffic.

## Run the lab

```bash
uv run dmxsf run systems/gpu-and-roofline
```

## Exercise

Keep the weight matrix fixed and change the batch size. Identify the first batch size at which the H100 bound becomes compute-limited.

## Full run (Apple M2 Max CPU, Python 3.12.14, seed 0)

```text
# GPU roofline: bytes, FLOPs and tile reuse
  course: https://dmxsf.com/wiki/systems/gpu-and-roofline/
          https://dmxsf.com/en/wiki/systems/gpu-and-roofline/

## 1. Compute ceilings and ridges
  [ok] A100 80GB SXM ridge  (153.0 FLOP/byte)
  [ok] H100 SXM ridge  (295.2 FLOP/byte)

## 2. Weights shared across tokens
  batch  FLOP/byte  H100 ceiling TFLOP/s
  -----  ---------  --------------------
      1        1.0                   3.3
     16       15.9                  53.2
     64       62.1                 207.9
    153      142.4                 476.9
    295      257.9                 863.8
    512      409.6                 989.0
   2048     1024.0                 989.0
  32768     1927.5                 989.0
  [ok] one token remains bandwidth-bound and a large batch reaches the compute ceiling
  [ok] INT4 weight-only intensity is about four FLOP/byte for one token

## 3. Tiling changes traffic with the same FLOPs
  [ok] 128-square matrices, one output store and 32-square input tiles  (294,912 vs 98,304 bytes)

All checks passed. (0.0s)
```
