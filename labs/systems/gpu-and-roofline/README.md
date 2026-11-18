# GPU 屋顶线：字节、运算与分块复用

[English](README.en.md)

[GPU 与屋顶线模型：为什么大模型常常卡在显存带宽上](https://dmxsf.com/wiki/systems/gpu-and-roofline/) · [实验脚本](run.py) · [库代码](../../../src/dmxsf/systems/roofline.py)

从 A100 和 H100 的硬件规格算转折点，再核对线性层和分块矩阵乘法的数据搬运量。

## 运行命令

```bash
uv run dmxsf run systems/gpu-and-roofline
```

## 练习

保持权重矩阵不变，改变批大小，找出 H100 模型中首先进入算力瓶颈的批大小。

## 完整运行（Apple M2 Max CPU，Python 3.12.14，种子 0）

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
