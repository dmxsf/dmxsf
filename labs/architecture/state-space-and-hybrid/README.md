# 状态空间与混合架构：等价扫描与显存

[English](README.en.md)

[状态空间模型与混合架构：用固定大小的状态换掉不断增长的 KV Cache](https://dmxsf.com/wiki/architecture/state-space-and-hybrid/) · [实验脚本](run.py) · [库代码](../../../src/dmxsf/architecture/state_space.py)

对比同一选择性更新的递推与显式因果形式，并分开核算混合架构的状态和缓存。

## 运行命令

```bash
uv run dmxsf run architecture/state-space-and-hybrid
```

## 练习

改变序列长度与注意力层数，比较常驻循环状态与随上下文增长的显存项。

## 完整运行（Apple M2 Max CPU，Python 3.12.14，种子 0）

```text
# State space and hybrids: equivalent scans and memory
  course: https://dmxsf.com/wiki/architecture/state-space-and-hybrid/
          https://dmxsf.com/en/wiki/architecture/state-space-and-hybrid/

## 1. Recurrence and explicit causal mixing
  [ok] the two evaluation orders agree in float64  (largest gap 1.8e-15)
  [ok] a zero decay resets history at that step

## 2. A hybrid cache ledger
  [ok] doubling sequence length doubles KV storage and leaves the state fixed  ((134217728, 786432) vs (268435456, 786432) bytes)

All checks passed. (0.0s)
```
