# 测试时计算：独立采样、重排与投票

[English](README.en.md)

[测试时计算：把算力花在推理上](https://dmxsf.com/wiki/frontier/test-time-compute/) · [实验脚本](run.py) · [库代码](../../../src/dmxsf/evaluation/test_time.py)

比较独立与完全相关的重复采样，并估算有噪选择器对 Best-of-N 的影响。

## 运行命令

```bash
uv run dmxsf run frontier/test-time-compute
```

## 练习

把单次正确率 p 设在 0.5 两侧，比较多数投票在哪些情况下提升或拉低正确率。

## 完整运行（Apple M2 Max CPU，Python 3.12.14，种子 0）

```text
# Test-time compute: independent draws, reranking and majority voting
  course: https://dmxsf.com/wiki/frontier/test-time-compute/
          https://dmxsf.com/en/wiki/frontier/test-time-compute/

## 1. More samples only help when the draws add information
  [ok] independent oracle coverage is 0.83193, binary majority is 0.16308  (oracle 0.83193, majority 0.16308)
  [ok] repeating the same draw leaves coverage at the single-draw rate

## 2. A noisy reranker loses attainable coverage
  [ok] the selector helps over one sample and falls short of the oracle  (reranked 0.607; oracle 0.831)

All checks passed. (0.0s)
```
