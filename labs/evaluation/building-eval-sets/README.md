# 构建评测集：标签噪声、一致率与题目信息

[English](README.en.md)

[构建评测集：题目来源、标注质量与题目的信息量](https://dmxsf.com/wiki/evaluation/building-eval-sets/) · [实验脚本](run.py) · [库代码](../../../src/dmxsf/evaluation/eval_sets.py)

模拟四选一标签噪声来校对公式，计算 Cohen κ，并找到 2PL 题目最能区分的能力点。

## 运行命令

```bash
uv run dmxsf run evaluation/building-eval-sets
```

## 练习

把标签噪声从 10% 降到 5%，核对真实四个百分点的差距还剩多少。

## 完整运行（Apple M2 Max CPU，Python 3.12.14，种子 0）

```text
# Evaluation sets: noisy labels, agreement and item information
  course: https://dmxsf.com/wiki/evaluation/building-eval-sets/
          https://dmxsf.com/en/wiki/evaluation/building-eval-sets/

## 1. What an independently wrong label does to scores
  [ok] the simulation reproduces predicted agreement near 72.7%  (0.728 vs 0.727)
  [ok] the article's 6.49% label error requires about 1.198x as many items  (1.198)

## 2. Agreement and information
  [ok] 90% raw agreement with balanced marginals gives kappa 0.8
  [ok] the 2PL information peaks at difficulty with value discrimination squared over four

All checks passed. (0.0s)
```
