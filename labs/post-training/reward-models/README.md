# 奖励模型：成对打分与标签噪声

[English](README.en.md)

[奖励模型：把人类偏好变成可以优化的分数](https://dmxsf.com/wiki/post-training/reward-models/) · [实验脚本](run.py) · [库代码](../../../src/dmxsf/post_training/reward_models.py)

在连通比较图上拟合 Bradley–Terry 分数，再核对独立标注噪声下的一致率和表现上限。

## 运行命令

```bash
uv run dmxsf run post-training/reward-models
```

## 练习

把翻转概率设为 0.25，比较单标注准确率上限和两人一致率。

## 完整运行（Apple M2 Max CPU，Python 3.12.14，种子 0）

```text
# Reward models: pairwise scores and label noise
  course: https://dmxsf.com/wiki/post-training/reward-models/
          https://dmxsf.com/en/wiki/post-training/reward-models/

## 1. Recover relative scores from pairwise choices
  [ok] relative scores are recovered up to a single global shift  (max gap 3.3e-16)

## 2. Independent symmetric label flips
  [ok] 10% flips yield 90% single-label ceiling and 82% two-label agreement  (observed agreement 0.818)

All checks passed. (0.0s)
```
