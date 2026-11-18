# 强化学习基础：策略梯度、基线与 GAE

[English](README.en.md)

[强化学习基础：策略梯度、优势估计与 PPO](https://dmxsf.com/wiki/post-training/rl-fundamentals/) · [实验脚本](run.py) · [库代码](../../../src/dmxsf/post_training/rl_basics.py)

枚举小策略，核对数值梯度，观察 GAE 的 λ 如何改变有限轨迹中的优势。

## 运行命令

```bash
uv run dmxsf run post-training/rl-fundamentals
```

## 练习

换一个常数基线，重新核对期望梯度并比较每个动作贡献。

## 完整运行（Apple M2 Max CPU，Python 3.12.14，种子 0）

```text
# Policy gradients: exact checks, baselines and GAE
  course: https://dmxsf.com/wiki/post-training/rl-fundamentals/
          https://dmxsf.com/en/wiki/post-training/rl-fundamentals/

## 1. An enumerated policy with three actions
  [ok] score-function gradient agrees with the finite difference of expected reward  ([ 0.25 -0.15 -0.1 ])
  [ok] a constant action-independent baseline preserves the exact gradient

## 2. Generalized advantage estimation
  [ok] lambda zero uses one TD error and lambda one accumulates the trajectory  ([2.62 1.8  2.  ])

All checks passed. (0.0s)
```
