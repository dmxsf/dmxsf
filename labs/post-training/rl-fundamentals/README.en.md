# Policy gradients: exact checks, baselines and GAE

[中文](README.md)

[Reinforcement Learning Basics: Policy Gradients, Advantages and PPO](https://dmxsf.com/en/wiki/post-training/rl-fundamentals/) · [Lab script](run.py) · [Library code](../../../src/dmxsf/post_training/rl_basics.py)

Enumerate a tiny policy, compare gradients against finite differences, and vary lambda in a finite GAE trajectory.

## Run the lab

```bash
uv run dmxsf run post-training/rl-fundamentals
```

## Exercise

Replace the constant baseline with a different constant. Check the expected gradient and compare the per-action updates.

## Full run (Apple M2 Max CPU, Python 3.12.14, seed 0)

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
