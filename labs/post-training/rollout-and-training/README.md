# 推理强化学习系统：采样与训练调度

[English](README.en.md)

[课程页](https://dmxsf.com/wiki/post-training/rollout-and-training/) · [实现](../../../src/dmxsf/post_training/rollout_pipeline.py)

回放串行与重叠调度，核对版本滞后和丢弃批次。

```bash
uv run dmxsf run post-training/rollout-and-training
```

改动滞后上限，报告总耗时、使用批次数和丢弃批次数。

## 完整运行 (Apple M2 Max, Python 3.12.14)

```text
# Reasoning RL systems: rollout and learner scheduling
  course: https://dmxsf.com/wiki/post-training/rollout-and-training/
          https://dmxsf.com/en/wiki/post-training/rollout-and-training/

## One actor and one learner
serial: t=57, used=6, lag=(0, 0, 0, 0, 0, 0)
bounded: t=29, used=4, lag=(0, 1, 2, 0, 1, 2)
permissive: t=39, used=6, lag=(0, 1, 2, 1, 2, 3)
  [ok] serial accepts every batch with zero lag
  [ok] bounded run drops stale work
  [ok] overlap saves wall time at permissive lag

All checks passed. (0.0s)
```
