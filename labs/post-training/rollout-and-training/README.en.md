# Reasoning RL systems: rollout and learner scheduling

[中文](README.md)

[Course page](https://dmxsf.com/en/wiki/post-training/rollout-and-training/) · [Implementation](../../../src/dmxsf/post_training/rollout_pipeline.py)

Replay serial and overlapping jobs, and verify policy lag and dropped batches.

```bash
uv run dmxsf run post-training/rollout-and-training
```

Vary the lag limit and report elapsed time, accepted batches, and drops.

## Full run (Apple M2 Max, Python 3.12.14)

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
