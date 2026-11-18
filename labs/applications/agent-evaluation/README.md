# Agent 评测：重复成功与终态判分

[English](README.en.md)

[课程页](https://dmxsf.com/wiki/applications/agent-evaluation/) · [实现](../../../src/dmxsf/applications/agent_metrics.py)

对比相同单次成功率下的重复成功，并判分四种日历终态。

```bash
uv run dmxsf run applications/agent-evaluation
```

改成每个任务不同的试验数，再检查任务权重和重复事件的判分。

## 完整运行 (Apple M2 Max, Python 3.12.14)

```text
# Agent evaluation: repeatability and terminal state
  course: https://dmxsf.com/wiki/applications/agent-evaluation/
          https://dmxsf.com/en/wiki/applications/agent-evaluation/

## Repeated attempts
  [ok] equal one-attempt success
  [ok] two-attempt reliability differs

## Four calendar traces
  [ok] already-ordered and reordered responses are equivalent
  [ok] only one terminal state is correct

All checks passed. (0.0s)
```
