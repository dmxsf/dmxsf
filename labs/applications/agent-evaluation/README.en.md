# Agent evaluation: repeatability and terminal state

[中文](README.md)

[Course page](https://dmxsf.com/en/wiki/applications/agent-evaluation/) · [Implementation](../../../src/dmxsf/applications/agent_metrics.py)

Compare equal one-run scores and validate four calendar traces.

```bash
uv run dmxsf run applications/agent-evaluation
```

Vary trial counts by task, then verify task weighting and duplicate-event grading.

## Full run (Apple M2 Max, Python 3.12.14)

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
