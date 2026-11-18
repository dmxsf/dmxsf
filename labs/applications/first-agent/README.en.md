# First agent: a bounded tool loop

[中文](README.md)

[Course page](https://dmxsf.com/en/wiki/applications/first-agent/) · [Implementation](../../../src/dmxsf/applications/first_agent.py)

Run a read-only lookup loop and check success, absence, and the step budget.

```bash
uv run dmxsf run applications/first-agent
```

Try an absent book and reduce the loop to one step.

## Full run (Apple M2 Max, Python 3.12.14)

```text
# First agent: a bounded tool loop
  course: https://dmxsf.com/wiki/applications/first-agent/
          https://dmxsf.com/en/wiki/applications/first-agent/

## A local library lookup
  [ok] answer follows the tool observation  ((('atlas', 'shelf A'),))
  [ok] missing input gets an explicit result
  [ok] bounded run stops

All checks passed. (0.0s)
```
