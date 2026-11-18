# Agent context and memory: budget and precedence

[中文](README.md)

[Course page](https://dmxsf.com/en/wiki/applications/context-and-memory/) · [Implementation](../../../src/dmxsf/applications/context_memory.py)

Keep tool groups whole and verify overrides and deletion on versioned records.

```bash
uv run dmxsf run applications/context-and-memory
```

Change the budget from 25 to 30 and compare memory with and without a deletion marker.

## Full run (Apple M2 Max, Python 3.12.14)

```text
# Agent context and memory: budget and precedence
  course: https://dmxsf.com/wiki/applications/context-and-memory/
          https://dmxsf.com/en/wiki/applications/context-and-memory/

## Context budget
  [ok] no partial tool group enters the context  ([2])
  [ok] the whole tool group fits with enough budget

## Memory precedence
  [ok] current request takes precedence
  [ok] a deletion does not resurrect an older value

All checks passed. (0.0s)
```
