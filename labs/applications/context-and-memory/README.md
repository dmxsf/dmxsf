# Agent 上下文与记忆：预算与优先级

[English](README.en.md)

[课程页](https://dmxsf.com/wiki/applications/context-and-memory/) · [实现](../../../src/dmxsf/applications/context_memory.py)

整组保留工具轮，验证版本记录覆盖与删除。

```bash
uv run dmxsf run applications/context-and-memory
```

把预算从 25 改到 30，再对照删除标记存在和缺失时读出的值。

## 完整运行 (Apple M2 Max, Python 3.12.14)

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
