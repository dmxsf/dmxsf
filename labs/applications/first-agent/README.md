# 第一个 Agent：有界工具循环

[English](README.en.md)

[课程页](https://dmxsf.com/wiki/applications/first-agent/) · [实现](../../../src/dmxsf/applications/first_agent.py)

回放只读查找循环，验证成功、缺失与步数上限。

```bash
uv run dmxsf run applications/first-agent
```

换一个目录中不存在的书名，再将循环步数改成一。

## 完整运行 (Apple M2 Max, Python 3.12.14)

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
