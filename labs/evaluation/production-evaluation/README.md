# 生产评测：配对回放与发布门槛

[English](README.en.md)

[课程页](https://dmxsf.com/wiki/evaluation/production-evaluation/) · [实现](../../../src/dmxsf/evaluation/production.py)

在同一批任务上比较旧版和候选版的结果。尝试补齐新语言切片，再改变 `max_loss`，区分覆盖不足与实际退化。

```bash
uv run dmxsf run evaluation/production-evaluation
```

## 完整运行输出（Apple M2 Max / Python 3.12.14）

```text
# Production evaluation: replay and release gates
  course: https://dmxsf.com/wiki/evaluation/production-evaluation/
          https://dmxsf.com/en/wiki/evaluation/production-evaluation/

## Overall and per-slice changes
overall: n=102, change=+6.9%
common: n=90, change=+10.0%
critical: n=10, change=-20.0%
new locale: n=2, change=+0.0%
  [ok] overall change is positive
  [ok] critical slice blocks release
  [ok] missing evidence blocks release

All checks passed. (0.0s)
```
