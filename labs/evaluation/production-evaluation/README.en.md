# Production evaluation: replay and release gates

[中文](README.md)

[Course page](https://dmxsf.com/en/wiki/evaluation/production-evaluation/) · [Implementation](../../../src/dmxsf/evaluation/production.py)

Compare baseline and candidate on the same tasks. Add observations for the new locale and vary `max_loss` to distinguish insufficient evidence from a regression.

```bash
uv run dmxsf run evaluation/production-evaluation
```

## Full run on Apple M2 Max with Python 3.12.14

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
