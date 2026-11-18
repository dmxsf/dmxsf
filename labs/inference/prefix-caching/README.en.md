# Prefix caching: cache reuse versus queues

[中文](README.md)

[Course page](https://dmxsf.com/en/wiki/inference/prefix-caching/) · [Implementation](../../../src/dmxsf/inference/prefix_routing.py)

Replay bursty and spaced arrivals with two cache routing policies.

```bash
uv run dmxsf run inference/prefix-caching
```

Shrink the spacing and find where median latency, worst latency, and hits diverge.

## Full run (Apple M2 Max, Python 3.12.14)

```text
# Prefix caching: cache reuse versus queues
  course: https://dmxsf.com/wiki/inference/prefix-caching/
          https://dmxsf.com/en/wiki/inference/prefix-caching/

## Burst and spaced requests
  [ok] fixed affinity matches the wiki  (RoutingResult(hits=6, median_ms=10.0, worst_ms=18))
  [ok] finish-time routing matches the wiki  (RoutingResult(hits=5, median_ms=9.5, worst_ms=13))
  [ok] with idle warm workers both routes agree

All checks passed. (0.0s)
```
