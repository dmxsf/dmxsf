# 前缀缓存：命中与排队

[English](README.en.md)

[课程页](https://dmxsf.com/wiki/inference/prefix-caching/) · [实现](../../../src/dmxsf/inference/prefix_routing.py)

用两种缓存路由回放突发与等间隔请求。

```bash
uv run dmxsf run inference/prefix-caching
```

缩短到达间隔，比较中位数、最大值和命中数从哪里开始分离。

## 完整运行 (Apple M2 Max, Python 3.12.14)

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
