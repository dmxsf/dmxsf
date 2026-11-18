# 世界模型：观察与重规划

[English](README.en.md)

[课程页](https://dmxsf.com/wiki/multimodal/world-models/) · [实现](../../../src/dmxsf/multimodal/world_models.py)

在有隐藏障碍的网格上对比固定计划和反馈重规划。

```bash
uv run dmxsf run multimodal/world-models
```

移动墙的位置，再比较两个策略的路径和终态。

## 完整运行 (Apple M2 Max, Python 3.12.14)

```text
# World models: plan, observe, and replan
  course: https://dmxsf.com/wiki/multimodal/world-models/
          https://dmxsf.com/en/wiki/multimodal/world-models/

## Hidden wall
open loop: ((0, 0), (0, 1), (0, 1), (0, 1))
replanned: ((0, 0), (0, 1), (0, 1), (1, 1), (1, 2), (1, 3), (0, 3))
  [ok] open loop stalls
  [ok] feedback reaches the goal in six actions

All checks passed. (0.0s)
```
