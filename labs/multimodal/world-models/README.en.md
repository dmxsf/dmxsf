# World models: plan, observe, and replan

[中文](README.md)

[Course page](https://dmxsf.com/en/wiki/multimodal/world-models/) · [Implementation](../../../src/dmxsf/multimodal/world_models.py)

Compare fixed execution with feedback on a grid with a hidden wall.

```bash
uv run dmxsf run multimodal/world-models
```

Move the wall and compare traces and final states.

## Full run (Apple M2 Max, Python 3.12.14)

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
