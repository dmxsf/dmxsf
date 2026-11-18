# First inference service: a local HTTP endpoint

[中文](README.md)

[Course page](https://dmxsf.com/en/wiki/inference/first-inference-service/) · [Implementation](../../../src/dmxsf/inference/first_service.py)

Send real local HTTP requests to a toy decoder and check success and errors.

```bash
uv run dmxsf run inference/first-inference-service
```

Set the output limit to one, then try a prompt that immediately reaches the stop condition.

## Full run (Apple M2 Max, Python 3.12.14)

```text
# First inference service: a local HTTP endpoint
  course: https://dmxsf.com/wiki/inference/first-inference-service/
          https://dmxsf.com/en/wiki/inference/first-inference-service/

## Loopback HTTP
  [ok] decode returns two tokens  ((200, {'tokens': ['two', 'three']}))
  [ok] invalid limit returns 400
  [ok] unknown route returns 404

All checks passed. (0.0s)
```
