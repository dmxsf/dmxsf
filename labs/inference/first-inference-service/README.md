# 第一个推理服务：本机 HTTP 接口

[English](README.en.md)

[课程页](https://dmxsf.com/wiki/inference/first-inference-service/) · [实现](../../../src/dmxsf/inference/first_service.py)

向演示推理器发送真实的本机 HTTP 请求并检查状态码。

```bash
uv run dmxsf run inference/first-inference-service
```

把上限改成一，观察输出；再试已经结束的提示。

## 完整运行 (Apple M2 Max, Python 3.12.14)

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
