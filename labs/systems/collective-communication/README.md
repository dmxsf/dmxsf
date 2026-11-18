# 集合通信：模拟环形 All-Reduce

[English](README.en.md)

[集合通信：all-reduce 为什么有这么多种算法](https://dmxsf.com/wiki/systems/collective-communication/) · [实验脚本](run.py) · [库代码](../../../src/dmxsf/systems/collectives.py)

逐轮追踪 Reduce-Scatter 与 All-Gather，再用 α-β 模型核对每个 rank 的通信量。

## 运行命令

```bash
uv run dmxsf run systems/collective-communication
```

## 练习

保持消息大小不变，比较 4 个和 8 个 rank 的通信字节及轮数。

## 完整运行（Apple M2 Max CPU，Python 3.12.14，种子 0）

```text
# Collective communication: simulate a ring all-reduce
  course: https://dmxsf.com/wiki/systems/collective-communication/
          https://dmxsf.com/en/wiki/systems/collective-communication/

## 1. Reduce-scatter then all-gather
  [ok] every rank receives the elementwise global sum
  [ok] eight ranks need 14 rounds and send 1.75 vectors per rank  (14 rounds, 448 bytes per rank)

## 2. Count rounds and bytes separately
  [ok] small messages pay mainly the 14 startup latencies  (70.004480 us)

All checks passed. (0.0s)
```
