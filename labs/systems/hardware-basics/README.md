# 硬件入门：容量与复制时间

[English](README.en.md)

[课程页](https://dmxsf.com/wiki/systems/hardware-basics/) · [实现](../../../src/dmxsf/systems/hardware_basics.py)

用一致的单位算张量容量和理想复制时间。

```bash
uv run dmxsf run systems/hardware-basics
```

把第一维加倍，再对照用字节数除以搬运速率的结果。

## 完整运行 (Apple M2 Max, Python 3.12.14)

```text
# Hardware basics: storage and copy time
  course: https://dmxsf.com/wiki/systems/hardware-basics/
          https://dmxsf.com/en/wiki/systems/hardware-basics/

## One square matrix
  [ok] two element sizes  (2097152, 4194304 bytes)
  [ok] ideal copy bound  (3.906 ms)

All checks passed. (0.0s)
```
