# Hardware basics: storage and copy time

[中文](README.md)

[Course page](https://dmxsf.com/en/wiki/systems/hardware-basics/) · [Implementation](../../../src/dmxsf/systems/hardware_basics.py)

Compute tensor sizes and a copy-time lower bound with explicit units.

```bash
uv run dmxsf run systems/hardware-basics
```

Double the first dimension and recompute storage and ideal copy time.

## Full run (Apple M2 Max, Python 3.12.14)

```text
# Hardware basics: storage and copy time
  course: https://dmxsf.com/wiki/systems/hardware-basics/
          https://dmxsf.com/en/wiki/systems/hardware-basics/

## One square matrix
  [ok] two element sizes  (2097152, 4194304 bytes)
  [ok] ideal copy bound  (3.906 ms)

All checks passed. (0.0s)
```
