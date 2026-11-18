# State space and hybrids: equivalent scans and memory

[中文](README.md)

[State Space Models and Hybrids: Trading the KV Cache for a Fixed-Size State](https://dmxsf.com/en/wiki/architecture/state-space-and-hybrid/) · [Lab script](run.py) · [Library code](../../../src/dmxsf/architecture/state_space.py)

Compare recurrent and explicit causal forms of the same selective update and account for hybrid state and cache.

## Run the lab

```bash
uv run dmxsf run architecture/state-space-and-hybrid
```

## Exercise

Change sequence length and the number of attention layers. Compare the fixed recurrent state and the term that grows with context.

## Full run (Apple M2 Max CPU, Python 3.12.14, seed 0)

```text
# State space and hybrids: equivalent scans and memory
  course: https://dmxsf.com/wiki/architecture/state-space-and-hybrid/
          https://dmxsf.com/en/wiki/architecture/state-space-and-hybrid/

## 1. Recurrence and explicit causal mixing
  [ok] the two evaluation orders agree in float64  (largest gap 1.8e-15)
  [ok] a zero decay resets history at that step

## 2. A hybrid cache ledger
  [ok] doubling sequence length doubles KV storage and leaves the state fixed  ((134217728, 786432) vs (268435456, 786432) bytes)

All checks passed. (0.0s)
```
