# Collective communication: simulate a ring all-reduce

[中文](README.md)

[Collective Communication: Why All-Reduce Has So Many Algorithms](https://dmxsf.com/en/wiki/systems/collective-communication/) · [Lab script](run.py) · [Library code](../../../src/dmxsf/systems/collectives.py)

Track each chunk through reduce-scatter and all-gather, then compare the observed per-rank bytes with the alpha-beta model.

## Run the lab

```bash
uv run dmxsf run systems/collective-communication
```

## Exercise

Run four and eight ranks at the same message size. Compare transmitted bytes and the number of rounds.

## Full run (Apple M2 Max CPU, Python 3.12.14, seed 0)

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
