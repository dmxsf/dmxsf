# Synthetic data: rare-event loss and verified selection

[中文](README.md)

[Synthetic Data and Self-Improvement: When Models Make Data, and When It Collapses](https://dmxsf.com/en/wiki/frontier/synthetic-data-and-self-improvement/) · [Lab script](run.py) · [Library code](../../../src/dmxsf/pretraining/synthetic.py)

Measure support lost when each generation replaces the previous one, then see how an independent verifier filters synthetic labels.

## Run the lab

```bash
uv run dmxsf run frontier/synthetic-data-and-self-improvement
```

## Exercise

Increase the sample budget and compare the closed-form missing mass with a replacement chain.

## Full run (Apple M2 Max CPU, Python 3.12.14, seed 0)

```text
# Synthetic data: rare-event loss and verified selection
  course: https://dmxsf.com/wiki/frontier/synthetic-data-and-self-improvement/
          https://dmxsf.com/en/wiki/frontier/synthetic-data-and-self-improvement/

## 1. Once a category disappears, pure replacement cannot rediscover it
  [ok] first-generation missing mass matches the exact occupancy formula  (0.0259 vs 0.0237)
  [ok] support never grows along a replacement chain

## 2. Inject a correctness signal
  [ok] independent arithmetic verification retains only correctly labelled examples  (2/4 accepted)

All checks passed. (0.0s)
```
