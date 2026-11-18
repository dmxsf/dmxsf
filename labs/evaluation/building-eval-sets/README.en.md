# Evaluation sets: noisy labels, agreement and item information

[中文](README.md)

[Building Evaluation Sets: Question Sources, Label Quality and Item Information](https://dmxsf.com/en/wiki/evaluation/building-eval-sets/) · [Lab script](run.py) · [Library code](../../../src/dmxsf/evaluation/eval_sets.py)

Check a four-option label-noise formula by simulation, compute Cohen's kappa, and locate a 2PL item's most informative difficulty.

## Run the lab

```bash
uv run dmxsf run evaluation/building-eval-sets
```

## Exercise

Reduce label noise from 10% to 5% and measure how much of a four-point true gap survives.

## Full run (Apple M2 Max CPU, Python 3.12.14, seed 0)

```text
# Evaluation sets: noisy labels, agreement and item information
  course: https://dmxsf.com/wiki/evaluation/building-eval-sets/
          https://dmxsf.com/en/wiki/evaluation/building-eval-sets/

## 1. What an independently wrong label does to scores
  [ok] the simulation reproduces predicted agreement near 72.7%  (0.728 vs 0.727)
  [ok] the article's 6.49% label error requires about 1.198x as many items  (1.198)

## 2. Agreement and information
  [ok] 90% raw agreement with balanced marginals gives kappa 0.8
  [ok] the 2PL information peaks at difficulty with value discrimination squared over four

All checks passed. (0.0s)
```
