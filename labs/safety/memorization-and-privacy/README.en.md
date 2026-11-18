# Privacy audit: exposure for synthetic canaries

[中文](README.md)

[Memorization and Privacy: What Models Remember, How to Measure and Reduce It](https://dmxsf.com/en/wiki/safety/memorization-and-privacy/) · [Lab script](run.py) · [Library code](../../../src/dmxsf/evaluation/privacy.py)

Compare rank exposure and top-one generation using an artificial candidate distribution, then measure uniform-rank exposure.

All candidates and scores are synthetic. Use the Wiki experiment for its separately measured small-model fine-tuning curves.

## Run the lab

```bash
uv run dmxsf run safety/memorization-and-privacy
```

## Exercise

Give the canary the same score as the leading candidate. See how conservative tie handling changes rank.

## Full run (Apple M2 Max CPU, Python 3.12.14, seed 0)

```text
# Privacy audit: exposure for synthetic canaries
  course: https://dmxsf.com/wiki/safety/memorization-and-privacy/
          https://dmxsf.com/en/wiki/safety/memorization-and-privacy/

## 1. Distinguish score rank from exact output
  [ok] the synthetic canary ranks second and has about 12.29 bits of exposure  (rank 2, 12.29 bits)
  [ok] a tie receives the last rank in the tied group

## 2. Use a uniform rank as a reference
  [ok] the uniform-rank reference is about 1.44 bits  (1.4419 bits)

All checks passed. (0.0s)
```
