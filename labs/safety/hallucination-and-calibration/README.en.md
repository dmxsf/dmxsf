# Calibration: withheld answers, confidence and accuracy

[中文](README.md)

[Hallucination and Calibration: Why Models Sound Sure and Are Wrong](https://dmxsf.com/en/wiki/safety/hallucination-and-calibration/) · [Lab script](run.py) · [Library code](../../../src/dmxsf/evaluation/calibration.py)

Fit temperature on held-out calibration examples and plot the trade-off between coverage and error using synthetic scores.

## Run the lab

```bash
uv run dmxsf run safety/hallucination-and-calibration
```

## Exercise

Choose a higher confidence threshold and compare answered-item accuracy with total coverage.

## Full run (Apple M2 Max CPU, Python 3.12.14, seed 0)

```text
# Calibration: withheld answers, confidence and accuracy
  course: https://dmxsf.com/wiki/safety/hallucination-and-calibration/
          https://dmxsf.com/en/wiki/safety/hallucination-and-calibration/

## 1. Use calibration data to choose temperature, then evaluate separately
  [ok] temperature reduces held-out negative log likelihood and preserves class decisions  (T=3, 0.781 -> 0.603)

## 2. Abstain below a confidence threshold
  [ok] in this synthetic world abstention reduces coverage and improves answered accuracy  (coverage 100.00% -> 16.15%, accuracy 67.04% -> 84.94%)
  [ok] no retained answers means no conditional accuracy

All checks passed. (0.0s)
```
