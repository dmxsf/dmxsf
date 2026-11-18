# Test-time compute: independent draws, reranking and majority voting

[中文](README.md)

[Test-Time Compute: Spending Compute on Inference](https://dmxsf.com/en/wiki/frontier/test-time-compute/) · [Lab script](run.py) · [Library code](../../../src/dmxsf/evaluation/test_time.py)

Compare independent and perfectly correlated draws and compute how a noisy selector changes best-of-N.

## Run the lab

```bash
uv run dmxsf run frontier/test-time-compute
```

## Exercise

Vary single-sample correctness p above and below 0.5. Identify where majority voting helps and where it amplifies a bad answer.

## Full run (Apple M2 Max CPU, Python 3.12.14, seed 0)

```text
# Test-time compute: independent draws, reranking and majority voting
  course: https://dmxsf.com/wiki/frontier/test-time-compute/
          https://dmxsf.com/en/wiki/frontier/test-time-compute/

## 1. More samples only help when the draws add information
  [ok] independent oracle coverage is 0.83193, binary majority is 0.16308  (oracle 0.83193, majority 0.16308)
  [ok] repeating the same draw leaves coverage at the single-draw rate

## 2. A noisy reranker loses attainable coverage
  [ok] the selector helps over one sample and falls short of the oracle  (reranked 0.607; oracle 0.831)

All checks passed. (0.0s)
```
