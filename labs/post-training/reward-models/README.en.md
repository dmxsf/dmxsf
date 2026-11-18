# Reward models: pairwise scores and label noise

[中文](README.md)

[Reward Models: Turning Human Preferences into a Score to Optimize](https://dmxsf.com/en/wiki/post-training/reward-models/) · [Lab script](run.py) · [Library code](../../../src/dmxsf/post_training/reward_models.py)

Fit Bradley–Terry scores on a connected comparison graph and reproduce the agreement ceiling under independent label flips.

## Run the lab

```bash
uv run dmxsf run post-training/reward-models
```

## Exercise

Increase the flip probability to 0.25 and compare the single-label accuracy ceiling and annotator agreement.

## Full run (Apple M2 Max CPU, Python 3.12.14, seed 0)

```text
# Reward models: pairwise scores and label noise
  course: https://dmxsf.com/wiki/post-training/reward-models/
          https://dmxsf.com/en/wiki/post-training/reward-models/

## 1. Recover relative scores from pairwise choices
  [ok] relative scores are recovered up to a single global shift  (max gap 3.3e-16)

## 2. Independent symmetric label flips
  [ok] 10% flips yield 90% single-label ceiling and 82% two-label agreement  (observed agreement 0.818)

All checks passed. (0.0s)
```
