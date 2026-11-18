# Benchmark contamination: detect wording overlap and audit slices

[中文](README.md)

[Benchmark Contamination: How It Happens, How to Detect It, How to Avoid It](https://dmxsf.com/en/wiki/evaluation/benchmark-contamination/) · [Lab script](run.py) · [Library code](../../../src/dmxsf/evaluation/contamination.py)

Build real overlap flags on synthetic benchmark questions and audit how averaging hides contaminated and clean slices.

The detector actually runs. Slice scores are illustrative. Use the Wiki experiment for the three-seed fine-tuning results.

## Run the lab

```bash
uv run dmxsf run evaluation/benchmark-contamination
```

## Exercise

Shorten the number of words in the overlap window. See whether the paraphrases are caught and whether clean items are flagged.

## Full run (Apple M2 Max CPU, Python 3.12.14, seed 0)

```text
# Benchmark contamination: detect wording overlap and audit slices
  course: https://dmxsf.com/wiki/evaluation/benchmark-contamination/
          https://dmxsf.com/en/wiki/evaluation/benchmark-contamination/

## 1. Detect verbatim leaks and inspect paraphrases
  [ok] the long exact window catches this verbatim slice and misses the paraphrased slice  (13-gram; hits by slice [8, 0, 0])

## 2. Report results by data slice
  slice       count  baseline correct  candidate correct
  ----------  -----  ----------------  -----------------
    verbatim      8                 4                  8
  paraphrase      8                 4                  6
       clean      8                 4                  4
  [ok] the demonstration aggregate gain is 25 points and the clean gain is zero

All checks passed. (0.0s)
```
