# Data engineering: build and audit a dedup filter

[中文](README.md)

[Pretraining Data Engineering: Filtering, Deduplication and Mixing](https://dmxsf.com/en/wiki/pretraining/data-engineering/) · [Lab script](run.py) · [Library code](../../../src/dmxsf/pretraining/dedup.py)

Compare exact duplicates and shingle similarity, then read the banding curve before choosing a deduplication threshold.

## Run the lab

```bash
uv run dmxsf run pretraining/data-engineering
```

## Exercise

Replace two words instead of one. Record the resulting shingle Jaccard score and banding probability.

## Full run (Apple M2 Max CPU, Python 3.12.14, seed 0)

```text
# Data engineering: build and audit a dedup filter
  course: https://dmxsf.com/wiki/pretraining/data-engineering/
          https://dmxsf.com/en/wiki/pretraining/data-engineering/

## 1. Exact matching and near duplicates
  [ok] exact-match misses a one-word edit, shingle matching sees seven shared triples  (Jaccard 0.5385)
  [ok] disjoint short documents have zero overlap

## 2. Read the banding curve
  Jaccard  candidate probability
  -------  ---------------------
   0.5000                  5.33%
   0.7500                 77.16%
   0.9000                 99.96%
    1.000                100.00%
  [ok] 14 bands of 8 rows favor close pairs but have no hard cutoff

All checks passed. (0.0s)
```
