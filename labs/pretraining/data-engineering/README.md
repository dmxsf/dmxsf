# 数据工程：构建并审计去重过滤器

[English](README.en.md)

[预训练数据工程：过滤、去重与配比](https://dmxsf.com/wiki/pretraining/data-engineering/) · [实验脚本](run.py) · [库代码](../../../src/dmxsf/pretraining/dedup.py)

比较精确重复与词片相似度，用分桶概率曲线审视去重阈值。

## 运行命令

```bash
uv run dmxsf run pretraining/data-engineering
```

## 练习

改两个词而非一个，记录词片 Jaccard 分数与分桶命中概率。

## 完整运行（Apple M2 Max CPU，Python 3.12.14，种子 0）

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
