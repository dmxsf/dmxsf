# 基准污染：检测措辞重合并分切片复核

[English](README.en.md)

[基准污染：怎么造出来、怎么检测、怎么避免](https://dmxsf.com/wiki/evaluation/benchmark-contamination/) · [实验脚本](run.py) · [库代码](../../../src/dmxsf/evaluation/contamination.py)

用合成题计算真实重合标记，再分切片检验总体平均怎样掩盖污染。

重合检测实际运行，分切片分数为示意。Wiki 中三种子微调结果使用正文实验。

## 运行命令

```bash
uv run dmxsf run evaluation/benchmark-contamination
```

## 练习

缩短重合窗口的词数，检查改写能否检出，以及干净题是否被误报。

## 完整运行（Apple M2 Max CPU，Python 3.12.14，种子 0）

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
