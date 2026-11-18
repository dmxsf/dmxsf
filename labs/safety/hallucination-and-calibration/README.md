# 幻觉与校准：弃答、置信度与准确率

[English](README.en.md)

[幻觉与校准：模型为什么会一本正经地胡说](https://dmxsf.com/wiki/safety/hallucination-and-calibration/) · [实验脚本](run.py) · [库代码](../../../src/dmxsf/evaluation/calibration.py)

用留出的校准样本拟合温度，在合成分数上观察回答率与错误率的折衷。

## 运行命令

```bash
uv run dmxsf run safety/hallucination-and-calibration
```

## 练习

提高置信门槛，比较已答题准确率与整体回答率。

## 完整运行（Apple M2 Max CPU，Python 3.12.14，种子 0）

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
