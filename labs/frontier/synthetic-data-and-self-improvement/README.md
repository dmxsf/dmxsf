# 合成数据：长尾消失与验证筛选

[English](README.en.md)

[合成数据与自我改进：模型给模型造数据，何时有效，何时塌缩](https://dmxsf.com/wiki/frontier/synthetic-data-and-self-improvement/) · [实验脚本](run.py) · [库代码](../../../src/dmxsf/pretraining/synthetic.py)

测量逐代替换数据时消失的长尾类别，再用独立验证器筛选合成标签。

## 运行命令

```bash
uv run dmxsf run frontier/synthetic-data-and-self-improvement
```

## 练习

增大样本预算，对比缺失质量公式与逐代替换的模拟结果。

## 完整运行（Apple M2 Max CPU，Python 3.12.14，种子 0）

```text
# Synthetic data: rare-event loss and verified selection
  course: https://dmxsf.com/wiki/frontier/synthetic-data-and-self-improvement/
          https://dmxsf.com/en/wiki/frontier/synthetic-data-and-self-improvement/

## 1. Once a category disappears, pure replacement cannot rediscover it
  [ok] first-generation missing mass matches the exact occupancy formula  (0.0259 vs 0.0237)
  [ok] support never grows along a replacement chain

## 2. Inject a correctness signal
  [ok] independent arithmetic verification retains only correctly labelled examples  (2/4 accepted)

All checks passed. (0.0s)
```
