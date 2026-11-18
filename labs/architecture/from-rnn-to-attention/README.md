# 从 RNN 到注意力：梯度与检索路径

[English](README.en.md)

[从 RNN 到注意力：Transformer 为什么出现](https://dmxsf.com/wiki/architecture/from-rnn-to-attention/) · [实验脚本](run.py) · [库代码](../../../src/dmxsf/architecture/recurrent.py)

对比循环链路和门控记忆链路的梯度，再用注意力找回此前的位置。

## 运行命令

```bash
uv run dmxsf run architecture/from-rnn-to-attention
```

## 练习

把遗忘门从 0.99 改成 0.9，比较 100 步后的梯度，再降低注意力分数的缩放倍数。

## 完整运行（Apple M2 Max CPU，Python 3.12.14，种子 0）

```text
# From RNN to attention: gradients and retrieval paths
  course: https://dmxsf.com/wiki/architecture/from-rnn-to-attention/
          https://dmxsf.com/en/wiki/architecture/from-rnn-to-attention/

## 1. Gradient paths through time
  [ok] backprop agrees with multiplying 100 local derivatives  (tanh 7.889e-31, gated 0.366)

## 2. Read an earlier value without the recurrent chain
  [ok] the most relevant earlier key receives over 99.9% of the mass  (target 0, recovered 0.0007)
  [ok] the returned value is near the selected value  (absolute error 0.000737)

All checks passed. (0.0s)
```
