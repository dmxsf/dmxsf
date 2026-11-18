# 神经网络基础：梯度与 XOR

[English](README.en.md)

[神经网络基础：从线性层到反向传播](https://dmxsf.com/wiki/foundations/neural-network-basics/) · [实验脚本](run.py) · [库代码](../../../src/dmxsf/foundations/neural_network.py)

用数值差分独立核对反传，再比较线性层和带隐藏层的网络怎样学习 XOR。

## 运行命令

```bash
uv run dmxsf run foundations/neural-network-basics
```

## 练习

把学习率设为零并重跑 XOR，比较交叉熵与预测。

## 完整运行（Apple M2 Max CPU，Python 3.12.14，种子 0）

```text
# Neural networks: gradients and XOR
  course: https://dmxsf.com/wiki/foundations/neural-network-basics/
          https://dmxsf.com/en/wiki/foundations/neural-network-basics/

## 1. Backpropagation by finite differences
  [ok] every weight agrees with finite differences  (loss 0.5908, max gap 5.8e-11)

## 2. What the nonlinearity changes
  [ok] a linear score cannot classify all four XOR points  (2/4)
  [ok] tanh can learn all four points  ([(0.0006, 4), (0.0006, 4), (0.0006, 4)])

All checks passed. (0.3s)
```
