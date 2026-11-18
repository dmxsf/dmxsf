# Neural networks: gradients and XOR

[中文](README.md)

[Neural Network Basics: From Linear Layers to Backpropagation](https://dmxsf.com/en/wiki/foundations/neural-network-basics/) · [Lab script](run.py) · [Library code](../../../src/dmxsf/foundations/neural_network.py)

Check manual backpropagation with finite differences, then fit XOR with and without a hidden layer.

## Run the lab

```bash
uv run dmxsf run foundations/neural-network-basics
```

## Exercise

Set the learning rate to zero and repeat the XOR experiment. Compare the final cross-entropy and predictions.

## Full run (Apple M2 Max CPU, Python 3.12.14, seed 0)

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
