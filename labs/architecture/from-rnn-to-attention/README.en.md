# From RNN to attention: gradients and retrieval paths

[中文](README.md)

[From RNNs to Attention: Why the Transformer Appeared](https://dmxsf.com/en/wiki/architecture/from-rnn-to-attention/) · [Lab script](run.py) · [Library code](../../../src/dmxsf/architecture/recurrent.py)

Compare the gradient through a recurrent chain and a gated memory path, then recover an earlier value with attention.

## Run the lab

```bash
uv run dmxsf run architecture/from-rnn-to-attention
```

## Exercise

Change the forget gate from 0.99 to 0.9 and compare the gradient after 100 steps. Then lower attention's score scale.

## Full run (Apple M2 Max CPU, Python 3.12.14, seed 0)

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
