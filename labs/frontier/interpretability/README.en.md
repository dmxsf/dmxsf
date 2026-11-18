# Interpretability: superposed features and causal patching

[中文](README.md)

[Mechanistic Interpretability: Superposition, Sparse Autoencoders and Circuits](https://dmxsf.com/en/wiki/frontier/interpretability/) · [Lab script](run.py) · [Library code](../../../src/dmxsf/architecture/interpretability.py)

Measure interference between nonorthogonal features and patch one controlled activation to explain a toy prediction change.

The toy features are known by construction. Use the Wiki code for its GPT-2 interventions and sparse autoencoder experiment.

## Run the lab

```bash
uv run dmxsf run frontier/interpretability
```

## Exercise

Increase the feature overlap and compare the leakage into the second feature with the effect of patching feature zero.

## Full run (Apple M2 Max CPU, Python 3.12.14, seed 0)

```text
# Interpretability: superposed features and causal patching
  course: https://dmxsf.com/wiki/frontier/interpretability/
          https://dmxsf.com/en/wiki/frontier/interpretability/

## 1. More features than representation dimensions
  [ok] activating only the first feature creates a 0.6 projection onto the second  ([1.  0.6 0. ])

## 2. Intervene and compare with an irrelevant control
  [ok] patching the removed activation restores the output vector  ([0.  0.8 1. ] -> [2. 2. 1.])
  [ok] patching an activation unchanged by corruption leaves the baseline intact
The activations are named by construction; a real model would require a separate feature-discovery step.

All checks passed. (0.0s)
```
