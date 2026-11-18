# Diffusion and flow matching: verify a Gaussian path

[中文](README.md)

[Diffusion Models and Flow Matching: Generating Images, Actions and Text from Noise](https://dmxsf.com/en/wiki/multimodal/diffusion-and-flow-matching/) · [Lab script](run.py) · [Library code](../../../src/dmxsf/multimodal/diffusion_flow.py)

Measure the forward diffusion variance and integrate a simple analytic flow, checking how Euler error changes with step count.

This exercise uses an analytic Gaussian velocity field. Use the Wiki code for the trained two-moons experiments.

## Run the lab

```bash
uv run dmxsf run multimodal/diffusion-and-flow-matching
```

## Exercise

Double the Euler step count again. Check whether the standard-deviation error roughly halves.

## Full run (Apple M2 Max CPU, Python 3.12.14, seed 0)

```text
# Diffusion and flow matching: verify a Gaussian path
  course: https://dmxsf.com/wiki/multimodal/diffusion-and-flow-matching/
          https://dmxsf.com/en/wiki/multimodal/diffusion-and-flow-matching/

## 1. The variance of a closed-form noising step
  [ok] the measured variance matches the analytical 0.91  (0.9139 vs 0.9100)
  [ok] the noise schedule endpoints return the original signal and pure noise

## 2. More Euler evaluations reduce discretization error
  Euler steps  output std  absolute error
  -----------  ----------  --------------
            8     0.67895         0.12105
           16     0.73725         0.06275
           32     0.76802         0.03198
           64     0.78385         0.01615
  [ok] for this flow, Euler estimates converge toward target std 0.8

All checks passed. (0.0s)
```
