# 扩散与流匹配：核对高斯路径

[English](README.en.md)

[扩散模型与流匹配：从噪声一步步生成图像、动作与文本](https://dmxsf.com/wiki/multimodal/diffusion-and-flow-matching/) · [实验脚本](run.py) · [库代码](../../../src/dmxsf/multimodal/diffusion_flow.py)

实测扩散前向方差，积分一条可解析的流，再比较欧拉误差怎样随步数变化。

这个实验使用可解析的高斯速度场。Wiki 中训练两个半月模型的实验使用正文代码。

## 运行命令

```bash
uv run dmxsf run multimodal/diffusion-and-flow-matching
```

## 练习

把欧拉步数再次翻倍，检查标准差误差是否大致减半。

## 完整运行（Apple M2 Max CPU，Python 3.12.14，种子 0）

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
