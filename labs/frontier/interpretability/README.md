# 机制可解释性：叠加特征与因果替换

[English](README.en.md)

[机制可解释性：叠加、稀疏自编码器与回路](https://dmxsf.com/wiki/frontier/interpretability/) · [实验脚本](run.py) · [库代码](../../../src/dmxsf/architecture/interpretability.py)

测量非正交特征的干扰，再替换一个受控激活，解释玩具预测为何变化。

特征由实验明确构造。Wiki 中 GPT-2 的干预与稀疏自编码器实验使用正文代码。

## 运行命令

```bash
uv run dmxsf run frontier/interpretability
```

## 练习

增大特征重叠度，比较第二个特征的串扰与替换第一个特征后的预测变化。

## 完整运行（Apple M2 Max CPU，Python 3.12.14，种子 0）

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
