# 隐私审计：合成金丝雀的曝光度

[English](README.en.md)

[记忆与隐私：模型记住了什么，怎样量出来，怎样降下来](https://dmxsf.com/wiki/safety/memorization-and-privacy/) · [实验脚本](run.py) · [库代码](../../../src/dmxsf/evaluation/privacy.py)

用完全合成的候选分布比较排名曝光度与首选输出，再计算均匀排名的参照值。

所有候选和分数均为合成数据。Wiki 中真实小模型的微调曲线使用正文实验。

## 运行命令

```bash
uv run dmxsf run safety/memorization-and-privacy
```

## 练习

令金丝雀和领先候选同分，观察保守并列处理后的排名。

## 完整运行（Apple M2 Max CPU，Python 3.12.14，种子 0）

```text
# Privacy audit: exposure for synthetic canaries
  course: https://dmxsf.com/wiki/safety/memorization-and-privacy/
          https://dmxsf.com/en/wiki/safety/memorization-and-privacy/

## 1. Distinguish score rank from exact output
  [ok] the synthetic canary ranks second and has about 12.29 bits of exposure  (rank 2, 12.29 bits)
  [ok] a tie receives the last rank in the tied group

## 2. Use a uniform rank as a reference
  [ok] the uniform-rank reference is about 1.44 bits  (1.4419 bits)

All checks passed. (0.0s)
```
