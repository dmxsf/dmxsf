# 公式速查：参数量、显存与算力

[English](README.en.md)

[大模型速查表：参数量、算力、显存与解码速度怎么估](https://dmxsf.com/wiki/start/cheat-sheet/) · [实验脚本](run.py) · [库代码](../../../src/dmxsf/architecture/configs.py)

交叉核对公开模型参数量、KV 显存、训练 FLOPs 和 pass@k。

## 运行命令

```bash
uv run dmxsf run start/cheat-sheet
```

## 练习

修改 Llama 3 8B 的 KV 头数，重算每 token 和 8192 token 的显存。

## 完整运行（Apple M2 Max CPU，Python 3.12.14，种子 0）

```text
# LLM formula sheet: parameters, memory and compute
  course: https://dmxsf.com/wiki/start/cheat-sheet/
          https://dmxsf.com/en/wiki/start/cheat-sheet/

## 1. Count from actual model shapes without allocating storage
  [ok] Llama 2 7B  (6,738,415,616)
  [ok] Llama 3 8B  (8,030,261,248)
  [ok] Mistral 7B  (7,241,732,096)
  [ok] Qwen2.5 7B  (7,615,616,512)
  [ok] Llama 3 70B  (70,553,706,496)

## 2. KV cache and training budget
  [ok] Llama 3 8B at BF16 needs 128 KiB/token and 1 GiB at 8192 tokens
  [ok] 405B x 15.6T tokens is about 3.8e25 FLOP  (3.79e+25, 26.6M GPU-hours)
  [ok] pass@k matches direct combinations

All checks passed. (1.0s)
```
