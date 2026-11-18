# LLM formula sheet: parameters, memory and compute

[中文](README.md)

[LLM Cheat Sheet: Estimating Parameters, Compute, Memory and Decoding Speed](https://dmxsf.com/en/wiki/start/cheat-sheet/) · [Lab script](run.py) · [Library code](../../../src/dmxsf/architecture/configs.py)

Cross-check released-model parameter counts, KV bytes, training FLOPs and pass@k.

## Run the lab

```bash
uv run dmxsf run start/cheat-sheet
```

## Exercise

Change the KV head count in the Llama 3 8B example. Recompute the per-token and 8192-token KV storage.

## Full run (Apple M2 Max CPU, Python 3.12.14, seed 0)

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
