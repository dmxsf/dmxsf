# KV Cache

课程页面：[KV Cache 与推理性能优化](https://dmxsf.com/wiki/inference/kv-cache/)  
[English](README.en.md)

显存公式对照真实缓存张量；有缓存与无缓存的解码结果完全一致；节省的量用两种方式衡量：处理的 token 位置数（精确）与 CPU 计时（看比值）；再看预填充与解码的差别。

## 运行

```bash
uv run python labs/inference/kv-cache/run.py            # 完整运行，几秒
uv run python labs/inference/kv-cache/run.py --quick    # 缩小规模的冒烟测试，几秒
```

也可以写成 `uv run dmxsf run kv-cache`（同样可以加 `--quick`）。`--seed N` 换随机种子，默认 0。脚本分节打印结果，并用 `[ok]` / `[FAIL]` 逐条检查课程页面里的结论：全部通过时退出码为 0，有一条失败就是 1；只有完整运行才做的检查在 `--quick` 下标为 `[skip]`。

## 代码

| 位置 | 内容 |
| --- | --- |
| `labs/inference/kv-cache/run.py` | 实验脚本，按编号分节 |
| `src/dmxsf/inference/kv_cache.py` | KV Cache：记住每层的键和值，让每个新 token 只花一步而不是重算全部 |

单元测试：`uv run pytest tests/test_inference_kv_cache.py`

## 看什么

- 第 1 节：KV Cache 的内存公式与页面的数字：Llama 3 8B 每个 token 128 KiB，8,192 个 token 1 GiB，32 条序列 32 GiB，比权重（16.1 GB）还大；没有 GQA 的 Llama 2 7B 每个 token 512 KiB，是它的 4 倍；KV 头数与精度相乘：MHA fp16 的 512 KiB 到 GQA-8 int8 的 64 KiB，再到 MQA int4 的 4 KiB。
- 第 2 节：公式对真实张量：预填充 100 个 token 后，4 / 2 / 1 个 KV 头的缓存是 409,600 / 204,800 / 102,400 字节，与公式一字不差；预分配的缓存从第一个 token 起就占满整个长度。
- 第 3 节：有无缓存的解码一致：64 个贪心 token 完全相同，增量路径与整段重算的 logits 最大差 8.3e-07；为写 64 个 token 处理的位置数是 6,112 对 128（48 倍）。
- 第 4 节：时间（读比值，不读绝对值）：4.8M 参数的 GPT 写 256 个 token，没有缓存时每个 token 的耗时（每三分之一段的中位数）随序列变长而上升（10.2 → 17.0 ms），有缓存时几乎不变（1.7 → 1.5 ms），整段快 8.3 倍；预填充每个 token 0.06 ms，解码 1.72 ms，预填充每 token 便宜 31 倍：解码要读全部权重与整个缓存才产出一个 token，受内存与开销限制，这就是服务系统要把许多序列合成一批的原因。

## 练习

改了参数再运行：核对页面数字的检查会变成 `[FAIL]`，这是预期的，看数字本身怎么变。

1. 第 1 节的 `for bytes_ in (2, 1, 0.5)` 里加一个 0.25，并在表头 `["attention", "fp16", "int8", "int4"]` 里补一列 int2：MQA 加 2 比特缓存时每个 token 只要多少 KiB？
2. 第 4 节 `p_big, n_big = (32, 32) if QUICK else (128, 256)` 里把 256 改成 384：没有缓存时最后三分之一的每 token 耗时又涨多少？有缓存的那条呢？

## 注意

- 第 4 节的毫秒与 GFLOP/s 是这台机器上的计时；检查只看比值与方向（加速至少 3 倍、预填充每 token 更便宜），阈值放得很宽。

## 输出

下面是在 Apple M2 Max（12 核，仅用 CPU）、Python 3.12、PyTorch 2.15.0 上的完整运行结果。换一台机器，涉及训练的数字可能在末位上不同，计时类的数字取决于机器；检查用的是阈值与容差，不要求逐位相同。

<details>
<summary>完整运行的输出（`labs/inference/kv-cache/run.py`）</summary>

```text
# KV cache: the memory formula, exactness, the saving, prefill against decode
  course: https://dmxsf.com/wiki/inference/kv-cache/
          https://dmxsf.com/en/wiki/inference/kv-cache/

## 1. The memory formula and the page's numbers (exact)
  fp16 cache; the shape is layers x kv_heads x head_dim:
  model         shape          KiB per token  GiB per 8,192 tokens  GiB for 32 sequences  fp16 weights (GB)
  ------------  -------------  -------------  --------------------  --------------------  -----------------
    Llama 2 7B  32 x 32 x 128            512                  4.00                 128.0               13.5
    Llama 3 8B   32 x 8 x 128            128                  1.00                  32.0               16.1
   Llama 3 70B   80 x 8 x 128            320                  2.50                  80.0              141.1
  Qwen2.5-0.5B    24 x 2 x 64             12                  0.09                   3.0                1.0
  [ok] Llama 3 8B: 131,072 bytes = 128 KiB per token, 1 GiB per 8,192 tokens, 32 GiB for 32 sequences
  [ok] ...which is more than the weights (16.1 GB); Llama 2 7B (32 KV heads) needs 512 KiB per token, 4x as much  (weights 16.06 GB)

  KiB per token for the Llama 3 8B shape, by number of KV heads and cache precision:
  attention             fp16   int8   int4 
  --------------------  -----  -----  -----
     MHA (32 KV heads)  512.0  256.0  128.0
  GQA (8, as released)  128.0   64.0   32.0
               GQA (4)   64.0   32.0   16.0
               MQA (1)   16.0    8.0    4.0
  [ok] fewer KV heads and fewer bits multiply: MHA fp16 512 KiB -> GQA-8 int8 64 KiB -> MQA int4 4 KiB

## 2. The formula against real tensors
  the 0.86M byte-level GPT, fp32, after a 100-token prefill:
  model       bytes in the cache  formula
  ----------  ------------------  -------
  4 KV heads              409600   409600
  2 KV heads              204800   204800
  1 KV heads              102400   102400
  [ok] the cache holds exactly what the formula says (and GQA shrinks it in proportion)
  [ok] a preallocated cache costs its full length from the first token (reserve for the longest request)  (524,288 bytes before any token is stored)

## 3. Cached and uncached decoding agree
  [ok] the same 64 greedy tokens with and without the cache
  [ok] logits from the incremental path equal a full recompute up to float rounding  (max difference 8.3e-07)

  token positions processed to write 64 tokens after a 64-token prompt: without a cache 6,112, with one 128 (48x).
  [ok] counted positions match the page's arithmetic: sum of (P + t) without a cache, P + n with one  (6,112 and 128)

## 4. Time: the saving, and prefill against decode (CPU; read the ratios)
  a 4.8M-parameter GPT (6 layers, d = 256) writing 256 tokens after a 128-token prompt; median ms per new token, by third of the run:
            first third  middle third  last third  whole run (s)
  --------  -----------  ------------  ----------  -------------
  no cache         10.2          14.6        17.0           3.58
  KV cache          1.7           1.6         1.5           0.42
  [ok] the cost per token rises through the run without a cache, and rises more slowly with one  (last third over first: no cache x1.67, cache x0.89)
  [ok] the whole run is at least 3x faster with the cache  (8.3x)
  the whole run is 8.3x faster with the cache (prefill included).

  prefill: 7.1 ms for 128 tokens = 0.06 ms per token, about 171 GFLOP/s;
  decode:  1.72 ms per token, about 5.6 GFLOP/s.
  [ok] prefill handles a token far cheaper than decode does (parallel over the prompt)  (31x cheaper per token)
  decode reads the weights and the whole cache to produce one token; at batch 1 that is
  memory and overhead bound, which is why serving systems batch many sequences per step.

All checks passed. (4.4s)
```

</details>
