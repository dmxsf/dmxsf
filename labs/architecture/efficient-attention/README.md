# 高效注意力：GQA、MLA 与 FlashAttention

课程页面：[高效注意力：MQA、GQA、MLA 与 FlashAttention](https://dmxsf.com/wiki/architecture/efficient-attention/)  
[English](README.en.md)

KV Cache 公式与课程页的表格、真实张量对拍；MLA 缓存潜向量；FlashAttention 的分块与在线 softmax；滑动窗口注意力；把 MHA 模型通过头的均值池化转成 GQA 并微调。

## 运行

```bash
uv run python labs/architecture/efficient-attention/run.py            # 完整运行，约 55 秒
uv run python labs/architecture/efficient-attention/run.py --quick    # 缩小规模的冒烟测试，几秒
```

也可以写成 `uv run dmxsf run efficient-attention`（同样可以加 `--quick`）。`--seed N` 换随机种子，默认 0。脚本分节打印结果，并用 `[ok]` / `[FAIL]` 逐条检查课程页面里的结论：全部通过时退出码为 0，有一条失败就是 1；只有完整运行才做的检查在 `--quick` 下标为 `[skip]`。

## 代码

| 位置 | 内容 |
| --- | --- |
| `labs/architecture/efficient-attention/run.py` | 实验脚本，按编号分节 |
| `src/dmxsf/architecture/attention.py` | 缩放点积注意力、多头 / 分组查询注意力与因果掩码 |
| `src/dmxsf/architecture/mla.py` | 多头潜在注意力（MLA），DeepSeek-V2 与 V3 用的注意力 |
| `src/dmxsf/inference/kv_cache.py` | KV Cache：记住每层的键和值，让每个新 token 只花一步而不是重算全部 |

单元测试：`uv run pytest tests/test_architecture_attention.py tests/test_architecture_mla.py tests/test_inference_kv_cache.py`

## 看什么

- 第 1 节：课程页的 KV Cache 表由公式算出，每 token 的大小是 512 / 128 / 320 / 504 / 67.5 / 68.6 KiB（Llama 2 7B、3 8B、3 70B、3 405B、DeepSeek-V2、V3），128K 上下文下 64 / 16 / 40 / 63 / 8.44 / 8.58 GiB；DeepSeek-V3 若用 MHA 是每 token 3.81 MiB，MLA 只有 1/56.9。
- 第 2 节：在真实的缓存张量上量出来的字节数与公式相等（MHA 4096 B、GQA-2 1024 B、MQA 512 B、MLA 1152 B）。
- 第 3 节：MLA 的“吸收”不改变结果（误差约 1e-16 到 3e-07），解码时只缓存 d_c + d_R = 80 个数；缓存越长吸收越划算，8192 个 token 时一步快 3.6 倍，但乘加次数少了约 30 倍，时间只少 3.6 倍：这么小的一步已经不再受乘加限制。
- 第 4 节：分块加在线 softmax 的注意力与标准注意力在 float64 下只差 4.6e-14，最大张量从 N×N（8192 时 256 MiB）降到 2 MiB；分块本身不减少计算量，跳过对角线以上的块（因果）才减少，只剩 36/64。
- 第 5 节：滑动窗口的信息每层向前传 window − 1 个位置（窗口 4、4 层可达 12）；缓存停在 30,720 B，不再随长度增长。
- 第 6 节：把 MHA 模型的头做平均池化变成 GQA 后损失从 1.933 跳到 2.660，只再训 25 步就回到 2.075，100 步回到 1.958；“平均池化”与“只保留第一个头”两种起点的差别（0.068）小于种子之间的波动（0.328），这个实验分不出谁更好，实验里写明了。

## 练习

改了参数再运行：核对页面数字的检查会变成 `[FAIL]`，这是预期的，看数字本身怎么变。

1. 把第 6 节 `steps_total, uptrain = (30, 5) if args.quick else (500, 25)` 里的 25 改成 100（转换之后继续训练 100 步和 400 步），损失能追上从头训练的 GQA-2 吗？
2. 第 5 节把窗口从 16 改成 64，缓存停在多少字节？与公式（window − 1 个 token）一致吗？

## 输出

下面是在 Apple M2 Max（12 核，仅用 CPU）、Python 3.12、PyTorch 2.15.0 上的完整运行结果。换一台机器，涉及训练的数字可能在末位上不同，计时类的数字取决于机器；检查用的是阈值与容差，不要求逐位相同。

<details>
<summary>完整运行的输出（`labs/architecture/efficient-attention/run.py`）</summary>

```text
# Efficient attention
  course: https://dmxsf.com/wiki/architecture/efficient-attention/
          https://dmxsf.com/en/wiki/architecture/efficient-attention/

## 1. KV cache per token: the formulas against the course page's table
  model         attention             per token (BF16)  128K tokens
  ------------  --------------------  ----------------  -----------
    Llama 2 7B      MHA, 32 KV heads         512.0 KiB    64.00 GiB
    Llama 3 8B       GQA, 8 KV heads         128.0 KiB    16.00 GiB
   Llama 3 70B       GQA, 8 KV heads         320.0 KiB    40.00 GiB
  Llama 3 405B       GQA, 8 KV heads         504.0 KiB    63.00 GiB
   DeepSeek-V2  MLA, d_c=512, d_R=64          67.5 KiB     8.44 GiB
   DeepSeek-V3  MLA, d_c=512, d_R=64          68.6 KiB     8.58 GiB
  [ok] per-token sizes: 512 / 128 / 320 / 504 / 67.5 / 68.6 KiB
  [ok] 128K-token caches: 64 / 16 / 40 / 63 / 8.44 / 8.58 GiB
  [ok] the formula agrees with kv_bytes_per_token on the real Llama configs
  DeepSeek-V3 as MHA: 3.81 MiB per token; MLA caches 68.6 KiB, 1/56.9
  [ok] course page: 3.81 MiB as MHA, 1/56.9 with MLA, and 576 numbers = 2.25 GQA groups

  Tensor parallelism: a KV head cannot be split, so fewer KV heads than GPUs means copies.
  8-way tensor parallel  KV heads per GPU  KV heads stored in total
  ---------------------  ----------------  ------------------------
      MHA (64 KV heads)                 8                        64
       GQA (8 KV heads)                 1                         8
        MQA (1 KV head)                 1                         8
  [ok] with 8 GPUs MQA stores as many KV heads as GQA-8 (the copies cancel the saving)

## 2. The same numbers, measured on real caches
  4 layers, 8 query heads of 16 dims, 100 tokens, float32 (4 bytes)
  design               numbers / token / layer  measured bytes / token
  -------------------  -----------------------  ----------------------
                  MHA                      256                  4096 B
      GQA, 2 KV heads                       64                  1024 B
                  MQA                       32                   512 B
  MLA (d_c=64, d_R=8)                       72                  1152 B
  [ok] measured cache sizes equal the formulas (2 g d_h L x 4 bytes; (d_c + d_R) L x 4 bytes)
  [ok] MLA with d_c = 4 d_h, d_R = d_h/2 caches as much as 2.25 GQA groups, whatever the number of heads

## 3. MLA: caching a latent vector, and never rebuilding the keys
  NumPy toy (d=256, 8 heads, d_c=48): max |explicit - absorbed| = 6.1e-16
  numbers cached per token and layer: MHA 512, MLA 48 (without the RoPE key)
  [ok] the course page's absorption check (error ~1e-16)
  full module with causal masking and decoupled RoPE, float32:
    absorbed vs rebuilt keys, whole sequence : max error 3.0e-07
    absorbed decoding from the latent cache  : max error 3.0e-07
    cache after 48 tokens: 80 numbers per token and layer
  [ok] absorbed decoding reproduces the rebuilt-keys result
  [ok] the cache holds d_c + d_R = 80 numbers per token

  Why absorb? One decode step with the cache already full (random weights, CPU):
  8 heads, d_c=128, median of 15 steps
  cached  rebuild MACs  time     absorbed MACs  time     faster by
  ------  ------------  -------  -------------  -------  ---------
     512        33.9 M  0.64 ms          1.2 M  0.40 ms       1.6x
    2048       135.5 M  1.22 ms          4.5 M  0.44 ms       2.8x
    8192       542.1 M  3.62 ms         17.9 M  1.02 ms       3.6x
  The arithmetic drops by an order of magnitude more than the time does: a step this small is no longer limited by multiply-adds.
  [ok] absorption makes a decode step at least 1.5x faster at the longest length  (3.6x at 8192 tokens)

## 4. FlashAttention's algorithm: tiles and an online softmax
  N=1000, d=64, tiles of 128: max |standard - tiled| = 4.6e-14 (float64)
  score matrix 7.6 MiB, one tile 128 KiB (float64)
  [ok] the page's numbers: 7.6 MiB for the full matrix, 128 KiB per tile, and an exact result
  [ok] the gradients of q, k and v agree too  (max difference 5.3e-15)

  The largest tensor any operation creates (float32, d=64, tiles of 128):
  N     standard    tiled   
  ----  ----------  --------
   512    1.00 MiB  0.12 MiB
  2048   16.00 MiB  0.50 MiB
  8192  256.00 MiB  2.00 MiB
  [ok] standard attention builds an N x N matrix; the tiled version's biggest tensor is its N x d output

  Matrix-multiply FLOPs, N=1024, d=64, 4 heads (forward):
  version        GFLOPs  relative
  -------------  ------  --------
       standard   1.074    1.0000
          tiled   1.074    1.0000
  tiled, causal   0.604    0.5625
  [ok] tiling alone does not reduce the arithmetic: the same 4 N^2 d per head
  [ok] skipping the tiles above the diagonal (causal) does: 36/64 of the work

## 5. Sliding-window attention: bounded cache, and information still travels
  window 4: the last position that still 'hears' a change to token 0
  layers  last position  layers x (window - 1)
  ------  -------------  ---------------------
       1              3                      3
       2              6                      6
       3              9                      9
       4             12                     12
  [ok] information travels window - 1 positions per layer (a window of 4 over 4 layers reaches 12)
  Mistral 7B: 32 layers x 4096 = 131,072 tokens of theoretical reach; a 32K sequence needs 8x less cache

  Decoding 128 tokens one by one, window 16 (float32, 2 layers):
  tokens so far  full cache  sliding cache
  -------------  ----------  -------------
              8    16,384 B       16,384 B
             16    32,768 B       30,720 B
             32    65,536 B       30,720 B
             64   131,072 B       30,720 B
            128   262,144 B       30,720 B
  [ok] the sliding cache holds window - 1 tokens and gives the same logits as recomputing everything  (max logit difference 7.2e-07)

## 6. From an MHA model to GQA: mean-pool the heads, then train a little
  2 layers, 4 query heads, 500 steps, then 25 more (5%); validation loss in nats per byte over 3 seed(s)
  model                             mean loss  range over seeds
  --------------------------------  ---------  ----------------
                      MHA, trained      1.933     1.927 - 1.937
    MHA, 25 more steps (reference)      1.937     1.930 - 1.946
       GQA-2, trained from scratch      1.935     1.905 - 1.959
         MHA -> GQA-2, mean-pooled      2.660     2.606 - 2.718
                 ... then 25 steps      2.075     2.069 - 2.080
            ... or 100 steps (20%)      1.958     1.948 - 1.963
     MHA -> GQA-2, first head kept      2.592     2.425 - 2.753
    ... then 25 steps (first head)      2.066     2.035 - 2.122
  The two starting points differ by 0.068 nats, less than the seed-to-seed range 0.328:
  this experiment cannot say which is better. (The GQA paper reports mean-pooling ahead on T5-XXL; nothing here contradicts that.)
  [ok] 25 steps repair most of the damage of the conversion (MHA -> GQA-2, mean-pooled)  (loss 2.660 -> 2.075, MHA 1.933)
  [ok] 25 steps repair most of the damage of the conversion (MHA -> GQA-2, first head kept)  (loss 2.592 -> 2.066, MHA 1.933)
  [ok] the more steps after the conversion, the lower the loss  (2.660 > 2.075 > 1.958)
  After 100 steps the converted model is +0.024 nats from the GQA-2 model trained from scratch (range of the latter 1.905 - 1.959); the checkpoint it started from cost 500 steps.

All checks passed. (102.7s)
```

</details>
