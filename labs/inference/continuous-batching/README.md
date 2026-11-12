# 推理服务：连续批处理、分页 KV 与前缀缓存

课程页面：[连续批处理与 PagedAttention：推理服务如何同时服务很多请求](https://dmxsf.com/wiki/inference/continuous-batching/) · [LLM 推理服务：延迟指标、实测方法与引擎选型](https://dmxsf.com/wiki/inference/serving/)  
[English](README.en.md)

Roofline 与精确的批量解码；静态与连续批处理的纸面推演和真实模型对比；带写时复制的分页 KV 池；基于哈希的前缀缓存；一个请求的成本（TTFT、TPOT、容量）。看比值，不看毫秒。

## 运行

```bash
uv run python labs/inference/continuous-batching/run.py            # 完整运行，约 10 秒
uv run python labs/inference/continuous-batching/run.py --quick    # 缩小规模的冒烟测试，几秒
```

也可以写成 `uv run dmxsf run continuous-batching`（同样可以加 `--quick`）。`--seed N` 换随机种子，默认 0。脚本分节打印结果，并用 `[ok]` / `[FAIL]` 逐条检查课程页面里的结论：全部通过时退出码为 0，有一条失败就是 1；只有完整运行才做的检查在 `--quick` 下标为 `[skip]`。

这个实验用到缓存的小模型：字节级基础模型（84 万参数）。第一次运行会先在 TinyShakespeare 上训练并缓存到 `~/.cache/dmxsf`（环境变量 `DMXSF_CACHE` 可以改位置），之后直接读取。

## 代码

| 位置 | 内容 |
| --- | --- |
| `labs/inference/continuous-batching/run.py` | 实验脚本，按编号分节 |
| `src/dmxsf/inference/serving.py` | 推理服务：按槽位批量解码、迭代级调度、分页 KV 内存与前缀缓存 |

单元测试：`uv run pytest tests/test_inference_serving.py`

## 看什么

- 第 1 节：为什么批处理划算：H100 的屋脊线是 295 FLOP/字节；Qwen2.5-7B 一条 4,096 token 的序列要读 224 MiB 的 KV，权重是 14.2 GiB，所以 65 条序列读的 KV 就与权重一样多。在这台 CPU 上，一步解码 16 条序列的总吞吐是 1 条的 11.6 倍（每条请求的速度是单独跑时的 0.72）；批里长度各不相同的五条序列，与各自单独解码的结果一致（logits 最大差 1.5e-05）。
- 第 2 节：纸面上的调度器（精确）：页面的 48 个请求，连续批处理 52.5 秒、88.6 token/秒，静态批处理 68.6 秒、67.9 token/秒；时间花在哪里：预填充 16.5 秒，两种调度一样；解码在连续批处理下是理想值的 81%，静态下是 56%（每一批要等 218、453、192 个 token 的最长输出）；页面小组件下面的几条结论在同一个模拟器里重算：输出都等长时连续批处理没有优势，步时与批大小无关时省 25%。
- 第 3 节：同一个循环在真实模型上：模拟器只拿这台机器自己测出的步时表和预填充时间，预测两种调度的完成时间，误差在 25% 以内（这次是连续 −0.8%，静态 −1.5%；GitHub 的 Linux 运行器上最大差 11%）；每个请求得到的字节与单独解码完全一致；连续批处理的完成时间是静态的 0.68，平均完成时间是 0.65；只数步数：95 步对 170 步，少 44%，这一部分来自调度，与硬件无关。
- 第 4 节：KV 显存的账：Qwen2.5-7B 每 token 57,344 字节，每条序列预留 8,192 token 时一张 80 GiB 的卡放得下 123 条，按 16 token 一块分页、每条 1,000 token 时是 999 条；页面的 48 个请求共 10,800 个 token，预留最大长度只用上 2.7% 的内存，分页是 97.0%（每个请求浪费 7.0 个 token，从不超过 15）；块表能把物理上不相邻的块拼回一致的注意力（逐位相同），并行采样共享提示的块（11 块对 20 块）。
- 第 5 节：前缀缓存：文档放前面时，第一个请求算 112 字节，之后的请求复用 96 字节、只算 16 字节，预填充快 2.1 倍（1.19 对 2.45 毫秒）；每次都变的 8 字节头放在前面，什么也复用不了；复用缓存不改变 logits（最大差 0）。
- 第 6 节：一个请求的代价：端到端 = 排队 + 预填充 + (N − 1) × TPOT，公式与实测相差在 10% 以内；预填充每字节比逐条解码快 10 倍以上；同样的代码，A / B 的基准比值在几次试验里最多偏离 1 达 5.1%：比这更小的两种真实配置之差，这个基准看不出来；利特尔定律 L = λW 在模拟的排队系统里成立（相对误差 0.01%）。

## 练习

改了参数再运行：核对页面数字的检查会变成 `[FAIL]`，这是预期的，看数字本身怎么变。

1. 第 1 节 `context, reps = 64, 8 if QUICK else 30` 里把 64 改成 120（每条序列的上下文更长，要读的 KV 更多）：16 条序列的总吞吐还是单条的 11 倍吗？
2. 把 `PAGE_STALL = 0.343` 改成 1.0（一个 128 token 的提示让整批停多久）：连续批处理对静态批处理的优势变大还是变小？

## 注意

- 第 1、3、5、6 节里的毫秒与吞吐是这台机器上的计时，每次运行略有不同（检查只看比值与方向，阈值放得很宽）；这个实验对机器的负载最敏感，跑它的时候不要同时做别的吃 CPU 的事。同一个实验还覆盖了“推理服务”一页，页面链接在上面。

## 输出

下面是在 Apple M2 Max（12 核，仅用 CPU）、Python 3.12、PyTorch 2.15.0 上的完整运行结果。换一台机器，涉及训练的数字可能在末位上不同，计时类的数字取决于机器；检查用的是阈值与容差，不要求逐位相同。

<details>
<summary>完整运行的输出（`labs/inference/continuous-batching/run.py`）</summary>

```text
# Serving: continuous batching, paged KV memory, prefix cache, TTFT, capacity
  course: https://dmxsf.com/wiki/inference/continuous-batching/
          https://dmxsf.com/en/wiki/inference/continuous-batching/
  course: https://dmxsf.com/wiki/inference/serving/
          https://dmxsf.com/en/wiki/inference/serving/

## 1. Why batching pays: the roofline, and a batched decode step that is exact
  H100 SXM: 989 TFLOPS / 3.35 TB/s = 295 FLOP per byte. A linear layer applied to B tokens
  does B FLOP for every byte of BF16 weight it reads, so below B of about 295 a step is paid for by
  reading the weights and extra sequences ride along almost free. The KV cache cannot be shared that way:
  Qwen2.5-7B reads 224 MiB of it per 4,096-token sequence against 14.2 GiB of weights, so 65
  sequences read as much KV as weights.
  [ok] 295 FLOP/B; Qwen2.5-7B: 224 MiB of KV per 4,096-token sequence, 14.2 GiB of weights, 65 sequences  (295.2, 224 MiB, 14.19 GiB, 64.8)

  one decode step on the CPU, 64 tokens of context per sequence (median of 28):
  batch  step ms  total tok/s  per-request tok/s  total vs batch 1
  -----  -------  -----------  -----------------  ----------------
      1     1.03          969                969              1.0x
      2     1.12        1,790                895              1.8x
      3     1.13        2,657                886              2.7x
      4     1.16        3,443                861              3.6x
      6     1.21        4,948                825              5.1x
      8     1.24        6,435                804              6.6x
     12     1.34        8,962                747              9.2x
     16     1.43       11,209                701             11.6x
     24     1.86       12,879                537             13.3x
     32     2.03       15,791                493             16.3x
  [ok] batching pays on this machine too: 16 sequences give more than 3x the tokens per second of one  (11.6x the total throughput; each request runs at 0.72 of its solo speed)
  [ok] five sequences of 5, 13, 29, 41 and 60 tokens decoded in one batch match each decoded alone  (largest logit difference 1.5e-05; same next byte in 50 of 50 steps)

## 2. The page's scheduler on paper: static against continuous batching (exact)
  48 requests, output tokens: min 15, median 61, mean 97, max 453, total 4,656
  [ok] the page's workload: min 15, median 61, mean 97, max 453, total 4,656  ((15, 61.0, 97, 453, 4656))

  16 slots; a 128-token prompt stalls the whole batch for 0.343 s; step times from the page's table:
  mode        makespan s  tok/s  mean done s  p90 done s
  ----------  ----------  -----  -----------  ----------
  continuous        52.5   88.6         29.9        47.0
      static        68.6   67.9         37.7        64.0
  [ok] continuous 52.5 s, 88.6 tok/s, 29.9, 47.0 and static 68.6, 67.9, 37.7, 64.0, as printed on the page  (continuous [52.5, 88.6, 29.9, 47.0], static [68.6, 67.9, 37.7, 64.0])

  where the time goes: 4,656 tokens at 16 sequences of 100.3 ms would take 29.2 s; prefill takes 16.5 s in both schedules.
  decoding takes 36.1 s (81% of ideal) when continuous and 52.1 s (56%) when static. Static batches
  of 16 wait for outputs of 218, 453 and 192 tokens; in the continuous schedule the last request starts at 37.9 s,
  then 14.6 s of drain while the batch shrinks.
  [ok] ideal 29 s, prefill 16.5 s, decoding 36 s against 52 s (81% against 56% of ideal); batches wait for 218, 453, 192; last start 38 s, 14.6 s drain  (29.2, 16.5, 36.1, 52.1, [218, 453, 192], 37.9, 14.6)

  the bullets under the page's widget, recomputed (same simulator):
  scenario                         continuous s  tok/s  static s  tok/s  continuous saves
  -------------------------------  ------------  -----  --------  -----  ----------------
               the page's setting          52.5     89      68.6     68               23%
          every output 100 tokens          46.6    103      46.6    103                0%
   step time independent of batch          28.4    164      38.0    123               25%
  step time proportional to batch         132.4     35     132.4     35                0%
     1,024-token prompts (2.18 s)         140.7     33     156.7     30               10%
                          2 slots         110.6     42     118.6     39                7%
  [ok] equal outputs: both 46.6 s. Batch-independent steps: 164 and 123 tok/s, 25% saved. Proportional steps: no difference. 1,024-token prompts: 105 s of prefill, 141 s, 10%. Two slots: 7%  (46.6/46.6, 164/123, 132.4/132.4, 140.7 (10.2%), 6.7%)
  [ok] the page's exercise: 4 slots, one request of 100 steps and nine of 10 take 120 steps static, 100 continuous  ([120.0, 100.0])

## 3. The same loop on a real model
  the page's workload scaled to this model's 128 positions: 48 requests, 16-byte prompts, outputs of 3 to 91 bytes (931 in all),
  16 slots, greedy decoding, median of 3 alternating runs. The simulation uses this machine's own step table (section 1) and
  its own prefill time (0.99 ms for 16 bytes):
  mode                   makespan ms  bytes/s  mean done ms  p90 done ms  mean TTFT ms
  ----------  ---------  -----------  -------  ------------  -----------  ------------
  continuous       real          168    5,527            82          132            44
  continuous  simulated          167    5,572            83          130              
      static       real          249    3,739           126          224            94
      static  simulated          245    3,794           124          220              
  [ok] every request gets exactly the bytes it would get decoded alone, in both schedules and every run  (48 requests, 3 runs each)
  [ok] continuous batching finishes the batch of requests sooner than static batching, and their average sooner  (makespan 0.68 of static, mean completion 0.65 of static)
  [ok] the simulator, fed only the measured step table and prefill time, predicts both makespans within 25%  (continuous: -0.8%, static: -1.5%)
  counted in steps alone, with no clock involved: continuous 95 decode steps, static 170, so 44% fewer; that saving comes from the schedule, not from the hardware.

## 4. KV memory: the ledger, reserving against paging, the block table
  quantity (Qwen2.5-7B, 80 GiB card)               value                    
  -----------------------------------------------  -------------------------
                               KV cache per token               57,344 bytes
         a 4,096-token slot / a 65,536-token pool         224 MiB / 3.50 GiB
                        parameters / FP16 weights         7.616 B / 14.2 GiB
   KV pool (90% of memory, 4 GiB for activations)   53.8 GiB = 1.01 M tokens
   sequences when every one reserves 8,192 tokens                        123
  sequences in 16-token blocks, 1,000 tokens each                        999
      OPT-13B per token (2 x 40 x 5120 x 2 bytes)  819,200 bytes, 14.3x more
  [ok] 57,344 bytes per token, 224 MiB, 3.50 GiB, 7.616 B and 14.2 GiB, 53.8 GiB = 1.01 M tokens, 123 against 999 sequences, OPT-13B 14x  (57344, 53.81 GiB, 123, 999, 14.29)
  [ok] exercise: Llama 3 8B needs 131,072 bytes per token, 1 GiB per 8,192-token request, and a 24 GiB card holds 9  (131,072 bytes, 9.04 GiB left, 9.04 requests)

  the page's 48 requests hold 10,800 tokens in all. Reserving 8,192 for each puts 2.7% of the reserved
  memory to use; 16-token blocks allocated as needed put 97.0% to use, a waste of 7.0 tokens per request (never more than 15).
  [ok] paged allocation wastes under 16 tokens per request and keeps over 90% of memory in use; reserving the maximum keeps under 5%  (97.0% against 2.7%)

  three sequences of 10, 19 and 23 tokens grow interleaved in a pool of 16 blocks of 4 tokens; their block tables:
  sequence  tokens  block table (physical blocks)
  --------  ------  -----------------------------
         0      10                      [0, 3, 6]
         1      19               [1, 4, 7, 9, 11]
         2      23          [2, 5, 8, 10, 12, 13]
  [ok] the blocks of a sequence are not next to each other, yet gathering them gives back the dense keys and values, and attention is bit-for-bit the same  (14 blocks in use for 52 tokens; a dense layout reserving 32 tokens each would take 24)

  parallel sampling: four samples of 5 tokens continue one 12-token prompt. Sharing the prompt's blocks they use 11 blocks,
  20 if each owned a copy. A fork of a sequence whose last block is half full copies that block the first time either side writes
  into it (1 copy, the original is untouched).
  [ok] sharing the prompt: 3 prompt blocks plus 2 per sample, no copies; writing into a shared half-full block copies it once and leaves the original unchanged  (11 blocks against 20)
  [ok] llama.cpp: -c 65536 -np 16 gives 4,096 tokens per slot, so a 6,000-token request is refused, unless the cache is unified (65,536)  (4096 against 65536)
  [ok] exercise: -c 32768 -np 8 gives 4,096 per slot, so 20,000 tokens are refused; unified or -np 1 gives 32,768; -c 160000 is needed to keep eight slots  (4096, 32768, 32768, 160000)

## 5. The prefix cache: the same beginning is computed once
  a 96-byte document (six 16-byte blocks) and a 16-byte question, four requests per case, each with a different question;
  B puts 8 bytes that change every time in front of the document. The cache starts empty in each case; times are the median of 9 repetitions:
  case                      request  reused  computed  prefill ms
  ------------------------  -------  ------  --------  ----------
     A doc first, cache on        1       0       112        2.58
     A doc first, cache on        2      96        16        1.22
     A doc first, cache on        3      96        16        1.18
     A doc first, cache on        4      96        16        1.18
  B header first, cache on        1       0       120        2.66
  B header first, cache on        2       0       120        2.68
  B header first, cache on        3       0       120        2.65
  B header first, cache on        4       0       120        2.65
    C doc first, cache off        1       0       112        2.57
    C doc first, cache off        2       0       112        2.41
    C doc first, cache off        3       0       112        2.38
    C doc first, cache off        4       0       112        2.43
  [ok] A: the first request computes all 112 bytes, the next three reuse the 96 of the document and compute 16; B: nothing is ever reused; C: nothing is reused  (A [(0, 112), (96, 16), (96, 16), (96, 16)], B (0, 120))
  [ok] reusing cached keys and values does not change the logits  (largest difference from a full prefill 0.0e+00 over 12 requests)
  [ok] a request that hits the cache prefills faster than one that does not  (1.19 ms against 2.45 ms (2.1x); 16 of 112 bytes computed, not 1/7 of the time: fixed costs per forward pass)
  [ok] the page's exercise: with 16-token blocks, a second request with the same 1,000-token system prompt reuses 992 tokens; one with a timestamp in front reuses 0; another tenant's salt reuses 0  (992, 0, 0)

## 6. What a request costs: TTFT, TPOT, the two regimes, capacity
  one slot, four identical requests at t = 0 (16-byte prompt, 40 new bytes). Alone, one request has a prefill of 1.10 ms and a TPOT of 0.98 ms:
  request  queue ms  prefill ms  TTFT ms  TPOT ms  end-to-end ms  queue + prefill + (N-1) TPOT, ms
  -------  --------  ----------  -------  -------  -------------  --------------------------------
        1       0.0        1.10      1.1     0.99           39.5                              39.5
        2      39.5        1.00     40.5     1.00           78.3                              79.0
        3      78.3        1.06     79.3     0.99          118.2                             117.8
        4     118.2        1.09    119.3     1.00          158.1                             157.7
  [ok] end-to-end = queue + prefill + (N - 1) x TPOT: with prefill and TPOT measured alone, the formula is within 10% for all four requests  (-0.1%, +1.0%, -0.3%, -0.3%)

  prefill speed against prompt length (decoding one sequence runs at 969 bytes/s):
  prompt bytes  prefill ms  bytes/s
  ------------  ----------  -------
             1        0.62    1,603
             4        0.77    5,202
            16        1.04   15,400
            64        1.83   34,964
           120        2.90   41,361
  [ok] prefill, which handles every prompt position in one pass, is more than 10x faster per byte than decoding one sequence, and gets faster per byte as the prompt grows  (41,361 against 969 bytes/s)
  the page's arithmetic                                        value                          
  -----------------------------------------------------------  -------------------------------
                    M2 Max decode ceiling, 400 GB/s / 4.68 GB                      85 tokens/s
                                42.3 tokens/s sustained reads  198 GB/s = 49% of the bandwidth
                                64.8 tokens/s when cool reads                   303 GB/s = 76%
                 exercise: 3,000-token prompt, 200 tokens out   TTFT 6.19 s, end to end 10.9 s
  exercise: 3.35 TB/s, FP16 weights 15.23 GB / Q4_K_M 4.68 GB      220 / 715.8 tokens/s, 3.25x
  [ok] 85 tokens/s ceiling, 303 GB/s = 76% at 64.8 tokens/s, 6.19 s and 10.9 s, 220 and 715 tokens/s (715.8, which the page truncates), 3.3x

  A and B are the same code (20 runs of 10 batched steps each, 5 trials). What does a benchmark say about their ratio B / A,
  depending on the order they run in?
  order            B / A in each trial            range         
  ---------------  -----------------------------  --------------
  A first, then B  0.975 1.025 1.008 0.949 0.982  0.949 to 1.025
  A, B, A, B, ...  1.002 1.046 0.957 0.998 1.002  0.957 to 1.046
  alternating did not give the narrower range on this run: the machine drifted little, so order mattered little.
  across all trials B / A strayed up to 5.1% from 1 with identical code: a difference between two real configurations
  smaller than that is not a difference this benchmark can see.

  Little's law in a simulated queue (Poisson arrivals, exponential service, 4 servers):
  utilization  arrivals/s  W: mean time in system  L: time-average number  lambda x W
  -----------  ----------  ----------------------  ----------------------  ----------
        0.375       1.501                   1.030                   1.546       1.546
        0.625       2.501                   1.214                   3.036       3.036
        0.875       3.502                   2.534                   8.873       8.874
  [ok] L = lambda x W: the time-average number of requests in the system equals the arrival rate times the mean time in it  (largest relative gap 0.01%)

  the page's capacity exercise: 17 tokens/s per user, 200 tokens per answer, 0.8 requests/s at the peak. One machine serves 4 at once
  (17.3 tokens/s each; at 6 it falls to 12.2), a request stays W = 0.47 + 199 / 17.3 = 12.0 s, so L = 0.8 x 12.0 = 9.6
  requests are in flight and 3 machines are needed; one machine sustains 0.33 requests/s.
  [ok] 4 requests per machine, W = 12.0 s, L = 9.6, 3 machines, 0.33 requests/s per machine  (4, 11.97, 9.58, 3)

All checks passed. (10.5s)
```

</details>
