# 量化

课程页面：[模型量化：INT8、INT4、GPTQ 与 AWQ 的原理与实测](https://dmxsf.com/wiki/inference/quantization/)  
[English](README.en.md)

Llama 3 8B 在 80 GB 显卡上的账本；取整网格；以及在小模型上的实验：就近取整、SmoothQuant、AWQ 式缩放搜索、GPTQ、KV Cache 量化，其中的离群值是人为构造并明确标注的。

## 运行

```bash
uv run python labs/inference/quantization/run.py            # 完整运行，约 20 秒
uv run python labs/inference/quantization/run.py --quick    # 缩小规模的冒烟测试，几秒
```

也可以写成 `uv run dmxsf run quantization`（同样可以加 `--quick`）。`--seed N` 换随机种子，默认 0。脚本分节打印结果，并用 `[ok]` / `[FAIL]` 逐条检查课程页面里的结论：全部通过时退出码为 0，有一条失败就是 1；只有完整运行才做的检查在 `--quick` 下标为 `[skip]`。

这个实验用到缓存的小模型：字节级基础模型（84 万参数）。第一次运行会先在 TinyShakespeare 上训练并缓存到 `~/.cache/dmxsf`（环境变量 `DMXSF_CACHE` 可以改位置），之后直接读取。

## 代码

| 位置 | 内容 |
| --- | --- |
| `labs/inference/quantization/run.py` | 实验脚本，按编号分节 |
| `src/dmxsf/inference/quantization.py` | 量化：课程页的算法用纯 PyTorch 实现，按“模拟量化”应用 |

单元测试：`uv run pytest tests/test_inference_quantization.py`

## 看什么

- 第 1 节：Llama 3 8B 在一张 80 GB 卡上的账（精确）：权重从 16.1 GB 缩到 5.7 GB，只缩 2.8 倍而不是 4 倍（嵌入层仍是 BF16），解码上限从 133 升到 430 token/秒，能同时放的 8K 序列在 KV 也量化之前几乎不动（52 → 61 → 121）。
- 第 2 节：页面小组件的数字逐个对上：2 到 8 比特的 SQNR 是 1.1、9.6、16.9、23.5、29.8、35.6、42.1 dB；一个离群值把 INT4 从 16.9 dB 拉到 3.7 dB，分组（128 / 64 / 16）找回 11.6 / 14.7 / 19.1 dB。
- 第 3 节：就近取整在真实（小）模型上：INT8 每行一个缩放因子看不出差别（−0.0004 比特/字节）；INT4 整张量共用一个缩放是每行一个的两倍多（+0.1326 对 +0.0459）；低于 4 比特就崩：INT3（g32）的损失是 INT4（g32）的 10 倍（+0.1630 对 +0.0162），INT2 每字节丢掉 1.7 个比特（+1.7262，70% 的权重被取整成 0）；只量化嵌入层的代价（+0.0505）比量化全部 24 个线性层（+0.0162）还大，而它只占 3.9% 的参数。
- 第 4 节：激活更难量化：人为造一个离群通道，逐张量的 W8A8 随它变大而崩（100 倍时 +3.18 比特/字节），逐 token 的缩放缓和但不能消除（+0.297）；自然的模型里没有离群值，W8A8 几乎免费。
- 第 5 节：SmoothQuant 把离群值挪进权重，所有离群大小下都回到 0.01 比特/字节以内；GPTQ 的输出误差最低，损失比就近取整低得多（INT4 +0.0035 对 +0.0162，INT3 +0.0575 对 +0.1630）；AWQ 式搜索在自然模型上没有帮助（1.16 倍），在有离群值的模型上帮助明显（0.52 倍）；校准数据：用 Python 源码校准与用 Shakespeare 一样好，随机字节丢掉大部分收益，几个千分位的差别不值得解读。
- 第 6 节：KV Cache 的量化：INT8 的键和值免费，INT4 时键比值更脆弱（只量化 K 是 +0.0120，只量化 V 是 +0.0055），INT2 不可用；造出键的离群通道，逐 token 的缩放付出代价（30 倍时 +2.749），逐通道的不在乎它多大（始终 +0.0100）。

## 练习

改了参数再运行：核对页面数字的检查会变成 `[FAIL]`，这是预期的，看数字本身怎么变。

1. 第 3 节的评测窗口数 `EVAL_WINDOWS = 16 if QUICK else 128` 改成 512：INT4 各行的差值（delta）还稳定吗，排序有变化吗？
2. 第 2 节的分组表 `groups = [...]` 里加 `("g8", 8)`：每个权重的有效比特数涨到多少？SQNR 又涨多少？

## 注意

- 模型是 84 万参数的字节级 GPT，指标是留出的 Shakespeare 上的比特/字节（页面用的是 WikiText-2 的困惑度，二者是对数关系）；这么小的模型没有页面里 0.5B 模型那样的大离群通道，所以第 4、6 节用“造出来”的离群通道演示机制。
- 这个实验把当前 Python 解释器自带的标准库源码当作“代码语料”，所以换一个 Python 版本，涉及代码语料的数字会略有不同。

## 输出

下面是在 Apple M2 Max（12 核，仅用 CPU）、Python 3.12、PyTorch 2.15.0 上的完整运行结果。换一台机器，涉及训练的数字可能在末位上不同，计时类的数字取决于机器；检查用的是阈值与容差，不要求逐位相同。

<details>
<summary>完整运行的输出（`labs/inference/quantization/run.py`）</summary>

```text
# Quantization: the ledger, the grid, weights, activations, calibration, the KV cache
  course: https://dmxsf.com/wiki/inference/quantization/
          https://dmxsf.com/en/wiki/inference/quantization/

## 1. The ledger: Llama 3 8B on one 80 GB card (exact)
  parameters: 8.03B in all; decoder linear layers 6.98B, input embedding 0.53B, output layer 0.53B.
  [ok] the three groups and the norm gains add up to the model's total  (8,030,261,248)
  KV cache: 131072 bytes per token (BF16), 1.07 GB for one sequence of 8,192 tokens.
  weights + KV  weights GB  read per step GB  tokens/s bound  8K sequences
  ------------  ----------  ----------------  --------------  ------------
   BF16 + BF16        16.1              15.0             133            52
   INT4 + BF16         5.7               4.6             430            61
   INT4 + INT8         5.7               4.6             430           121
  [ok] the page's table: 16.1 / 5.7 / 5.7 GB, 15.0 / 4.6 / 4.6 GB per step, 133 / 430 / 430 tokens/s, 52 / 61 / 121 sequences  ({'BF16 + BF16': (16.1, 15.0, 133, 52), 'INT4 + BF16': (5.7, 4.6, 430, 61), 'INT4 + INT8': (5.7, 4.6, 430, 121)})
  weights shrink 2.8x, not 4x (the embeddings stay BF16); the decode bound rises 3.2x; the number
  of sequences barely moves until the cache itself is quantized.

## 2. Rounding to a grid: the page's widget, number for number
  1,024 fixed-seed normal weights (the widget's own sample), symmetric, one scale:
  bits  SQNR dB  gain over previous bit
  ----  -------  ----------------------
     2      1.1                        
     3      9.6                    +8.4
     4     16.9                    +7.4
     5     23.5                    +6.5
     6     29.8                    +6.3
     7     35.6                    +5.8
     8     42.1                    +6.5
  [ok] SQNR for 2 to 8 bits is 1.1, 9.6, 16.9, 23.5, 29.8, 35.6, 42.1 dB

  INT4, symmetric, by group size; the last column replaces one weight by 20 (20 sigma):
  scale shared by  effective bits  SQNR dB  SQNR dB with an outlier
  ---------------  --------------  -------  -----------------------
     whole tensor           4.000     16.9                      3.7
             g256           4.062     17.4                      9.2
             g128           4.125     18.1                     11.6
              g64           4.250     18.9                     14.7
              g32           4.500     19.9                     17.3
              g16           5.000     21.2                     19.1
  [ok] one outlier takes INT4 from 16.9 dB to 3.7 dB; groups of 128 / 64 / 16 win back 11.6 / 14.7 / 19.1 dB

  asymmetric against symmetric at INT4, g128: 19.6 dB (4.156 bits) against 18.1 dB (4.125 bits).
  [ok] asymmetric: 19.6 dB at 4.156 bits; symmetric: 18.1 dB at 4.125 bits
  [ok] a 7B model: 14 GB in BF16, about 3.6 GB at 4.125 bits per weight

## 3. Round to nearest on a real model: bits, granularity, sensitivity
  the course's byte-level GPT: 842,496 parameters, of which 786,432 are in the 24 decoder linear layers; the tied embedding has 32,768 (3.9%).
  held-out Shakespeare: 2.3085 bits/byte in float32. The page's metric is perplexity on WikiText-2;
  bits per byte is its logarithm per character, so 'delta' below is a log-ratio.
  scheme       effective bits  linear weights KiB  bits/byte  delta    rounded to 0
  -----------  --------------  ------------------  ---------  -------  ------------
  INT8 tensor           8.000                 768     2.3090  +0.0005          1.8%
     INT8 row           8.094                 777     2.3081  -0.0004          1.1%
  INT4 tensor           4.000                 384     2.4410  +0.1326         32.0%
     INT4 row           4.094                 393     2.3543  +0.0459         18.8%
      INT4 64           4.250                 408     2.3378  +0.0293         16.6%
      INT4 32           4.500                 432     2.3247  +0.0162         14.6%
      INT4 16           5.000                 480     2.3200  +0.0115         12.6%
       INT4 8           6.000                 576     2.3129  +0.0044         10.4%
      INT3 32           3.500                 336     2.4715  +0.1630         33.1%
      INT3 16           4.000                 384     2.4197  +0.1113         28.8%
      INT2 16           3.000                 288     4.0347  +1.7262         70.1%
  max|w| / median|w| per layer: median 8.3, largest 12.5 (the page's 0.5B model: 27 and 117).
  [ok] INT8 with one scale per row is invisible  (-0.0004 bits/byte)
  [ok] INT4 with one scale for the whole tensor is more than twice as bad as one per row  (+0.1326 against +0.0459)
  [ok] smaller groups always help, but each extra bit buys less than the one before  (saved per extra bit: 0.106, 0.052, 0.009, 0.007)
  [ok] round to nearest breaks below 4 bits: INT3 costs 5x INT4 (g32), INT2 loses half a bit per byte  (INT3 +0.163, INT4 +0.016, INT2 +1.726)

  INT4 g32 on one kind of layer at a time (delta bits/byte):
  layer kind  q_proj   k_proj   v_proj   o_proj   fc       proj   
  ----------  -------  -------  -------  -------  -------  -------
       delta  +0.0012  -0.0002  +0.0032  +0.0022  +0.0063  +0.0027
  ...and on one block at a time:
  block  0        1        2        3      
  -----  -------  -------  -------  -------
  delta  +0.0088  +0.0006  +0.0019  +0.0038

  all 24 linear layers at INT4 g32: +0.0162; adding the tied embedding / output matrix: +0.0667
  (the single-block rows above sum to +0.0152: roughly additive).
  [ok] rounding the embedding alone costs more than rounding all 24 linear layers, with a fraction of the parameters  (+0.0505 against +0.0162; 3.9% of the parameters)
  INT4 g32 linear layers, the rest in 16 bits: 542 KiB, 33% of the 16-bit model (the page's 0.5B model: 46%).

## 4. Activations: why they are harder, and an outlier you can dial
  what enters four linear layers of block 2 (one 128-byte window):
  input        median |x|  median of per-token max  max  top channel / median  top channel is the max in
  -----------  ----------  -----------------------  ---  --------------------  -------------------------
  q/k/v input       0.555                     3.08  5.5                   3.2                        27%
      o input       0.215                     1.07  2.4                   2.3                         5%
     fc input       0.793                     3.85  6.3                   1.9                         1%
   proj input       0.042                     2.42  4.2                   3.5                         1%
  the page's Qwen2.5-0.5B has a channel 21 times the median that is the largest entry in 44-58% of the
  tokens; here the strongest channel is 3.5 times the median.

  natural model, delta bits/byte: W8A16 -0.0004; W8A8 one scale per tensor +0.0024; one per token -0.0003.
  [ok] on this small model activation INT8 is nearly free even with one scale per tensor: there is no outlier to pay for  (+0.0024 bits/byte)

  constructed outlier: one residual channel made c times larger (the model computes the same function):
  channel scale  W8A16    W8A8 per tensor  W8A8 per token
  -------------  -------  ---------------  --------------
             1x  -0.0004          +0.0024         -0.0003
            10x  -0.0002          +0.0276         +0.0013
            30x  +0.0005          +0.3808         +0.0208
           100x  +0.0002          +3.1761         +0.2970
  [ok] the constructed models compute what the original does (max logit difference under 1e-3)  (bits/byte moved by at most 8.6e-08)
  [ok] an outlier channel wrecks per-tensor activation INT8: the damage grows with its size, past a bit per byte at 100x  (delta +0.002, +0.028, +0.381, +3.176)
  [ok] one scale per token softens the damage but does not remove it  (at 100x: +0.297 against +3.176)

## 5. Calibration: SmoothQuant, an AWQ-style search, GPTQ
  SmoothQuant on the 30x model, W8A8 with one scale per tensor (alpha is the knob):
  alpha  delta bits/byte
  -----  ---------------
   none          +0.3808
    0.3          +0.0018
    0.5          +0.0014
    0.7          +0.0007
    0.8          +0.0012
    0.9          +0.0020
    1.0          +0.0015
  alpha fixed in advance at 0.7, every outlier size: 1x +0.0009, 10x +0.0008, 30x +0.0007, 100x +0.0008
  [ok] smoothing moves the outlier into the weights and recovers the loss: under 0.01 bits/byte at every size  (largest +0.0009; unsmoothed at 100x it was +3.176)

  weight quantization with calibration data (64 windows of the training set, groups of 32):
  model          weights   method     relative output error  delta bits/byte  seconds
  -------------  --------  ---------  ---------------------  ---------------  -------
  natural model  INT4 g32        RTN                 0.0035          +0.0162      0.1
  natural model  INT4 g32  AWQ-style                 0.0033          +0.0188      0.2
  natural model  INT4 g32       GPTQ                 0.0012          +0.0035      0.3
  natural model  INT3 g32        RTN                 0.0193          +0.1630      0.1
  natural model  INT3 g32  AWQ-style                 0.0179          +0.1628      0.2
  natural model  INT3 g32       GPTQ                 0.0070          +0.0575      0.3
    30x outlier  INT4 g32        RTN                 0.0050          +0.0404      0.1
    30x outlier  INT4 g32  AWQ-style                 0.0036          +0.0210      0.2
    30x outlier  INT4 g32       GPTQ                 0.0017          +0.0119      0.4
    30x outlier  INT3 g32        RTN                 0.0206          +0.1878      0.1
    30x outlier  INT3 g32  AWQ-style                 0.0188          +0.1871      0.2
    30x outlier  INT3 g32       GPTQ                 0.0075          +0.0437      0.3
  [ok] output error: GPTQ is the lowest and the AWQ-style search is never above round to nearest
  [ok] GPTQ beats round to nearest in bits per byte by more than 2x, at INT4 and INT3  (INT4: +0.0035 against +0.0162, INT3: +0.0575 against +0.1630)
  the AWQ-style search at INT4 leaves 1.16x of RTN's loss on the natural model and 0.52x with the constructed outlier:
  it does not help on the first and helps on the second (below 0.8x counts as help).
  [ok] with an outlier channel the AWQ-style search cuts round-to-nearest's INT4 loss by 20% or more  (0.52x)

  GPTQ, groups of 32, as the calibration data changes (delta bits/byte; RTN: INT4 +0.0162, INT3 +0.1630):
  calibration data           INT4     INT3   
  -------------------------  -------  -------
     Shakespeare, 1 windows  +0.0035  +0.0961
     Shakespeare, 4 windows  +0.0125  +0.0717
    Shakespeare, 16 windows  +0.0036  +0.0603
    Shakespeare, 64 windows  +0.0035  +0.0575
   Shakespeare, 256 windows  +0.0119  +0.0505
  Python source, 64 windows  +0.0025  +0.0572
   random bytes, 64 windows  +0.0042  +0.1052
  [ok] INT3: more calibration data helps (256 windows against 1), and random bytes lose much of the gain  (+0.096 -> +0.051; random +0.105)
  INT4 GPTQ ranges over 0.0099 bits/byte across the 7 calibration sets, all below RTN's loss: differences of a few thousandths between two
  quantized models of this size are not worth interpreting. Python source calibrates about as well as Shakespeare here (INT3 +0.0572 against +0.0575):
  the page's warning about domain mismatch is not visible in this model, which is no evidence against it for a large one.

## 6. The KV cache: keys and values
  natural model; the cache holds 2,048 bytes per token in 16 bits (formula: 2,048):
  cache format                bytes per token, scales included  delta bits/byte
  --------------------------  --------------------------------  ---------------
                  K 16, V 16                              2048          +0.0000
              K INT8, V INT8                              1088          -0.0000
              K INT4, V INT4                               576          +0.0169
                K INT4, V 16                              1312          +0.0120
                K 16, V INT4                              1312          +0.0055
  K INT8 per channel, V INT8                              1088          +0.0001
  K INT4 per channel, V INT4                               576          +0.0154
              K INT2, V INT2                               320          +1.1218
  [ok] INT8 keys and values per token are free here  (-0.0000)
  [ok] at INT4 the keys are the more fragile half: rounding K alone costs more than 1.5x rounding V alone  (K +0.0120, V +0.0055)
  [ok] INT2 is not usable  (+1.122)
  per-channel scales for K change INT4 from +0.0169 to +0.0154 here (the page's model: perplexity 62.46 to 18.00).
  What a per-channel scale is for are key channels that are far larger than the rest, so build them:
  [ok] every key-outlier copy computes the same function (max logit difference under 1e-3)

  constructed outlier: one key channel per head made c times larger, only K rounded (delta bits/byte):
  key channel scale  INT8 per token  INT8 per channel  INT4 per token  INT4 per channel
  -----------------  --------------  ----------------  --------------  ----------------
                 1x         -0.0001           +0.0000         +0.0120           +0.0100
                10x         +0.0006           +0.0000         +0.7045           +0.0100
                30x         +0.0100           +0.0000         +2.7486           +0.0100
  [ok] a per-channel scale does not care how big the channel is (identical at every size); a per-token scale pays for it  (INT4 at 30x: +2.749 per token, +0.0100 per channel)
  the per-channel numbers need a whole group of 32 tokens before they can be rounded: a real cache
  keeps the newest tokens in 16 bits (KIVI); this evaluation rounds whole windows, so it is optimistic.

All checks passed. (18.8s)
```

</details>
