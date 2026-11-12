# Transformer Block 与现代架构

课程页面：[Transformer Block 完整拆解](https://dmxsf.com/wiki/architecture/transformer-block/) · [现代大模型架构：从 GPT-2 到 Llama、Qwen 与 DeepSeek](https://dmxsf.com/wiki/architecture/modern-llm-architecture/)  
[English](README.en.md)

数清一个 GPT-2 small 的 Block（12d²+13d 个参数）；看真实模型的参数分布；比较 Pre-Norm 与 Post-Norm 在不同深度的信号与梯度；再从 GPT-2 到 Llama，一次只改一处。

## 运行

```bash
uv run python labs/architecture/transformer-block/run.py            # 完整运行，约 3 分钟
uv run python labs/architecture/transformer-block/run.py --quick    # 缩小规模的冒烟测试，几秒
```

也可以写成 `uv run dmxsf run transformer-block`（同样可以加 `--quick`）。`--seed N` 换随机种子，默认 0。脚本分节打印结果，并用 `[ok]` / `[FAIL]` 逐条检查课程页面里的结论：全部通过时退出码为 0，有一条失败就是 1；只有完整运行才做的检查在 `--quick` 下标为 `[skip]`。

## 代码

| 位置 | 内容 |
| --- | --- |
| `labs/architecture/transformer-block/run.py` | 实验脚本，按编号分节 |
| `src/dmxsf/architecture/gpt.py` | 一个可配置的 Decoder-only Transformer：从 GPT-2 到 Llama |
| `src/dmxsf/architecture/layers.py` | 归一化层与前馈块 |
| `src/dmxsf/architecture/configs.py` | 公开模型的架构配置，以及它们报告的参数量 |

单元测试：`uv run pytest tests/test_architecture_gpt.py tests/test_architecture_layers.py`

## 看什么

- 第 1 节：GPT-2 small 的一个块是 12d² + 13d = 7,087,872 个参数（前馈 66.6%，注意力 33.3%），12 个块加嵌入表共 124,439,808，与课程页一致。
- 第 2 节：七个公开模型的参数构成，只建模型的形状、不加载权重：Llama 3 8B 的前馈占 70.2%、注意力 16.7%、词表 13.1%；GQA 让 Llama 3 8B 每个 token 的 KV Cache 是 128 KiB，Llama 2 7B 是 512 KiB。
- 第 3 节：Pre-LN 的梯度随深度按 1/√L 缩小（48 层时 0.1354），残差流越来越大（RMS 7.386）；Post-LN 的梯度不缩小（0.9117）；把每个残差分支乘 1/√N，大小稳定在 1.136。
- 第 4 节：从 GPT-2 风格一项项换成 Llama 配方：RoPE 的收益最明确（−0.314 nats），其余几项的变化都落在种子之间的波动（最大 0.063）以内；完整配方比 GPT-2 风格好 0.231 nats，GQA 把每 token 的 KV Cache 从 512 B 减到 256 B，损失没有明显变化。

## 练习

改了参数再运行：核对页面数字的检查会变成 `[FAIL]`，这是预期的，看数字本身怎么变。

1. 第 4 节的差别大多小于种子波动。把 `[args.seed + i for i in range(3)]`（第 4 节的 `seeds`）里的 3 改成 10，看哪些项能从噪声里浮出来。
2. 第 3 节把 `depths` 改成 `(6, 12, 24, 48, 96)`：Pre-LN 的“梯度 × √L”还是常数吗？

## 输出

下面是在 Apple M2 Max（12 核，仅用 CPU）、Python 3.12、PyTorch 2.15.0 上的完整运行结果。换一台机器，涉及训练的数字可能在末位上不同，计时类的数字取决于机器；检查用的是阈值与容差，不要求逐位相同。

<details>
<summary>完整运行的输出（`labs/architecture/transformer-block/run.py`）</summary>

```text
# The Transformer block and the modern recipe
  course: https://dmxsf.com/wiki/architecture/transformer-block/
          https://dmxsf.com/en/wiki/architecture/transformer-block/
  course: https://dmxsf.com/wiki/architecture/modern-llm-architecture/
          https://dmxsf.com/en/wiki/architecture/modern-llm-architecture/

## 1. One GPT-2 small block: 12 d^2 + 13 d parameters
  part          parameters  share
  ------------  ----------  -----
     attention   2,362,368  33.3%
  feed-forward   4,722,432  66.6%
         norms       3,072   0.0%
  total 7,087,872; the formula 12d^2 + 13d gives 7,087,872
  12 blocks + token table + position table + final norm = 124,439,808
  [ok] the numbers on the course page (GPT-2 small is 124,439,808 parameters)
  [ok] output shape is unchanged and positions 0-5 ignore everything after them

## 2. Where real models keep their parameters (shapes only, no weights)
  model         params  feed-forward  attention  vocab tables  KV cache / token
  ------------  ------  ------------  ---------  ------------  ----------------
   GPT-2 small   0.12B         45.5%      22.8%         31.0%            36 KiB
  Qwen2.5-0.5B   0.49B         63.5%       8.9%         27.6%            12 KiB
    Llama 2 7B   6.74B         64.2%      31.9%          3.9%           512 KiB
    Llama 3 8B   8.03B         70.2%      16.7%         13.1%           128 KiB
    Mistral 7B   7.24B         77.8%      18.5%          3.6%           128 KiB
    Qwen2.5 7B   7.62B         74.9%      10.8%         14.3%            56 KiB
   Llama 3 70B  70.55B         79.9%      17.1%          3.0%           320 KiB
  [ok] all seven totals equal the published parameter counts  (asserted above)
  [ok] course page: Llama 3 8B is 70.2% feed-forward, 16.7% attention, 13.1% vocabulary
  [ok] course page: GPT-2 small spends 31.0% on its token table, Llama 2 7B 3.9%, Llama 3 70B 3.0%
  [ok] course page: KV cache per token is 512 / 128 / 320 KiB for Llama 2 7B / Llama 3 8B / 70B
  Grouped-query attention is the difference: Llama 2 7B caches 32 KV heads, Llama 3 8B 8.

## 3. Pre-norm or post-norm: signal and gradient at initialisation, by depth
  random tokens, d=256, 8 sequences of 64, mean of 8 seed(s)
  RMS  = size of the last layer's output
  grad = norm of the gradient of the last feed-forward's output matrix
  layers  post RMS  post grad  pre RMS  pre grad  pre+s RMS  pre+s grad
  ------  --------  ---------  -------  --------  ---------  ----------
       6     1.000     0.8198    2.347    0.4075      1.137      0.8112
      12     1.000     0.8447    3.415    0.2833      1.136      0.8153
      24     1.000     0.8784    5.029    0.1950      1.137      0.8137
      48     1.000     0.9117    7.386    0.1354      1.136      0.8155
  [ok] post-LN: the gradient does not shrink with depth
  [ok] pre-LN: the gradient shrinks like 1/sqrt(L) (gradient x sqrt(L) stays within 15%)
  [ok] pre-LN: the residual stream keeps growing with depth
  [ok] pre-LN: gradient x RMS is nearly constant (between 0.9 and 1.1)
  [ok] scaling the residual branches by 1/sqrt(N) keeps the size steady (spread under 0.05)  (1.136 to 1.137)
  [ok] all 12 cells agree with the course page's table (RMS within 0.05, gradient within 0.03)  (largest differences: RMS 0.005, gradient 0.005)

## 4. From GPT-2 to Llama, one change at a time
  2 layers, d_model 64, 500 steps (lr 0.003, gradient clipping 1) on Shakespeare bytes, 3 seed(s); validation loss in nats per byte
  variant                                                  parameters  KV / token  mean loss  spread   change vs row above
  -------------------------------------------------------  ----------  ----------  ---------  -------  -------------------
  GPT-2 style: LayerNorm, GELU, learned positions, biases     120,576       512 B      2.071  0.03791                     
                                                + RMSNorm     120,256       512 B      2.080  0.01865          +0.009 same
        + SwiGLU (hidden size cut to keep the MLP's size)     120,168       512 B      2.137  0.01324          +0.057 same
                      + RoPE instead of learned positions     116,072       512 B      1.823  0.04301        -0.314 better
                                              + no biases     114,752       512 B      1.860  0.06328          +0.037 same
                   + grouped-query attention (2 KV heads)     106,560       256 B      1.840  0.02142          -0.020 same
  'spread' is max - min over seeds; a row is called better or worse only when it moves the mean by more than the largest spread, 0.063
  [ok] the full recipe beats GPT-2 style by more than the spread  (0.231 nats)
  [ok] swapping learned positions for RoPE is a clear gain  (0.314 nats)
  [ok] grouped-query attention halves the KV cache without a clear loss  (-0.020 nats)

All checks passed. (302.4s)
```

</details>
