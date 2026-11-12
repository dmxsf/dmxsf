# 投机解码

课程页面：[投机解码：用小模型打草稿，让大模型一次验证多个 token](https://dmxsf.com/wiki/inference/speculative-decoding/)  
[English](README.en.md)

保持目标模型精确分布的接受/拒绝规则：在课程页的玩具分布上、用精确代数、在真实模型上逐一验证；Leviathan 等人的公式；带 KV Cache 的草稿/目标字节模型对；以及什么时候不划算。

## 运行

```bash
uv run python labs/inference/speculative-decoding/run.py            # 完整运行，约 35 秒
uv run python labs/inference/speculative-decoding/run.py --quick    # 缩小规模的冒烟测试，几秒
```

也可以写成 `uv run dmxsf run speculative-decoding`（同样可以加 `--quick`）。`--seed N` 换随机种子，默认 0。脚本分节打印结果，并用 `[ok]` / `[FAIL]` 逐条检查课程页面里的结论：全部通过时退出码为 0，有一条失败就是 1；只有完整运行才做的检查在 `--quick` 下标为 `[skip]`。

这个实验用到缓存的小模型：字节级基础模型（84 万参数）、字节级小模型（草稿模型）。第一次运行会先在 TinyShakespeare 上训练并缓存到 `~/.cache/dmxsf`（环境变量 `DMXSF_CACHE` 可以改位置），之后直接读取。

## 代码

| 位置 | 内容 |
| --- | --- |
| `labs/inference/speculative-decoding/run.py` | 实验脚本，按编号分节 |
| `src/dmxsf/inference/speculative.py` | 投机解码：小模型起草，大模型一次前向验证多个 token |

单元测试：`uv run pytest tests/test_inference_speculative.py`

## 看什么

- 第 1 节：页面里 A100 的账（精确）：读一遍 14 GB 权重 6.87 ms，算一个 token 0.045 ms，算五个位置 0.22 ms，153 FLOP/字节才能喂饱芯片：多验证几个位置几乎是免费的，直到批大小接近 150。
- 第 2 节：四个 token 的分布上，拒绝后从残差分布重新采样恰好还原目标分布 p；从 p 本身重采样则不行（token 0 只有 0.375，应是 0.50）；接受率 Σmin(p, q) = 0.75，模拟的输出频率 [0.5, 0.25, 0.149, 0.1]，每轮 3.054 个 token 对公式的 3.051。
- 第 3 节：Leviathan 等人的公式（定理 3.8）与小组件的数字：α = 0.8、c = 0.05 时，γ = 4 加速 2.80 倍、最佳的 γ = 8 是 3.09 倍、γ = 16 回落到 2.72 倍；草稿贵 6 倍（c = 0.3）最佳 γ 缩到 3、加速降到 1.55 倍；α = 0.3、γ = 4 是 0.65 倍，比不用还慢。
- 第 4 节：真实的草稿模型（124,672 个参数）与目标模型（842,496 个）：一个草稿步是目标步成本的 0.41（c = 0.41，小模型在 CPU 上受每层开销限制，不同于 A100 上受内存带宽限制）；贪心的投机解码与目标模型单独解码逐字节相同（120 次运行全部一致）；每个 γ 需要的目标模型前向次数都少于普通解码（γ = 4 时每字节 0.33 次），公式用实测的 α 预测每轮字节数误差在 1.5% 以内。
- 第 4 节的时钟列：这次完整运行里没有一个 γ 比不用投机解码快（0.68 倍到 0.88 倍，最好的是 γ = 4 的 0.88 倍），而公式（已经把草稿的成本 c = 0.41 算进去）用实测的 α 预测 γ = 2 能到 1.29 倍；剩下的差距在公式没有算的地方：循环自己的开销（采样、缓存裁剪、记账）。第一个采样字节的频率与目标模型自己的分布吻合（卡方 p = 0.665）。
- 第 5 节：什么时候不划算：接受率低、草稿贵，什么都赚不到（α = 0.3、c = 0.5 时连 γ = 1 也亏，0.87 倍）；第三个条件是大批量，见第 1 节的账。

## 练习

改了参数再运行：核对页面数字的检查会变成 `[FAIL]`，这是预期的，看数字本身怎么变。

1. 第 4 节 `gammas = (2, 4) if QUICK else (1, 2, 4, 6, 8)` 里加一个 3：时钟上最快的 γ 是哪个？它与公式的选择一致吗？
2. 第 3 节 `for a in (0.6, 0.8, 0.9)` 里加一个 0.95：最佳 γ 与加速倍数各是多少？

## 注意

- 时钟列（“wall-clock speedup”）每次运行都会摆动，机器的负载与温度都有影响：同一台机器上，一次完整运行里 γ = 2 达到 1.11 倍（不用投机解码时每字节 0.92 ms），这里附的这次是 0.83 倍（每字节 0.63 ms），每个 γ 都比不用慢。读“目标模型前向次数”那一列和公式，别读时钟。

## 输出

下面是在 Apple M2 Max（12 核，仅用 CPU）、Python 3.12、PyTorch 2.15.0 上的完整运行结果。换一台机器，涉及训练的数字可能在末位上不同，计时类的数字取决于机器；检查用的是阈值与容差，不要求逐位相同。

<details>
<summary>完整运行的输出（`labs/inference/speculative-decoding/run.py`）</summary>

```text
# Speculative decoding: the rule, the formulas, a real draft and target
  course: https://dmxsf.com/wiki/inference/speculative-decoding/
          https://dmxsf.com/en/wiki/inference/speculative-decoding/

## 1. Why the extra positions are almost free (the page's A100 ledger, exact)
  quantity                            value                           how                      
  ----------------------------------  ------------------------------  -------------------------
                        BF16 weights                           14 GB   7e9 parameters x 2 bytes
              time to read them once  6.87 ms (146 tokens/s at most)         14 GB / 2,039 GB/s
               compute for one token                        0.045 ms  2 x 7e9 FLOP / 312 TFLOPS
             compute for 5 positions                         0.22 ms               5 x 0.045 ms
  FLOP per byte to saturate the chip                             153           312e12 / 2,039e9
  [ok] 6.87 ms to read the weights (146 tokens/s), 0.045 ms to compute a token, 0.22 ms for five, 153 FLOP/byte
  reading the weights costs 153x one token's arithmetic, so checking gamma + 1 positions in the same
  pass adds little. It stops being free around a batch of 150, where the arithmetic catches up.

## 2. The rule on the page's four-token distribution
  exact, no sampling: what one position yields under each repair rule
  token  p (target)  q (draft)  min(p, q)  residual  output, residual  output, refill from p
  -----  ----------  ---------  ---------  --------  ----------------  ---------------------
      0        0.50       0.25       0.25     1.000            0.5000                 0.3750
      1        0.25       0.25       0.25     0.000            0.2500                 0.3125
      2        0.15       0.25       0.15     0.000            0.1500                 0.1875
      3        0.10       0.25       0.10     0.000            0.1000                 0.1250
  [ok] refilling a rejected position from the residual gives back p exactly; refilling from p itself does not  (token 0 would come out at 0.375 instead of 0.50, tokens 1-3 at 0.312, 0.188, 0.125)
  the draft already covers the target's mass on tokens where q >= p; refilling from p counts that mass a second
  time, and token 0, which the draft under-proposes, ends up short.
  [ok] the acceptance rate is sum(min(p, q)) = 0.75 and the residual is (1, 0, 0, 0)

  the page's simulation: output frequencies [0.5   0.25  0.149 0.1  ] against p [0.5  0.25 0.15 0.1 ]; acceptance 0.750 (theory 0.75);
  3.054 tokens per round against the formula's 3.051.
  [ok] the page's printed numbers: [0.5, 0.25, 0.149, 0.1], 0.750, 3.054 and 3.051  ([0.5, 0.25, 0.149, 0.1], 0.750, 3.054, 3.051)

## 3. The formulas (Leviathan et al., theorem 3.8) and the widget's numbers
  speedup over plain decoding, c = 0.05 (a draft step costs 5% of a target step):
               gamma 1  gamma 2  gamma 4  gamma 8  gamma 16  best gamma: speedup
  -----------  -------  -------  -------  -------  --------  -------------------
  alpha = 0.6     1.52     1.78     1.92     1.77      1.39              4: 1.92
  alpha = 0.8     1.71     2.22     2.80     3.09      2.72              8: 3.09
  alpha = 0.9     1.81     2.46     3.41     4.38      4.63             13: 4.67
  [ok] alpha 0.9, gamma 10, free draft: 6.86x; alpha 0.8, c 0.05: gamma 4 gives 2.80x, the best gamma is 8 (3.09x), gamma 16 falls to 2.72x
  [ok] with a free draft alpha 0.8 caps at 1/(1 - 0.8) = 5x: gamma 4 reaches 3.36x, gamma 32 about 5.0x  (gamma 32: 4.997)
  [ok] alpha 0.6 at gamma 4: 1.92x, and the target does 1.49x (alpha 0.8) or 2.17x (alpha 0.6) positions per kept token
  [ok] a draft six times as dear (c = 0.3) shrinks the best gamma to 3 and the speedup to 1.55x; alpha 0.3 with gamma 4 is 0.65x, slower than not using it
  [ok] with gamma = 1 the speedup is (1 + alpha) / (1 + c), above 1 exactly when alpha > c
  [ok] simulated rounds with independent acceptance average the formula's number of tokens  (largest gap 0.004 tokens over three (alpha, gamma) pairs)
  [ok] DeepSeek-V3's 1.8x with one extra token (gamma = 1) at 85-90% acceptance means c between 0.03 and 0.06  (0.028 to 0.056)

## 4. A real draft and target
  target: 842,496 parameters, 4 layers, d = 128, 2.308 bits/byte on held-out text.
  draft:  124,672 parameters, 2 layers, d = 64,  2.494 bits/byte.

  time of one forward pass on the CPU with 64 bytes already cached, by number of new positions:
  new positions      1      2      3      5      9    
  -----------------  -----  -----  -----  -----  -----
         target, ms   0.95   1.13   1.07   1.04   1.12
    vs one position  1.00x  1.18x  1.12x  1.09x  1.17x
          draft, ms   0.40   0.38   0.41   0.44   0.45
  one draft step costs 0.41 of a target step here (c = 0.41); a 0.86M model on a CPU is limited
  by per-layer overhead, not by memory bandwidth, so this is a different regime from the A100 ledger.
  [ok] five positions in one target pass cost well under five single steps  (1.09x one step)

  on 128 held-out windows (every position, teacher-forced), the chance that a drafted byte is kept,
  beta = sum(min(p, q)), by sampling temperature:
  temperature  beta 
  -----------  -----
          1.0  0.778
          0.7  0.770
          0.4  0.755
  the two models pick the same most likely byte at 73.7% of positions (greedy decoding).
  [ok] greedy speculative decoding writes the same 64 bytes as the target alone, for every gamma and prompt  (120 of 120 runs)

  sampling at temperature 1 from 24 prompts, 64 new bytes each. Plain decoding: 0.63 ms per byte.
  gamma  alpha measured  drafts kept  bytes per round  formula at that alpha  target passes per byte  fewer target passes  formula speedup (c = 0.41)  wall-clock speedup
  -----  --------------  -----------  ---------------  ---------------------  ----------------------  -------------------  --------------------------  ------------------
      1           0.771         0.77             1.77                   1.77                   0.583                1.71x                       1.25x               0.80x
      2           0.766         0.68             2.35                   2.35                   0.446                2.24x                       1.29x               0.83x
      4           0.779         0.55             3.22                   3.23                   0.333                3.00x                       1.21x               0.88x
      6           0.797         0.48             3.86                   3.92                   0.282                3.55x                       1.12x               0.79x
      8           0.791         0.39             4.14                   4.20                   0.270                3.71x                       0.97x               0.68x
  [ok] every gamma needs fewer target passes per byte than plain decoding (which needs one)  (gamma 1: 0.58, gamma 2: 0.45, gamma 4: 0.33, gamma 6: 0.28, gamma 8: 0.27)
  fastest on the clock: gamma 4 (0.88x); the formula, fed the measured alpha and c, picks gamma 2 (1.29x).
  no setting beats plain decoding on this machine. With c = 0.41 the formula's best possible speedup at this alpha is 1.31x (gamma 2), before the loop's own work
  (sampling, cache cropping, bookkeeping), which the formula leaves out and this lab did not time separately.
  bytes per round against the formula at the measured alpha: -1.5% to -0.0% (the formula assumes every
  position accepts with the same probability; a gap means contexts differ in how easy they are to guess).
  [ok] the formula, fed the measured alpha, predicts the bytes per round within 10%  (gamma 2: 2.35 against 2.35, gamma 4: 3.22 against 3.23, gamma 6: 3.86 against 3.92, gamma 8: 4.14 against 4.20)
  [ok] the acceptance rate of the real loop at gamma 1 agrees with the offline beta at temperature 1  (alpha 0.771, beta 0.778)

  the first sampled byte over 2,000 runs (gamma 4), against the target's own next-byte distribution:
    byte  32 (   ' '): target 0.737, speculative 0.721
    byte  46 (   '.'): target 0.108, speculative 0.114
    byte  44 (   ','): target 0.050, speculative 0.050
  chi-square 8.5 on 11 degrees of freedom, p = 0.665 (cells expected under 5 pooled).
  [ok] the first byte's frequencies match the target's distribution (p above 0.001)

## 5. When it does not pay
  best achievable speedup by acceptance rate (rows) and draft cost (columns):
               c = 0.02          c = 0.1           c = 0.3          c = 0.6        
  -----------  ----------------  ----------------  ---------------  ---------------
  alpha = 0.3   1.34x (gamma 3)   1.18x (gamma 1)          no gain          no gain
  alpha = 0.5   1.79x (gamma 4)   1.46x (gamma 2)  1.15x (gamma 1)          no gain
  alpha = 0.7   2.76x (gamma 8)   1.98x (gamma 4)  1.37x (gamma 2)  1.06x (gamma 1)
  alpha = 0.9  6.37x (gamma 19)  3.43x (gamma 10)  1.87x (gamma 5)  1.23x (gamma 2)
  [ok] a low acceptance rate and a dear draft leave nothing to gain: with alpha 0.3 and c 0.5 even gamma = 1 loses  (best: gamma 1, 0.87x)
  and the page's third condition, a large batch, is the ledger of section 1: once the batch is near 150
  tokens per weight read the chip is busy, and checking gamma extra positions per sequence costs real time.

All checks passed. (35.5s)
```

</details>
