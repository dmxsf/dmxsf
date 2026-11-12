# 推理强化学习（GRPO 与可验证奖励）

课程页面：[推理强化学习：RLVR 与 GRPO](https://dmxsf.com/wiki/post-training/reasoning-rl/)  
[English](README.en.md)

验证器、组内优势、GRPO 损失、裁剪与 k3 KL 估计量的精确核对；然后是真实的 RLVR 训练：小模型仅凭精确匹配给出的 0/1 奖励完成策略更新。

## 运行

```bash
uv run python labs/post-training/reasoning-rl/run.py            # 完整运行，约 3 分钟
uv run python labs/post-training/reasoning-rl/run.py --quick    # 缩小规模的冒烟测试，几秒
```

也可以写成 `uv run dmxsf run reasoning-rl`（同样可以加 `--quick`）。`--seed N` 换随机种子，默认 0。脚本分节打印结果，并用 `[ok]` / `[FAIL]` 逐条检查课程页面里的结论：全部通过时退出码为 0，有一条失败就是 1；只有完整运行才做的检查在 `--quick` 下标为 `[skip]`。

这个实验用到缓存的小模型：字节级基础模型（84 万参数）。第一次运行会先在 TinyShakespeare 上训练并缓存到 `~/.cache/dmxsf`（环境变量 `DMXSF_CACHE` 可以改位置），之后直接读取。

## 代码

| 位置 | 内容 |
| --- | --- |
| `labs/post-training/reasoning-rl/run.py` | 实验脚本，按编号分节 |
| `src/dmxsf/post_training/reasoning.py` | 推理强化学习：可验证奖励、GRPO 优势与损失、采样加更新的循环 |

单元测试：`uv run pytest tests/test_post_training_reasoning.py`

## 看什么

- 第 1 节：页面里的验证器给出 1、0、0、1、0（最后一个是假阴性：`\boxed{0.5}` 与 1/2 相等，字符串比较却判错）；意图任务上，“包含标签就算对”的宽松验证器会奖励把所有标签都列出来的回答，精确匹配不会。
- 第 2 节：组内优势就是页面的数字（[1, 0, 0, 1] 得 ±0.8659，全对的组为 0，奖励放大 10 倍不变）；样本标准差与总体标准差恰好差 √(G/(G−1))；pass rate 为 5% 时 G=16 的组有 44.0% 全错，没有信号；除以标准差会把“几乎全对”和“几乎全错”的提示推得更重（16 个里对 1 个是 3.998 倍，对 8 个是 1.936 倍）。
- 第 3 节：按回答平均的损失让长的错误答案每个 token 只受到 1/10 的惩罚（0.087 对 0.0087），token 级平均则一视同仁；PPO 裁剪只挡住“已经推得够远”的方向；k3 估计 KL 无偏且不为负，k1 无偏但会取到负值（−3.265），靠近参考时 k3 的方差是 k1 的 0.8%。
- 第 4 节：在意图任务上做真实的可验证奖励强化学习：参考模型（SFT 后）58.0%，80 步、10,240 个采样回答后贪心准确率 85.3%（换一个种子 85.7%），平均奖励 0.56 → 0.83，有信号的组从 39% 降到 26%；从没有做过 SFT 的预训练模型出发，1,280 个回答里一个奖励都没有，权重一点也不动：强化学习起步前，策略得先偶尔答对。
- 第 4 节末尾：只保留“有时对、有时错”的 136 条训练样本，自己池子里的奖励更高（0.92），留出准确率却更低（78.0% 和 76.3%，全量是 85.3% 和 85.7%）；拿同样 360 条正确答案做 SFT 是 99.3%：RLVR 每个回答只看到一个比特，它是为有检查器、没有示范的情形准备的。

## 练习

改了参数再运行：核对页面数字的检查会变成 `[FAIL]`，这是预期的，看数字本身怎么变。

1. 第 4 节 `PPS, GROUP, LR, BETA = (4, 4, 1e-4, 0.04) if QUICK else (16, 8, 1e-4, 0.04)` 里把 GROUP 的 8 改成 4 或 16：有信号的组的比例和最终准确率各怎么变？
2. 把同一行的 `BETA` 从 0.04 改成 0：KL（k3）会涨到多少？留出的 bits/byte 会比 3.92 更差吗？

## 注意

- 任务是合成的意图分类（回答只是一个标签），模型是 84 万参数的字节级模型；页面里数学题上的 GRPO 结果需要真实的大模型，这里没有复现。

## 输出

下面是在 Apple M2 Max（12 核，仅用 CPU）、Python 3.12、PyTorch 2.15.0 上的完整运行结果。换一台机器，涉及训练的数字可能在末位上不同，计时类的数字取决于机器；检查用的是阈值与容差，不要求逐位相同。

<details>
<summary>完整运行的输出（`labs/post-training/reasoning-rl/run.py`）</summary>

```text
# Reasoning RL: verifiers, GRPO advantages and loss, and a real RLVR run
  course: https://dmxsf.com/wiki/post-training/reasoning-rl/
          https://dmxsf.com/en/wiki/post-training/reasoning-rl/

## 1. The verifier
  response                     gold  reward  why                                            
  ---------------------------  ----  ------  -----------------------------------------------
                 so \boxed{3}     3       1                                          correct
                 so \boxed{4}     3       0                                            wrong
              the answer is 3     3       0                       right answer, wrong format
  \boxed{4} no wait \boxed{3}     3       1                              the last box counts
                  \boxed{0.5}   1/2       0  string equality is not mathematical equivalence
  [ok] the page's verifier: 1, 0, 0, 1, 0 (the last row is a false negative)

  the intent task, gold label card_fee:
  answer                                 exact match  'contains the label'
  -------------------------------------  -----------  --------------------
                               card_fee            1                     1
  card_fee card_limit transfer_declined            0                     1
  [ok] a leaky verifier pays for listing every label; exact match does not

## 2. Group advantages
  rewards of three groups of four samples and their advantages (the page's code):
  rewards       A1       A2       A3       A4     
  ------------  -------  -------  -------  -------
  [1, 0, 0, 1]  +0.8659  -0.8659  -0.8659  +0.8659
  [1, 0, 0, 0]  +1.4997  -0.4999  -0.4999  -0.4999
  [1, 1, 1, 1]  +0.0000  +0.0000  +0.0000  +0.0000
  [ok] these are the numbers the page prints (4 decimals), and a unanimous group has no signal
  [ok] multiplying every reward by 10 changes nothing (only the comparison inside the group counts)

  the advantage of a correct answer in the group [1, 0, 0, 1]:
  normalisation                 A      
  ----------------------------  -------
  sample std (G - 1), the page  +0.8659
            population std (G)  +0.9998
        mean only, no division  +0.5000
  [ok] sample and population std differ by exactly sqrt(G / (G - 1)) = 1.1547 for G = 4

  chance that a group of G samples is mixed (some right, some wrong) at pass rate p:
  pass rate p  G = 4  G = 8  G = 16  G = 64
  -----------  -----  -----  ------  ------
           5%  18.5%  33.7%   56.0%   96.2%
          20%  58.9%  83.2%   97.2%  100.0%
          50%  87.5%  99.2%  100.0%  100.0%
          80%  58.9%  83.2%   97.2%  100.0%
  [ok] the page's numbers: 87.5% at p = 50%, G = 4; about 44% of groups all wrong at p = 5%, G = 16  (all wrong at 5%: 44.0%)
  [ok] the formula agrees with 100,000 simulated groups (p = 20%, G = 8)  (0.832 against 0.832)

  how much total push a group gets (sum of |A| over the group, G = 16):
  right     mean only  divided by std  ratio
  --------  ---------  --------------  -----
   1 of 16      1.875           7.497  3.998
   2 of 16      3.500           10.24  2.927
   4 of 16      6.000           13.41  2.236
   8 of 16      8.000           15.49  1.936
  12 of 16      6.000           13.41  2.236
  15 of 16      1.875           7.497  3.998
  [ok] dividing by the std boosts nearly-solved and nearly-unsolved prompts (the bias Dr. GRPO points out)  (ratio 4.00 at 1 of 16 and 1.94 at 8 of 16)

## 3. The loss
  two wrong answers (A = -0.87), 10 and 100 tokens long: push-down weight per token
  loss averaged                             short answer  long answer
  ----------------------------------------  ------------  -----------
  per answer, then over answers (the page)       0.08700     0.008700
                    over all tokens (DAPO)        0.8700       0.8700
  [ok] per-answer averaging punishes a long wrong answer 10x less per token; token-level does not  (0.087 against 0.0087 per token)

  PPO clipping (eps = 0.2): does the token's probability still get pushed?
  advantage  ratio new/old  push
  ---------  -------------  ----
         +1            1.5  none
         +1            0.5    up
         -1            0.5  none
         -1            1.5  down
  [ok] clipping stops only the pushes that already went far enough in the intended direction

  KL(pi || ref) of two distributions over 6 outcomes, and what each estimator averages to:
  estimator                      expectation over samples from pi  smallest value
  -----------------------------  --------------------------------  --------------
                       exact KL                             2.449               -
             k1 = log(pi / ref)                             2.449          -3.265
  k3 = ref/pi - log(ref/pi) - 1                             2.449         0.04938
  [ok] k3 is unbiased and never negative; k1 is unbiased but goes negative
  [ok] for policies close to the reference, k3 has far lower variance than k1 (200 random pairs)  (median Var(k3) / Var(k1) = 0.008)

## 4. A real RLVR run on the intent task
  reference: pretrained model + SFT on 60 messages (300 steps): accuracy 58.0% on 300 test messages, 3.80 bits/byte.
  RLVR: 16 messages x 8 sampled answers per step, reward 1 for an exact match, lr 0.0001, beta 0.04,
  80 steps = 10,240 sampled answers, one update per batch (so the ratio is 1 and the clip never binds).

  training curve (sampled at temperature 1, averages over windows of steps):
  steps  mean reward  groups with signal  entropy (nats)  answer tokens  k3 KL
  -----  -----------  ------------------  --------------  -------------  -----
   1-20         0.56                0.39            0.05          15.19  0.049
  21-40         0.70                0.38            0.04          15.36  0.060
  41-60         0.77                0.33            0.04          15.15  0.103
  61-80         0.83                0.26            0.03          15.11  0.097
  after training: greedy accuracy 85.3% (reference 58.0%), 3.92 bits/byte (reference 3.80);
  k3 per step: median 0.062, largest 0.236: no heavy tail in this run.
  another seed: accuracy 85.7% (range 0.3 points over 2 runs; one test message is 0.33 points)
  [ok] the mean reward rises over training (last window against first, by more than 0.15)  (0.56 -> 0.83)
  [ok] the share of groups with a signal falls as the pass rate rises  (0.39 -> 0.26)
  [ok] RLVR beats the reference by more than the seed range (mean of 2 seeds)  (58.0% -> 85.5%)

  the same loop from the pretrained model without SFT (10 steps): 0 of 1,280 sampled answers earned a reward, groups with a signal 0%, largest weight change 0.0e+00.
  [ok] no answer is ever right, so no group has a signal and the weights do not move at all  (RL cannot start from a policy that never succeeds)

  pass rates of the reference on the 360 training messages (8 samples each): all wrong 32%, all right 30%, mixed 38%.
  training pool            mean reward, last window  groups with signal, first window  test accuracy (2 seeds)
  -----------------------  ------------------------  --------------------------------  -----------------------
         all 360 messages                      0.82                              0.38             85.3%, 85.7%
  only the 136 mixed ones                      0.92                              0.42             78.0%, 76.3%
  The filter gives each step more signal at the start and a higher reward on its own
  pool, but the pool is smaller and saturates sooner: read the last column for what it
  did to held-out accuracy. (Two seeds per row.)

  for scale: SFT on the 360 correct answers themselves reaches 99.3% (200 steps, lr 1e-4). RLVR saw one bit per sampled answer, not
  the answer; it is the tool for when a checker exists and demonstrations do not.

All checks passed. (206.6s)
```

</details>
