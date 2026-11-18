# 混合专家（MoE）

课程页面：[混合专家（MoE）：路由、负载均衡与稀疏激活](https://dmxsf.com/wiki/architecture/mixture-of-experts/)  
[English](README.en.md)

只计算被路由到的 token 的 MoE 层；在真实模块上数总参数与激活参数；能跑起来的 DeepSeek-V3 风格玩具模型；专家容量；负载均衡损失；解码时一个批次会激活多少专家；以及在字节数据上训练的 MoE 与同算力、同参数量稠密模型的对比。

## 运行

```bash
uv run python labs/architecture/mixture-of-experts/run.py            # 完整运行，约 3 分钟
uv run python labs/architecture/mixture-of-experts/run.py --quick    # 缩小规模的冒烟测试，约 10 秒
```

也可以写成 `uv run dmxsf run mixture-of-experts`（同样可以加 `--quick`）。`--seed N` 换随机种子，默认 0。脚本分节打印结果，并用 `[ok]` / `[FAIL]` 逐条检查课程页面里的结论：全部通过时退出码为 0，有一条失败就是 1；只有完整运行才做的检查在 `--quick` 下标为 `[skip]`。

## 代码

| 位置 | 内容 |
| --- | --- |
| `labs/architecture/mixture-of-experts/run.py` | 实验脚本，按编号分节 |
| `src/dmxsf/architecture/moe.py` | 混合专家：路由器、top-k 选择、负载均衡、容量与共享专家 |
| `src/dmxsf/architecture/deepseek.py` | DeepSeek-V3 风格的语言模型：MLA、共享加路由专家、前几层稠密 |

单元测试：`uv run pytest tests/test_architecture_moe.py tests/test_architecture_deepseek.py`

## 看什么

- 第 1 节：只计算被选中的 token 与稠密计算的结果相同（误差 5.6e-17），工作量是 k/E；三种门控（Switch、Mixtral、DeepSeek-V3）的权重和分别是 0.376、1.000、2.5。
- 第 2 节：在真实模块上数参数（只建形状、不分配内存）：Mixtral 8x7B 总共 46.70B、每 token 激活 12.88B，DeepSeek-V3 总共 671.03B、激活 37.55B，与课程页一致；“8x7B”并不是 8×7B：8 个完整的 Mistral-7B 是 57.9B，而它是 1.61B 共享部分加 8 个 5.64B 的专家。
- 第 6 节：解码时一个批次会碰到多少专家：Mixtral 批大小 1、4、16、64 依次用到 2.00、5.47、7.92、8.00 个，读到 27.3%、69.2%、98.8%、99.7% 的权重；DeepSeek-V3 要到批大小 22 才有一半专家被用到。
- 第 7 节：路由偏好越大，最忙的专家越忙（倾斜 1 时是平均的 2.665 倍，29.5% 的分配被容量丢掉）；DeepSeek-V3 的偏置法不用损失项就把负载压回采样噪声的水平（1.19 倍）；步长太小太慢（γ = 0.001，100 步后最忙的专家仍有平均负载的 3.0 倍，γ = 0.01 时只有 1.16 倍），太大则振荡（γ = 0.2 时负载的抖动是 γ = 0.01 的 6 倍），并且在这个种子上比不用偏置还差（1.99 倍对 1.75 倍；5 个种子里有 3 个不是这样，这一条是观察，不成立时打印 `[differs]`）。
- 第 8 节：在字节上训练的小 MoE 与同成本、同大小的稠密模型比：MoE（有辅助损失）比同激活参数的稠密 FFN 128 只高 0.018 nats，落在种子波动 0.061 以内，实验写明“没有可以声称的差别”；比同总参数的稠密 FFN 512 差 0.109 nats，大于波动。

## 练习

改了参数再运行：核对页面数字的检查会变成 `[FAIL]`，这是预期的，看数字本身怎么变。

1. 第 7 节把偏置更新步长 `gamma` 扫得更细（0.003、0.02、0.1），哪个最接近最优？
2. 第 8 节把种子数加到 6，“MoE 与稠密 FFN 128 没有差别”的结论会变吗？

## 输出

下面是在 Apple M2 Max（12 核，仅用 CPU）、Python 3.12、PyTorch 2.15.0 上的完整运行结果。换一台机器，涉及训练的数字可能在末位上不同，计时类的数字取决于机器；检查用的是阈值与容差，不要求逐位相同。

<details>
<summary>完整运行的输出（`labs/architecture/mixture-of-experts/run.py`）</summary>

```text
# Mixture of experts: routing, load balance and the parameter accounts
  course: https://dmxsf.com/wiki/architecture/mixture-of-experts/
          https://dmxsf.com/en/wiki/architecture/mixture-of-experts/

## 1. A MoE layer computes only the tokens an expert was chosen for
  T=16 tokens, E=4 experts, k=2: max |sparse - dense| = 5.6e-17 (float64)
  token-expert pairs: sparse 32, dense 64, ratio 0.50
  [ok] computing only the chosen tokens gives the dense result
  [ok] 32 pairs against 64: the sparse form does k/E of the expert work
  measured FLOPs: sparse 50,176, dense 99,328 (router 1,024 + pairs x 1,536)
  [ok] FLOPs = router + (token-expert pairs) x 6 d f, counted by torch, not assumed

  one token, six experts, logits [2.0, 0.5, 1.2, -0.3, 1.7, 0.1]:
  gate               experts  weights         sum   
  -----------------  -------  --------------  ------
       Switch (k=1)      [0]         [0.376]  0.3760
      Mixtral (k=2)   [0, 4]  [0.574, 0.426]   1.000
  DeepSeek-V3 (k=2)   [0, 4]  [1.276, 1.224]   2.500
  [ok] Mixtral's two weights sum to 1; DeepSeek-V3's sum to its route scale 2.5
  a bias of -1 on expert 0 changes the choice [0, 4] -> [4, 2]; the weights [1.31, 1.19] are still the raw scores, normalised
  [ok] the bias decides who is chosen and never enters a weight
  [ok] one expert and k=1 is an ordinary feed-forward network

## 2. Total and active parameters, counted on real modules
  built Mixtral 8x7B and DeepSeek-V3 on the meta device in 9.3s
  Mixtral 8x7B: 46,702,792,704 parameters; 12,879,925,248 active per token (12.88B; 12.75B without the input table)
      the page's closed form has no norm gains (266,240 here): without them 12.75B, 27.3% of the total
  DeepSeek-V3: 671,026,404,352 parameters; 37,552,282,624 active per token (37.55B; 36.63B without the input table)
      the page's closed form has no norm gains (1,006,592 here): without them 36.62B, 5.5% of the total
  [ok] totals and active parameters of the course page: 46.70B / 12.88B and 671.03B / 37.55B
  [ok] without the input table and the norm gains: 12.75B and 36.62B, 27.3% and 5.5% of the total  (counting the gains moves DeepSeek-V3 from 36.62B to 36.63B: the page's figure leaves them out)
  part            Mixtral 8x7B    DeepSeek-V3    
  --------------  --------------  ---------------
  routed experts  45.10B (96.6%)  653.91B (97.4%)
  shared experts           0.00B            2.55B
       attention           1.34B           11.41B
       dense FFN           0.00B            1.19B
         routers              1M             106M
  vocab matrices           0.26B            1.85B
           total          46.70B          671.03B
  [ok] the breakdown of the course page (experts 45.10B / 653.91B, attention 1.34B / 11.41B, ...)
  Mixtral's routers: 1,048,576 parameters, 0.0022% of the total
  Mixtral = 1.61B shared + 8 x 5.64B of experts; eight full Mistral-7B models would be 57.9B
  with k=1 the active parameters are 7,242,780,672; Mistral 7B has 7,241,732,096
  [ok] '8x7B' is not 8 x 7B, and k=1 gives the dense Mistral 7B plus the routers  (difference = 1,048,576 = the routers)
  6ND with 14.8T tokens: 3.3e+24 FLOPs for the MoE, 6.0e+25 for a dense 671B (18x)
  [ok] one routed expert 44.04M; the activated FFN width (8+1) x 2048 = the dense layers' 18432; 3.3e24 vs 6.0e25
  [ok] possible expert combinations: C(8,2) = 28 and C(256,8) = 4.1e14

## 3. The same classes at toy size: a DeepSeek-V3-style model that runs
  toy: 425,728 parameters, 204,544 active per token, 3 MoE layers after 1 dense layer
  decoding token by token from the latent cache: max logit error 1.8e-07; the cache holds 96 numbers per token = 4 x (16 + 8)
  [ok] MLA attention + shared and routed experts: decoding from the cache equals the full pass
  nodes a token is sent to (256 experts on 8 nodes, k=8): free routing 5.31 on average (exact 5.29), at most 8; node-limited 3.97, at most 4
  [ok] node-limited routing sends every token to at most 4 nodes

## 4. Capacity: how many assignments an expert may take
  case         T     k  E   factor  average load  capacity
  -----------  ----  -  --  ------  ------------  --------
  course page   256  2   8   1.250         64.00        80
         quiz   512  2  16   1.250         64.00        80
   exercise 2  1024  8  64   1.250         128.0       160
  285 assignments for one expert with capacity 160: 160 kept, 125 dropped
  [ok] capacities 80 / 80 / 160, and 285 - 160 = 125 assignments dropped  (the first 160 arrivals are the ones kept)

## 5. The balance loss N * sum(f_i P_i)
  perfectly even f, five random P: loss = 1.000000, 1.000000, 1.000000, 1.000000, 1.000000
  every token on one expert: loss = 8.000 (N = 8)
  random router, k=2: loss = 1.029; d loss / d p[t, i] = N f_i / T, f = [0.141, 0.117, 0.109, 0.148, 0.141, 0.164, 0.055, 0.125]
  [ok] f uniform gives exactly 1 whatever P is; a collapsed router gives N
  [ok] the gradient reaches the router through P only: d loss / d p[t, i] = N f_i / T

## 6. Decoding: how many experts does a batch touch?
  model         batch  experts used / layer  simulated  weights read
  ------------  -----  --------------------  ---------  ------------
  Mixtral 8x7B      1                  2.00       2.00         27.3%
  Mixtral 8x7B      4                  5.47       5.46         69.2%
  Mixtral 8x7B     16                  7.92       7.93         98.8%
  Mixtral 8x7B     64                  8.00       8.00         99.7%
   DeepSeek-V3      1                  8.00       8.00          5.5%
   DeepSeek-V3      4                 30.53      30.52         14.0%
   DeepSeek-V3     16                101.96     102.04         41.2%
   DeepSeek-V3     64                222.44     222.16         87.1%
  [ok] the course page's table: 2.00 / 5.47 / 7.92 / 8.00 and 8.00 / 30.53 / 101.96 / 222.44 experts
  [ok] weights read per decoding step: 27.3% / 69.2% / 98.8% / 99.7% and 5.5% / 14.0% / 41.2% / 87.1%
  [ok] a Monte Carlo over random routers agrees with the formula
  exercise 3: more than half of the 256 experts need B >= 21.8: B=21 -> 124.6, B=22 -> 128.7
  [ok] B = 22 (B = 21 gives 124.6)
  with a popularity-skewed router, a batch of 64 touches 130.0 experts, not 222.4: skew only helps
  [ok] non-uniform routing touches fewer experts than the uniform estimate

## 7. Imbalance, capacity and bias balancing on synthetic router scores
  8 experts, k=2, 256 tokens per batch, capacity factor 1.25; mean over 40 batches

  tilt  busiest / average  dropped  balance loss
  ----  -----------------  -------  ------------
     0              1.190     0.1%         1.005
   0.4              1.755    10.2%         1.151
     1              2.665    29.5%         1.668
  [ok] more preference, more imbalance: busiest expert, drops and balance loss all rise with the tilt
  with no preference at all the busiest expert is still above average, from sampling alone: 1.39x at 64 tokens, 1.19x at 256 tokens, 1.10x at 1024 tokens  (the course page's single runs: 1.31x, 1.16x, 1.09x)
  [ok] that sampling floor shrinks with the batch and matches the page within 0.1
  tilt 0.4, dropped assignments by capacity factor: 1.0: 21.4%, 1.25: 10.2%, 1.5: 3.6%, 2.0: 0.0%
  the same router with k=1 (Switch) at factor 1.25: 14.6% dropped, against 10.2% for k=2
  [ok] a larger capacity factor drops fewer assignments; k=1 drops more than k=2

  bias balancing (DeepSeek-V3 rule, scores are sigmoids), 200 steps, fresh batch each step:
  tilt  gamma  @25    @100   last 20%  jitter   dropped
  ----  -----  -----  -----  --------  -------  -------
   0.4  0.001  1.641  1.181     1.175  0.06786     0.1%
   0.4   0.01  1.256  1.194     1.195  0.05666     0.1%
   0.4   0.05  1.372  1.366     1.351   0.1210     1.7%
   0.4    0.2  1.950  2.009     1.987   0.3548    17.0%
   1.5  0.001  3.138  3.016     2.435   0.1486    16.3%
   1.5   0.01  2.116  1.163     1.202  0.06437     0.1%
   1.5   0.05  1.472  1.562     1.441   0.1457     2.9%
   1.5    0.2  2.456  2.531     2.502   0.4962    25.1%
  (columns: busiest / average expert load; jitter = its standard deviation in the last 20%)
  tilt 0.4, gamma 0.01: Spearman correlation between the learned bias and the popularity -0.98; most popular expert 4, lowest bias 4
  [ok] a reasonable step reaches the sampling floor without any loss term  (1.19x against 1.19x with no preference)
  [ok] the learned bias opposes the popularity
  [ok] a step that is too small is slow (tilt 1.5: after 100 steps gamma 0.001 leaves the busiest expert at more than 1.5x the load it has with gamma 0.01)  (3.02x against 1.16x)
  [ok] a step that is too large oscillates: with gamma 0.2 the busiest expert carries more load and its load jumps about more than with gamma 0.01 (tilt 0.4)  (1.99x against 1.19x, jitter 0.355 against 0.057)
  [ok] ...and with tilt 0.4 it is worse than no bias at all  (1.99x against 1.75x)
  64 experts, k=8, 1024 tokens, tilt 0.4: busiest expert 3.10x and 16.8% dropped with no bias; after 100 steps of gamma 0.005: 1.24x and 0.0%
  [ok] the same mechanism balances a DeepSeek-V3-shaped layer

## 8. Trained on bytes: a MoE against dense models of the same cost and the same size
  2 layers, d_model 64, 500 steps, validation loss in nats per byte, 3 seed(s)
  model                      params   active   FLOPs/token  ms/step  val loss  range over seeds
  -------------------------  -------  -------  -----------  -------  --------  ----------------
             dense, FFN 128   98,624   98,624         229k     18.3     1.964     1.956 - 1.969
  MoE 8x64, top-2, aux 0.01  247,104   99,648         231k     46.6     1.982     1.943 - 2.003
    MoE 8x64, top-2, no aux  247,104   99,648         231k     35.0     1.931     1.922 - 1.942
             dense, FFN 512  246,080  246,080         524k     21.2     1.873     1.866 - 1.883
  MoE                        busiest / average load  dropped at factor 1.25
  -------------------------  ----------------------  ----------------------
  MoE 8x64, top-2, aux 0.01                   1.268                    0.6%
    MoE 8x64, top-2, no aux                   2.656                   29.6%
  MoE (aux 0.01) minus dense FFN 128: +0.018 nats, within the seed range 0.061: no difference we can claim
  MoE (aux 0.01) minus dense FFN 512: +0.109 nats, larger than the seed range 0.061
  MoE without aux minus MoE with aux: -0.051 nats, within the seed range 0.061: no difference we can claim
  [ok] same active parameters as the dense model, plus the routers; the total is that of the wide dense FFN  (99,648 = 98,624 + 1024)
  [ok] measured FLOPs per token: the MoE costs what the small dense model costs, the wide one more  (1.009x and 2.29x of the small dense model)
  [ok] every variant learned (well below the 5.55 nats of a uniform guess)  (worst 1.982)

All checks passed. (205.1s)
```

</details>
