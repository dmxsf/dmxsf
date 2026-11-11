# 扩展定律

课程页面：[扩展定律：算力、参数与数据如何分配](https://dmxsf.com/wiki/pretraining/scaling-laws/)  
[English](README.en.md)

重算课程页的幂律与计算最优配比；推理成本如何改变最优模型；一次真实的小型扩展实验（按规模调学习率、训练网格、拟合 L(N, D)、预测没跑过的配置）；以及 12 个点的拟合能识别什么、不能识别什么。

## 运行

```bash
uv run python labs/pretraining/scaling-laws/run.py            # 完整运行，约 2 分钟
uv run python labs/pretraining/scaling-laws/run.py --quick    # 缩小规模的冒烟测试，几秒
```

也可以写成 `uv run dmxsf run scaling-laws`（同样可以加 `--quick`）。`--seed N` 换随机种子，默认 0。脚本分节打印结果，并用 `[ok]` / `[FAIL]` 逐条检查课程页面里的结论：全部通过时退出码为 0，有一条失败就是 1；只有完整运行才做的检查在 `--quick` 下标为 `[skip]`。

## 代码

| 位置 | 内容 |
| --- | --- |
| `labs/pretraining/scaling-laws/run.py` | 实验脚本，按编号分节 |
| `src/dmxsf/pretraining/scaling.py` | Chinchilla 的参数化损失、算力最优的配比与推理成本 |
| `src/dmxsf/pretraining/accounting.py` | 一次训练的算术：FLOPs、MFU、GPU 小时，以及模型状态与激活的字节数 |

单元测试：`uv run pytest tests/test_pretraining_scaling.py tests/test_pretraining_accounting.py`

## 看什么

- 第 1 节：用 Chinchilla 的系数重算页面的表（1e21 FLOPs → 2.8B 参数、0.06T token；1e25 → 312.1B、5.34T），闭式解与暴力搜索的最小值一致；谷底很平，最优规模减半或加倍只多损失不到 0.02（最差 0.0100）；Llama 3 8B 比同预算的最优点高 0.05。
- 第 2 节：把“部署后要服务多少 token”算进去，最优模型从 46.9B（不服务）一路缩小到 9.6B（服务 1e15 token），训练 token 从 0.88T 涨到 50.6T，token/参数比从 19 到 5264，这就是小模型过训练的账。
- 第 3 节是我们自己的小实验：4 种宽度 × 3 种数据量 × 2 个种子，每个格子都调过学习率。数据最少时最大的模型并不最好（最好的是 N = 176,128）；最优学习率随模型变大而下降（0.1 → 0.001）；不调学习率的代价（最小的模型 +0.293）比放大一档模型的收益（0.192）还大。
- 第 3 节的拟合 E = 1.41、α = 0.48、β = 0.45（均方残差 0.068）；留出最大的一次运行再预测：预测 1.947、实测 1.847，误差 0.101，比“和上一格一样”的 0.390 好，但没有小于种子间的范围（0.292）：这个网格只能说明方法，说不上结论。
- 第 4 节：已知真值的合成数据上，系数（A、B、β）比它们合出的预测不确定得多（B 的 10–90% 区间半宽是真值的 232%，外推损失只有 1.8%）；按实测的噪声，拟合出的最优规模多付的损失只有 0.0009（中位数）；噪声放大十倍后，系数完全失控，中位数是 0.054，最差的十分之一在 0.251 以上，已经超过砍掉九成算力的代价（0.109）。比较预测，不要比较系数。

## 练习

改了参数再运行：核对页面数字的检查会变成 `[FAIL]`，这是预期的，看数字本身怎么变。

1. 第 2 节 `scaling.best_model(2.0, served)` 里的 2.0 是目标损失：改成 1.9，最优模型会变大还是变小，各行的 token/参数比怎么变？
2. 把 `SEEDS = 1 if args.quick else 2` 里的 2 改成 4（要多等几分钟），中位数种子差、拟合系数和留出预测的误差各自怎么变？

## 输出

下面是在 Apple M2 Max（12 核，仅用 CPU）、Python 3.12、PyTorch 2.15.0 上的完整运行结果。换一台机器，涉及训练的数字可能在末位上不同，计时类的数字取决于机器；检查用的是阈值与容差，不要求逐位相同。

<details>
<summary>完整运行的输出（`labs/pretraining/scaling-laws/run.py`）</summary>

```text
# Scaling laws: the formulas, the serving cost, and a tiny study of our own
  course: https://dmxsf.com/wiki/pretraining/scaling-laws/
          https://dmxsf.com/en/wiki/pretraining/scaling-laws/

## 1. Power laws and the compute-optimal split
  [ok] Kaplan's N exponent 0.076: ten times the parameters multiply the loss by 0.84  (0.8395)
  C (FLOPs)  N_opt   D_opt  D / N  loss 
  ---------  ------  -----  -----  -----
      1e+21    2.8B  0.06T   21.6  2.306
      1e+22    9.0B  0.18T   20.4  2.141
      1e+23   29.4B  0.57T   19.2  2.032
   5.76e+23   72.2B  1.33T   18.4  1.974
      1e+25  312.1B  5.34T   17.1  1.912
  [ok] the page's table: 2.8B 0.06T 21.6 2.306 ... 312.1B 5.34T 17.1 1.912
  [ok] exponents 0.51 and 0.49: ten times the budget buys about 3.2x the parameters and 3.1x the data  (N x3.26, D x3.07)
  [ok] the closed form is the brute-force minimum of L(N, C/6N) (the four IsoFLOP minima of the page's figure)  (2.8B, 9.0B, 29.4B, 95.9B)
  [ok] the valley is flat: half or double the optimal size costs less than 0.02 in loss  (worst +0.0100)
  [ok] Llama 3 8B (D/N = 1875) sits 0.05 above the compute-optimal loss at its own budget  (optimal at this budget would be 81B on 1.5T tokens; an extrapolation)
  [ok] D = 20N and C = 6ND give C = 120 N^2: 1.2e21 FLOPs -> 3.2B (quiz), 1e22 -> 9.1B and 180B tokens (exercise)  (the last is 183B, which the page rounds to 180B)
  [ok] exercise 2: a 7B model on 3T tokens is 1.26e23 FLOPs, 88 thousand H100 hours at MFU 40%  (88,473 GPU hours)
  [ok] 6ND and 184,320 A100 hours put Llama 2 7B at MFU 39%

## 2. Counting the serving cost: 6ND to train plus 2NT to serve
  tokens served  best N  training tokens  tokens / parameter
  -------------  ------  ---------------  ------------------
              0   46.9B            0.88T                  19
          1e+12   34.3B            1.25T                  36
          1e+13   20.0B            3.01T                 150
          1e+14   12.7B           11.12T                 876
          1e+15    9.6B           50.60T                5264
  [ok] the page's table: 46.9B 0.88T 19 | 34.3B 1.25T 36 | 20.0B 3.01T 150 | 12.7B 11.12T 876 | 9.6B 50.60T 5264
  [ok] with nothing served the best model is the Chinchilla one, about 19 tokens per parameter
  [ok] more serving means a smaller model on more data, monotonically
  [ok] tokens per parameter of the page's models: 1.7, 20, 143, 286, 1875

## 3. A tiny scaling study on TinyShakespeare
  4 sizes (N = 12,288, 40,960, 176,128, 884,736) x 3 token budgets, 8 x 64 = 512 tokens per step, 2 seed(s) per cell;
  validation loss in nats per byte on the held-out 5% of the text

  learning-rate sweep at 131,072 tokens (the best of each row is marked *):
               0.0003  0.001   0.003   0.01   0.03    0.1   
  -----------  ------  ------  ------  -----  ------  ------
   d_model 16                   3.002  2.807   2.784  2.709*
   d_model 32                   2.501  2.553  2.464*   2.806
   d_model 64           2.462  2.283*  2.506                
  d_model 128   2.602  2.223*   2.328                       

  mean validation loss over seeds (rows: N = matrix-multiply parameters, columns: training tokens D):
  N \ D    32,768  131,072  524,288
  -------  ------  -------  -------
   12,288   3.208    2.674    2.269
   40,960   3.133    2.482    2.114
  176,128   3.077    2.298    1.919
  884,736   3.151    2.236    1.847
  seed-to-seed range over the 12 cells: median 0.029, largest 0.292 nats
  [ok] more tokens always help: at every size the loss falls with D
  [ok] with the most data every size step helps: loss falls with N
  [ok] with the fewest tokens (64 steps) the biggest model is not the best: it cannot use its capacity  (best at 32,768 tokens: N = 176,128)
  [ok] the best learning rate falls with model size  (0.1 for N = 12,288, 0.001 for N = 884,736)

  what an untuned learning rate costs (3e-3 for every size, 131,072 tokens):
  N        tuned lr  tuned loss  loss at 3e-3  cost of not tuning
  -------  --------  ----------  ------------  ------------------
   12,288       0.1       2.709         3.002              +0.293
   40,960      0.03       2.464         2.501              +0.038
  176,128     0.003       2.283         2.283              +0.000
  884,736     0.001       2.223         2.328              +0.105
  [ok] not tuning the learning rate costs as much as a size step: the curve then shows the recipe, not the scale  (+0.293 for the smallest model, against 0.192 gained by the next size up)

  fit of L = E + A/N^alpha + B/D^beta to all 12 cells: E = 1.41, alpha = 0.48, beta = 0.45, rms residual 0.068 nats
  (the page's coefficients, E = 1.8172, alpha = 0.3478, beta = 0.3658, describe subword tokens of web text; bytes of one play are another setting, so these are different numbers, not wrong ones)
  [ok] a loss floor and two decaying terms describe the grid  (rms 0.068)

  held out the biggest run (N = 884,736, D = 524,288, 2.8e+12 FLOPs, 5.0x the compute of the next smaller model at the same D):
  predicted 1.947, measured 1.847 (seeds 1.841 - 1.853), the previous cell of this size was 2.236
  [ok] the fit predicts a run it never saw better than 'same as the previous cell of that size'  (error 0.101 against 0.390)
  the error is within the larger of the seed range (0.292) and the fit's rms (0.068) when the biggest run is left out

  at C = 1.3e+11 FLOPs the fit's best split is N = 19,330 on D = 1,110,969 tokens (D / N = 57; the page's rule of thumb is 20; the corpus has 1,059,624 training bytes)
  cell near that budget     FLOPs    D / N  loss 
  ------------------------  -------  -----  -----
   N = 40,960, D = 524,288  1.3e+11   12.8  2.114
  N = 176,128, D = 131,072  1.4e+11    0.7  2.298
   N = 884,736, D = 32,768  1.7e+11    0.0  3.151
  [ok] among the cells near that budget the fit and the measurements pick the same winner  (N = 40,960)

## 4. What a small grid can and cannot identify (synthetic data, known truth)
  12 cells of the page's law (N 1e7 - 1e9, D 3e9 - 1e11), 30 noisy copies of the grid, fitted each time;
  noise sigma = 0.025 (the median seed-to-seed range of section 3) and ten times that; 10th - 90th percentile of the fitted values:
  quantity               truth      sigma 0.025           +-    sigma 0.253            +-    
  ---------------------  ---------  --------------------  ----  ---------------------  ------
                      E      1.817         1.689 - 1.894    6%      0.0002148 - 2.376     65%
                      A        482         347.2 - 622.9   29%      21.12 - 7.642e+04  >1000%
                  alpha     0.3478       0.3256 - 0.3632    5%        0.1083 - 0.6645     80%
                      B       2085     669.8 - 1.034e+04  232%      10.35 - 2.688e+14  >1000%
                   beta     0.3658       0.3061 - 0.4426   19%         0.06392 - 1.55    203%
  loss at N=1e9, D=1e12      2.259         2.214 - 2.294    2%          1.943 - 2.487     12%
        N_opt at C=1e23  2.945e+10  1.88e+10 - 4.916e+10   52%  1.051e+09 - 1.078e+12  >1000%
  [ok] measured noise: the coefficients are far less certain than the predictions they combine into  (uncertainty of A and B 232%, of the extrapolated loss 1.8% (half-width of the 10 - 90% range))
  [ok] ten times the noise: the coefficients are far less certain than the predictions they combine into  (uncertainty of A and B >1000%, of the extrapolated loss 12.0% (half-width of the 10 - 90% range))
  [ok] measured noise: a typical fitted N_opt costs a few hundredths of a nat (the valley is flat)  (loss penalty at C = 1e23 under the true law: median 0.0009, worst tenth 0.0039 nats or more)
  with ten times the noise the median penalty is 0.054 and the worst tenth lose 0.251 or more; for scale, a tenfold cut in the compute budget costs 0.109
  so a fitted 'alpha = 0.34' deserves less trust than the curve it came from; compare predictions, not coefficients

All checks passed. (136.4s)
```

</details>
