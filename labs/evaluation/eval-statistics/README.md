# 评测统计

课程页面：[评测的统计学：pass@k、置信区间与所需题数](https://dmxsf.com/wiki/evaluation/eval-statistics/)  
[English](README.en.md)

pass@k 对照暴力枚举；95% 区间对照真实覆盖率；配对比较与样本量表对照真实检验的功效；最后在留出文本上比较两个真实模型。

## 运行

```bash
uv run python labs/evaluation/eval-statistics/run.py            # 完整运行，几秒
uv run python labs/evaluation/eval-statistics/run.py --quick    # 缩小规模的冒烟测试，几秒
```

也可以写成 `uv run dmxsf run eval-statistics`（同样可以加 `--quick`）。`--seed N` 换随机种子，默认 0。脚本分节打印结果，并用 `[ok]` / `[FAIL]` 逐条检查课程页面里的结论：全部通过时退出码为 0，有一条失败就是 1；只有完整运行才做的检查在 `--quick` 下标为 `[skip]`。

这个实验用到缓存的小模型：字节级基础模型（84 万参数）、字节级小模型（草稿模型）。第一次运行会先在 TinyShakespeare 上训练并缓存到 `~/.cache/dmxsf`（环境变量 `DMXSF_CACHE` 可以改位置），之后直接读取。

## 代码

| 位置 | 内容 |
| --- | --- |
| `labs/evaluation/eval-statistics/run.py` | 实验脚本，按编号分节 |
| `src/dmxsf/evaluation/stats.py` | 评测统计：pass@k、分数的误差、配对比较与样本量 |

单元测试：`uv run pytest tests/test_evaluation_stats.py`

## 看什么

- 第 1 节：pass@k 的无偏估计在页面的例子里是 0.4200 / 0.6833 / 0.8000，朴素估计是 0.4200 / 0.6478 / 0.7246（k > 1 时偏低）；它与枚举全部 k 元子集的平均逐位相等；在 40 道已知成功率的题上重复 4,000 次，无偏估计均值落在真值上，朴素估计在 pass@10 上低 0.0430。
- 第 2 节：HumanEval（164 题）在准确率 80% 时 95% 区间是 ±6.1 个点，GSM8K ±2.2，MMLU ±0.7；模拟 40,000 次，区间的覆盖率是 0.9493 与 0.9485。
- 第 3 节：两个模型在同一批题上，配对分析的区间是 [+0.011, +0.136]（不含 0），当作独立样本算是 [−0.014, +0.161]（含 0）；重复 3,000 次，配对找到真实差距的次数是 57.5%，不配对只有 26.8%。
- 第 4 节：差距减半，需要的题数约四倍（2 个点到 1 个点：×3.93）；页面表里的题数下，真实的检验在 80% 左右的次数里能找到差距（0.795、0.800、0.788）。
- 第 5 节：用两个真实的字节模型（草稿和基础）在 434 个窗口上比：相关 0.95，配对的区间半宽 0.0152，不配对 0.0566；这些窗口来自同一份文本，并不独立，区间应当读作真实不确定性的下界。

## 练习

改了参数再运行：核对页面数字的检查会变成 `[FAIL]`，这是预期的，看数字本身怎么变。

1. 第 3 节里两个模型的相关来自共享的题目难度：把 `difficulty = page.normal(scale=3.0, size=n_q)` 的 3.0 改成 1.0 或 6.0，相关系数怎么变，配对与不配对的区间半宽谁受的影响更大？
2. 第 4 节 `for gap in (5, 3, 2, 1)` 里加一个 0.5：需要多少道题？这比现实里常见的基准大多少？

## 输出

下面是在 Apple M2 Max（12 核，仅用 CPU）、Python 3.12、PyTorch 2.15.0 上的完整运行结果。换一台机器，涉及训练的数字可能在末位上不同，计时类的数字取决于机器；检查用的是阈值与容差，不要求逐位相同。

<details>
<summary>完整运行的输出（`labs/evaluation/eval-statistics/run.py`）</summary>

```text
# Evaluation statistics: pass@k, the error of a score, paired tests, sample sizes
  course: https://dmxsf.com/wiki/evaluation/eval-statistics/
          https://dmxsf.com/en/wiki/evaluation/eval-statistics/

## 1. pass@k: the page's example, then the estimator against brute force
           unbiased  naive 
  -------  --------  ------
   pass@1    0.4200  0.4200
   pass@5    0.6833  0.6478
  pass@10    0.8000  0.7246
  [ok] the page's table: unbiased 0.4200 / 0.6833 / 0.8000, naive 0.4200 / 0.6478 / 0.7246
  [ok] pass@k equals the average over every possible k-subset of the n samples  (largest gap 0.0e+00 over 5 cases)
  40 problems with known success rates, n = 10 samples each, 4000 repeats:
           true value  unbiased  naive   naive - true
  -------  ----------  --------  ------  ------------
   pass@1      0.4692    0.4692  0.4692       -0.0000
   pass@2      0.6138    0.6138  0.5994       -0.0145
   pass@5      0.7631    0.7627  0.7322       -0.0309
  pass@10      0.8341    0.8334  0.7911       -0.0430
  [ok] the unbiased estimator averages to the true value (within 3.5 standard errors)
  [ok] the naive estimate is below the truth for every k > 1

## 2. The error of one score
  benchmark  questions  95% interval at 80%
  ---------  ---------  -------------------
  HumanEval        164    80% +- 6.1 points
      GSM8K       1319    80% +- 2.2 points
       MMLU      14079    80% +- 0.7 points
  [ok] +-6.1, +-2.2 and +-0.7 points, as on the page

  How often the interval contains the true accuracy (40,000 simulated runs):
  questions  coverage of the 95% interval
  ---------  ----------------------------
        164                        0.9493
       1319                        0.9485
  [ok] the 95% interval covers about 95% of the time (0.93 to 0.96)  (0.9493 and 0.9485)

## 3. Comparing two models: the same questions, so pair them
  A 0.829   B 0.756   correlation 0.50   gap +0.073
  interval          low     high    half-width
  ----------------  ------  ------  ----------
          unpaired  -0.014  +0.161       0.088
            paired  +0.011  +0.136       0.062
  paired bootstrap  +0.012  +0.140       0.064
  [ok] the page's simulation: A 0.829, B 0.756, correlation 0.50, gap +0.073
  [ok] unpaired interval [-0.014, +0.161] contains 0, paired [+0.011, +0.136] does not
  [ok] the bootstrap over questions agrees with the paired formula: [0.012, 0.140]

  The same comparison repeated 3,000 times with new questions each time:
  analysis  runs in which A is found better than B
  --------  --------------------------------------
    paired                                   0.575
  unpaired                                   0.268
  [ok] pairing finds the real difference more often than treating the runs as independent  (0.575 against 0.268)

## 4. How many questions does it take?
  gap       paired (correlation 0.5)  unpaired (correlation 0)
  --------  ------------------------  ------------------------
  5 points                       548                     1,091
  3 points                     1,472                     2,940
  2 points                     3,256                     6,507
   1 point                    12,792                    25,580
  [ok] the page's table: 548 / 1,091, 1,472 / 2,940, 3,256 / 6,507, 12,792 / 25,580
  [ok] halving the gap takes about four times the questions (2 points -> 1 point)  (x3.93)

  Power of a real test at the table's size (4,000 simulated benchmarks each):
  case                                 questions  found the gap
  -----------------------------------  ---------  -------------
    5 points, correlation 0.5, paired        548          0.795
  5 points, correlation 0.0, unpaired      1,091          0.800
    3 points, correlation 0.5, paired      1,472          0.788
  [ok] at the size the table gives, the test finds the gap about 80% of the time (0.75 to 0.85)  (0.795, 0.800, 0.788)

## 5. Two real models on held-out text (base against draft)
  434 windows of 128 bytes; score = bits per byte (lower is better)
  draft 2.480   base 2.303   correlation between the two 0.95
  interval for draft - base  low     high    half-width
  -------------------------  ------  ------  ----------
                   unpaired  +0.120  +0.233      0.0566
                     paired  +0.161  +0.192      0.0152
           paired bootstrap  +0.161  +0.192      0.0154
  [ok] the two models' window scores are positively correlated, so pairing narrows the interval  (correlation 0.95; half-width 0.0152 against 0.0566)
  [ok] the paired normal interval and the bootstrap interval agree to within 0.01 bits
  [ok] the paired interval for draft - base excludes zero: the draft really is worse  (draft - base = +0.177 bits per byte, interval [+0.161, +0.192])

  Windows are the 'questions' here; they all come from one text, so they are not independent
  in the way the formulas assume (neighbouring speeches share characters and style). The
  intervals are best read as lower bounds on the real uncertainty.

All checks passed. (1.4s)
```

</details>
