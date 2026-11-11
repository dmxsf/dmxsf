# LLM 评委

课程页面：[LLM-as-a-Judge：用模型当评委，以及它的偏差与校准](https://dmxsf.com/wiki/evaluation/llm-as-judge/)  
[English](README.en.md)

一个偏差可调的评委替身，所以每个数字都有精确答案：交换顺序能抵消位置偏差却抵消不了长度偏差；Cohen's kappa 能识破永远猜多数类的评委；按长度校正的 Bradley-Terry 排行榜。

## 运行

```bash
uv run python labs/evaluation/llm-as-judge/run.py            # 完整运行，几秒
uv run python labs/evaluation/llm-as-judge/run.py --quick    # 缩小规模的冒烟测试，几秒
```

也可以写成 `uv run dmxsf run llm-as-judge`（同样可以加 `--quick`）。`--seed N` 换随机种子，默认 0。脚本分节打印结果，并用 `[ok]` / `[FAIL]` 逐条检查课程页面里的结论：全部通过时退出码为 0，有一条失败就是 1；只有完整运行才做的检查在 `--quick` 下标为 `[skip]`。

## 代码

| 位置 | 内容 |
| --- | --- |
| `labs/evaluation/llm-as-judge/run.py` | 实验脚本，按编号分节 |
| `src/dmxsf/evaluation/judge.py` | LLM-as-a-judge：带偏差的（模拟）评审、一致性指标与 Bradley-Terry 评分 |

单元测试：`uv run pytest tests/test_evaluation_judge.py`

## 看什么

- 第 1 节：位置偏差下，只判一次、随机顺序时与真值一致 0.753（精确值 0.757）；两种顺序都判再取一致的：0.380 的题成了平局，其余的一致率 0.914。偏差越大，只判一次越差（偏差 2.0 时 0.558），交换后一致率保持在 0.99，代价是平局升到 0.882。
- 第 2 节：没有偏差时 1/2 + arctan(s/σ)/π = 78.5%，两种顺序取平均是 83.6%（噪声除以 √2）；位置偏差降低一次判断的一致率，却不改变平均后的结果。
- 第 3 节：长度偏差交换也消不掉：同样质量、长度四倍的两个回答，只有位置偏差时交换后 95.7% 成了平局（正确的判决），有长度偏差时较长的回答两种顺序都赢（99.7%），交换后不报平局。
- 第 4 节：一致率 0.750、随机一致 0.418，Cohen's κ = 0.570；总说“A 赢”的懒裁判一致率 0.900 而 κ = 0；一致率随质量差距从 0.565 升到 0.998，一个总数 0.785 把这些都藏起来了。
- 第 5 节：从两两投票到排行榜：Bradley-Terry 加自助法（B 1031.0 [1006.8, 1052.3] 等），B − A 的区间 [6.8, 52.3] 不含 0，真值都落在区间里；打乱顺序，在线 Elo 的 D − B 在 178.9 到 240.2 之间摆动，BT 不变。
- 第 6 节：偏爱长回答的裁判让话痨模型 E（实际和 A 一样好）排到 C 之上（1175 对 1120）；把长度差作为协变量放进回归，E 回到 1017，系数 0.302（真值 0.35）。

## 练习

改了参数再运行：核对页面数字的检查会变成 `[FAIL]`，这是预期的，看数字本身怎么变。

1. 第 1 节 `for bias in (0.0, 0.25, 0.5, 1.0, 2.0)` 里加一个 4.0：交换后的一致率和平局率各到多少？
2. 第 6 节里把 `n_battles` 的 6000 改成 600：长度系数的估计还准吗？E 还能回到 A 的水平吗？

## 注意

- 裁判是模拟的（质量差距、噪声、位置偏差、长度偏差都是参数），不是真正的语言模型：这样才有可以对照的真值。

## 输出

下面是在 Apple M2 Max（12 核，仅用 CPU）、Python 3.12、PyTorch 2.15.0 上的完整运行结果。换一台机器，涉及训练的数字可能在末位上不同，计时类的数字取决于机器；检查用的是阈值与容差，不要求逐位相同。

<details>
<summary>完整运行的输出（`labs/evaluation/llm-as-judge/run.py`）</summary>

```text
# LLM-as-a-judge: bias, agreement, and Bradley-Terry leaderboards
  course: https://dmxsf.com/wiki/evaluation/llm-as-judge/
          https://dmxsf.com/en/wiki/evaluation/llm-as-judge/

## 1. Swapping the order: the page's simulation, then the exact answer
  procedure                                    simulated (20,000 pairs)  exact (integral over the gap)
  -------------------------------------------  ------------------------  -----------------------------
  judge once, random order: agrees with truth                     0.753                          0.757
         swap both orders: pairs left as ties                     0.380                          0.382
      swap both orders: agreement on the rest                     0.914                          0.915
  [ok] the page's numbers: 0.753 once; 0.380 ties and 0.914 agreement when swapped
  [ok] the simulation agrees with the exact integral to within 0.01

  The price of swapping grows with the bias (noise 0.8, exact):
  position bias  judge once  ties when swapped  agreement when swapped
  -------------  ----------  -----------------  ----------------------
          0.000       0.785              0.291                   0.902
         0.2500       0.778              0.315                   0.906
         0.5000       0.757              0.382                   0.915
          1.000       0.688              0.577                   0.944
          2.000       0.558              0.882                   0.990
  [ok] judging once gets worse as the bias grows; swapping keeps agreement high but ties rise

## 2. Averaging both orders, and the closed forms on the page
  quality gap sd 1.0, noise 0.8, 200,000 pairs per row:
  position bias  judge once  average of both orders
  -------------  ----------  ----------------------
          0.000       0.785                   0.836
         0.5000       0.757                   0.836
  [ok] no bias: 1/2 + arctan(s/sigma)/pi = 78.5% once, and 83.6% averaging both orders (noise / sqrt 2)
  [ok] the simulations match those closed forms to within 0.01  (0.785 and 0.836)
  [ok] a position bias lowers the one-shot agreement but leaves the averaged one unchanged

## 3. A length bias is not cancelled by swapping
  two answers of equal quality, x four times as long; the right verdict is 'tie':
  judge               x wins, judged once  x wins swapped  y wins swapped  tie swapped
  ------------------  -------------------  --------------  --------------  -----------
  position bias only                0.499           0.021           0.022        0.957
    length bias only                0.998           0.997           0.000        0.003
  [ok] position bias: swapping turns almost every pair into a tie (the right verdict)  (0.957 ties)
  [ok] length bias: the longer answer wins in both orders and swapping reports no tie  (x wins 0.997)

## 4. Agreement, kappa, and how many human labels are needed
  rows: human (A wins, B wins, tie); columns: judge
  agreement 0.750, chance agreement 0.418, Cohen's kappa 0.570
  a judge that always says 'A': agreement 0.900, kappa 0.000
  [ok] the page's numbers: 0.750 / 0.418 / 0.570, and 0.900 agreement with kappa 0 for the lazy judge
  [ok] labels needed for +-5 points: 246 at 80% agreement, 385 at 50%

  Agreement with the truth by size of the quality gap (the same judge throughout):
  |gap|        pairs  agreement  expected
  -----------  -----  ---------  --------
    0 to 0.25  39656      0.565     0.562
  0.25 to 0.5  37262      0.678     0.679
     0.5 to 1  59501      0.816     0.817
       1 to 2  54387      0.950     0.950
       2 to 8   9194      0.998     0.997
  one overall number would be 0.785, which hides all of that
  [ok] agreement rises with the gap, from near a coin flip to near certainty  (0.565 -> 0.998)

## 5. From pairwise votes to a leaderboard: Bradley-Terry with a bootstrap
  model  true Elo  estimate  95% interval    
  -----  --------  --------  ----------------
      A      1000    1000.0  [1000.0, 1000.0]
      B      1050    1031.0  [1006.8, 1052.3]
      C      1120    1118.6  [1095.0, 1140.9]
      D      1250    1246.8  [1223.2, 1270.5]
  B - A: 31.0 points, interval [6.8, 52.3]
  [ok] the page's table: B 1031.0 [1006.8, 1052.3], C 1118.6 [1095.0, 1140.9], D 1246.8 [1223.2, 1270.5]
  [ok] the interval of the difference B - A is [6.8, 52.3] and excludes 0
  [ok] every true rating lies inside its interval
  [ok] the Newton fit sits at the maximum of the likelihood (gradient is zero)  (largest gradient entry 3.3e-13)
  [ok] a 400-point Elo gap is 10 to 1 odds, and the Elo scale is 400 / ln 10 per coefficient

  The same 3,000 battles in five random orders; D minus B in rating points:
  order  online Elo (K=4)  Bradley-Terry
  -----  ----------------  -------------
      0             240.2          215.8
      1             227.6          215.8
      2             178.9          215.8
      3             208.8          215.8
      4             202.6          215.8
  [ok] Bradley-Terry does not depend on the order of the battles; online Elo does  (spread 4.5e-13 against 61.2 points)

## 6. The judge's bias goes into the leaderboard unless you model it
  model  true Elo  naive Bradley-Terry  length-controlled
  -----  --------  -------------------  -----------------
      A      1000                 1000               1000
      B      1050                 1066               1066
      C      1120                 1120               1120
      D      1250                 1253               1253
      E      1000                 1175               1017
  estimated length coefficient 0.302 (true 0.35); E is the wordy model
  [ok] a length-loving judge puts the wordy model E above C, though E is as good as A  (naive E 1175, C 1120, A 1000)
  [ok] adding the length difference as a covariate recovers the true order and the coefficient  (controlled E 1017, coefficient 0.302)

  This is the idea behind length-controlled win rates (Dubois et al., 2024): fit a model of the
  judge's preference, then read the leaderboard at a length difference of zero. It only works
  to the extent that the effect of length is what the regression says it is.

All checks passed. (0.6s)
```

</details>
