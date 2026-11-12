# SFT 数据：损失看到什么

课程页面：[SFT 数据构造与质量控制](https://dmxsf.com/wiki/post-training/sft-data/)  
[English](README.en.md)

SFT 损失到底在平均什么（-100 掩码、打包与填充）；用埋好答案的数据做健康检查；四个对照实验：同配置重复运行的波动、每类样本数、错误标签、提示部分算不算损失与回放通用文本。

## 运行

```bash
uv run python labs/post-training/sft-data/run.py            # 完整运行，约 3 分钟
uv run python labs/post-training/sft-data/run.py --quick    # 缩小规模的冒烟测试，约 15 秒
```

也可以写成 `uv run dmxsf run sft-data`（同样可以加 `--quick`）。`--seed N` 换随机种子，默认 0。脚本分节打印结果，并用 `[ok]` / `[FAIL]` 逐条检查课程页面里的结论：全部通过时退出码为 0，有一条失败就是 1；只有完整运行才做的检查在 `--quick` 下标为 `[skip]`。

这个实验用到缓存的小模型：字节级基础模型（84 万参数）。第一次运行会先在 TinyShakespeare 上训练并缓存到 `~/.cache/dmxsf`（环境变量 `DMXSF_CACHE` 可以改位置），之后直接读取。

## 代码

| 位置 | 内容 |
| --- | --- |
| `labs/post-training/sft-data/run.py` | 实验脚本，按编号分节 |
| `src/dmxsf/post_training/sft.py` | 监督微调的机制：损失掩码、批次、打包与 SFT 循环 |
| `src/dmxsf/post_training/data_quality.py` | 检查指令数据集：规则、近重复（MinHash + LSH）与 n-gram 污染 |
| `src/dmxsf/post_training/intents.py` | 合成的客户消息任务：后训练各实验里 banking77 的替身 |

单元测试：`uv run pytest tests/test_post_training_sft.py tests/test_post_training_data_quality.py tests/test_post_training_intents.py`

## 看什么

- 第 1–2 节：一条样本的 ids、标签与 -100 掩码（提示 95 个 token、答案 12 个，监督的只占 11%）；结束符（换行）也要监督，它是模型知道何时停的唯一信号；按 token 平均（4.5944）与按序列平均（4.7516）是不同的数。
- 第 1 节末尾：同一个模型，换一种提示措辞，准确率从 99.0% 掉到 85.0%（合法标签 91.3%）；多一个空格则没有影响：模板属于数据，要和数据一起记版本。
- 第 3 节：打包必须带文档掩码并重置位置（与分开跑的最大 logit 差 1.3e-05；直接拼接是 9.6，文档互相看见）；随机批次浪费 75.8%–83.5% 的位置，按长度分组 6.2%–6.8%，打包 0.1%。
- 第 4 节：在植入已知缺陷的 540 条数据里检查：规则只给出候选（“输出 ≤ 2 个词”把 15 条正确的一字答案也算进去）；MinHash + LSH 找到暴力法发现的 83 对里的 83 对，没有凭空多出；污染检测 13-gram 抓得到逐字拷贝却抓不到改过的（0/20）和只留一段的（1/20），n=8 都能抓到，代价是误报率。
- 第 5–7 节：什么都不改，只换训练种子或数据子集，准确率也从 90.3% 到 99.0%（极差 8.7 个点，测试 300 条所以一条是 0.33%），小于这个极差的差别都不是发现；每类意图的样本数 1 → 2 → 5 → 10 → 20 → 50 对应 18% → 18% → 40% → 77% → 99% → 100%；错标 10% / 20% / 40% 使准确率降到 84.0% / 53.0% / 42.0%。
- 第 8 节：损失放在整个序列上而不是只放在答案上，准确率从 99.0% 掉到 24.3%（答案只占约八分之一）；每隔一步混入普通文本（回放）保住了通用能力（2.32 比特/字节，基础模型是 2.31），目标准确率仍在波动范围内（97.3%）。

## 练习

改了参数再运行：核对页面数字的检查会变成 `[FAIL]`，这是预期的，看数字本身怎么变。

1. 把 `STEPS = 40 if QUICK else 300` 里的 300 改成 600：每类只有 1–2 条样本的两行，准确率会上升吗，还是在背样本？
2. `rates = (0.2,) if QUICK else (0.1, 0.2, 0.4)` 里加一个 0.05：最少错标多少，才使准确率的下降超过第 5 节量出的 8.7 个点的极差？
3. 第 4 节污染检测的 `for n in (5, 8, 13)` 加上 10：它在“只留一段 10 个词”与误报率之间怎么取舍？

## 注意

- 任务是合成的意图分类，在一个 84 万参数的字节级模型上微调；规模很小，换种子的波动就有 8.7 个点，所以实验里先量噪声底，再谈差别。页面上的数字来自 0.5B 的真实模型，这里检验的是方法与趋势。

## 输出

下面是在 Apple M2 Max（12 核，仅用 CPU）、Python 3.12、PyTorch 2.15.0 上的完整运行结果。换一台机器，涉及训练的数字可能在末位上不同，计时类的数字取决于机器；检查用的是阈值与容差，不要求逐位相同。

<details>
<summary>完整运行的输出（`labs/post-training/sft-data/run.py`）</summary>

```text
# SFT data: the loss mask, a data health check, and what data size, noise, masking and replay do
  course: https://dmxsf.com/wiki/post-training/sft-data/
          https://dmxsf.com/en/wiki/post-training/sft-data/

## 1. One sample: ids, labels and the -100 mask
  message 'hi, my credit card is capped today' -> card_limit
  95 prompt tokens, 12 answer tokens (supervised share 11%)
  position  token  label
  --------  -----  -----
        92    'n'   -100
        93    't'   -100
        94    ':'   -100
        95    ' '     32
        96    'c'     99
        97    'a'     97
        98    'r'    114
  [ok] every prompt position is -100 and every answer position carries its token
  [ok] the stop token (newline) is supervised: it is the only signal for where to stop

  the model fine-tuned on 20 messages per intent, asked in other words:
  prompt at inference time           accuracy  valid label
  ---------------------------------  --------  -----------
                 the trained prompt     99.0%       100.0%
  one extra space before the answer     99.0%       100.0%
      another wording of the prompt     85.0%        91.3%
  (one model, one run: the pattern is the point, not the exact gap. The template is part of
   the data and should be versioned with it.)

## 2. What the loss averages over
  a batch of 8 examples, answers of 10 to 18 bytes; the base model's loss on it:
  averaging                          loss (nats)
  ---------------------------------  -----------
     per supervised token (default)       4.5944
  per sequence, then over sequences       4.7516
  prompt not masked (all positions)       4.1014
  [ok] the default loss is the sum over supervised positions divided by their count  (4.594379 vs 4.594379)
  [ok] unequal answer lengths make the two averages differ (equal lengths would not)  (4.5944 vs 4.7516)

## 3. Packing and padding
  four speeches (32, 18, 32, 24 bytes) packed into one row of 106:
  packing                                          max |logit difference| vs four separate passes
  -----------------------------------------------  ----------------------------------------------
          block-diagonal mask + positions restart                                         1.3e-05
  plain concatenation (no mask, positions run on)                                         9.6e+00
  [ok] packing with a document mask and restarted positions equals separate passes  (1.3e-05)
  [ok] plain concatenation changes the numbers: documents see each other  (9.6e+00)

  lengths of the 7,222 speeches in TinyShakespeare (cut at 1,024 bytes): mean 148, median 83, longest 1024
  padding share of all positions  random batches  grouped by length
  ------------------------------  --------------  -----------------
                        batch 16           75.8%               6.2%
                        batch 64           83.5%               6.8%
     packed into 1,024-byte rows            0.1%                  -
  [ok] random batches waste far more positions than length-grouped ones  (76% vs 6% at batch 16)
  [ok] packing (with the mask above) wastes less than either  (0.1%)

## 4. Health check: rules, near-duplicates, contamination (planted ground truth)
  540 records made of Shakespeare passages; planted defects:
  rule                 flagged  planted found  planted  flagged, not planted
  -------------------  -------  -------------  -------  --------------------
         empty output       10             10       10                     0
    output <= 2 words       25             25       25                     0
     placeholder text       25             25       25                     0
  input is only a URL       10             10       10                     0
  refusal-like phrase       20             15       15                     5
  [ok] each rule finds every defect planted for it
  'output <= 2 words' counts the 15 correct one-word answers as well: rules only produce candidates, and
  a person has to read them. The refusal rule also flags 5 passages where a character simply says 'I cannot'.

  pairs with Jaccard >= 0.8 by brute force over all 145,530 pairs: 83 (40 planted);
  MinHash + LSH (16 bands x 4 rows) found 83 of them and 0 pairs that are not
  [ok] LSH finds (almost) every pair brute force finds, and invents none  (83 of 83)

  the LSH S-curve: how often two texts with a given Jaccard share a band
  Jaccard  measured  1-(1-J^4)^16
  -------  --------  ------------
     0.30     0.085         0.120
     0.50     0.640         0.655
     0.60     0.905         0.891
     0.69     0.993         0.986
     0.80     1.000         1.000
  [ok] the measured candidate rate follows the S-curve formula  (400 trials per row; every row within three binomial standard deviations (+0.01))

  contamination: 60 'benchmark questions'; 20 copied verbatim into the training text, 20 with
  every 8th word replaced, 20 with only a 10-word stretch left; 'clean' questions were never copied
  n-gram  verbatim found  edited found  10-word overlap found  clean flagged
  ------  --------------  ------------  ---------------------  -------------
       5           20/20         20/20                  20/20           2.0%
       8           20/20         14/20                  20/20           0.0%
      13           20/20          0/20                   1/20           0.0%
  [ok] 13-grams find every verbatim copy and miss the edited ones  (verbatim 20/20, edited 0/20)
  [ok] shorter n-grams catch the partial overlaps that 13-grams (almost) all miss  (10-word overlap found: n=8 20/20, n=13 1/20)
  the price of a short n: clean questions flagged 2.0% at n=5, 0.0% at n=8, 0.0% at n=13.

## 5. Experiment: how much does a run vary when nothing is changed?
  run              accuracy  shakespeare bits/byte
  ---------------  --------  ---------------------
          default     99.0%                   3.73
  training seed 1     93.0%                   3.77
  training seed 2     90.3%                   3.78
    data subset 2     97.7%                   3.82
    data subset 3     94.3%                   3.83
  accuracy 94.9% on average, range 90.3%-99.0%; 300 test messages, so one message is 0.33%.
  The pretrained model scores 2.31 bits/byte on held-out Shakespeare: the 'forgetting' baseline.
  Differences below the range are not findings.

## 6. Experiment: examples per intent (300 steps each, all else equal)
  per intent  examples  accuracy  valid label  shakespeare bits/byte
  ----------  --------  --------  -----------  ---------------------
           1        12     18.0%         100%                   3.56
           2        24     18.0%         100%                   3.63
           5        60     40.3%         100%                   3.79
          10       120     77.0%         100%                   3.80
          20       240     99.0%         100%                   3.73
          50       600    100.0%         100%                   3.71
  [ok] accuracy rises with data, and each step up past 20 per intent is worth less than the first ones  (18%  ->  18%  ->  40%  ->  77%  ->  99%  ->  100%)

## 7. Experiment: wrong labels
  labels replaced  accuracy  shakespeare bits/byte
  ---------------  --------  ---------------------
               0%     99.0%                   3.73
              10%     84.0%                   3.79
              20%     53.0%                   3.81
              40%     42.0%                   3.89
  [ok] every noise level costs more than the run-to-run range, and more noise costs more  (drops 10.9 pts, 41.9 pts, 52.9 pts; range 8.7 pts)
  about 1.32 accuracy points per 1% of wrong labels at 40%; the page's 0.5B model lost 0.4

## 8. Experiment: loss on the prompt, and replaying general text
  setting                                           accuracy  shakespeare bits/byte  optimizer steps      
  ------------------------------------------------  --------  ---------------------  ---------------------
                  default: loss on the answer only     99.0%                   3.73                    300
                        loss on the whole sequence     24.3%                   4.33                    300
  answer-only + every other step plain Shakespeare     97.3%                   2.32  600 (300 of them SFT)
  (the pretrained model: 2.31 bits/byte before any fine-tuning)
  [ok] replaying pretraining-style text keeps held-out Shakespeare much closer to the base model  (2.32 vs 3.73 bits/byte (base 2.31))
  [ok] ...without costing target accuracy beyond the run-to-run range (same number of SFT steps)  (97.3% vs runs 90.3%-99.0%)
  loss on the prompt: accuracy +74.7 points for the answer-only default (run-to-run range 8.7).
  A likely reason, not isolated here: the prompt is about eight times longer than the answer,
  so it dominates the loss and the 300 steps. On the page's 0.5B model the same switch moved
  accuracy by 1.3 points (85.8% to 84.5%).

All checks passed. (266.3s)
```

</details>
