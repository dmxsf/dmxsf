# 基准污染

课程页面：[LLM 评测方法论：基准、数据污染与评测流程](https://dmxsf.com/wiki/evaluation/llm-evaluation/)  
[English](README.en.md)

课程页的 n-gram 污染检测器；一个答案只能靠记忆得到的基准；模型的训练流里三分之一原样泄漏、三分之一改写、三分之一没有；检测器能抓到什么，以及“只在干净子集上重测”会得出什么结论。

## 运行

```bash
uv run python labs/evaluation/llm-evaluation/run.py            # 完整运行，约 2 分钟
uv run python labs/evaluation/llm-evaluation/run.py --quick    # 缩小规模的冒烟测试，几秒
```

也可以写成 `uv run dmxsf run llm-evaluation`（同样可以加 `--quick`）。`--seed N` 换随机种子，默认 0。脚本分节打印结果，并用 `[ok]` / `[FAIL]` 逐条检查课程页面里的结论：全部通过时退出码为 0，有一条失败就是 1；只有完整运行才做的检查在 `--quick` 下标为 `[skip]`。

这个实验用到缓存的小模型：字节级基础模型（84 万参数）。第一次运行会先在 TinyShakespeare 上训练并缓存到 `~/.cache/dmxsf`（环境变量 `DMXSF_CACHE` 可以改位置），之后直接读取。

## 代码

| 位置 | 内容 |
| --- | --- |
| `labs/evaluation/llm-evaluation/run.py` | 实验脚本，按编号分节 |
| `src/dmxsf/evaluation/contamination.py` | 基准污染：n-gram 检测与一个可以测量的迷你泄漏 |

单元测试：`uv run pytest tests/test_evaluation_contamination.py`

## 看什么

- 第 1–2 节：页面上的 13-gram 检测器对原题命中 9 次，改写后的同义题 0 次；三个基准的误差棒：HumanEval ±7.7 个点、GSM8K ±2.7、MMLU ±0.8（准确率 50% 时的最坏情形）。
- 第 3–4 节：造一个“答案只能靠背”的基准（三位随机数字，瞎猜千分之一），把 240 行泄漏的题放进训练数据，训练 1,000 步。逐字泄漏的题答对 95.8% 和 100.0%，没泄漏的 0.0%，改写后泄漏的 10.4% 和 20.8%：改写后的泄漏传到了基准的措辞上，但弱得多。
- 第 3 节：GPT-3 的配方（题长的第 5 百分位，取 8 到 13 之间）给出 N = 13，只抓逐字泄漏，改写的全部放过；N = 8 全抓，因为基准自己的模板句（“in the north wing of the old archive”）也在泄漏的训练行里，误报率 100%。
- 第 5 节：只留“干净”的子集再测，分数低于整个基准（5.2% 对 35.4%），说明检查确实察觉到了污染，但仍比瞎猜高 52 到 104 倍：被标成干净的子集里还有改写的泄漏。真实基准上你并不知道瞎猜的水平。
- 第 6 节：同一批逐字泄漏的题换一种提问格式：把答案前的标记从 `A:` 换成 `Answer:`，分数从 95.8% 掉到 25.0%；答案后少一个空格，掉到 0.0%；前面加 `Question:` 则几乎不变。格式敏感是间接信号，探针得碰对地方才看得出。

## 练习

改了参数再运行：核对页面数字的检查会变成 `[FAIL]`，这是预期的，看数字本身怎么变。

1. 第 3 节的 `for n in (13, 8)` 里加一个 10：它在“抓住改写泄漏”与“误报率”之间怎么取舍？
2. 把 `steps = 60 if QUICK else 1000` 里的 1000 改成 300：逐字泄漏还能被背下来吗？改写泄漏的转移还在吗？

## 注意

- 这是一个自己造的基准和一个 84 万参数的字节级模型：泄漏的多少、模型会不会背，都是可控的真值，真实基准上看不到。完整运行训练两个模型，各约 1.5 分钟。

## 输出

下面是在 Apple M2 Max（12 核，仅用 CPU）、Python 3.12、PyTorch 2.15.0 上的完整运行结果。换一台机器，涉及训练的数字可能在末位上不同，计时类的数字取决于机器；检查用的是阈值与容差，不要求逐位相同。

<details>
<summary>完整运行的输出（`labs/evaluation/llm-evaluation/run.py`）</summary>

```text
# Benchmark contamination: an n-gram detector, and a leak you can measure
  course: https://dmxsf.com/wiki/evaluation/llm-evaluation/
          https://dmxsf.com/en/wiki/evaluation/llm-evaluation/

## 1. The page's detector on its own example
  verbatim   13-gram hits  9  ->  dirty
  rephrased  13-gram hits  0  ->  clean
  unrelated  13-gram hits  0  ->  clean
  [ok] 9 hits for the verbatim question; the rephrased one, same meaning, has none

## 2. How big is a benchmark's error bar?
  benchmark  questions  worst-case 95% half-width (accuracy 50%)
  ---------  ---------  ----------------------------------------
  HumanEval        164                              +-7.7 points
      GSM8K       1319                              +-2.7 points
       MMLU      14079                              +-0.8 points
  (the formula and the paired comparison are in the eval-statistics lab; keep these
   widths in mind when reading the table below, which has 48 questions per group)

## 3. A benchmark whose answers can only be memorised
  benchmark wording : What is the access code of vault N-7890 in the north wing of the old archive?
  one paraphrase    : Vault N-7890 sits in the old archive's north wing; which access code opens it?
  training line     : Vault N-7890 sits in the old archive's north wing; which access code opens it? A: 041
  the answer is three random digits, so a guess is right 1 time in 1,000.

  [ok] the GPT-3 recipe (5th percentile of the question length, between 8 and 13) gives N = 13  (17 words per question)
  Share of benchmark questions the detector flags as dirty:
  detector  never leaked  leaked verbatim  leaked as paraphrases
  --------  ------------  ---------------  ---------------------
    N = 13            0%             100%                     0%
     N = 8          100%             100%                   100%
  [ok] N = 13 flags exactly the verbatim leak: the paraphrases pass as clean
  [ok] N = 8 flags everything: the benchmark's own template ('in the north wing of the old archive') is in the leaked lines

## 4. Train with the leak, then ask the benchmark
  seed 0: training 1000 steps on 240 leaked lines ...
  seed 1: training 1000 steps on 240 leaked lines ...

  Exact match of all three digits, and the loss per digit in bits (guessing = 3.32):
  seed  group                  exact match  bits per digit
  ----  ---------------------  -----------  --------------
     0           never leaked         0.0%            7.66
     0        leaked verbatim        95.8%            0.10
     0  leaked as paraphrases        10.4%            3.28
     1           never leaked         0.0%            9.21
     1        leaked verbatim       100.0%            0.07
     1  leaked as paraphrases        20.8%            1.84
  [ok] the verbatim leak is almost fully retained (80% or more exact answers) in every run  (96%, 100%)
  [ok] questions never leaked are answered at chance level (at most 2% exact)  (0.0%, 0.0%)
  [ok] paraphrased leaks transfer to the benchmark wording: at least 5 points above never-leaked  (10.4% against 0.0%, 20.8% against 0.0%)
  [ok] the paraphrase leak is weaker than the verbatim one

## 5. The clean-subset check, applied to this benchmark
  seed  full benchmark  flagged dirty  'clean' subset  pure guessing
  ----  --------------  -------------  --------------  -------------
     0           35.4%          95.8%            5.2%           0.1%
     1           40.3%         100.0%           10.4%           0.1%
  The clean subset scores below the full benchmark, so the check does notice the
  contamination. But it is still 52 to 104 times above guessing: the paraphrased leak is in there,
  labelled clean. Here we know the guessing rate; on a real benchmark you do not.
  [ok] the 'clean' subset scores below the full benchmark: the check notices the contamination  (5.2% against 35.4%, 10.4% against 40.3%)
  [ok] ...yet it is at least 10 times above guessing (the paraphrased leak is in it)  (5.2%, 10.4%)

## 6. The pattern of the scores: where a leak is fragile
  Exact match on the verbatim-leaked questions, asked in different formats:
  prompt                     seed 0  seed 1
  -------------------------  ------  ------
             as in training   95.8%  100.0%
  'Answer:' instead of 'A:'   25.0%   29.2%
       'Question:' in front   95.8%  100.0%
        no space after 'A:'    0.0%    0.0%
  [ok] changing the marker just before the answer cuts the leaked score to less than half
  [ok] adding text in front of the question changes almost nothing (within 20%)

  A score that collapses when 'A:' becomes 'Answer:', or when a space goes missing, is the
  memory of a string rather than the ability to answer. But the same model does not mind a
  new prefix: it keys on what comes right before the answer. Format sensitivity is an
  indirect sign, and a probe has to touch the right place to show it.

All checks passed. (163.9s)
```

</details>
