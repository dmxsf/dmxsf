# Evaluation statistics: pass@k, the error of a score, paired tests, sample sizes

Course pages: [The Statistics of Evaluation: pass@k, Confidence Intervals and Sample Size](https://dmxsf.com/en/wiki/evaluation/eval-statistics/)  
[中文](README.md)

pass@k against brute-force enumeration, the 95% interval against how often it actually covers, paired comparisons and the sample-size table against the power of a real test, then two real models compared on held-out text.

## Run

```bash
uv run python labs/evaluation/eval-statistics/run.py            # full run, a few seconds
uv run python labs/evaluation/eval-statistics/run.py --quick    # a shrunk smoke test, a few seconds
```

You can also write `uv run dmxsf run eval-statistics` (and add `--quick`). `--seed N` changes the random seed, default 0. The script prints its results section by section and checks the course page's claims one by one with `[ok]` / `[FAIL]`: the exit status is 0 when all pass and 1 when any fails; checks that only make sense in the full run are marked `[skip]` under `--quick`.

This lab uses cached small models: the byte-level base model (842,496 parameters), the small draft model. The first run trains them on TinyShakespeare and caches them in `~/.cache/dmxsf` (set `DMXSF_CACHE` to move it); later runs read them back.

## Code

| File | What it holds |
| --- | --- |
| `labs/evaluation/eval-statistics/run.py` | the lab script, numbered sections |
| `src/dmxsf/evaluation/stats.py` | Evaluation statistics: pass@k, the error of a benchmark score, paired comparisons, sample sizes |

Unit tests: `uv run pytest tests/test_evaluation_stats.py`

## What to look for

- Section 1: the unbiased pass@k estimator gives 0.4200 / 0.6833 / 0.8000 on the page's example, the naive one 0.4200 / 0.6478 / 0.7246 (too low for k > 1); it equals the average over every k-subset, bit for bit; on 40 problems with known success rates repeated 4,000 times the unbiased estimate averages to the truth and the naive one is 0.0430 low at pass@10.
- Section 2: HumanEval (164 questions) at 80% accuracy has a 95% interval of +-6.1 points, GSM8K +-2.2, MMLU +-0.7; over 40,000 simulated runs the interval covers the truth 0.9493 and 0.9485 of the time.
- Section 3: two models on the same questions: the paired interval is [+0.011, +0.136] (excludes 0), treated as independent it is [-0.014, +0.161] (contains 0); repeated 3,000 times, pairing finds the real difference 57.5% of the time, not pairing 26.8%.
- Section 4: halving the gap takes about four times the questions (2 points to 1 point: x3.93); at the sizes of the page's table a real test finds the gap about 80% of the time (0.795, 0.800, 0.788).
- Section 5: two real byte-level models (draft and base) on 434 windows: correlation 0.95, paired half-width 0.0152 against 0.0566 unpaired; the windows come from one text and are not independent, so read the intervals as lower bounds on the real uncertainty.

## Exercises

Run again after a change: the checks that compare with the page's numbers turn into `[FAIL]`; that is expected, read how the numbers themselves move.

1. The correlation of the two models in section 3 comes from the question difficulty they share: change the 3.0 in `difficulty = page.normal(scale=3.0, size=n_q)` to 1.0 or 6.0: how does the correlation change, and which of the paired and the unpaired half-width is affected more?
2. Add 0.5 to `for gap in (5, 3, 2, 1)` in section 4: how many questions does it take, and how much larger is that than the benchmarks you meet in practice?

## Output

The complete runs below were made on an Apple M2 Max (12 cores, CPU only) with Python 3.12 and PyTorch 2.15.0. On another machine the numbers that involve training can differ in the last digits and timings depend on the machine; the checks use thresholds and tolerances, not bit-for-bit equality.

<details>
<summary>Output of the full run (`labs/evaluation/eval-statistics/run.py`)</summary>

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

All checks passed. (1.2s)
```

</details>
