# Benchmark contamination: an n-gram detector, and a leak you can measure

Course pages: [LLM Evaluation Methodology: Benchmarks, Contamination and a Workflow](https://dmxsf.com/en/wiki/evaluation/llm-evaluation/)  
[中文](README.md)

The page's n-gram contamination detector on its own example; a benchmark whose answers can only be memorised; a model trained with a third of it leaked verbatim, a third paraphrased and a third absent; what the detector flags and what the clean-subset check concludes.

## Run

```bash
uv run python labs/evaluation/llm-evaluation/run.py            # full run, about 2 minutes
uv run python labs/evaluation/llm-evaluation/run.py --quick    # a shrunk smoke test, a few seconds
```

You can also write `uv run dmxsf run llm-evaluation` (and add `--quick`). `--seed N` changes the random seed, default 0. The script prints its results section by section and checks the course page's claims one by one with `[ok]` / `[FAIL]`: the exit status is 0 when all pass and 1 when any fails; checks that only make sense in the full run are marked `[skip]` under `--quick`.

This lab uses cached small models: the byte-level base model (842,496 parameters). The first run trains them on TinyShakespeare and caches them in `~/.cache/dmxsf` (set `DMXSF_CACHE` to move it); later runs read them back.

## Code

| File | What it holds |
| --- | --- |
| `labs/evaluation/llm-evaluation/run.py` | the lab script, numbered sections |
| `src/dmxsf/evaluation/contamination.py` | Benchmark contamination: n-gram detection and a miniature leak you can measure |

Unit tests: `uv run pytest tests/test_evaluation_contamination.py`

## What to look for

- Sections 1-2: the page's 13-gram detector hits the verbatim question 9 times and its rephrased twin 0 times; the error bars of three benchmarks: HumanEval +-7.7 points, GSM8K +-2.7, MMLU +-0.8 (the worst case, at 50% accuracy).
- Sections 3-4: a benchmark whose answers can only be memorised (three random digits, a guess is right 1 time in 1,000), with 240 leaked lines in the training data and 1,000 training steps. The verbatim leak is answered 95.8% and 100.0%, the never-leaked questions 0.0%, the leaked paraphrases 10.4% and 20.8%: a leaked paraphrase does transfer to the benchmark's wording, but far more weakly.
- Section 3: the GPT-3 recipe (the 5th percentile of the question length, between 8 and 13) gives N = 13, which flags only the verbatim leak and lets every paraphrase pass; N = 8 flags everything, because the benchmark's own template ('in the north wing of the old archive') is in the leaked training lines: 100% false alarms.
- Section 5: scoring only the 'clean' subset gives less than the full benchmark (5.2% against 35.4%), so the check does notice the contamination, but it is still 52 to 104 times above guessing: the paraphrased leak is in the subset that was labelled clean. On a real benchmark you do not know the guessing rate.
- Section 6: the same verbatim-leaked questions in another format: change the marker before the answer from `A:` to `Answer:` and the score falls from 95.8% to 25.0%; drop the space after `A:` and it falls to 0.0%; put `Question:` in front and almost nothing changes. Format sensitivity is an indirect sign and a probe has to touch the right place to show it.

## Exercises

Run again after a change: the checks that compare with the page's numbers turn into `[FAIL]`; that is expected, read how the numbers themselves move.

1. Add 10 to `for n in (13, 8)` in section 3: how does it trade catching the paraphrased leak against false alarms?
2. Change the 1000 in `steps = 60 if QUICK else 1000` to 300: is the verbatim leak still memorised, and does the transfer of the paraphrased leak survive?

## Notes

- The benchmark is made up and the model is a byte-level one of 842,496 parameters: the size of the leak and whether the model memorises it are known truths that a real benchmark does not give you. The full run trains two models of about 1.5 minutes each.

## Output

The complete runs below were made on an Apple M2 Max (12 cores, CPU only) with Python 3.12 and PyTorch 2.15.0. On another machine the numbers that involve training can differ in the last digits and timings depend on the machine; the checks use thresholds and tolerances, not bit-for-bit equality.

<details>
<summary>Output of the full run (`labs/evaluation/llm-evaluation/run.py`)</summary>

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
