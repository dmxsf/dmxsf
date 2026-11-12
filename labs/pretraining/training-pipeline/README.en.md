# The pretraining pipeline on a laptop: monitoring, data mix, resuming, leakage

Course pages: [The LLM Training Pipeline: From Data to Deployment](https://dmxsf.com/en/wiki/pretraining/training-pipeline/)  
[中文](README.md)

Two free checks before spending compute, a baseline run and how to read its log, then three experiments on a laptop: the data mix, four ways to resume from a checkpoint (only one is exact), and validation text leaking into the training data.

## Run

```bash
uv run python labs/pretraining/training-pipeline/run.py            # full run, about 3 minutes
uv run python labs/pretraining/training-pipeline/run.py --quick    # a shrunk smoke test, about 20 seconds
```

You can also write `uv run dmxsf run training-pipeline` (and add `--quick`). `--seed N` changes the random seed, default 0. The script prints its results section by section and checks the course page's claims one by one with `[ok]` / `[FAIL]`: the exit status is 0 when all pass and 1 when any fails; checks that only make sense in the full run are marked `[skip]` under `--quick`.

## Code

| File | What it holds |
| --- | --- |
| `labs/pretraining/training-pipeline/run.py` | the lab script, numbered sections |
| `src/dmxsf/pretraining/train.py` | A small, explicit training loop: schedule, clipping, monitoring, checkpoints |
| `src/dmxsf/data.py` | Tiny corpora and batching. Everything here works offline |

Unit tests: `uv run pytest tests/test_pretraining_train.py tests/test_data.py`

## What to look for

- Section 1: with N(0, 0.02^2) initialisation the first loss is 8.01 bits, i.e. log2 256; with N(0, 1) and tied embeddings it is 47.84 bits (the page's 0.86M model starts at 108.0). A single batch can be memorised (7.86 -> 0.02 bits), so the code, the loss and the optimizer work.
- Section 2: in the monitoring table the validation loss falls at every evaluation (3.89 -> 3.24) and clipping is a start-up event: 24% of the first 100 steps are clipped against 5.0% afterwards (an observation, not a check: with seed 2 it is the other way round, the gradient norm climbs to 1.4 in the second half and nearly every step is clipped, as with the lower peak learning rate of the page's exercise 6, and training is fine).
- Section 3: the data mix decides what the model is good at: a 90% share of prose beats a 10% share by 0.52 bits on prose, and code by 0.61 bits; the mean over both domains hardly moves (3.014 / 3.238 / 2.965), so one number would have hidden the trade.
- Section 4: with weights, optimizer state and step restored, the resumed run equals the uninterrupted one bit for bit; forget any one of them and the weights differ (by 0.22, 0.64, 0.44) while the validation score does not flag them: none is worse than the correct resume (worst +0.005 bits, a change of seed alone gives 0.131); the two runs that restart the learning rate even score better (-0.132 and -0.133): when under-trained, the score cannot tell a bug from a benefit.
- Section 5: 4.1 KB of held-out code in the training data: the leaked chunk falls from 3.582 to 2.267 bits (37% lower) while on this seed the rest of the held-out code hardly moves (2.908 -> 2.964); the page's larger leak (60 KB into a 0.86M model) lifted sibling files of the same package too, which is not seen at this small size, and cannot be ruled out either (see the note below).

## Exercises

Run again after a change: the checks that compare with the page's numbers turn into `[FAIL]`; that is expected, read how the numbers themselves move.

1. Add `(0.99, S0)` to the loop `for share, seed in [(0.9, S0), (0.1, S0), (0.5, SEEDS[-1])]` of section 3: how much worse is prose when it is only 1% of the data than at 50%?
2. Change the leak of section 5 from 4096 bytes to 40 KB: does the rest of the held-out code start to improve?

## Notes

- Two statements about the size of an effect are observations that change with the seed and the machine, and print `[differs]` when they fail: 'clipping is a start-up event' of section 2 (with seed 2 it is the other way round) and 'the part of the held-out code that did not leak is unaffected' of section 5. The latter moved against the control by +0.06, +0.34, +0.28, +0.30 and -0.41 bits per byte over five runs (seeds 0, 1, 2 on an Apple M2 Max; Python 3.10 and 3.12 on a GitHub Linux runner): no fixed direction, while the baseline and the control, which saw no leak, differ by 0.04 to 0.32 bits between themselves in four of them. A 4 KB leak shows no spillover at this size and does not prove there is none: one training run cannot tell spillover from training noise.
- This lab uses the source of the running interpreter's standard library as its code corpus, so the numbers that involve it differ a little between Python versions.

## Output

The complete runs below were made on an Apple M2 Max (12 cores, CPU only) with Python 3.12 and PyTorch 2.15.0. On another machine the numbers that involve training can differ in the last digits and timings depend on the machine; the checks use thresholds and tolerances, not bit-for-bit equality.

<details>
<summary>Output of the full run (`labs/pretraining/training-pipeline/run.py`)</summary>

```text
# The pretraining pipeline on a laptop: monitoring, data mix, resuming, leakage
  course: https://dmxsf.com/wiki/pretraining/training-pipeline/
          https://dmxsf.com/en/wiki/pretraining/training-pipeline/

## 1. Two free checks before you train
  the page's model: 858,880 parameters; one batch of 16 x 128 bytes, 60 steps of AdamW
  initialisation                   loss at step 0 (bits)  first step below 8 bits  clipped steps
  -------------------------------  ---------------------  -----------------------  -------------
        N(0, 0.02^2), GPT-2 style                   8.01                        1       16 of 50
  N(0, 1), nn.Embedding's default                  47.84                       30       50 of 50
  [ok] with a sensible initialisation the first loss is log2(256) = 8 bits  (8.01)
  [ok] with N(0, 1) and tied embeddings the model starts far worse than uniform guessing  (47.8 bits; the page's 0.86M model starts at 108.0)

  one batch, 300 steps, constant learning rate: loss 7.86 -> 0.02 -> 0.02 -> 0.02 bits (steps 1, 100, 200, 300)
  [ok] the model can memorise a single batch: the code, the loss and the optimizer all work  (7.86 -> 0.02 bits)

## 2. A baseline run: 1200 steps, 50% prose + 50% code, and how to read its log
  124,672-parameter byte-level GPT, 2,048 bytes per step (2.46 MB in all); prose 1.06 MB (TinyShakespeare, last 5% held out), code 10.81 MB (Python standard library, held-out files)
  step  lr       train loss  val prose  val code  grad norm (median)  clipped  
  ----  -------  ----------  ---------  --------  ------------------  ---------
   200  5.7e-03        3.93       3.91      3.86                0.50  25 of 200
   400  4.8e-03        3.65       3.73      3.64                0.42   0 of 200
   600  3.4e-03        3.53       3.63      3.48                0.43   3 of 200
   800  2.0e-03        3.40       3.50      3.38                0.44   1 of 200
  1000  9.8e-04        3.31       3.37      3.26                0.57   6 of 200
  1200  6.0e-04        3.20       3.28      3.19                0.77  44 of 200
  largest gradient norm before clipping: 5.9 at step 21; clipped in 24% of the first 100 steps against 5.0% of the rest
  [ok] the loss starts at log2(256) = 8 bits  (7.93)
  [ok] validation loss falls at every evaluation (prose + code)  (3.89 > 3.69 > 3.55 > 3.44 > 3.32 > 3.24)
  [ok] no overfitting yet: the validation mean is close to the training loss  (validation mean 3.24, training 3.20 bits)
  [ok] clipping is a start-up event: the largest gradient norm comes in the first 100 steps and clipped steps are more frequent early than later  (peak at step 21; 24% of the first 100 steps against 5.0% of the rest)

## 3. Experiment 1: the data mix decides what the model is good at
  prose : code  seed  prose bpb  code bpb  mean 
  ------------  ----  ---------  --------  -----
     90% : 10%     0      2.800     3.228  3.014
     50% : 50%     0      3.285     3.190  3.238
     10% : 90%     0      3.316     2.614  2.965
     50% : 50%     1      3.152     3.061  3.106
  [ok] a 90% share beats a 10% share for that domain  (gain 0.52 bits on prose, 0.61 bits on code)
  [ok] ...by more than the seed range of the 50:50 runs  (seed range 0.13 bits on prose, 0.13 bits on code)
  the mean over both domains hardly moves (3.014 / 3.238 / 2.965 for 90:10 / 50:50 / 10:90, seed range 0.131): one number would have hidden the trade

## 4. Experiment 2: resuming from step 600
  resume                                 prose bpb  code bpb  mean   max |w - baseline|  loss, first 20 steps
  -------------------------------------  ---------  --------  -----  ------------------  --------------------
                 uninterrupted baseline      3.285     3.190  3.238                   0                 3.524
             weights + optimizer + step      3.285     3.190  3.238            0.00e+00                 3.524
                           weights only      3.292     3.193  3.243            2.21e-01                 3.564
  forgot the step (data and lr restart)      3.193     3.017  3.105            6.38e-01                 3.562
                data right, lr restarts      3.178     3.030  3.104            4.42e-01                 3.514
  [ok] with weights, optimizer state and step restored, the resumed run equals the uninterrupted one bit for bit  (largest weight difference 0.0)
  [ok] each of the other three ends with different weights  (0.22, 0.64, 0.44)
  [ok] ...and the validation score does not flag them: none is worse than the correct resume by more than a change of seed  (worst +0.005 bits for the three, 0.131 between seeds 0 and 1)
  the runs that restart the learning rate even score *better* than the correct resume (forgot the step (data and lr restart): -0.132; data right, lr restarts: -0.133): at this step the model is still under-trained, so a higher learning rate helps.
  The score cannot tell a bug from a benefit.

## 5. Experiment 3: validation text in the training data
  leaked: the first 4.1 KB of the held-out code; the remaining 567 KB never enter training. 20% of the training windows come from the extra data.
  model                              leaked chunk (bpb)  rest of held-out code (bpb)
  ---------------------------------  ------------------  ---------------------------
            baseline: no extra data               3.828                        3.145
     control: 4 KB of training code               3.582                        2.908
  leaked: the 4 KB of held-out code               2.267                        2.964
  [ok] the leaked text scores far better than the same model without the leak  (3.582 -> 2.267 bits: 37% lower, and the model did not become 30% better at code)
  [ok] the rest of the held-out code stays where the control had it (no spillover at this size)  (2.908 -> 2.964; the page's larger leak (60 KB into 0.86M parameters) lifted sibling files in the same package)

All checks passed. (209.0s)
```

</details>
