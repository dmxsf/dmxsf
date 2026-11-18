# Optimizers, schedules, mixed precision and loss spikes

Course pages: [Optimizers and Training Stability: AdamW, Warmup, Mixed Precision, Loss Spikes](https://dmxsf.com/en/wiki/foundations/optimizers-and-training-stability/)  
[中文](README.md)

SGD, Adam and AdamW written out and checked against PyTorch in float64; why Adam wins on an ill-conditioned quadratic; decoupled weight decay; learning-rate schedules; mixed precision and master weights; one corrupted batch and what clipping and beta2 do about it.

## Run

```bash
uv run python labs/foundations/optimizers-and-training-stability/run.py            # full run, a few seconds
uv run python labs/foundations/optimizers-and-training-stability/run.py --quick    # a shrunk smoke test, a few seconds
```

You can also write `uv run dmxsf run optimizers-and-training-stability` (and add `--quick`). `--seed N` changes the random seed, default 0. The script prints its results section by section and checks the course page's claims one by one with `[ok]` / `[FAIL]`: the exit status is 0 when all pass and 1 when any fails; checks that only make sense in the full run are marked `[skip]` under `--quick`.

## Code

| File | What it holds |
| --- | --- |
| `labs/foundations/optimizers-and-training-stability/run.py` | the lab script, numbered sections |
| `src/dmxsf/foundations/optim.py` | SGD, Adam and AdamW from scratch, plus learning-rate schedules and gradient clipping |

Unit tests: `uv run pytest tests/test_foundations_optim.py`

## What to look for

- Section 1: our SGD, Adam and AdamW differ from `torch.optim` by 5.6e-17 in float64.
- Section 2: on a quadratic with condition number 100, SGD needs 162 steps, momentum 103, Adam 24: Adam divides each coordinate by its own gradient scale.
- Section 3: AdamW gives two weights whose gradients differ 1000x the same trajectory, while Adam + L2 shrinks the small-gradient weight to about 0 (-0.0104 against +0.9322).
- Section 4: the course page's schedule numbers (cosine and linear average 0.548, WSD 0.885, about 1.6x cosine).
- Section 6: pure BF16 rounds the update away (it is smaller than half a BF16 step) and stalls at a loss of 7.281; BF16 compute with FP32 master weights reaches 0.01978, as good as FP32's 0.01943.
- Section 7: one batch with labels x1000: without clipping the loss is still 1000x the control 250 steps later; clipping to norm 1 keeps the peak under 1% of the unclipped one.

## Exercises

Run again after a change: the checks that compare with the page's numbers turn into `[FAIL]`; that is expected, read how the numbers themselves move.

1. Change the SGD `lr=0.019` of section 2 to 0.021 (above the stability limit 2/100): what happens?
2. Add `(None, 0.99)` to the `for clip, beta2 in [...]` list of section 7: does it forget the bad batch at a speed between 0.999 and 0.95?

## Output

The complete runs below were made on an Apple M2 Max (12 cores, CPU only) with Python 3.12 and PyTorch 2.15.0. On another machine the numbers that involve training can differ in the last digits and timings depend on the machine; the checks use thresholds and tolerances, not bit-for-bit equality.

<details>
<summary>Output of the full run (`labs/foundations/optimizers-and-training-stability/run.py`)</summary>

```text
# Optimizers, schedules, mixed precision and loss spikes
  course: https://dmxsf.com/wiki/foundations/optimizers-and-training-stability/
          https://dmxsf.com/en/wiki/foundations/optimizers-and-training-stability/

## 1. Our optimizers against PyTorch's (float64, 100 steps)
  optimizer     max |ours - torch|
  ------------  ------------------
  SGD+momentum             5.6e-17
          Adam             5.6e-17
         AdamW             5.6e-17
  [ok] all three agree with torch.optim to 1e-10

## 2. Why Adam: an ill-conditioned quadratic f = (100 x^2 + y^2) / 2
  optimizer                                        steps until f < 1e-3
  -----------------------------------------------  --------------------
  SGD, lr 0.019 (the largest stable one is 2/100)                   162
                     SGD + momentum 0.9, lr 0.019                   103
                                    Adam, lr 0.05                    24
  The curvature differs 100x between the axes: one learning rate cannot suit both for SGD,
  while Adam divides each coordinate by its own gradient scale.
  [ok] Adam gets there first

## 3. AdamW decouples weight decay; Adam + L2 does not
  AdamW      weights after 300 steps (both started at 1): +0.6921 (gradient x0.01)  +0.6921 (gradient x10)
  Adam + L2  weights after 300 steps (both started at 1): -0.0104 (gradient x0.01)  +0.9322 (gradient x10)
  [ok] AdamW gives the two weights the same trajectory (scale-invariant)
  [ok] Adam + L2 shrinks the small-gradient weight much more than the large-gradient one

## 4. Learning-rate schedules
  schedule  mean    at 25%  at 50%  at 75%  last step
  --------  ------  ------  ------  ------  ---------
    cosine  0.5475  0.9051  0.5872  0.2452     0.1000
    linear  0.5475  0.8106  0.5737  0.3369     0.1000
       wsd  0.8850   1.000   1.000   1.000     0.1000
   invsqrt  0.3722  0.4472  0.3162  0.2582     0.2236
  20,000 steps, 1,000 warmup steps, floor 10% of the peak, WSD decays over the last 20%
  [ok] course page: cosine and linear average 0.548, WSD 0.885 (about 1.6x cosine)
  [ok] course page: inverse square root is 0.447 / 0.316 / 0.258 and ends at 0.224

## 5. The course page's training loop: AdamW + warmup/cosine + clipping
  step   0  lr 3.00e-06  grad_norm 7.770  loss 16.4080
  step  99  lr 3.00e-04  grad_norm 7.666  loss 16.0073
  step 100  lr 3.00e-04  grad_norm 7.664  loss 15.9993
  step 500  lr 1.88e-04  grad_norm 6.959  loss 13.4174
  step 999  lr 3.00e-05  grad_norm 6.674  loss 12.4340
  [ok] same five lines as the course page
  [ok] our loop and torch's end with the same weights  (max difference 6.0e-08)

## 6. Mixed precision: why the master weights stay in FP32
  format  max       smallest normal  machine eps
  ------  --------  ---------------  -----------
    FP32   3.4e+38          1.2e-38      1.2e-07
    FP16  6.55e+04          6.1e-05      0.00098
    BF16  3.39e+38          1.2e-38       0.0078
  [ok] values on the course page
  a gradient of 1e-08: FP16 stores 0; with loss scaling x65536 it survives as 9.997e-09
  [ok] FP16 underflows to 0 without loss scaling
  [ok] loss scaling brings it back (to 3 digits)

  Same regression, same AdamW (lr 1e-3), three ways to hold the weights:
  loss at step 0: 16.41
  weights held in                                                        loss after training
  ---------------------------------------------------------------------  -------------------
                                                                   FP32              0.01943
  BF16 only (the update is smaller than half a BF16 step: rounded away)                7.281
                            BF16 copy for compute + FP32 master weights              0.01978
  [ok] pure BF16 stalls far above FP32  (375x the FP32 loss)
  [ok] BF16 compute with FP32 master weights matches FP32
  model state per parameter: bf16 weight 2 + bf16 grad 2 + fp32 master 4 + m 4 + v 4 = 16 bytes
  a 7B model: 112 GB (104 GiB) before activations
  [ok] 16 bytes per parameter, 112 GB for 7B

## 7. One corrupted batch: what clipping and beta2 do
  one batch at step 400 has labels x1000: its gradient norm is 7,981
  (loss on clean data; 'control' = the same run without the bad batch)
  setting                peak     +50 steps  control  step 699  control
  ---------------------  -------  ---------  -------  --------  -------
  clip off, beta2 0.999  5.3e-01    5.3e-01  4.8e-04   4.2e-01  3.7e-08
  clip 1.0, beta2 0.999  1.8e-03    1.6e-05  1.0e-12   4.3e-13  7.4e-13
   clip off, beta2 0.95  5.6e-02    5.5e-02  2.3e-06   1.9e-05  1.1e-06
   clip 1.0, beta2 0.95  3.6e-03    4.2e-05  7.6e-07   2.2e-06  1.3e-06
  [ok] without clipping the bad batch is not forgotten: 1000x the control loss 250 steps on
  [ok] clipping to norm 1 keeps the peak below 1% of the unclipped peak
  [ok] beta2 = 0.95 forgets the spike sooner than 0.999 (no clipping)

All checks passed. (2.3s)
```

</details>
