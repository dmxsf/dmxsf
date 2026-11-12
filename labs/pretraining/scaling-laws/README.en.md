# Scaling laws: the formulas, the serving cost, and a tiny study of our own

Course pages: [Scaling Laws: How to Split Compute Between Parameters and Data](https://dmxsf.com/en/wiki/pretraining/scaling-laws/)  
[中文](README.md)

The page's power laws and compute-optimal split recomputed, what serving cost does to the best model, a real tiny scaling study (tune the learning rate per size, train a grid, fit L(N, D), predict an unseen run), and what a 12-point fit can and cannot identify.

## Run

```bash
uv run python labs/pretraining/scaling-laws/run.py            # full run, about 2 minutes
uv run python labs/pretraining/scaling-laws/run.py --quick    # a shrunk smoke test, a few seconds
```

You can also write `uv run dmxsf run scaling-laws` (and add `--quick`). `--seed N` changes the random seed, default 0. The script prints its results section by section and checks the course page's claims one by one with `[ok]` / `[FAIL]`: the exit status is 0 when all pass and 1 when any fails; checks that only make sense in the full run are marked `[skip]` under `--quick`.

## Code

| File | What it holds |
| --- | --- |
| `labs/pretraining/scaling-laws/run.py` | the lab script, numbered sections |
| `src/dmxsf/pretraining/scaling.py` | Chinchilla's parametric loss, the compute-optimal split, and the cost of serving |
| `src/dmxsf/pretraining/accounting.py` | The arithmetic of a training run: FLOPs, MFU, GPU hours, and the bytes of model state and activations |

Unit tests: `uv run pytest tests/test_pretraining_scaling.py tests/test_pretraining_accounting.py`

## What to look for

- Section 1: Chinchilla's coefficients recompute the page's table (1e21 FLOPs -> 2.8B parameters on 0.06T tokens; 1e25 -> 312.1B on 5.34T), and the closed form is the brute-force minimum; the valley is flat: half or double the optimal size costs under 0.02 (worst 0.0100); Llama 3 8B sits 0.05 above the optimum at its budget.
- Section 2: count the tokens that will be served and the best model shrinks from 46.9B (nothing served) to 9.6B (1e15 tokens served) while the training tokens grow from 0.88T to 50.6T and tokens per parameter from 19 to 5264: that is the arithmetic behind small, heavily over-trained models.
- Section 3 is our own small study: 4 widths x 3 token budgets x 2 seeds, learning rate tuned in every cell. With the fewest tokens the biggest model is not the best (that is N = 176,128 and the biggest trails it by 0.074; from the smallest to the biggest size the loss falls by only 0.056, against 0.422 with the most tokens); the best learning rate falls as the model grows (0.1 -> 0.001); not tuning it costs more (+0.293 for the smallest model) than a size step buys (0.192).
- The fit of section 3: E = 1.41, alpha = 0.48, beta = 0.45 (rms 0.068); leaving the biggest run out and predicting it: 1.947 predicted, 1.847 measured, an error of 0.101 against 0.390 for "same as the previous cell", but not smaller than the seed range (0.292): a grid this small demonstrates the method, it proves nothing.
- Section 4: on synthetic data with a known truth the coefficients (A, B, beta) are far less certain than the predictions they combine into (the half-width of B's 10-90% range is 232% of the truth, that of the extrapolated loss 1.8%). At the measured noise the extra loss of a mis-fitted compute-optimal size is only 0.0009 (median); with ten times the noise the coefficients are useless, the median is 0.054 and the worst tenth pays 0.251 or more, more than cutting the budget tenfold (0.109). Compare predictions, not coefficients.

## Exercises

Run again after a change: the checks that compare with the page's numbers turn into `[FAIL]`; that is expected, read how the numbers themselves move.

1. In section 2, `scaling.best_model(2.0, served)` takes the target loss: change 2.0 to 1.9. Does the best model grow or shrink, and how do the tokens-per-parameter ratios change?
2. Change the 2 in `SEEDS = 1 if args.quick else 2` to 4 (a few minutes longer): how do the median seed spread, the fitted coefficients and the error of the held-out prediction change?

## Output

The complete runs below were made on an Apple M2 Max (12 cores, CPU only) with Python 3.12 and PyTorch 2.15.0. On another machine the numbers that involve training can differ in the last digits and timings depend on the machine; the checks use thresholds and tolerances, not bit-for-bit equality.

<details>
<summary>Output of the full run (`labs/pretraining/scaling-laws/run.py`)</summary>

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
  [ok] a bigger model buys much less when data is scarce: from the smallest to the biggest size the loss falls less with the fewest tokens than with the most  (0.056 nats at 32,768 tokens against 0.422 at 524,288)
  [ok] with the fewest tokens (64 steps) the biggest model has no clear lead: it cannot use its capacity  (best at 32,768 tokens: N = 176,128; the biggest trails by 0.074)
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

All checks passed. (150.2s)
```

</details>
