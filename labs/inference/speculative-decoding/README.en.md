# Speculative decoding: the rule, the formulas, a real draft and target

Course pages: [Speculative Decoding: Draft with a Small Model, Verify with a Large One](https://dmxsf.com/en/wiki/inference/speculative-decoding/)  
[中文](README.md)

The accept/reject rule that keeps the target's exact distribution, checked on the page's toy distribution, by exact algebra and on real models; the formulas of Leviathan et al.; a draft and target byte-level pair run with KV caches; and when it does not pay.

## Run

```bash
uv run python labs/inference/speculative-decoding/run.py            # full run, about 35 seconds
uv run python labs/inference/speculative-decoding/run.py --quick    # a shrunk smoke test, a few seconds
```

You can also write `uv run dmxsf run speculative-decoding` (and add `--quick`). `--seed N` changes the random seed, default 0. The script prints its results section by section and checks the course page's claims one by one with `[ok]` / `[FAIL]`: the exit status is 0 when all pass and 1 when any fails; checks that only make sense in the full run are marked `[skip]` under `--quick`.

This lab uses cached small models: the byte-level base model (842,496 parameters), the small draft model. The first run trains them on TinyShakespeare and caches them in `~/.cache/dmxsf` (set `DMXSF_CACHE` to move it); later runs read them back.

## Code

| File | What it holds |
| --- | --- |
| `labs/inference/speculative-decoding/run.py` | the lab script, numbered sections |
| `src/dmxsf/inference/speculative.py` | Speculative decoding: a small model drafts, the big one verifies several tokens in one pass |

Unit tests: `uv run pytest tests/test_inference_speculative.py`

## What to look for

- Section 1: the page's A100 ledger (exact): reading the 14 GB of weights once takes 6.87 ms, computing a token 0.045 ms and five positions 0.22 ms, and 153 FLOP per byte are needed to saturate the chip: checking extra positions is nearly free until the batch nears 150.
- Section 2: on the four-token distribution, resampling a rejected position from the residual gives back the target distribution p exactly, resampling from p itself does not (token 0 would come out at 0.375 instead of 0.50); the acceptance rate is sum(min(p, q)) = 0.75, the simulated output frequencies are [0.5, 0.25, 0.149, 0.1] and a round yields 3.054 tokens against the formula's 3.051.
- Section 3: the formulas of Leviathan et al. (theorem 3.8) and the widget's numbers: at alpha = 0.8 and c = 0.05, gamma 4 gives 2.80x, the best gamma is 8 (3.09x) and gamma 16 falls to 2.72x; a draft six times as dear (c = 0.3) shrinks the best gamma to 3 and the speedup to 1.55x; alpha 0.3 with gamma 4 is 0.65x, slower than not using it.
- Section 4: a real draft (124,672 parameters) and target (842,496): one draft step costs 0.41 of a target step (c = 0.41; a small model on a CPU is limited by per-layer overhead, not by memory bandwidth as on an A100); greedy speculative decoding writes the same bytes as the target alone (all 120 runs); every gamma needs fewer target passes per byte than plain decoding (0.33 at gamma 4) and the formula fed the measured alpha predicts the bytes per round within 1.5%.
- The clock column of section 4: in this full run no gamma beats plain decoding (0.68x to 0.88x, the best being gamma 4 at 0.88x), while the formula, which already charges the draft's cost (c = 0.41), predicts 1.29x for gamma 2 from the measured alpha; what is left is what the formula does not count: the loop's own work (sampling, cache cropping, bookkeeping). The first sampled byte follows the target's own distribution (chi-square p = 0.665).
- Section 5: when it does not pay: a low acceptance rate and a dear draft leave nothing to gain (at alpha 0.3 and c 0.5 even gamma = 1 loses, 0.87x); the third condition, a large batch, is the ledger of section 1.

## Exercises

Run again after a change: the checks that compare with the page's numbers turn into `[FAIL]`; that is expected, read how the numbers themselves move.

1. Add 3 to `gammas = (2, 4) if QUICK else (1, 2, 4, 6, 8)` in section 4: which gamma is fastest on the clock, and does it agree with the formula's choice?
2. Add 0.95 to `for a in (0.6, 0.8, 0.9)` in section 3: what are the best gamma and the speedup?

## Notes

- The clock column ('wall-clock speedup') swings from run to run with the load and the temperature of the machine: on this same machine one full run had gamma 2 at 1.11x (plain decoding took 0.92 ms per byte) and the one attached here has 0.83x (0.63 ms per byte), every gamma slower than plain decoding. Read the 'target passes per byte' column and the formula, not the clock.

## Output

The complete runs below were made on an Apple M2 Max (12 cores, CPU only) with Python 3.12 and PyTorch 2.15.0. On another machine the numbers that involve training can differ in the last digits and timings depend on the machine; the checks use thresholds and tolerances, not bit-for-bit equality.

<details>
<summary>Output of the full run (`labs/inference/speculative-decoding/run.py`)</summary>

```text
# Speculative decoding: the rule, the formulas, a real draft and target
  course: https://dmxsf.com/wiki/inference/speculative-decoding/
          https://dmxsf.com/en/wiki/inference/speculative-decoding/

## 1. Why the extra positions are almost free (the page's A100 ledger, exact)
  quantity                            value                           how                      
  ----------------------------------  ------------------------------  -------------------------
                        BF16 weights                           14 GB   7e9 parameters x 2 bytes
              time to read them once  6.87 ms (146 tokens/s at most)         14 GB / 2,039 GB/s
               compute for one token                        0.045 ms  2 x 7e9 FLOP / 312 TFLOPS
             compute for 5 positions                         0.22 ms               5 x 0.045 ms
  FLOP per byte to saturate the chip                             153           312e12 / 2,039e9
  [ok] 6.87 ms to read the weights (146 tokens/s), 0.045 ms to compute a token, 0.22 ms for five, 153 FLOP/byte
  reading the weights costs 153x one token's arithmetic, so checking gamma + 1 positions in the same
  pass adds little. It stops being free around a batch of 150, where the arithmetic catches up.

## 2. The rule on the page's four-token distribution
  exact, no sampling: what one position yields under each repair rule
  token  p (target)  q (draft)  min(p, q)  residual  output, residual  output, refill from p
  -----  ----------  ---------  ---------  --------  ----------------  ---------------------
      0        0.50       0.25       0.25     1.000            0.5000                 0.3750
      1        0.25       0.25       0.25     0.000            0.2500                 0.3125
      2        0.15       0.25       0.15     0.000            0.1500                 0.1875
      3        0.10       0.25       0.10     0.000            0.1000                 0.1250
  [ok] refilling a rejected position from the residual gives back p exactly; refilling from p itself does not  (token 0 would come out at 0.375 instead of 0.50, tokens 1-3 at 0.312, 0.188, 0.125)
  the draft already covers the target's mass on tokens where q >= p; refilling from p counts that mass a second
  time, and token 0, which the draft under-proposes, ends up short.
  [ok] the acceptance rate is sum(min(p, q)) = 0.75 and the residual is (1, 0, 0, 0)

  the page's simulation: output frequencies [0.5   0.25  0.149 0.1  ] against p [0.5  0.25 0.15 0.1 ]; acceptance 0.750 (theory 0.75);
  3.054 tokens per round against the formula's 3.051.
  [ok] the page's printed numbers: [0.5, 0.25, 0.149, 0.1], 0.750, 3.054 and 3.051  ([0.5, 0.25, 0.149, 0.1], 0.750, 3.054, 3.051)

## 3. The formulas (Leviathan et al., theorem 3.8) and the widget's numbers
  speedup over plain decoding, c = 0.05 (a draft step costs 5% of a target step):
               gamma 1  gamma 2  gamma 4  gamma 8  gamma 16  best gamma: speedup
  -----------  -------  -------  -------  -------  --------  -------------------
  alpha = 0.6     1.52     1.78     1.92     1.77      1.39              4: 1.92
  alpha = 0.8     1.71     2.22     2.80     3.09      2.72              8: 3.09
  alpha = 0.9     1.81     2.46     3.41     4.38      4.63             13: 4.67
  [ok] alpha 0.9, gamma 10, free draft: 6.86x; alpha 0.8, c 0.05: gamma 4 gives 2.80x, the best gamma is 8 (3.09x), gamma 16 falls to 2.72x
  [ok] with a free draft alpha 0.8 caps at 1/(1 - 0.8) = 5x: gamma 4 reaches 3.36x, gamma 32 about 5.0x  (gamma 32: 4.997)
  [ok] alpha 0.6 at gamma 4: 1.92x, and the target does 1.49x (alpha 0.8) or 2.17x (alpha 0.6) positions per kept token
  [ok] a draft six times as dear (c = 0.3) shrinks the best gamma to 3 and the speedup to 1.55x; alpha 0.3 with gamma 4 is 0.65x, slower than not using it
  [ok] with gamma = 1 the speedup is (1 + alpha) / (1 + c), above 1 exactly when alpha > c
  [ok] simulated rounds with independent acceptance average the formula's number of tokens  (largest gap 0.004 tokens over three (alpha, gamma) pairs)
  [ok] DeepSeek-V3's 1.8x with one extra token (gamma = 1) at 85-90% acceptance means c between 0.03 and 0.06  (0.028 to 0.056)

## 4. A real draft and target
  target: 842,496 parameters, 4 layers, d = 128, 2.308 bits/byte on held-out text.
  draft:  124,672 parameters, 2 layers, d = 64,  2.494 bits/byte.

  time of one forward pass on the CPU with 64 bytes already cached, by number of new positions:
  new positions      1      2      3      5      9    
  -----------------  -----  -----  -----  -----  -----
         target, ms   0.95   1.13   1.07   1.04   1.12
    vs one position  1.00x  1.18x  1.12x  1.09x  1.17x
          draft, ms   0.40   0.38   0.41   0.44   0.45
  one draft step costs 0.41 of a target step here (c = 0.41); a 0.86M model on a CPU is limited
  by per-layer overhead, not by memory bandwidth, so this is a different regime from the A100 ledger.
  [ok] five positions in one target pass cost well under five single steps  (1.09x one step)

  on 128 held-out windows (every position, teacher-forced), the chance that a drafted byte is kept,
  beta = sum(min(p, q)), by sampling temperature:
  temperature  beta 
  -----------  -----
          1.0  0.778
          0.7  0.770
          0.4  0.755
  the two models pick the same most likely byte at 73.7% of positions (greedy decoding).
  [ok] greedy speculative decoding writes the same 64 bytes as the target alone, for every gamma and prompt  (120 of 120 runs)

  sampling at temperature 1 from 24 prompts, 64 new bytes each. Plain decoding: 0.63 ms per byte.
  gamma  alpha measured  drafts kept  bytes per round  formula at that alpha  target passes per byte  fewer target passes  formula speedup (c = 0.41)  wall-clock speedup
  -----  --------------  -----------  ---------------  ---------------------  ----------------------  -------------------  --------------------------  ------------------
      1           0.771         0.77             1.77                   1.77                   0.583                1.71x                       1.25x               0.80x
      2           0.766         0.68             2.35                   2.35                   0.446                2.24x                       1.29x               0.83x
      4           0.779         0.55             3.22                   3.23                   0.333                3.00x                       1.21x               0.88x
      6           0.797         0.48             3.86                   3.92                   0.282                3.55x                       1.12x               0.79x
      8           0.791         0.39             4.14                   4.20                   0.270                3.71x                       0.97x               0.68x
  [ok] every gamma needs fewer target passes per byte than plain decoding (which needs one)  (gamma 1: 0.58, gamma 2: 0.45, gamma 4: 0.33, gamma 6: 0.28, gamma 8: 0.27)
  fastest on the clock: gamma 4 (0.88x); the formula, fed the measured alpha and c, picks gamma 2 (1.29x).
  no setting beats plain decoding on this machine. With c = 0.41 the formula's best possible speedup at this alpha is 1.31x (gamma 2), before the loop's own work
  (sampling, cache cropping, bookkeeping), which the formula leaves out and this lab did not time separately.
  bytes per round against the formula at the measured alpha: -1.5% to -0.0% (the formula assumes every
  position accepts with the same probability; a gap means contexts differ in how easy they are to guess).
  [ok] the formula, fed the measured alpha, predicts the bytes per round within 10%  (gamma 2: 2.35 against 2.35, gamma 4: 3.22 against 3.23, gamma 6: 3.86 against 3.92, gamma 8: 4.14 against 4.20)
  [ok] the acceptance rate of the real loop at gamma 1 agrees with the offline beta at temperature 1  (alpha 0.771, beta 0.778)

  the first sampled byte over 2,000 runs (gamma 4), against the target's own next-byte distribution:
    byte  32 (   ' '): target 0.737, speculative 0.721
    byte  46 (   '.'): target 0.108, speculative 0.114
    byte  44 (   ','): target 0.050, speculative 0.050
  chi-square 8.5 on 11 degrees of freedom, p = 0.665 (cells expected under 5 pooled).
  [ok] the first byte's frequencies match the target's distribution (p above 0.001)

## 5. When it does not pay
  best achievable speedup by acceptance rate (rows) and draft cost (columns):
               c = 0.02          c = 0.1           c = 0.3          c = 0.6        
  -----------  ----------------  ----------------  ---------------  ---------------
  alpha = 0.3   1.34x (gamma 3)   1.18x (gamma 1)          no gain          no gain
  alpha = 0.5   1.79x (gamma 4)   1.46x (gamma 2)  1.15x (gamma 1)          no gain
  alpha = 0.7   2.76x (gamma 8)   1.98x (gamma 4)  1.37x (gamma 2)  1.06x (gamma 1)
  alpha = 0.9  6.37x (gamma 19)  3.43x (gamma 10)  1.87x (gamma 5)  1.23x (gamma 2)
  [ok] a low acceptance rate and a dear draft leave nothing to gain: with alpha 0.3 and c 0.5 even gamma = 1 loses  (best: gamma 1, 0.87x)
  and the page's third condition, a large batch, is the ledger of section 1: once the batch is near 150
  tokens per weight read the chip is busy, and checking gamma extra positions per sequence costs real time.

All checks passed. (35.5s)
```

</details>
