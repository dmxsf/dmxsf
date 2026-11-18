# Attention, by hand and in a model

Course pages: [Attention Basics: How a Transformer Reads Context](https://dmxsf.com/en/wiki/architecture/attention-basics/)  
[中文](README.md)

The page's three-position example by hand, why scores are divided by sqrt(d) (the dot product has variance d), masks, heads and positions, and a small model that has to use attention to copy what followed the last occurrence.

## Run

```bash
uv run python labs/architecture/attention-basics/run.py            # full run, about 35 seconds
uv run python labs/architecture/attention-basics/run.py --quick    # a shrunk smoke test, a few seconds
```

You can also write `uv run dmxsf run attention-basics` (and add `--quick`). `--seed N` changes the random seed, default 0. The script prints its results section by section and checks the course page's claims one by one with `[ok]` / `[FAIL]`: the exit status is 0 when all pass and 1 when any fails; checks that only make sense in the full run are marked `[skip]` under `--quick`.

## Code

| File | What it holds |
| --- | --- |
| `labs/architecture/attention-basics/run.py` | the lab script, numbered sections |
| `src/dmxsf/architecture/attention.py` | Scaled dot-product attention, multi-head / grouped-query attention and the causal mask |

Unit tests: `uv run pytest tests/test_architecture_attention.py`

## What to look for

- Section 1: the page's three-position example: weights [0.40, 0.20, 0.40] and output [6.4, 4.4].
- Section 2: the variance of a dot product is about d (1,030.6 at d = 1024); dividing by sqrt(d) brings it back to about 1; without scaling the softmax saturates as d grows (largest weight -> 1), with it nothing changes.
- Section 3: the causal mask hides the future; heads do not change the parameter count (always 4 d^2); shuffling the tokens only shuffles the outputs: attention alone cannot see order.
- Section 4: a small model that must use attention to copy what followed the last occurrence sits at chance for a long time and then drops abruptly from 3.2 to 0.5; layer 0 has a previous-token head (0.67), layer 1 induction heads (0.92), and switching either off sends the loss back above 4.8.

## Exercises

Run again after a change: the checks that compare with the page's numbers turn into `[FAIL]`; that is expected, read how the numbers themselves move.

1. Change `n_layer=2` in the section 4 config to 1: can the model still learn to copy? (Hint: induction heads need two layers working together.)
2. Add 4096 to the `for d in (4, 16, 64, 256, 1024)` of section 2: how do the variance and the largest weight change?

## Notes

- How many steps the induction heads need varies with the seed and the machine: of 14 seeds tried on an Apple M2 Max, 13 formed between steps 1,100 and 1,800 and the 14th (`--seed 13`) had not after 8,000; on a GitHub Linux runner `--seed 0` had not after 3,000 steps either. So the script trains at most 3,000 steps per round and, if the repeat has not clicked, starts again from a fresh seed (plus 100), up to three times, and says so in the output (`--seed 13` prints 'training again with seed 113'); after the click it trains 300 steps more (at least 2,000 in all). The step of the abrupt drop in the table belongs to this run, it is not a universal number.

## Output

The complete runs below were made on an Apple M2 Max (12 cores, CPU only) with Python 3.12 and PyTorch 2.15.0. On another machine the numbers that involve training can differ in the last digits and timings depend on the machine; the checks use thresholds and tolerances, not bit-for-bit equality.

<details>
<summary>Output of the full run (`labs/architecture/attention-basics/run.py`)</summary>

```text
# Attention, by hand and in a model
  course: https://dmxsf.com/wiki/architecture/attention-basics/
          https://dmxsf.com/en/wiki/architecture/attention-basics/

## 1. The course page's three-position example
  scores / sqrt(2) = [0.71, 0.0, 0.71]
  weights          = [0.4, 0.2, 0.4]
  output           = [6.4, 4.4]
  [ok] weights [0.40, 0.20, 0.40] and output [6.4, 4.4], as on the page

## 2. Why divide by sqrt(d): the dot product has variance d
  d     var(q.k)  var after scaling  max weight, raw  max weight, scaled
  ----  --------  -----------------  ---------------  ------------------
     4     3.945             0.9862           0.5264              0.3422
    16     15.76             0.9851           0.7413              0.3517
    64     64.00              1.000           0.8776              0.3618
   256     260.4              1.017           0.9388              0.3637
  1024   1,030.6              1.006           0.9670              0.3624
  [ok] var(q.k) is about d
  [ok] scaling brings the variance back to about 1
  [ok] without scaling the softmax saturates (largest weight -> 1); with it nothing changes with d

## 3. Masks, heads and positions
  causal attention weights of a 5-token sequence (row = the token that is looking):
    1.00 0.00 0.00 0.00 0.00
    0.27 0.73 0.00 0.00 0.00
    0.01 0.01 0.98 0.00 0.00
    0.08 0.18 0.43 0.30 0.00
    0.01 0.02 0.07 0.03 0.88
  [ok] a token never attends to the future
  [ok] parameters: 4 d^2 whatever the number of heads (d = 64: 16,384)  (1, 4 and 8 heads: [16384])
  [ok] shuffling the tokens only shuffles the outputs: attention alone cannot see order  (max difference 2.4e-07)

## 4. A model that has to use attention: copy what came after the last time
  2 layers x 4 heads, d_model 64, 115,008 parameters; vocabulary 128, ln(vocab) = 4.852
  step  loss on the repeat  loss elsewhere
  ----  ------------------  --------------
   200               4.638           4.875
   400               4.436           4.879
   600               4.209           4.892
   800               4.102           4.888
  1000               3.652           4.883
  1200               3.244           4.894
  1400              0.5145           4.959
  1600              0.4486           4.922
  1800              0.3625           4.918
  2000              0.3294           4.918
  the repeat's loss fell by 2.38 nats between step 1200 and step 1300
  [ok] it starts at chance
  [ok] then falls abruptly, by more than 1.5 nats within 100 steps
  [ok] the repeat is predicted far better than chance  (0.33 nats)
  [ok] everything else stays at chance (it is random)

  attention paid by the tokens inside the repeat, averaged over 256 sequences:
  head            to the previous token  to the token after the earlier copy
  --------------  ---------------------  -----------------------------------
  layer 0 head 0              0.0009116                              0.01513
  layer 0 head 1                 0.6700                            0.0003745
  layer 0 head 2                0.02275                              0.03975
  layer 0 head 3              0.0002441                              0.01278
  layer 1 head 0              4.280e-06                               0.2698
  layer 1 head 1              0.0002490                               0.9216
  layer 1 head 2              1.293e-07                              0.04565
  layer 1 head 3              0.0002228                               0.1041
  [ok] layer 0 has a previous-token head (> 0.3)
  [ok] layer 1 has induction heads (> 0.7)

  loss on the repeat when one head is switched off (all heads on: 0.329):
  head switched off  loss on the repeat
  -----------------  ------------------
     layer 0 head 0              0.5097
     layer 0 head 1               5.195
     layer 0 head 2              0.3171
     layer 0 head 3              0.5210
     layer 1 head 0              0.3607
     layer 1 head 1               4.800
     layer 1 head 2              0.5825
     layer 1 head 3               1.118
  [ok] switching off the right head costs at least 1 nat  (worst 5.19)

All checks passed. (32.3s)
```

</details>
