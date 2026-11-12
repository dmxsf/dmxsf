# Decoding and sampling: temperature, top-k, top-p, min-p, beam search, penalties

Course pages: [Decoding and Sampling: Temperature, Top-k, Top-p and Beam Search](https://dmxsf.com/en/wiki/inference/decoding-and-sampling/)  
[中文](README.md)

The page's four-token example reproduced exactly (temperature, top-k, top-p, min-p, and why the order matters), whether samples follow the processed distribution, what each knob does to real text, beam search, repetition penalties, and whether batch shape changes greedy output.

## Run

```bash
uv run python labs/inference/decoding-and-sampling/run.py            # full run, about 10 seconds
uv run python labs/inference/decoding-and-sampling/run.py --quick    # a shrunk smoke test, a few seconds
```

You can also write `uv run dmxsf run decoding-and-sampling` (and add `--quick`). `--seed N` changes the random seed, default 0. The script prints its results section by section and checks the course page's claims one by one with `[ok]` / `[FAIL]`: the exit status is 0 when all pass and 1 when any fails; checks that only make sense in the full run are marked `[skip]` under `--quick`.

This lab uses cached small models: the byte-level base model (842,496 parameters). The first run trains them on TinyShakespeare and caches them in `~/.cache/dmxsf` (set `DMXSF_CACHE` to move it); later runs read them back.

## Code

| File | What it holds |
| --- | --- |
| `labs/inference/decoding-and-sampling/run.py` | the lab script, numbered sections |
| `src/dmxsf/inference/generate.py` | Autoregressive generation with a KV cache: the loop every chat product runs |

Unit tests: `uv run pytest tests/test_inference_generate.py`

## What to look for

- Section 1: every row of the page's logits [4, 3, 1, 0] (temperature, top-k, top-p, min-p) matches to three decimals; the order matters: temperature first and then top-p keeps 2 tokens, top-p first and then temperature keeps 4.
- Section 2: 200,000 draws per setting land within 4 standard errors of the processed distribution (worst z = 1.8), and the tokens that were cut are never drawn.
- Section 3: what the settings do to text, on the byte-level GPT: greedy decoding loops most (0.64 distinct 4-grams, against 0.98 for sampling at T >= 1); at T = 2 only 37.4% of the words are real (87.1% at T = 1); T = 1.5 with top-p 0.9 gets real words back (77.0%, against 61.9% without); the noise: T = 1 again with another seed gives 87.8%, about a point away.
- Section 4: beam search finds a more probable sequence than greedy (on 8 of 8 prompts), 0.82 bits/byte against 1.15 for greedy and 1.95 for the real text: the most probable text is not the most natural.
- Section 5: on a vocabulary of 256 symbols repetition penalties are far harsher than on a real 100k-token one (a strength of 2.0 already drops real words to 95.6%, greedy has 99.1%): read it as a warning about strength, not as a recipe.
- Section 6: the same 64 prompts run as one batch and one at a time give logits that differ by 0 (bit for bit) and no greedy choice differs: on this CPU the batch shape does not change the arithmetic. The page's warning (other hardware picks different kernels for different batch sizes) cannot be shown here.

## Exercises

Run again after a change: the checks that compare with the page's numbers turn into `[FAIL]`; that is expected, read how the numbers themselves move.

1. Replace `Z = torch.tensor([[4.0, 3.0, 1.0, 0.0]])` by flatter logits, for example `[2.0, 1.8, 1.5, 0.0]`: how many tokens does top-p 0.9 keep now?
2. In section 3 change `n_prompts, plen, new = (8, 48, 24) if QUICK else (64, 48, 80)` to `(64, 8, 120)` (the model's context is 128 bytes, so prompt plus continuation cannot exceed it): with a longer continuation, does greedy decoding loop more, and does the share of real words at T = 1 fall?

## Output

The complete runs below were made on an Apple M2 Max (12 cores, CPU only) with Python 3.12 and PyTorch 2.15.0. On another machine the numbers that involve training can differ in the last digits and timings depend on the machine; the checks use thresholds and tolerances, not bit-for-bit equality.

<details>
<summary>Output of the full run (`labs/inference/decoding-and-sampling/run.py`)</summary>

```text
# Decoding and sampling: temperature, top-k, top-p, min-p, beam search, penalties
  course: https://dmxsf.com/wiki/inference/decoding-and-sampling/
          https://dmxsf.com/en/wiki/inference/decoding-and-sampling/

## 1. The page's example: logits z = [4, 3, 1, 0]
  setting             token 1  token 2  token 3  token 4
  ------------------  -------  -------  -------  -------
             T = 0.5    0.879    0.119    0.002    0.000
               T = 1    0.696    0.256    0.035    0.013
               T = 2    0.509    0.309    0.114    0.069
             top-k 2    0.731    0.269    0.000    0.000
           top-p 0.9    0.731    0.269    0.000    0.000
  T = 0.5, top-p 0.9    0.881    0.119    0.000    0.000
           min-p 0.1    0.731    0.269    0.000    0.000
  [ok] every row matches the page's numbers (three decimals)

  the order matters (top-p 0.99):
  order                         token 1  token 2  token 3  token 4
  ----------------------------  -------  -------  -------  -------
   temperature 0.5, then top-p    0.881    0.119    0.000    0.000
  top-p at T = 1, then T = 0.5    0.879    0.119    0.002    0.000
  [ok] temperature first keeps 2 tokens, nucleus first keeps 4: the order changes the result

## 2. Do the samples follow the processed distribution?
  setting             p1     p2     p3     p4     freq 1  freq 2  freq 3  freq 4  max |z|
  ------------------  -----  -----  -----  -----  ------  ------  ------  ------  -------
             top-k 2  0.731  0.269  0.000  0.000   0.733   0.267   0.000   0.000      1.7
  T = 0.5, top-p 0.9  0.881  0.119  0.000  0.000   0.880   0.120   0.000   0.000      1.8
    T = 2, min-p 0.2  0.547  0.331  0.122  0.000   0.547   0.331   0.123   0.000      0.9
  [ok] 200,000 draws per setting: frequencies within 4 standard errors, dropped tokens never drawn  (worst z = 1.8)

## 3. What the settings do to text (byte-level GPT on TinyShakespeare)
  64 prompts of 48 bytes from the held-out text, 80 new bytes each (about 842 words per setting):
  setting                  words in the training text  distinct 4-grams  bits/byte under the model
  -----------------------  --------------------------  ----------------  -------------------------
           greedy (T = 0)                       99.1%              0.64                       1.15
                  T = 0.5                       98.6%              0.94                       1.34
                    T = 1                       87.1%              0.98                       1.91
                  T = 1.5                       61.9%              0.99                       2.92
                    T = 2                       37.4%              1.00                       4.47
         T = 1, top-p 0.9                       94.5%              0.98                       1.62
        T = 1, min-p 0.05                       94.3%              0.98                       1.63
       T = 1.5, top-p 0.9                       77.0%              0.99                       2.35
  (the real continuation)                       94.1%              0.98                       2.33
  distinct 4-grams: share of byte 4-grams in a continuation that are unique (low = looping).
  bits/byte: how surprised the model (at T = 1) is by the text, which includes its own output.
  [ok] greedy loops more than any sampled setting at T >= 1 (distinct 4-grams)  (0.64 against 0.98)
  [ok] heating up lowers the share of real words (T = 2 against T = 1, by more than 10 points)  (37.4% against 87.1%)
  [ok] cutting the tail at T = 1.5 (top-p 0.9) recovers real words  (77.0% against 61.9%)
  noise: T = 1 again with another seed gives 87.8% real words, 0.98 distinct, 1.96 bits/byte (first run 87.1%, 0.98, 1.91).

## 4. Beam search: the most probable text is not the most natural
  continuation          bits/byte under the model  distinct 4-grams
  --------------------  -------------------------  ----------------
             real text                       1.95                 -
        sampled, T = 1                       1.87              0.98
                greedy                       1.15              0.71
  beam search, 4 beams                       0.82              0.70
  beam search found a strictly more probable sequence than greedy on 8 of 8 prompts (0 ties).
  [ok] greedy and beam output are far more probable than the real text (open-ended text is not the mode)  (real 1.95, greedy 1.15, beam 0.82 bits/byte)
  [ok] no beam is worse than greedy (not a guarantee of the algorithm, only what these prompts show)  (beam better on 8, equal on 0 of 8)

## 5. Repetition penalties on a 256-symbol vocabulary
  greedy decoding with  words in the training text  distinct 4-grams  bits/byte under the model
  --------------------  --------------------------  ----------------  -------------------------
                greedy                       99.1%              0.64                       1.15
        repetition 1.1                       99.7%              0.80                       1.08
        repetition 1.3                       98.3%              0.92                       1.14
        repetition 2.0                       95.6%              0.92                       1.71
         frequency 0.5                       96.6%              0.95                       1.35
          presence 0.5                       99.2%              0.80                       1.10
  A byte-level model has 256 symbols and the prompt already contains every common letter,
  so 'seen' is nearly everything: these penalties are far harsher than on a vocabulary of
  100k tokens. Read the table as a warning about strength, not as a recipe.
  [ok] a very strong repetition penalty (2.0) costs real words against plain greedy, by more than 2 points  (95.6% against 99.1%)

## 6. Does the batch shape change greedy decoding?
  the same 64 prompts, run as one batch and one at a time: max |logit difference| 0.0e+00,
  smallest gap between the best and second-best logit 2.3e-02, greedy choices that differ: 0.
  bit-identical here: on this CPU the arithmetic is the same for every batch shape, so nothing flips.
  The page's warning (a library may pick different kernels, and a different summation order, per
  batch size) is about other hardware; this lab cannot show it.
  [ok] batch shape changes the logits by at most float rounding (relative to the logit scale)  (0.0e+00)

All checks passed. (7.1s)
```

</details>
