# Position encodings and length extrapolation

Course pages: [Positional Encoding Explained: Absolute, Relative and RoPE](https://dmxsf.com/en/wiki/architecture/positional-encoding/) · [Long Context: Position Extrapolation, Training Recipes and Cost](https://dmxsf.com/en/wiki/architecture/long-context/)  
[中文](README.md)

Sinusoidal encodings as clocks, RoPE's score depending only on distance, the stretching recipes (position interpolation, NTK-aware, YaRN, the Llama 3.1 rule) computed side by side, and five position schemes trained on short text and tested on long text.

## Run

```bash
uv run python labs/architecture/positional-encoding/run.py            # full run, about 2 minutes
uv run python labs/architecture/positional-encoding/run.py --quick    # a shrunk smoke test, a few seconds
```

You can also write `uv run dmxsf run positional-encoding` (and add `--quick`). `--seed N` changes the random seed, default 0. The script prints its results section by section and checks the course page's claims one by one with `[ok]` / `[FAIL]`: the exit status is 0 when all pass and 1 when any fails; checks that only make sense in the full run are marked `[skip]` under `--quick`.

## Code

| File | What it holds |
| --- | --- |
| `labs/architecture/positional-encoding/run.py` | the lab script, numbered sections |
| `src/dmxsf/architecture/position.py` | Position encodings: sinusoidal, RoPE (with context-extension variants) and ALiBi |

Unit tests: `uv run pytest tests/test_architecture_position.py`

## What to look for

- Section 2: the RoPE score depends only on the distance: positions 7 apart score -7.5584 whether they are 3 and 10, 103 and 110 or 190 and 197, as on the page.
- Section 3: with training length 4096 and target 32768 the largest unseen phase is 1.00, 1.75 and 1.00 for PI, NTK-aware and YaRN (8.00 with no change); the page's YaRN segments (21 / 25 / 18) and the Llama 3.1 ones (29 / 6 / 29) match.
- Section 4: one model, five position schemes, trained on 64 bytes and tested up to 256: RoPE's loss climbs to 3.4 beyond twice the training length, ALiBi stays at about 1.77 everywhere; NTK-aware and YaRN stretched 4x without fine-tuning bring the far positions down to 2.0 and 1.9, while linear interpolation (PI) damages even the positions it was trained on (3.64).

## Exercises

Run again after a change: the checks that compare with the page's numbers turn into `[FAIL]`; that is expected, read how the numbers themselves move.

1. Re-run section 4 with other `--seed` values: is the ranking of the schemes stable? Which differences are smaller than the spread between seeds?
2. Change `s` in `dim, base, trained, s = 128, 10_000.0, 4096, 8.0` of section 3 to 32 (target length 131072): are NTK-aware's and YaRN's largest unseen phases still 1.75 and 1.00?

## Output

The complete runs below were made on an Apple M2 Max (12 cores, CPU only) with Python 3.12 and PyTorch 2.15.0. On another machine the numbers that involve training can differ in the last digits and timings depend on the machine; the checks use thresholds and tolerances, not bit-for-bit equality.

<details>
<summary>Output of the full run (`labs/architecture/positional-encoding/run.py`)</summary>

```text
# Position encodings and length extrapolation
  course: https://dmxsf.com/wiki/architecture/positional-encoding/
          https://dmxsf.com/en/wiki/architecture/positional-encoding/
  course: https://dmxsf.com/wiki/architecture/long-context/
          https://dmxsf.com/en/wiki/architecture/long-context/

## 1. Sinusoidal: every pair of dimensions is a clock with its own period
  32 clocks: periods from 6.3 to 47,117 positions
  [ok] moving k positions rotates each pair by a fixed angle (so offsets are easy to learn)

## 2. RoPE: the score depends only on the distance
  m=  3  n= 10  n-m=7  score -7.5584
  m= 13  n= 20  n-m=7  score -7.5584
  m=103  n=110  n-m=7  score -7.5584
  m=190  n=197  n-m=7  score -7.5584
  [ok] the same offset gives the same score at any position (page: -7.5584)
  [ok] RoFormer's adjacent pairing and Llama's rotate_half differ only by a permutation
  d=128, base 10,000: wavelengths run from 6.28 to 54,410 positions
  [ok] the fastest pair turns every 6.28 positions, the slowest every 54,410

## 3. Stretching RoPE: PI, NTK-aware, YaRN and the Llama 3.1 rule
  d=128, base 10,000, trained on 4096, target 32768 (s = 8)
  pairs that turn less than once inside the training length: 18 of 64 (pairs 46 to 63)
  method            pair 0 frequency  slowest pair divided by  largest unseen phase
  ----------------  ----------------  -----------------------  --------------------
  RoPE (no change)             1.000                    1.000                 8.000
                PI            0.1250                    8.000                 1.000
         NTK-aware             1.000                    8.000                 1.753
              YaRN             1.000                    8.000                 1.000
  [ok] the course page's table: 8.00 / 1.00 / 1.75 / 1.00
  YaRN: 21 pairs unchanged, 25 blended, 18 divided by s; attention factor 1.208 (q.k grows 1.46x)
  Llama 3.1 (base 500,000): 29 pairs divided by 8, 6 blended, 29 unchanged
  [ok] segments match the course page: YaRN 21/25/18, Llama 3.1 29/6/29
  NTK-aware base: 82,685   ALiBi slopes (8 heads): 1/2 ... 1/256

## 4. Train short, test long: five position schemes, same model otherwise
  2 layers, d_model 64, 1000 steps on Shakespeare bytes with 64-byte windows, 2 seed(s); loss in nats per byte
  evaluated on 256-byte windows, split by position (position < 64 was seen in training)
  scheme            pos 0-63  pos 64-127  pos 128-255
  ----------------  --------  ----------  -----------
           learned     1.827         n/a          n/a
        sinusoidal     1.990       3.271        3.386
              rope     1.606       1.824        3.428
  rope + linear x4     3.641       3.678        3.708
     rope + ntk x4     1.642       1.637        2.015
    rope + yarn x4     1.759       1.802        1.857
  rope + llama3 x4     1.938       2.059        2.038
             alibi     1.766       1.768        1.775
              none     2.155       2.393        2.505
  (learned positions exist only for the first 64 positions; the four rows below RoPE
   reuse the RoPE models, stretched 4x at test time with no further training)
  [ok] learned positions cannot even run past the training length
  [ok] ALiBi is flat: far positions cost the same as near ones (within 0.1 nats)
  [ok] RoPE collapses beyond about twice the training length (+1 nat)
  [ok] sinusoidal degrades beyond the training length too (+0.5 nats)
  [ok] PI without fine-tuning damages even the positions it was trained on (+0.5 nats)
  [ok] NTK-aware and YaRN stretch RoPE 4x with no training: far positions improve by 0.5+ nats

All checks passed. (156.2s)
```

</details>
