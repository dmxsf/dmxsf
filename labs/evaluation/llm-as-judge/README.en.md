# LLM-as-a-judge: bias, agreement, and Bradley-Terry leaderboards

Course pages: [LLM-as-a-Judge: Using a Model as the Grader, Its Biases and Calibration](https://dmxsf.com/en/wiki/evaluation/llm-as-judge/)  
[中文](README.md)

A judge stand-in with chosen biases, so every number has an exact answer: swapping the order cancels position bias but not length bias, Cohen's kappa exposes a judge that always guesses the majority class, and a Bradley-Terry leaderboard is corrected for length.

## Run

```bash
uv run python labs/evaluation/llm-as-judge/run.py            # full run, a few seconds
uv run python labs/evaluation/llm-as-judge/run.py --quick    # a shrunk smoke test, a few seconds
```

You can also write `uv run dmxsf run llm-as-judge` (and add `--quick`). `--seed N` changes the random seed, default 0. The script prints its results section by section and checks the course page's claims one by one with `[ok]` / `[FAIL]`: the exit status is 0 when all pass and 1 when any fails; checks that only make sense in the full run are marked `[skip]` under `--quick`.

## Code

| File | What it holds |
| --- | --- |
| `labs/evaluation/llm-as-judge/run.py` | the lab script, numbered sections |
| `src/dmxsf/evaluation/judge.py` | LLM-as-a-judge: a judge with biases (simulated), agreement measures, Bradley-Terry ratings |

Unit tests: `uv run pytest tests/test_evaluation_judge.py`

## What to look for

- Section 1: with a position bias, judging once in a random order agrees with the truth 0.753 (exact 0.757); judging in both orders and keeping what agrees leaves 0.380 of the pairs as ties and agrees on the rest 0.914. The larger the bias, the worse judging once (0.558 at bias 2.0), and swapping keeps agreement at 0.99 at the price of ties rising to 0.882.
- Section 2: with no bias 1/2 + arctan(s/sigma)/pi = 78.5% and averaging both orders gives 83.6% (the noise divided by sqrt 2); a position bias lowers the agreement of one judgement and leaves the averaged one unchanged.
- Section 3: a length bias is not cancelled by swapping: for two answers of equal quality, one four times as long, a position bias alone turns 95.7% of the pairs into ties when swapped (the right verdict), while a length bias lets the longer answer win in both orders (99.7%) and swapping reports no tie.
- Section 4: agreement 0.750, chance agreement 0.418, Cohen's kappa 0.570; a lazy judge that always says 'A' agrees 0.900 with kappa 0; agreement with the truth rises from 0.565 to 0.998 with the quality gap, which the single number 0.785 hides.
- Section 5: from pairwise votes to a leaderboard: Bradley-Terry with a bootstrap (B 1031.0 [1006.8, 1052.3] and so on); the interval of B - A, [6.8, 52.3], excludes 0 and every true rating lies inside its interval; shuffle the battles and online Elo's D - B swings between 178.9 and 240.2 while Bradley-Terry does not move.
- Section 6: a judge that loves long answers puts the wordy model E (really as good as A) above C (1175 against 1120); adding the length difference as a covariate in the regression brings E back to 1017, with a coefficient of 0.302 (true 0.35).

## Exercises

Run again after a change: the checks that compare with the page's numbers turn into `[FAIL]`; that is expected, read how the numbers themselves move.

1. Add 4.0 to `for bias in (0.0, 0.25, 0.5, 1.0, 2.0)` in section 1: what agreement and tie rate does swapping give?
2. In section 6 change the 6000 in `m, n_battles = 5, 6000` to 600: is the estimate of the length coefficient still good, and does E come back down to A's level?

## Notes

- The judge is simulated (the quality gap, the noise, the position and the length bias are parameters), not a language model: that is what gives a truth to compare with.

## Output

The complete runs below were made on an Apple M2 Max (12 cores, CPU only) with Python 3.12 and PyTorch 2.15.0. On another machine the numbers that involve training can differ in the last digits and timings depend on the machine; the checks use thresholds and tolerances, not bit-for-bit equality.

<details>
<summary>Output of the full run (`labs/evaluation/llm-as-judge/run.py`)</summary>

```text
# LLM-as-a-judge: bias, agreement, and Bradley-Terry leaderboards
  course: https://dmxsf.com/wiki/evaluation/llm-as-judge/
          https://dmxsf.com/en/wiki/evaluation/llm-as-judge/

## 1. Swapping the order: the page's simulation, then the exact answer
  procedure                                    simulated (20,000 pairs)  exact (integral over the gap)
  -------------------------------------------  ------------------------  -----------------------------
  judge once, random order: agrees with truth                     0.753                          0.757
         swap both orders: pairs left as ties                     0.380                          0.382
      swap both orders: agreement on the rest                     0.914                          0.915
  [ok] the page's numbers: 0.753 once; 0.380 ties and 0.914 agreement when swapped
  [ok] the simulation agrees with the exact integral to within 0.01

  The price of swapping grows with the bias (noise 0.8, exact):
  position bias  judge once  ties when swapped  agreement when swapped
  -------------  ----------  -----------------  ----------------------
          0.000       0.785              0.291                   0.902
         0.2500       0.778              0.315                   0.906
         0.5000       0.757              0.382                   0.915
          1.000       0.688              0.577                   0.944
          2.000       0.558              0.882                   0.990
  [ok] judging once gets worse as the bias grows; swapping keeps agreement high but ties rise

## 2. Averaging both orders, and the closed forms on the page
  quality gap sd 1.0, noise 0.8, 200,000 pairs per row:
  position bias  judge once  average of both orders
  -------------  ----------  ----------------------
          0.000       0.785                   0.836
         0.5000       0.757                   0.836
  [ok] no bias: 1/2 + arctan(s/sigma)/pi = 78.5% once, and 83.6% averaging both orders (noise / sqrt 2)
  [ok] the simulations match those closed forms to within 0.01  (0.785 and 0.836)
  [ok] a position bias lowers the one-shot agreement but leaves the averaged one unchanged

## 3. A length bias is not cancelled by swapping
  two answers of equal quality, x four times as long; the right verdict is 'tie':
  judge               x wins, judged once  x wins swapped  y wins swapped  tie swapped
  ------------------  -------------------  --------------  --------------  -----------
  position bias only                0.499           0.021           0.022        0.957
    length bias only                0.998           0.997           0.000        0.003
  [ok] position bias: swapping turns almost every pair into a tie (the right verdict)  (0.957 ties)
  [ok] length bias: the longer answer wins in both orders and swapping reports no tie  (x wins 0.997)

## 4. Agreement, kappa, and how many human labels are needed
  rows: human (A wins, B wins, tie); columns: judge
  agreement 0.750, chance agreement 0.418, Cohen's kappa 0.570
  a judge that always says 'A': agreement 0.900, kappa 0.000
  [ok] the page's numbers: 0.750 / 0.418 / 0.570, and 0.900 agreement with kappa 0 for the lazy judge
  [ok] labels needed for +-5 points: 246 at 80% agreement, 385 at 50%

  Agreement with the truth by size of the quality gap (the same judge throughout):
  |gap|        pairs  agreement  expected
  -----------  -----  ---------  --------
    0 to 0.25  39656      0.565     0.562
  0.25 to 0.5  37262      0.678     0.679
     0.5 to 1  59501      0.816     0.817
       1 to 2  54387      0.950     0.950
       2 to 8   9194      0.998     0.997
  one overall number would be 0.785, which hides all of that
  [ok] agreement rises with the gap, from near a coin flip to near certainty  (0.565 -> 0.998)

## 5. From pairwise votes to a leaderboard: Bradley-Terry with a bootstrap
  model  true Elo  estimate  95% interval    
  -----  --------  --------  ----------------
      A      1000    1000.0  [1000.0, 1000.0]
      B      1050    1031.0  [1006.8, 1052.3]
      C      1120    1118.6  [1095.0, 1140.9]
      D      1250    1246.8  [1223.2, 1270.5]
  B - A: 31.0 points, interval [6.8, 52.3]
  [ok] the page's table: B 1031.0 [1006.8, 1052.3], C 1118.6 [1095.0, 1140.9], D 1246.8 [1223.2, 1270.5]
  [ok] the interval of the difference B - A is [6.8, 52.3] and excludes 0
  [ok] every true rating lies inside its interval
  [ok] the Newton fit sits at the maximum of the likelihood (gradient is zero)  (largest gradient entry 3.3e-13)
  [ok] a 400-point Elo gap is 10 to 1 odds, and the Elo scale is 400 / ln 10 per coefficient

  The same 3,000 battles in five random orders; D minus B in rating points:
  order  online Elo (K=4)  Bradley-Terry
  -----  ----------------  -------------
      0             240.2          215.8
      1             227.6          215.8
      2             178.9          215.8
      3             208.8          215.8
      4             202.6          215.8
  [ok] Bradley-Terry does not depend on the order of the battles; online Elo does  (spread 4.5e-13 against 61.2 points)

## 6. The judge's bias goes into the leaderboard unless you model it
  model  true Elo  naive Bradley-Terry  length-controlled
  -----  --------  -------------------  -----------------
      A      1000                 1000               1000
      B      1050                 1066               1066
      C      1120                 1120               1120
      D      1250                 1253               1253
      E      1000                 1175               1017
  estimated length coefficient 0.302 (true 0.35); E is the wordy model
  [ok] a length-loving judge puts the wordy model E above C, though E is as good as A  (naive E 1175, C 1120, A 1000)
  [ok] adding the length difference as a covariate recovers the true order and the coefficient  (controlled E 1017, coefficient 0.302)

  This is the idea behind length-controlled win rates (Dubois et al., 2024): fit a model of the
  judge's preference, then read the leaderboard at a length difference of zero. It only works
  to the extent that the effect of length is what the regression says it is.

All checks passed. (0.6s)
```

</details>
