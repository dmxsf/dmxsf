# Reasoning RL: verifiers, GRPO advantages and loss, and a real RLVR run

Course pages: [Reinforcement Learning for Reasoning: RLVR and GRPO](https://dmxsf.com/en/wiki/post-training/reasoning-rl/)  
[中文](README.md)

The verifier, group advantages, the GRPO loss, clipping and the k3 KL estimator checked exactly; then a real RLVR run in which a tiny model is updated from nothing but exact-match 0/1 rewards.

## Run

```bash
uv run python labs/post-training/reasoning-rl/run.py            # full run, about 3 minutes
uv run python labs/post-training/reasoning-rl/run.py --quick    # a shrunk smoke test, a few seconds
```

You can also write `uv run dmxsf run reasoning-rl` (and add `--quick`). `--seed N` changes the random seed, default 0. The script prints its results section by section and checks the course page's claims one by one with `[ok]` / `[FAIL]`: the exit status is 0 when all pass and 1 when any fails; checks that only make sense in the full run are marked `[skip]` under `--quick`.

This lab uses cached small models: the byte-level base model (842,496 parameters). The first run trains them on TinyShakespeare and caches them in `~/.cache/dmxsf` (set `DMXSF_CACHE` to move it); later runs read them back.

## Code

| File | What it holds |
| --- | --- |
| `labs/post-training/reasoning-rl/run.py` | the lab script, numbered sections |
| `src/dmxsf/post_training/reasoning.py` | Reasoning RL: verifiable rewards, GRPO advantages and loss, and a rollout-and-update loop |

Unit tests: `uv run pytest tests/test_post_training_reasoning.py`

## What to look for

- Section 1: the page's verifier gives 1, 0, 0, 1, 0 (the last is a false negative: `\boxed{0.5}` equals 1/2 but string comparison says no); on the intent task a lenient 'contains the label' verifier pays for listing every label, exact match does not.
- Section 2: the group advantages are the page's numbers ([1, 0, 0, 1] gives +-0.8659, a unanimous group 0, multiplying the rewards by 10 changes nothing); the sample and the population standard deviation differ by exactly sqrt(G/(G-1)); at a pass rate of 5% 44.0% of the groups of G=16 are all wrong and carry no signal; dividing by the standard deviation pushes nearly-solved and nearly-unsolved prompts harder (3.998 times for 1 of 16 right, 1.936 for 8 of 16).
- Section 3: averaging per answer lets a long wrong answer be punished 10 times less per token (0.087 against 0.0087), averaging over tokens treats them alike; PPO clipping only stops pushes that already went far enough; the k3 estimate of the KL is unbiased and never negative, k1 is unbiased but can be negative (-3.265), and close to the reference k3 has 0.8% of k1's variance.
- Section 4: a real RLVR run on the intent task: the reference (after SFT) scores 58.0%; after 80 steps and 10,240 sampled answers greedy accuracy is 85.3% (another seed 85.7%), the mean reward goes 0.56 -> 0.83 and the share of groups with a signal falls from 39% to 26%; from the pretrained model without SFT not one of 1,280 answers earns a reward and the weights do not move at all: before RL can start the policy has to be right now and then.
- The end of section 4: keeping only the 136 training messages that are sometimes right and sometimes wrong gives a higher reward on its own pool (0.92) but a lower held-out accuracy (78.0% and 76.3%, against 85.3% and 85.7% for all); SFT on the same 360 correct answers reaches 99.3%: RLVR sees one bit per sampled answer, it is for when a checker exists and demonstrations do not.

## Exercises

Run again after a change: the checks that compare with the page's numbers turn into `[FAIL]`; that is expected, read how the numbers themselves move.

1. In section 4, change the 8 for GROUP in `PPS, GROUP, LR, BETA = (4, 4, 1e-4, 0.04) if QUICK else (16, 8, 1e-4, 0.04)` to 4 or 16: how do the share of groups with a signal and the final accuracy change?
2. Change BETA in the same line from 0.04 to 0: how high does the KL (k3) climb, and is the held-out bits/byte worse than 3.92?

## Notes

- The task is a synthetic intent classification (the answer is one label) on a byte-level model of 842,496 parameters; the page's GRPO results on maths problems need a real large model and are not reproduced here.

## Output

The complete runs below were made on an Apple M2 Max (12 cores, CPU only) with Python 3.12 and PyTorch 2.15.0. On another machine the numbers that involve training can differ in the last digits and timings depend on the machine; the checks use thresholds and tolerances, not bit-for-bit equality.

<details>
<summary>Output of the full run (`labs/post-training/reasoning-rl/run.py`)</summary>

```text
# Reasoning RL: verifiers, GRPO advantages and loss, and a real RLVR run
  course: https://dmxsf.com/wiki/post-training/reasoning-rl/
          https://dmxsf.com/en/wiki/post-training/reasoning-rl/

## 1. The verifier
  response                     gold  reward  why                                            
  ---------------------------  ----  ------  -----------------------------------------------
                 so \boxed{3}     3       1                                          correct
                 so \boxed{4}     3       0                                            wrong
              the answer is 3     3       0                       right answer, wrong format
  \boxed{4} no wait \boxed{3}     3       1                              the last box counts
                  \boxed{0.5}   1/2       0  string equality is not mathematical equivalence
  [ok] the page's verifier: 1, 0, 0, 1, 0 (the last row is a false negative)

  the intent task, gold label card_fee:
  answer                                 exact match  'contains the label'
  -------------------------------------  -----------  --------------------
                               card_fee            1                     1
  card_fee card_limit transfer_declined            0                     1
  [ok] a leaky verifier pays for listing every label; exact match does not

## 2. Group advantages
  rewards of three groups of four samples and their advantages (the page's code):
  rewards       A1       A2       A3       A4     
  ------------  -------  -------  -------  -------
  [1, 0, 0, 1]  +0.8659  -0.8659  -0.8659  +0.8659
  [1, 0, 0, 0]  +1.4997  -0.4999  -0.4999  -0.4999
  [1, 1, 1, 1]  +0.0000  +0.0000  +0.0000  +0.0000
  [ok] these are the numbers the page prints (4 decimals), and a unanimous group has no signal
  [ok] multiplying every reward by 10 changes nothing (only the comparison inside the group counts)

  the advantage of a correct answer in the group [1, 0, 0, 1]:
  normalisation                 A      
  ----------------------------  -------
  sample std (G - 1), the page  +0.8659
            population std (G)  +0.9998
        mean only, no division  +0.5000
  [ok] sample and population std differ by exactly sqrt(G / (G - 1)) = 1.1547 for G = 4

  chance that a group of G samples is mixed (some right, some wrong) at pass rate p:
  pass rate p  G = 4  G = 8  G = 16  G = 64
  -----------  -----  -----  ------  ------
           5%  18.5%  33.7%   56.0%   96.2%
          20%  58.9%  83.2%   97.2%  100.0%
          50%  87.5%  99.2%  100.0%  100.0%
          80%  58.9%  83.2%   97.2%  100.0%
  [ok] the page's numbers: 87.5% at p = 50%, G = 4; about 44% of groups all wrong at p = 5%, G = 16  (all wrong at 5%: 44.0%)
  [ok] the formula agrees with 100,000 simulated groups (p = 20%, G = 8)  (0.832 against 0.832)

  how much total push a group gets (sum of |A| over the group, G = 16):
  right     mean only  divided by std  ratio
  --------  ---------  --------------  -----
   1 of 16      1.875           7.497  3.998
   2 of 16      3.500           10.24  2.927
   4 of 16      6.000           13.41  2.236
   8 of 16      8.000           15.49  1.936
  12 of 16      6.000           13.41  2.236
  15 of 16      1.875           7.497  3.998
  [ok] dividing by the std boosts nearly-solved and nearly-unsolved prompts (the bias Dr. GRPO points out)  (ratio 4.00 at 1 of 16 and 1.94 at 8 of 16)

## 3. The loss
  two wrong answers (A = -0.87), 10 and 100 tokens long: push-down weight per token
  loss averaged                             short answer  long answer
  ----------------------------------------  ------------  -----------
  per answer, then over answers (the page)       0.08700     0.008700
                    over all tokens (DAPO)        0.8700       0.8700
  [ok] per-answer averaging punishes a long wrong answer 10x less per token; token-level does not  (0.087 against 0.0087 per token)

  PPO clipping (eps = 0.2): does the token's probability still get pushed?
  advantage  ratio new/old  push
  ---------  -------------  ----
         +1            1.5  none
         +1            0.5    up
         -1            0.5  none
         -1            1.5  down
  [ok] clipping stops only the pushes that already went far enough in the intended direction

  KL(pi || ref) of two distributions over 6 outcomes, and what each estimator averages to:
  estimator                      expectation over samples from pi  smallest value
  -----------------------------  --------------------------------  --------------
                       exact KL                             2.449               -
             k1 = log(pi / ref)                             2.449          -3.265
  k3 = ref/pi - log(ref/pi) - 1                             2.449         0.04938
  [ok] k3 is unbiased and never negative; k1 is unbiased but goes negative
  [ok] for policies close to the reference, k3 has far lower variance than k1 (200 random pairs)  (median Var(k3) / Var(k1) = 0.008)

## 4. A real RLVR run on the intent task
  reference: pretrained model + SFT on 60 messages (300 steps): accuracy 58.0% on 300 test messages, 3.80 bits/byte.
  RLVR: 16 messages x 8 sampled answers per step, reward 1 for an exact match, lr 0.0001, beta 0.04,
  80 steps = 10,240 sampled answers, one update per batch (so the ratio is 1 and the clip never binds).

  training curve (sampled at temperature 1, averages over windows of steps):
  steps  mean reward  groups with signal  entropy (nats)  answer tokens  k3 KL
  -----  -----------  ------------------  --------------  -------------  -----
   1-20         0.56                0.39            0.05          15.19  0.049
  21-40         0.70                0.38            0.04          15.36  0.060
  41-60         0.77                0.33            0.04          15.15  0.103
  61-80         0.83                0.26            0.03          15.11  0.097
  after training: greedy accuracy 85.3% (reference 58.0%), 3.92 bits/byte (reference 3.80);
  k3 per step: median 0.062, largest 0.236: no heavy tail in this run.
  another seed: accuracy 85.7% (range 0.3 points over 2 runs; one test message is 0.33 points)
  [ok] the mean reward rises over training (last window against first, by more than 0.15)  (0.56 -> 0.83)
  [ok] the share of groups with a signal falls as the pass rate rises  (0.39 -> 0.26)
  [ok] RLVR beats the reference by more than the seed range (mean of 2 seeds)  (58.0% -> 85.5%)

  the same loop from the pretrained model without SFT (10 steps): 0 of 1,280 sampled answers earned a reward, groups with a signal 0%, largest weight change 0.0e+00.
  [ok] no answer is ever right, so no group has a signal and the weights do not move at all  (RL cannot start from a policy that never succeeds)

  pass rates of the reference on the 360 training messages (8 samples each): all wrong 32%, all right 30%, mixed 38%.
  training pool            mean reward, last window  groups with signal, first window  test accuracy (2 seeds)
  -----------------------  ------------------------  --------------------------------  -----------------------
         all 360 messages                      0.82                              0.38             85.3%, 85.7%
  only the 136 mixed ones                      0.92                              0.42             78.0%, 76.3%
  The filter gives each step more signal at the start and a higher reward on its own
  pool, but the pool is smaller and saturates sooner: read the last column for what it
  did to held-out accuracy. (Two seeds per row.)

  for scale: SFT on the 360 correct answers themselves reaches 99.3% (200 steps, lr 1e-4). RLVR saw one bit per sampled answer, not
  the answer; it is the tool for when a checker exists and demonstrations do not.

All checks passed. (195.5s)
```

</details>
