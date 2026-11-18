# Preference optimization: the DPO loss, its closed form, and a real DPO run

Course pages: [Preference Optimization: From RLHF to DPO](https://dmxsf.com/en/wiki/post-training/preference-optimization/)  
[中文](README.md)

The DPO loss by hand, its closed-form optimum checked against random competitors and fitted comparisons, a real DPO run on a small byte-level model (learning rate, beta, chosen log-probabilities that fall, the LoRA-off reference), and reward hacking as the KL budget grows.

## Run

```bash
uv run python labs/post-training/preference-optimization/run.py            # full run, about 6 minutes
uv run python labs/post-training/preference-optimization/run.py --quick    # a shrunk smoke test, about 15 seconds
```

You can also write `uv run dmxsf run preference-optimization` (and add `--quick`). `--seed N` changes the random seed, default 0. The script prints its results section by section and checks the course page's claims one by one with `[ok]` / `[FAIL]`: the exit status is 0 when all pass and 1 when any fails; checks that only make sense in the full run are marked `[skip]` under `--quick`.

This lab uses cached small models: the byte-level base model (842,496 parameters). The first run trains them on TinyShakespeare and caches them in `~/.cache/dmxsf` (set `DMXSF_CACHE` to move it); later runs read them back.

## Code

| File | What it holds |
| --- | --- |
| `labs/post-training/preference-optimization/run.py` | the lab script, numbered sections |
| `src/dmxsf/post_training/preference.py` | Preference optimization: the DPO loss, its closed-form optimum, and a Bradley-Terry toy world |

Unit tests: `uv run pytest tests/test_post_training_preference.py`

## What to look for

- Section 1: the page's two examples (beta = 0.1 and 0.5, margins 0.15 and 0.75) recompute to 0.621 and 0.387; with the policy equal to the reference the loss is exactly ln 2; the loss sees only the margin: the log probabilities of chosen and rejected can both fall (-2 and -5) and the loss still drops (0.554 < 0.693).
- Section 2: the closed form pi* in a small world of six candidate answers: a smaller beta buys more reward (0.68 -> 3.00) and costs more KL (0.003 -> 3.51); fitting DPO on simulated comparisons, the total variation to the closed form falls from 0.459 at 100 comparisons to 0.014 at 100,000, and the implicit reward recovers the true one up to a constant (largest error 0.041).
- Section 3: a real DPO run on the intent task. The reference (after SFT) scores 58.0%, the default (beta = 0.1, lr 3e-5) 83.0%, three seeds average 84.4% (range 2.7 points); with a learning rate too large (3e-4) the reward accuracy is 100% while the chosen answers' log-probability FALLS by 18.1 nats and the model stops answering in the format (valid labels 7%); a larger beta stays closer to the reference (KL 2.26 / 2.05 / 0.90 / 0.47) and scores lower (83.7% -> 71.7%); DPO with LoRA works as well, but its 93.7% at lr 1e-3 pays with more forgetting (4.26 bits/byte).
- The end of section 3: on the same 360 correct answers plain SFT reaches 99.3% and DPO 84.4%: where a correct answer exists, SFT uses the data more directly; DPO is for the case where only comparisons exist.
- Section 4: the reward-hacking toy world: the proxy reward rises with KL all the way to 3.55 while the true reward peaks at an interior beta (0.7, KL 1.6 nats) at 0.73 and falls to 0.17: the KL penalty, early stopping on a held-out gold signal and refreshing the reward model are there for this.

## Exercises

Run again after a change: the checks that compare with the page's numbers turn into `[FAIL]`; that is expected, read how the numbers themselves move.

1. Add `2e-4` to `lrs = (DEFAULT_LR,) if QUICK else (1e-5, 3e-5, 1e-4, 3e-4)` in section 3: in which range of learning rates does the reward accuracy still rise while the accuracy starts to fall?
2. In section 4 change the 3 in `k_resp, n_tricks, overrate, penalty, sigma, worlds = 200, 3, 4.5, 2.0, 0.3, 200` to 10: is the peak of the true reward still there, and which way does it move?

## Notes

- The task is a synthetic intent classification and both the reference and the policy are byte-level models of 842,496 parameters; section 4 is a toy world built so that the proxy reward can be exploited: it shows the mechanism, not how large the effect is.

## Output

The complete runs below were made on an Apple M2 Max (12 cores, CPU only) with Python 3.12 and PyTorch 2.15.0. On another machine the numbers that involve training can differ in the last digits and timings depend on the machine; the checks use thresholds and tolerances, not bit-for-bit equality.

<details>
<summary>Output of the full run (`labs/post-training/preference-optimization/run.py`)</summary>

```text
# Preference optimization: the DPO loss, its closed form, and a real DPO run
  course: https://dmxsf.com/wiki/post-training/preference-optimization/
          https://dmxsf.com/en/wiki/post-training/preference-optimization/

## 1. The loss by hand (the page's numbers)
  beta    chosen reward  rejected reward  margin  loss    gradient weight
  ------  -------------  ---------------  ------  ------  ---------------
  0.1000        0.05000          -0.1000  0.1500  0.6210           0.4626
  0.5000         0.2500          -0.5000  0.7500  0.3869           0.3208
  [ok] loss 0.621 at beta = 0.1 and 0.387 at beta = 0.5 (the page's 0.15 and 0.75 margins)  (0.6210 and 0.3869)
  [ok] autograd gives the page's gradient: -beta * weight on chosen, +beta * weight on rejected  (weight 0.4626, max difference 0.0e+00)
  [ok] policy = reference: the loss is exactly ln 2 (the page's self-test)  (0.693147 vs 0.693147)
  [ok] the loss sees only the margin: both log-probabilities can fall (-2 and -5) and it still drops  (0.554 < 0.693)

## 2. The closed form: what DPO is trying to reach
  a made-up world: six candidate answers, a reference policy and a hidden reward;
  the best answer (reward 3) has only 3% of the reference's probability.
  beta     pi*(best)  expected reward  KL(pi* || ref) nats
  -------  ---------  ---------------  -------------------
    10.00    0.03795           0.6780             0.002945
    2.000    0.09092           0.9706              0.09409
   0.5000     0.6164            2.424                1.870
   0.2000     0.9832            2.982                3.401
  0.05000      1.000            3.000                3.507
  [ok] a smaller beta buys more reward and costs more KL, monotonically  (reward 0.68 -> 3.00, KL 0.003 -> 3.51)
  [ok] pi* beats 2,100 random distributions and 300 small perturbations of itself  (objective 1.4886; best rival 1.4686, best perturbation 1.4886)
  DPO fitted from simulated comparisons (beta = 0.5; pairs drawn from the reference; the winner chosen
  with probability sigmoid(r_i - r_j)), 5 draws per size:
  comparisons  TV(pi_DPO, pi*) mean  range over draws  max |implicit - true reward|
  -----------  --------------------  ----------------  ----------------------------
          100                0.4588    0.332 to 0.716                         3.322
        1,000                0.1006    0.017 to 0.173                        0.3015
       10,000               0.05365    0.026 to 0.077                        0.1101
      100,000               0.01406    0.005 to 0.023                       0.04053
  [ok] with enough comparisons DPO reaches the closed form (total variation < 0.05)  (0.459 at 100 comparisons -> 0.014 at 100,000)
  [ok] the implicit reward recovers the true reward up to a constant (the log Z that cancels)  (max error 0.041 on rewards spanning 3.0)

## 3. A real DPO run on the intent task
  reference model: the pretrained model after SFT on 60 messages (5 per intent, 300 steps):
  accuracy 58.0% on 300 test messages, valid labels 100%, 3.80 bits/byte on held-out Shakespeare.
  preference pairs: 720 = 360 new messages x 2; chosen = the correct label, rejected = a wrong label
  drawn from the reference's own probabilities over the eleven wrong labels (the mistakes it really makes).
  [ok] step 1 of a real run: loss ln 2 and mean gradient weight 0.5 (policy = reference)  (loss 0.693147, weight 0.5000)

  the default run (beta 0.1, lr 3e-05), training batches:
  steps    loss    reward acc  d log p chosen  d log p rejected  mean weight
  -------  ------  ----------  --------------  ----------------  -----------
     1-50  0.6049      0.6937          0.5155            -1.772       0.4468
   51-100  0.4414      0.9213           1.296            -5.518       0.3465
  101-150  0.3553      0.9725           1.624            -8.155       0.2888
  151-200  0.3156      0.9712           1.902            -9.663       0.2600
  [ok] pairs that are ordered correctly weigh less: the mean gradient weight falls  (0.45 -> 0.26)

  learning rate (beta 0.1, 200 steps, SFT used 3e-4):
             reward acc  d log p chosen  d log p rejected  accuracy  valid label  mass on 12 labels  KL to ref  bits/byte
  ---------  ----------  --------------  ----------------  --------  -----------  -----------------  ---------  ---------
   lr 1e-05        0.89           +1.16             -3.27     69.7%         100%               0.95       0.85       3.83
   lr 3e-05        0.97           +1.90             -9.66     83.0%          99%               0.91       2.05       3.89
  lr 0.0001        1.00           +2.15            -23.07     91.0%          93%               0.71       2.22       4.12
  lr 0.0003        1.00          -18.05            -90.19      7.0%           7%               0.01       3.84       4.80
  (the reference: accuracy 58.0%, valid 100%, mass on the 12 labels 0.97, 3.80 bits/byte)
  [ok] a 3x smaller learning rate moves the policy less (KL to the reference)  (KL 0.85 at 1e-5 vs 2.05 at 3e-5)
  [ok] too large: the chosen answers' log-probability FALLS although the reward accuracy is 100%  (chosen -18.1 nats, rejected -90.2, reward acc 1.00)
  [ok] ...and the model stops answering in the format (valid labels collapse)  (valid 7%, mass on the twelve labels 0.010)

  the default run with other seeds (pairs resampled, batch order changed): 83.0%, 84.7%, 85.7% (range 2.7 points; one test message is 0.33 points)
  [ok] DPO beats the reference by more than the seed range  (58.0% -> 84.4% on average (range 2.7 points))

  beta (lr 3e-05, 200 steps):
             reward acc  d log p chosen  d log p rejected  accuracy  valid label  mass on 12 labels  KL to ref  bits/byte
  ---------  ----------  --------------  ----------------  --------  -----------  -----------------  ---------  ---------
  beta 0.01        0.96           +1.73            -10.83     83.7%          99%               0.91       2.26       3.91
   beta 0.1        0.97           +1.90             -9.66     83.0%          99%               0.91       2.05       3.89
   beta 0.5        0.99           +1.97             -6.03     78.3%          97%               0.90       0.90       3.86
     beta 2        0.99           +1.47             -2.97     71.7%         100%               0.95       0.47       3.83
  [ok] a larger beta keeps the policy closer to the reference (KL to it)  (KL 2.26 / 2.05 / 0.90 / 0.47)

  LoRA (r = 8, all linear layers), the reference is the same model with the adapters off:
                  reward acc  d log p chosen  d log p rejected  accuracy  valid label  mass on 12 labels  KL to ref  bits/byte  trained parameters
  --------------  ----------  --------------  ----------------  --------  -----------  -----------------  ---------  ---------  ------------------
  lora lr 0.0003        0.96           +1.59            -10.14     77.7%          97%               0.88       2.02       3.94              73,728
   lora lr 0.001        1.00           +2.20            -21.98     93.7%          99%               0.88       2.29       4.26              73,728
   full lr 3e-05        0.97           +1.90             -9.66     83.0%          99%               0.91       2.05       3.89             842,496
  [ok] with the adapters off, the LoRA model's log-probabilities equal the reference's exactly  (no second copy of the weights is needed)

  for comparison, plain SFT on the same 360 correct answers (lr 1e-4, 200 steps; neither method tuned further):
                                  accuracy  valid label  mass on 12 labels  KL to ref  bits/byte
  ------------------------------  --------  -----------  -----------------  ---------  ---------
       SFT on the chosen answers     99.3%         100%               0.99       2.10       3.93
  DPO (default, mean of 3 seeds)     84.4%          98%               0.91       1.88       3.90
  Where a correct answer exists, SFT on it uses the data more directly. DPO is for the
  case where only comparisons exist; with a verifiable answer the next page's rule-based
  reward is the other route.

## 4. Reward hacking: a proxy reward and the KL budget (a toy world)
  200 candidate answers with true quality ~ N(0, 1) and a uniform reference. The proxy
  reward sees quality plus noise (sd 0.3), except for 3 'tricks' (think: very long,
  flattering answers) that are really 2 worse than average but that the proxy rates
  4.5 above their true value. Averages over 200 random worlds. The world is built so
  that the proxy can be exploited; it shows the mechanism, not how big the effect is.
  beta     KL (nats)  proxy reward  true reward  mass on the tricks
  -------  ---------  ------------  -----------  ------------------
    10.00   0.005946        0.1596      0.06293             0.01920
    3.000    0.06891        0.4484       0.2650             0.03394
    1.500     0.2989        0.9050       0.5131             0.07388
    1.000     0.7344         1.426       0.6783              0.1470
   0.7000      1.568         2.114       0.7293              0.2830
   0.5000      2.705         2.787       0.6155              0.4547
   0.3000      4.096         3.346       0.3594              0.6347
   0.1000      5.028         3.543       0.1872              0.7175
  0.05000      5.170         3.554       0.1747              0.7226
  [ok] the proxy reward rises with every bit of KL spent  (0.16 -> 3.55)
  [ok] the true reward peaks at an interior beta, then falls while the proxy keeps rising  (peak 0.73 at KL 1.6 nats (beta 0.7), 0.17 at KL 5.2)
  In a real run the true reward is not observable. That is why the page recommends a KL
  penalty, early stopping on a held-out gold signal, and refreshing the reward model.

All checks passed. (290.6s)
```

</details>
