# SFT data: the loss mask, a data health check, and what data size, noise, masking and replay do

Course pages: [SFT Data: Construction and Quality Control](https://dmxsf.com/en/wiki/post-training/sft-data/)  
[中文](README.md)

What the SFT loss averages over (the -100 mask, packing and padding), data health checks against planted ground truth, and four controlled experiments: run-to-run noise, examples per intent, wrong labels, loss on the prompt and replaying general text.

## Run

```bash
uv run python labs/post-training/sft-data/run.py            # full run, about 3 minutes
uv run python labs/post-training/sft-data/run.py --quick    # a shrunk smoke test, about 15 seconds
```

You can also write `uv run dmxsf run sft-data` (and add `--quick`). `--seed N` changes the random seed, default 0. The script prints its results section by section and checks the course page's claims one by one with `[ok]` / `[FAIL]`: the exit status is 0 when all pass and 1 when any fails; checks that only make sense in the full run are marked `[skip]` under `--quick`.

This lab uses cached small models: the byte-level base model (842,496 parameters). The first run trains them on TinyShakespeare and caches them in `~/.cache/dmxsf` (set `DMXSF_CACHE` to move it); later runs read them back.

## Code

| File | What it holds |
| --- | --- |
| `labs/post-training/sft-data/run.py` | the lab script, numbered sections |
| `src/dmxsf/post_training/sft.py` | Supervised fine-tuning, mechanically: loss masks, batches, packing, and the SFT loop |
| `src/dmxsf/post_training/data_quality.py` | Checking an instruction dataset: rules, near-duplicates (MinHash + LSH) and n-gram contamination |
| `src/dmxsf/post_training/intents.py` | A synthetic customer-message task: the stand-in for banking77 in the post-training labs |

Unit tests: `uv run pytest tests/test_post_training_sft.py tests/test_post_training_data_quality.py tests/test_post_training_intents.py`

## What to look for

- Sections 1-2: one sample's ids, labels and the -100 mask (95 prompt tokens, 12 answer tokens, 11% supervised); the stop token (a newline) is supervised, it is the model's only signal where to stop; averaging per token (4.5944) and per sequence (4.7516) are different numbers.
- The end of section 1: the same model, another wording of the prompt, and accuracy falls from 99.0% to 85.0% (valid label 91.3%); an extra space changes nothing: the template belongs to the data and should be versioned with it.
- Section 3: packing needs a document mask and restarted positions (largest logit difference to separate passes 1.3e-05; plain concatenation 9.6, the documents see each other); random batches waste 75.8%-83.5% of the positions, length-grouped ones 6.2%-6.8%, packing 0.1%.
- Section 4: a health check on 540 records with planted defects: rules only give candidates ('output <= 2 words' also counts the 15 correct one-word answers); MinHash + LSH finds 83 of the 83 pairs brute force finds and invents none; contamination: 13-grams catch verbatim copies but miss edited ones (0/20) and partial overlaps (1/20), n=8 catches them all, at the price of false alarms.
- Sections 5-7: change nothing but the training seed or the data subset and accuracy still ranges from 90.3% to 99.0% (8.7 points; 300 test messages, so one message is 0.33%): differences below that range are not findings; examples per intent 1 -> 2 -> 5 -> 10 -> 20 -> 50 give 18% -> 18% -> 40% -> 77% -> 99% -> 100%; 10% / 20% / 40% wrong labels drop accuracy to 84.0% / 53.0% / 42.0%.
- Section 8: loss on the whole sequence instead of the answer only drops accuracy from 99.0% to 24.3% (the answer is about an eighth of it); replaying plain text every other step keeps the general ability (2.32 bits/byte, the base model has 2.31) and the target accuracy stays inside the noise range (97.3%).

## Exercises

Run again after a change: the checks that compare with the page's numbers turn into `[FAIL]`; that is expected, read how the numbers themselves move.

1. Change the 300 in `STEPS = 40 if QUICK else 300` to 600: do the two rows with 1-2 examples per intent get better, or does the model just memorise them?
2. Add 0.05 to `rates = (0.2,) if QUICK else (0.1, 0.2, 0.4)`: how few wrong labels does it take before the drop in accuracy exceeds the 8.7-point range measured in section 5?
3. Add 10 to `for n in (5, 8, 13)` in the contamination check of section 4: how does it trade the 'only a 10-word stretch left' case against false alarms?

## Notes

- The task is a synthetic intent classification, fine-tuned on a byte-level model of 842,496 parameters; the scale is small and a change of seed alone moves accuracy by 8.7 points, so the lab measures the noise floor before talking about differences. The page's numbers come from a real 0.5B model; the lab tests the method and the trends.

## Output

The complete runs below were made on an Apple M2 Max (12 cores, CPU only) with Python 3.12 and PyTorch 2.15.0. On another machine the numbers that involve training can differ in the last digits and timings depend on the machine; the checks use thresholds and tolerances, not bit-for-bit equality.

<details>
<summary>Output of the full run (`labs/post-training/sft-data/run.py`)</summary>

```text
# SFT data: the loss mask, a data health check, and what data size, noise, masking and replay do
  course: https://dmxsf.com/wiki/post-training/sft-data/
          https://dmxsf.com/en/wiki/post-training/sft-data/

## 1. One sample: ids, labels and the -100 mask
  message 'hi, my credit card is capped today' -> card_limit
  95 prompt tokens, 12 answer tokens (supervised share 11%)
  position  token  label
  --------  -----  -----
        92    'n'   -100
        93    't'   -100
        94    ':'   -100
        95    ' '     32
        96    'c'     99
        97    'a'     97
        98    'r'    114
  [ok] every prompt position is -100 and every answer position carries its token
  [ok] the stop token (newline) is supervised: it is the only signal for where to stop

  the model fine-tuned on 20 messages per intent, asked in other words:
  prompt at inference time           accuracy  valid label
  ---------------------------------  --------  -----------
                 the trained prompt     99.0%       100.0%
  one extra space before the answer     99.0%       100.0%
      another wording of the prompt     85.0%        91.3%
  (one model, one run: the pattern is the point, not the exact gap. The template is part of
   the data and should be versioned with it.)

## 2. What the loss averages over
  a batch of 8 examples, answers of 10 to 18 bytes; the base model's loss on it:
  averaging                          loss (nats)
  ---------------------------------  -----------
     per supervised token (default)       4.5944
  per sequence, then over sequences       4.7516
  prompt not masked (all positions)       4.1014
  [ok] the default loss is the sum over supervised positions divided by their count  (4.594379 vs 4.594379)
  [ok] unequal answer lengths make the two averages differ (equal lengths would not)  (4.5944 vs 4.7516)

## 3. Packing and padding
  four speeches (32, 18, 32, 24 bytes) packed into one row of 106:
  packing                                          max |logit difference| vs four separate passes
  -----------------------------------------------  ----------------------------------------------
          block-diagonal mask + positions restart                                         1.3e-05
  plain concatenation (no mask, positions run on)                                         9.6e+00
  [ok] packing with a document mask and restarted positions equals separate passes  (1.3e-05)
  [ok] plain concatenation changes the numbers: documents see each other  (9.6e+00)

  lengths of the 7,222 speeches in TinyShakespeare (cut at 1,024 bytes): mean 148, median 83, longest 1024
  padding share of all positions  random batches  grouped by length
  ------------------------------  --------------  -----------------
                        batch 16           75.8%               6.2%
                        batch 64           83.5%               6.8%
     packed into 1,024-byte rows            0.1%                  -
  [ok] random batches waste far more positions than length-grouped ones  (76% vs 6% at batch 16)
  [ok] packing (with the mask above) wastes less than either  (0.1%)

## 4. Health check: rules, near-duplicates, contamination (planted ground truth)
  540 records made of Shakespeare passages; planted defects:
  rule                 flagged  planted found  planted  flagged, not planted
  -------------------  -------  -------------  -------  --------------------
         empty output       10             10       10                     0
    output <= 2 words       25             25       25                     0
     placeholder text       25             25       25                     0
  input is only a URL       10             10       10                     0
  refusal-like phrase       20             15       15                     5
  [ok] each rule finds every defect planted for it
  'output <= 2 words' counts the 15 correct one-word answers as well: rules only produce candidates, and
  a person has to read them. The refusal rule also flags 5 passages where a character simply says 'I cannot'.

  pairs with Jaccard >= 0.8 by brute force over all 145,530 pairs: 83 (40 planted);
  MinHash + LSH (16 bands x 4 rows) found 83 of them and 0 pairs that are not
  [ok] LSH finds (almost) every pair brute force finds, and invents none  (83 of 83)

  the LSH S-curve: how often two texts with a given Jaccard share a band
  Jaccard  measured  1-(1-J^4)^16
  -------  --------  ------------
     0.30     0.085         0.120
     0.50     0.640         0.655
     0.60     0.905         0.891
     0.69     0.993         0.986
     0.80     1.000         1.000
  [ok] the measured candidate rate follows the S-curve formula  (400 trials per row; every row within three binomial standard deviations (+0.01))

  contamination: 60 'benchmark questions'; 20 copied verbatim into the training text, 20 with
  every 8th word replaced, 20 with only a 10-word stretch left; 'clean' questions were never copied
  n-gram  verbatim found  edited found  10-word overlap found  clean flagged
  ------  --------------  ------------  ---------------------  -------------
       5           20/20         20/20                  20/20           2.0%
       8           20/20         14/20                  20/20           0.0%
      13           20/20          0/20                   1/20           0.0%
  [ok] 13-grams find every verbatim copy and miss the edited ones  (verbatim 20/20, edited 0/20)
  [ok] shorter n-grams catch the partial overlaps that 13-grams (almost) all miss  (10-word overlap found: n=8 20/20, n=13 1/20)
  the price of a short n: clean questions flagged 2.0% at n=5, 0.0% at n=8, 0.0% at n=13.

## 5. Experiment: how much does a run vary when nothing is changed?
  run              accuracy  shakespeare bits/byte
  ---------------  --------  ---------------------
          default     99.0%                   3.73
  training seed 1     93.0%                   3.77
  training seed 2     90.3%                   3.78
    data subset 2     97.7%                   3.82
    data subset 3     94.3%                   3.83
  accuracy 94.9% on average, range 90.3%-99.0%; 300 test messages, so one message is 0.33%.
  The pretrained model scores 2.31 bits/byte on held-out Shakespeare: the 'forgetting' baseline.
  Differences below the range are not findings.

## 6. Experiment: examples per intent (300 steps each, all else equal)
  per intent  examples  accuracy  valid label  shakespeare bits/byte
  ----------  --------  --------  -----------  ---------------------
           1        12     18.0%         100%                   3.56
           2        24     18.0%         100%                   3.63
           5        60     40.3%         100%                   3.79
          10       120     77.0%         100%                   3.80
          20       240     99.0%         100%                   3.73
          50       600    100.0%         100%                   3.71
  [ok] accuracy rises with data, and each step up past 20 per intent is worth less than the first ones  (18%  ->  18%  ->  40%  ->  77%  ->  99%  ->  100%)

## 7. Experiment: wrong labels
  labels replaced  accuracy  shakespeare bits/byte
  ---------------  --------  ---------------------
               0%     99.0%                   3.73
              10%     84.0%                   3.79
              20%     53.0%                   3.81
              40%     42.0%                   3.89
  [ok] every noise level costs more than the run-to-run range, and more noise costs more  (drops 10.9 pts, 41.9 pts, 52.9 pts; range 8.7 pts)
  about 1.32 accuracy points per 1% of wrong labels at 40%; the page's 0.5B model lost 0.4

## 8. Experiment: loss on the prompt, and replaying general text
  setting                                           accuracy  shakespeare bits/byte  optimizer steps      
  ------------------------------------------------  --------  ---------------------  ---------------------
                  default: loss on the answer only     99.0%                   3.73                    300
                        loss on the whole sequence     24.3%                   4.33                    300
  answer-only + every other step plain Shakespeare     97.3%                   2.32  600 (300 of them SFT)
  (the pretrained model: 2.31 bits/byte before any fine-tuning)
  [ok] replaying pretraining-style text keeps held-out Shakespeare much closer to the base model  (2.32 vs 3.73 bits/byte (base 2.31))
  [ok] ...without costing target accuracy beyond the run-to-run range (same number of SFT steps)  (97.3% vs runs 90.3%-99.0%)
  loss on the prompt: accuracy +74.7 points for the answer-only default (run-to-run range 8.7).
  A likely reason, not isolated here: the prompt is about eight times longer than the answer,
  so it dominates the loss and the 300 steps. On the page's 0.5B model the same switch moved
  accuracy by 1.3 points (85.8% to 84.5%).

All checks passed. (206.1s)
```

</details>
