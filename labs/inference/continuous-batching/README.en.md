# Serving: continuous batching, paged KV memory, prefix cache, TTFT, capacity

Course pages: [Continuous Batching and PagedAttention: How a Server Handles Many Requests at Once](https://dmxsf.com/en/wiki/inference/continuous-batching/) · [LLM Inference Serving: Latency Metrics, Measuring, and Choosing an Engine](https://dmxsf.com/en/wiki/inference/serving/)  
[中文](README.md)

Roofline and exact batched decoding, static against continuous scheduling on paper and on a real model, a paged KV pool with copy-on-write, a hash-based prefix cache, and what a request costs (TTFT, TPOT, capacity). Read the ratios, not the milliseconds.

## Run

```bash
uv run python labs/inference/continuous-batching/run.py            # full run, about 10 seconds
uv run python labs/inference/continuous-batching/run.py --quick    # a shrunk smoke test, a few seconds
```

You can also write `uv run dmxsf run continuous-batching` (and add `--quick`). `--seed N` changes the random seed, default 0. The script prints its results section by section and checks the course page's claims one by one with `[ok]` / `[FAIL]`: the exit status is 0 when all pass and 1 when any fails; checks that only make sense in the full run are marked `[skip]` under `--quick`.

This lab uses cached small models: the byte-level base model (842,496 parameters). The first run trains them on TinyShakespeare and caches them in `~/.cache/dmxsf` (set `DMXSF_CACHE` to move it); later runs read them back.

## Code

| File | What it holds |
| --- | --- |
| `labs/inference/continuous-batching/run.py` | the lab script, numbered sections |
| `src/dmxsf/inference/serving.py` | Serving: batched decoding over slots, iteration-level scheduling, paged KV memory, prefix caching |

Unit tests: `uv run pytest tests/test_inference_serving.py`

## What to look for

- Section 1: why batching pays: the H100's roofline is 295 FLOP per byte; Qwen2.5-7B reads 224 MiB of KV for a 4,096-token sequence against 14.2 GiB of weights, so 65 sequences read as much KV as weights. On this CPU one decode step with 16 sequences gives 11.4 times the tokens per second of one (each request runs at 0.71 of its solo speed); five sequences of different lengths decoded in one batch match each decoded alone (largest logit difference 1.5e-05).
- Section 2: the scheduler on paper (exact): the page's 48 requests take 52.5 s and 88.6 tokens/s under continuous batching, 68.6 s and 67.9 tokens/s under static batching. Where the time goes: prefill takes 16.5 s in both; decoding reaches 81% of the ideal when continuous and 56% when static (each batch waits for outputs of 218, 453 and 192 tokens). The bullets under the page's widget are recomputed in the same simulator: with equal outputs continuous batching has no edge, with batch-independent step times it saves 25%.
- Section 3: the same loop on a real model: given only this machine's own step table and prefill time, the simulator predicts both makespans within 15% (this run: continuous -3.6%, static -4.2%); every request gets exactly the bytes it would get alone; the makespan of continuous batching is 0.67 of static and the mean completion 0.65; counted in steps alone: 95 against 170, 44% fewer, which comes from the schedule and not the hardware.
- Section 4: the KV memory ledger: Qwen2.5-7B takes 57,344 bytes per token; on an 80 GiB card, reserving 8,192 tokens per sequence fits 123 sequences, paging in 16-token blocks with 1,000 tokens each fits 999. The page's 48 requests hold 10,800 tokens: reserving the maximum uses 2.7% of the memory, paging 97.0% (7.0 tokens wasted per request, never more than 15). A block table gathers blocks that are not next to each other into the same attention (bit for bit), and parallel sampling shares the prompt's blocks (11 blocks against 20).
- Section 5: the prefix cache: with the document first, the first request computes 112 bytes and the next ones reuse 96 and compute 16, prefilling 2.2 times faster (1.44 against 3.21 ms); with 8 changing bytes in front nothing is ever reused; reusing the cache does not change the logits (largest difference 0).
- Section 6: what a request costs: end-to-end = queue + prefill + (N - 1) x TPOT, the formula within 10% of the measurement; prefill is more than 10 times faster per byte than decoding one sequence. With identical code the benchmark's B / A ratio strayed by up to 10.9% from 1 over a few trials: a difference between two real configurations smaller than that is not one this benchmark can see; Little's law L = lambda x W holds in the simulated queue (relative gap 0.01%).

## Exercises

Run again after a change: the checks that compare with the page's numbers turn into `[FAIL]`; that is expected, read how the numbers themselves move.

1. In section 1 change the 64 in `context, reps = 64, 8 if QUICK else 30` to 120 (a longer context per sequence, more KV to read): do 16 sequences still give about 11 times the throughput of one?
2. Change `PAGE_STALL = 0.343` to 1.0 (how long a 128-token prompt stalls the whole batch): does the edge of continuous over static batching grow or shrink?

## Notes

- The milliseconds and throughputs of sections 1, 3, 5 and 6 are timings of this machine and differ a little from run to run (the checks look only at ratios and directions, with wide thresholds); this lab is the most sensitive to the load of the machine, so do not run anything else CPU-heavy while it runs. The same lab also covers the serving page, linked above.

## Output

The complete runs below were made on an Apple M2 Max (12 cores, CPU only) with Python 3.12 and PyTorch 2.15.0. On another machine the numbers that involve training can differ in the last digits and timings depend on the machine; the checks use thresholds and tolerances, not bit-for-bit equality.

<details>
<summary>Output of the full run (`labs/inference/continuous-batching/run.py`)</summary>

```text
# Serving: continuous batching, paged KV memory, prefix cache, TTFT, capacity
  course: https://dmxsf.com/wiki/inference/continuous-batching/
          https://dmxsf.com/en/wiki/inference/continuous-batching/
  course: https://dmxsf.com/wiki/inference/serving/
          https://dmxsf.com/en/wiki/inference/serving/

## 1. Why batching pays: the roofline, and a batched decode step that is exact
  H100 SXM: 989 TFLOPS / 3.35 TB/s = 295 FLOP per byte. A linear layer applied to B tokens
  does B FLOP for every byte of BF16 weight it reads, so below B of about 295 a step is paid for by
  reading the weights and extra sequences ride along almost free. The KV cache cannot be shared that way:
  Qwen2.5-7B reads 224 MiB of it per 4,096-token sequence against 14.2 GiB of weights, so 65
  sequences read as much KV as weights.
  [ok] 295 FLOP/B; Qwen2.5-7B: 224 MiB of KV per 4,096-token sequence, 14.2 GiB of weights, 65 sequences  (295.2, 224 MiB, 14.19 GiB, 64.8)

  one decode step on the CPU, 64 tokens of context per sequence (median of 28):
  batch  step ms  total tok/s  per-request tok/s  total vs batch 1
  -----  -------  -----------  -----------------  ----------------
      1     1.25          800                800              1.0x
      2     1.33        1,505                753              1.9x
      3     1.39        2,163                721              2.7x
      4     1.41        2,838                710              3.5x
      6     1.46        4,099                683              5.1x
      8     1.70        4,714                589              5.9x
     12     1.62        7,393                616              9.2x
     16     1.76        9,087                568             11.4x
     24     2.45        9,780                407             12.2x
     32     2.85       11,213                350             14.0x
  [ok] batching pays on this machine too: 16 sequences give more than 3x the tokens per second of one  (11.4x the total throughput; each request runs at 0.71 of its solo speed)
  [ok] five sequences of 5, 13, 29, 41 and 60 tokens decoded in one batch match each decoded alone  (largest logit difference 1.5e-05; same next byte in 50 of 50 steps)

## 2. The page's scheduler on paper: static against continuous batching (exact)
  48 requests, output tokens: min 15, median 61, mean 97, max 453, total 4,656
  [ok] the page's workload: min 15, median 61, mean 97, max 453, total 4,656  ((15, 61.0, 97, 453, 4656))

  16 slots; a 128-token prompt stalls the whole batch for 0.343 s; step times from the page's table:
  mode        makespan s  tok/s  mean done s  p90 done s
  ----------  ----------  -----  -----------  ----------
  continuous        52.5   88.6         29.9        47.0
      static        68.6   67.9         37.7        64.0
  [ok] continuous 52.5 s, 88.6 tok/s, 29.9, 47.0 and static 68.6, 67.9, 37.7, 64.0, as printed on the page  (continuous [52.5, 88.6, 29.9, 47.0], static [68.6, 67.9, 37.7, 64.0])

  where the time goes: 4,656 tokens at 16 sequences of 100.3 ms would take 29.2 s; prefill takes 16.5 s in both schedules.
  decoding takes 36.1 s (81% of ideal) when continuous and 52.1 s (56%) when static. Static batches
  of 16 wait for outputs of 218, 453 and 192 tokens; in the continuous schedule the last request starts at 37.9 s,
  then 14.6 s of drain while the batch shrinks.
  [ok] ideal 29 s, prefill 16.5 s, decoding 36 s against 52 s (81% against 56% of ideal); batches wait for 218, 453, 192; last start 38 s, 14.6 s drain  (29.2, 16.5, 36.1, 52.1, [218, 453, 192], 37.9, 14.6)

  the bullets under the page's widget, recomputed (same simulator):
  scenario                         continuous s  tok/s  static s  tok/s  continuous saves
  -------------------------------  ------------  -----  --------  -----  ----------------
               the page's setting          52.5     89      68.6     68               23%
          every output 100 tokens          46.6    103      46.6    103                0%
   step time independent of batch          28.4    164      38.0    123               25%
  step time proportional to batch         132.4     35     132.4     35                0%
     1,024-token prompts (2.18 s)         140.7     33     156.7     30               10%
                          2 slots         110.6     42     118.6     39                7%
  [ok] equal outputs: both 46.6 s. Batch-independent steps: 164 and 123 tok/s, 25% saved. Proportional steps: no difference. 1,024-token prompts: 105 s of prefill, 141 s, 10%. Two slots: 7%  (46.6/46.6, 164/123, 132.4/132.4, 140.7 (10.2%), 6.7%)
  [ok] the page's exercise: 4 slots, one request of 100 steps and nine of 10 take 120 steps static, 100 continuous  ([120.0, 100.0])

## 3. The same loop on a real model
  the page's workload scaled to this model's 128 positions: 48 requests, 16-byte prompts, outputs of 3 to 91 bytes (931 in all),
  16 slots, greedy decoding, median of 3 alternating runs. The simulation uses this machine's own step table (section 1) and
  its own prefill time (1.09 ms for 16 bytes):
  mode                   makespan ms  bytes/s  mean done ms  p90 done ms  mean TTFT ms
  ----------  ---------  -----------  -------  ------------  -----------  ------------
  continuous       real          207    4,495           102          163            54
  continuous  simulated          200    4,662            97          155              
      static       real          310    3,007           156          280           116
      static  simulated          297    3,139           149          266              
  [ok] every request gets exactly the bytes it would get decoded alone, in both schedules and every run  (48 requests, 3 runs each)
  [ok] continuous batching finishes the batch of requests sooner than static batching, and their average sooner  (makespan 0.67 of static, mean completion 0.65 of static)
  [ok] the simulator, fed only the measured step table and prefill time, predicts both makespans within 15%  (continuous: -3.6%, static: -4.2%)
  counted in steps alone, with no clock involved: continuous 95 decode steps, static 170, so 44% fewer; that saving comes from the schedule, not from the hardware.

## 4. KV memory: the ledger, reserving against paging, the block table
  quantity (Qwen2.5-7B, 80 GiB card)               value                    
  -----------------------------------------------  -------------------------
                               KV cache per token               57,344 bytes
         a 4,096-token slot / a 65,536-token pool         224 MiB / 3.50 GiB
                        parameters / FP16 weights         7.616 B / 14.2 GiB
   KV pool (90% of memory, 4 GiB for activations)   53.8 GiB = 1.01 M tokens
   sequences when every one reserves 8,192 tokens                        123
  sequences in 16-token blocks, 1,000 tokens each                        999
      OPT-13B per token (2 x 40 x 5120 x 2 bytes)  819,200 bytes, 14.3x more
  [ok] 57,344 bytes per token, 224 MiB, 3.50 GiB, 7.616 B and 14.2 GiB, 53.8 GiB = 1.01 M tokens, 123 against 999 sequences, OPT-13B 14x  (57344, 53.81 GiB, 123, 999, 14.29)
  [ok] exercise: Llama 3 8B needs 131,072 bytes per token, 1 GiB per 8,192-token request, and a 24 GiB card holds 9  (131,072 bytes, 9.04 GiB left, 9.04 requests)

  the page's 48 requests hold 10,800 tokens in all. Reserving 8,192 for each puts 2.7% of the reserved
  memory to use; 16-token blocks allocated as needed put 97.0% to use, a waste of 7.0 tokens per request (never more than 15).
  [ok] paged allocation wastes under 16 tokens per request and keeps over 90% of memory in use; reserving the maximum keeps under 5%  (97.0% against 2.7%)

  three sequences of 10, 19 and 23 tokens grow interleaved in a pool of 16 blocks of 4 tokens; their block tables:
  sequence  tokens  block table (physical blocks)
  --------  ------  -----------------------------
         0      10                      [0, 3, 6]
         1      19               [1, 4, 7, 9, 11]
         2      23          [2, 5, 8, 10, 12, 13]
  [ok] the blocks of a sequence are not next to each other, yet gathering them gives back the dense keys and values, and attention is bit-for-bit the same  (14 blocks in use for 52 tokens; a dense layout reserving 32 tokens each would take 24)

  parallel sampling: four samples of 5 tokens continue one 12-token prompt. Sharing the prompt's blocks they use 11 blocks,
  20 if each owned a copy. A fork of a sequence whose last block is half full copies that block the first time either side writes
  into it (1 copy, the original is untouched).
  [ok] sharing the prompt: 3 prompt blocks plus 2 per sample, no copies; writing into a shared half-full block copies it once and leaves the original unchanged  (11 blocks against 20)
  [ok] llama.cpp: -c 65536 -np 16 gives 4,096 tokens per slot, so a 6,000-token request is refused, unless the cache is unified (65,536)  (4096 against 65536)
  [ok] exercise: -c 32768 -np 8 gives 4,096 per slot, so 20,000 tokens are refused; unified or -np 1 gives 32,768; -c 160000 is needed to keep eight slots  (4096, 32768, 32768, 160000)

## 5. The prefix cache: the same beginning is computed once
  a 96-byte document (six 16-byte blocks) and a 16-byte question, four requests per case, each with a different question;
  B puts 8 bytes that change every time in front of the document. The cache starts empty in each case; times are the median of 9 repetitions:
  case                      request  reused  computed  prefill ms
  ------------------------  -------  ------  --------  ----------
     A doc first, cache on        1       0       112        2.99
     A doc first, cache on        2      96        16        1.42
     A doc first, cache on        3      96        16        1.46
     A doc first, cache on        4      96        16        1.43
  B header first, cache on        1       0       120        3.95
  B header first, cache on        2       0       120        3.69
  B header first, cache on        3       0       120        3.55
  B header first, cache on        4       0       120        3.72
    C doc first, cache off        1       0       112        3.40
    C doc first, cache off        2       0       112        3.08
    C doc first, cache off        3       0       112        3.23
    C doc first, cache off        4       0       112        3.11
  [ok] A: the first request computes all 112 bytes, the next three reuse the 96 of the document and compute 16; B: nothing is ever reused; C: nothing is reused  (A [(0, 112), (96, 16), (96, 16), (96, 16)], B (0, 120))
  [ok] reusing cached keys and values does not change the logits  (largest difference from a full prefill 0.0e+00 over 12 requests)
  [ok] a request that hits the cache prefills faster than one that does not  (1.44 ms against 3.21 ms (2.2x); 16 of 112 bytes computed, not 1/7 of the time: fixed costs per forward pass)
  [ok] the page's exercise: with 16-token blocks, a second request with the same 1,000-token system prompt reuses 992 tokens; one with a timestamp in front reuses 0; another tenant's salt reuses 0  (992, 0, 0)

## 6. What a request costs: TTFT, TPOT, the two regimes, capacity
  one slot, four identical requests at t = 0 (16-byte prompt, 40 new bytes). Alone, one request has a prefill of 1.34 ms and a TPOT of 1.11 ms:
  request  queue ms  prefill ms  TTFT ms  TPOT ms  end-to-end ms  queue + prefill + (N-1) TPOT, ms
  -------  --------  ----------  -------  -------  -------------  --------------------------------
        1       0.0        1.26      1.3     1.17           46.8                              44.6
        2      46.8        1.41     48.2     1.10           90.8                              91.3
        3      90.8        1.36     92.2     1.13          136.6                             135.4
        4     136.6        1.28    137.8     1.15          182.6                             181.1
  [ok] end-to-end = queue + prefill + (N - 1) x TPOT: with prefill and TPOT measured alone, the formula is within 10% for all four requests  (-4.7%, +0.6%, -0.8%, -0.8%)

  prefill speed against prompt length (decoding one sequence runs at 800 bytes/s):
  prompt bytes  prefill ms  bytes/s
  ------------  ----------  -------
             1        0.78    1,279
             4        0.89    4,497
            16        1.08   14,825
            64        2.10   30,536
           120        3.22   37,243
  [ok] prefill, which handles every prompt position in one pass, is more than 10x faster per byte than decoding one sequence, and gets faster per byte as the prompt grows  (37,243 against 800 bytes/s)
  the page's arithmetic                                        value                          
  -----------------------------------------------------------  -------------------------------
                    M2 Max decode ceiling, 400 GB/s / 4.68 GB                      85 tokens/s
                                42.3 tokens/s sustained reads  198 GB/s = 49% of the bandwidth
                                64.8 tokens/s when cool reads                   303 GB/s = 76%
                 exercise: 3,000-token prompt, 200 tokens out   TTFT 6.19 s, end to end 10.9 s
  exercise: 3.35 TB/s, FP16 weights 15.23 GB / Q4_K_M 4.68 GB      220 / 715.8 tokens/s, 3.25x
  [ok] 85 tokens/s ceiling, 303 GB/s = 76% at 64.8 tokens/s, 6.19 s and 10.9 s, 220 and 715 tokens/s (715.8, which the page truncates), 3.3x

  A and B are the same code (20 runs of 10 batched steps each, 5 trials). What does a benchmark say about their ratio B / A,
  depending on the order they run in?
  order            B / A in each trial            range         
  ---------------  -----------------------------  --------------
  A first, then B  0.998 0.997 0.909 0.891 0.995  0.891 to 0.998
  A, B, A, B, ...  0.996 1.001 0.999 0.989 0.989  0.989 to 1.001
  alternating gave the narrower range here, as the page advises.
  across all trials B / A strayed up to 10.9% from 1 with identical code: a difference between two real configurations
  smaller than that is not a difference this benchmark can see.

  Little's law in a simulated queue (Poisson arrivals, exponential service, 4 servers):
  utilization  arrivals/s  W: mean time in system  L: time-average number  lambda x W
  -----------  ----------  ----------------------  ----------------------  ----------
        0.375       1.501                   1.030                   1.546       1.546
        0.625       2.501                   1.214                   3.036       3.036
        0.875       3.502                   2.534                   8.873       8.874
  [ok] L = lambda x W: the time-average number of requests in the system equals the arrival rate times the mean time in it  (largest relative gap 0.01%)

  the page's capacity exercise: 17 tokens/s per user, 200 tokens per answer, 0.8 requests/s at the peak. One machine serves 4 at once
  (17.3 tokens/s each; at 6 it falls to 12.2), a request stays W = 0.47 + 199 / 17.3 = 12.0 s, so L = 0.8 x 12.0 = 9.6
  requests are in flight and 3 machines are needed; one machine sustains 0.33 requests/s.
  [ok] 4 requests per machine, W = 12.0 s, L = 9.6, 3 machines, 0.33 requests/s per machine  (4, 11.97, 9.58, 3)

All checks passed. (11.9s)
```

</details>
