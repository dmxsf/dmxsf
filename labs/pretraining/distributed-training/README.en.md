# Distributed training: gradients, ZeRO, tensor and pipeline parallelism

Course pages: [Distributed Training: Data, Tensor and Pipeline Parallelism](https://dmxsf.com/en/wiki/pretraining/distributed-training/)  
[中文](README.md)

Built and measured in a few processes on one machine: data-parallel gradient averaging, the ring all-reduce, data parallelism, ZeRO-1 and ZeRO-3 trained side by side (same weights, measured memory and communication), Megatron-style tensor parallelism, and pipeline schedules.

## Run

```bash
uv run python labs/pretraining/distributed-training/run.py            # full run, about 15 seconds
uv run python labs/pretraining/distributed-training/run.py --quick    # a shrunk smoke test, about 15 seconds
```

You can also write `uv run dmxsf run distributed-training` (and add `--quick`). `--seed N` changes the random seed, default 0. The script prints its results section by section and checks the course page's claims one by one with `[ok]` / `[FAIL]`: the exit status is 0 when all pass and 1 when any fails; checks that only make sense in the full run are marked `[skip]` under `--quick`.

## Code

| File | What it holds |
| --- | --- |
| `labs/pretraining/distributed-training/run.py` | the lab script, numbered sections |
| `src/dmxsf/pretraining/parallel.py` | The arithmetic of parallel training: ring all-reduce, pipeline schedules, communication costs |
| `src/dmxsf/pretraining/distributed.py` | Parallel training that really runs: several processes on one CPU, talking through gloo |

Unit tests: `uv run pytest tests/test_pretraining_parallel.py tests/test_pretraining_distributed.py`

## What to look for

- Section 1: data parallelism with 4 real processes (gloo). With equal lengths every way of averaging agrees with one process (error about 2.5e-07, float32 rounding); with unequal lengths 'every rank averages, then average the means' (what PyTorch DDP and gradient accumulation both do) is off by 36.5%, and dividing by all valid tokens brings it back to 1.6e-07; a real 2-layer decoder is biased too (37.9%).
- Section 2: a ring all-reduce sends 2(n-1)/n times the gradient per rank, nearly independent of the number of ranks, but needs 2(n-1) steps of latency.
- Section 3: data parallelism, ZeRO-1 and ZeRO-3 end 10 steps with exactly the weights of one process (6.9e-15, float64 rounding); each rank keeps 4.0 / 2.5 / 1.0 times the number of parameters; ZeRO-1 moves as much as data parallelism (1.5), ZeRO-3 1.5 times as much (2.25).
- Section 4: tensor parallelism splits the MLP first by columns, then by rows: error about 1e-15 and one all-reduce; activating the two partial sums and adding is wrong (error 1.2). A whole block (GQA, RoPE) split 2 or 4 ways gives the same result, with 2 all-reduces forward and 2 backward per layer.
- Sections 5-6: the pipeline bubble (p-1)/(m+p-1) matches the formula in the simulation, 1F1B has the same bubble as GPipe and changes the memory (live micro-batches on the first stage 8 -> 4); the bandwidths and FLOP rates of the communication section are illustrative, not measured on this machine.

## Exercises

Run again after a change: the checks that compare with the page's numbers turn into `[FAIL]`; that is expected, read how the numbers themselves move.

1. Make `UNEQUAL` more extreme, for example `[32, 2, 2, 2, 2, 2, 2, 2]`: how large does the bias of method A become? (A sequence of length 1 has no next token to predict, so that rank would average 0 / 0: use 2 at least.)
2. Add 8 to `for t in (2,) if quick else (2, 4)` in section 4: what error do you get, and why (look at `n_kv_head=4` in `TP_CFG`)? Change `n_head=8` and `n_kv_head=4` in `TP_CFG` of `src/dmxsf/pretraining/distributed.py` to 16 and 8 and try again: does a block split 8 ways still match one process?

## Output

The complete runs below were made on an Apple M2 Max (12 cores, CPU only) with Python 3.12 and PyTorch 2.15.0. On another machine the numbers that involve training can differ in the last digits and timings depend on the machine; the checks use thresholds and tolerances, not bit-for-bit equality.

<details>
<summary>Output of the full run (`labs/pretraining/distributed-training/run.py`)</summary>

```text
# Distributed training: gradients, ZeRO, tensor and pipeline parallelism
  course: https://dmxsf.com/wiki/pretraining/distributed-training/
          https://dmxsf.com/en/wiki/pretraining/distributed-training/

## 1. How data-parallel gradients are averaged (4 processes, gloo)
  8 sequences, 2 per process; error = |g - g_ref| / |g_ref| against one process, the whole batch,
  loss divided by the number of valid next-token predictions in it

  the page's toy model (embedding + linear):
  way of averaging                equal lengths  lengths [16, 14, 12, 12, 10, 8, 8, 4]
  ------------------------------  -------------  -------------------------------------
     A: mean of the ranks' means       2.45e-07                               3.65e-01
   B: divide by all valid tokens       2.45e-07                               1.59e-07
                  DDP as shipped       2.40e-07                               3.65e-01
  DDP, loss x world / all tokens       2.47e-07                               1.60e-07
           gradient accumulation       2.43e-07                               3.65e-01

  a real 2-layer decoder:
  way of averaging                equal lengths  lengths [16, 14, 12, 12, 10, 8, 8, 4]
  ------------------------------  -------------  -------------------------------------
     A: mean of the ranks' means       1.55e-07                               3.79e-01
   B: divide by all valid tokens       1.55e-07                               1.12e-07
                  DDP as shipped       1.58e-07                               3.79e-01
  DDP, loss x world / all tokens       1.57e-07                               1.15e-07
           gradient accumulation       1.55e-07                               3.79e-01

  valid predictions per rank [28, 22, 16, 10]: they should weigh 36.8%, 28.9%, 21.1%, 13.2%; averaging the means gives each 25%
  [ok] equal lengths: every way of averaging agrees with the reference to float32 rounding  (largest error 2.5e-07; the page reports 2.45e-07)
  [ok] the page's numbers: unequal lengths bias the average of the means by 36.5%  (0.365)
  [ok] dividing by all valid tokens brings the error back to rounding  (1.6e-07 on the toy model, 1.1e-07 on the decoder; the page reports 1.60e-07)
  [ok] PyTorch's DDP is method A, and 'loss x world / all tokens' is method B  (DDP 0.365 against 0.365; the recipe 1.6e-07)
  [ok] gradient accumulation has the same bias: time or space, the means are averaged the same way  (0.365 on the toy model, 0.379 on the decoder)
  [ok] a real decoder is biased too  (0.379)
  [ok] the rank with 10 valid tokens counts 25% instead of 13.2%, 1.9 times too much  (13.2%, x1.90)

## 2. Ring all-reduce, simulated
  n   all ranks hold the sum  sent per rank / size  2(n-1)/n  steps
  --  ----------------------  --------------------  --------  -----
   2                    True                1.0000    1.0000      2
   4                    True                1.5000    1.5000      6
   8                    True                1.7500    1.7500     14
  16                    True                1.8750    1.8750     30
  [ok] each rank sends 2(n-1)/n of the gradient: about 2S however many ranks, but 2(n-1) steps of latency

## 3. Data parallelism, ZeRO-1 and ZeRO-3 side by side (4 processes, 10 steps)
  a 139,584-parameter Llama-style decoder in float64, AdamW without weight decay, batch of 16 sequences x 64 bytes of TinyShakespeare, 4 ranks
  values held per rank, as a multiple of the number of parameters (what each rank keeps between steps):
                 weights  gradients  Adam state  total  sent / step  collectives per step            max |w - ref|
  -------------  -------  ---------  ----------  -----  -----------  ------------------------------  -------------
  data parallel    1.000      1.000       2.000  4.000        1.500                    1 all-reduce       6.82e-15
         ZeRO-1    1.000      1.000       0.500  2.500        1.500  1 reduce-scatter, 1 all-gather       6.82e-15
  ZeRO-3 / FSDP    0.250      0.250       0.500  1.000        2.250  4 reduce-scatter, 8 all-gather       6.87e-15
  (sent / step: elements a ring implementation sends per rank, in units of the 139,584 parameters)
  [ok] all three end with the weights of one process on the whole batch, on every rank  (largest difference 6.9e-15 (float64 rounding))
  [ok] ZeRO-1 keeps full weights and gradients but 1/n of the optimizer state  (Adam state 69,792 against 279,168)
  [ok] ZeRO-3 keeps 1/n of all three  (weights 34,896 of 139,584)
  [ok] ZeRO-1 moves exactly as much as data parallelism (reduce-scatter + all-gather = all-reduce)  (1.500 x psi against 1.500 x psi, 2(n-1)/n = 1.500)
  [ok] ZeRO-3 moves 1.5 times as much: all-gather going forward and coming back, reduce-scatter  (1.500x; each of the 4 units is gathered twice and its gradient scattered once)
  [ok] the same loss curve in all three  (loss 5.5614 -> 3.8035)

  the same sharding in mixed precision (2 + 2 + 12 bytes per parameter), the page's example:
  7.5B on 64 GPUs  GiB per GPU  GB (the ZeRO paper's unit)
  ---------------  -----------  --------------------------
    data parallel       111.76                      120.00
           ZeRO-1        29.25                       31.41
           ZeRO-2        15.50                       16.64
           ZeRO-3         1.75                        1.88
  [ok] 7.5B on 64 GPUs: 111.8, 29.2, 15.5, 1.75 GiB (the paper: 120, 31.4, 16.6, 1.88 GB)  (111.76, 29.25, 15.50, 1.75)
  [ok] 70B on 80 GiB cards: ZeRO-1 and ZeRO-2 never fit; ZeRO-3 needs 14 cards; 8 cards would need 130 GiB each  (130.4 GiB on 8 cards)

## 4. Tensor parallelism: first column, then row
  split of the MLP  Y = GeLU(XA) B                     max error
  ---------------------------------------------------  ---------
                         A by columns, B by rows, t=2   2.22e-15
                         A by columns, B by rows, t=4   2.66e-15
                         A by columns, B by rows, t=8   3.11e-15
              A by rows, activate each part, then add   1.20e+00
  A by rows, add, then activate (one more all-reduce)   1.78e-15
  [ok] column then row is exact: one all-reduce at the end of the MLP  (largest error 3.1e-15; the page reports 2.2e-15 ... 3.1e-15)
  [ok] activating partial sums is wrong (error of the order of the output)  (1.2e+00 against 1.8e-15 when the sum comes first; the page: 1.2e+00 and 1.8e-15)

  a whole pre-norm block (GQA attention with 8 query / 4 KV heads, RoPE, GELU MLP), 2 layers, float64, real processes:
  t  output    input grad  weight grads  all-reduces per layer (fwd + bwd)  matrix params per rank
  -  --------  ----------  ------------  ---------------------------------  ----------------------
  2  8.88e-16    3.69e-16      9.05e-16                              2 + 2        45,056 of 90,112
  4  8.88e-16    3.69e-16      1.16e-15                              2 + 2        22,528 of 90,112
  [ok] split over t ranks, the block gives the same output, input gradient and every weight gradient  (relative errors at rounding level, over every rank)
  [ok] the page's count: 2 all-reduces in the forward pass and 2 in the backward pass, per layer  (4 + 4 for 2 layers)
  [ok] every rank holds 1/t of the matrix weights

## 5. Pipeline schedules: bubbles and activation memory (1 time unit forward, 2 backward)
  p  m   step time  idle (simulated)  (p-1)/(m+p-1)  idle, 2 chunks per GPU  GPipe: live on stage 1  1F1B
  -  --  ---------  ----------------  -------------  ----------------------  ----------------------  ----
  4   4         21             42.9%          42.9%                   27.3%                       4     4
  4   8         33             27.3%          27.3%                   15.8%                       8     4
  4  16         57             15.8%          15.8%                    8.6%                      16     4
  4  32        105              8.6%           8.6%                    4.5%                      32     4
  8  32        117             17.9%          17.9%                    9.9%                      32     8
  [ok] the simulated time is (m+p-1)(t_fwd+t_bwd) and the idle share (p-1)/(m+p-1): the page's 21, 33, 57, 105, 117
  [ok] 1F1B has exactly the same bubble as GPipe
  [ok] what 1F1B changes is memory: p=4, m=8 keeps 8 micro-batches alive on the first stage under GPipe, 4 under 1F1B  (live activations per stage: GPipe [8, 8, 8, 8], 1F1B [4, 3, 2, 1])
  [ok] cutting the pipeline finer needs more micro-batches: m=32 gives 8.6% with 4 stages and 17.9% with 8
  [ok] exercise 2: 8 stages and at most 10% idle need at least 63 micro-batches  (10.00% at 63, 10.14% at 62)

## 6. What the communication costs for a 7B model (illustrative hardware numbers)
  Llama 2 7B shapes (6.738B parameters, 32 layers, hidden 4096, sequence 4096), BF16 gradients;
  bandwidths from the DeepSeek-V3 paper (NVLink 160 GB/s, InfiniBand 50 GB/s), an H100 at 989 TFLOPs and
  MFU 40%. These do not come from one machine: they give the orders of magnitude, nothing measured.
  data parallel, 64 GPUs  GB per GPU per step
  ----------------------  -------------------
   DDP / ZeRO-1 / ZeRO-2                 26.5
           ZeRO-3 / FSDP                 39.8

  tensor parallelism, one micro-batch of one sequence (ms; NVLink / InfiniBand are the transfer times):
  t  GB per GPU  NVLink ms  InfiniBand ms  compute ms
  -  ----------  ---------  -------------  ----------
  2         4.3         27             86         209
  4         6.4         40            129         105
  8         7.5         47            150          52

  pipeline parallelism: 67.1 MB per micro-batch across each stage boundary

  fixed global batch of 4M tokens, gradient sync 0.53 s (an upper bound on scaling):
  GPUs  tokens per GPU  compute s  exposed comm s  efficiency
  ----  --------------  ---------  --------------  ----------
    64           65536       6.70            0.00      100.0%
   128           32768       3.35            0.00      100.0%
   256           16384       1.67            0.00      100.0%
   512            8192       0.84            0.00      100.0%
  1024            4096       0.42            0.25       62.5%
  the backward pass hides the sync from about 7,788 tokens per GPU and step
  [ok] the page: 26.5 GB (DDP) and 39.8 GB (ZeRO-3), a factor 1.5
  [ok] the page: tensor parallelism at t=8 needs 7.5 GB, 47 ms on NVLink against 52 ms of compute, 150 ms on InfiniBand
  [ok] the page: a pipeline boundary moves 67.1 MB per micro-batch, two orders of magnitude less than t=8  (112x)
  [ok] the page: efficiency stays at 100% to 512 GPUs and falls to 62.5% at 1024; about 7,788 tokens per GPU hide the sync
  [ok] exercise 1: ring all-reduce of the 13.476 GB of gradients sends 23.6 GB per GPU on 8 GPUs and 26.9 GB on 1024  (14% more for 128 times the GPUs)
  [ok] exercise 4: doubling the hidden size halves tensor parallelism's communication per FLOP  (16 (t-1)/t h bytes against 72 h^2 FLOPs per token and layer)

All checks passed. (15.5s)
```

</details>
