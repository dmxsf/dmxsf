# Efficient attention

Course pages: [Efficient Attention: MQA, GQA, MLA and FlashAttention](https://dmxsf.com/en/wiki/architecture/efficient-attention/)  
[中文](README.md)

KV-cache formulas against the page's table and against real tensors, MLA caching a latent vector, FlashAttention's tiles and online softmax, sliding-window attention, and converting an MHA model to GQA by mean-pooling heads.

## Run

```bash
uv run python labs/architecture/efficient-attention/run.py            # full run, about 55 seconds
uv run python labs/architecture/efficient-attention/run.py --quick    # a shrunk smoke test, a few seconds
```

You can also write `uv run dmxsf run efficient-attention` (and add `--quick`). `--seed N` changes the random seed, default 0. The script prints its results section by section and checks the course page's claims one by one with `[ok]` / `[FAIL]`: the exit status is 0 when all pass and 1 when any fails; checks that only make sense in the full run are marked `[skip]` under `--quick`.

## Code

| File | What it holds |
| --- | --- |
| `labs/architecture/efficient-attention/run.py` | the lab script, numbered sections |
| `src/dmxsf/architecture/attention.py` | Scaled dot-product attention, multi-head / grouped-query attention and the causal mask |
| `src/dmxsf/architecture/mla.py` | Multi-head Latent Attention (MLA), the attention of DeepSeek-V2 and V3 |
| `src/dmxsf/inference/kv_cache.py` | KV caches: remember every layer's keys and values so each new token costs one step, not a re-run |

Unit tests: `uv run pytest tests/test_architecture_attention.py tests/test_architecture_mla.py tests/test_inference_kv_cache.py`

## What to look for

- Section 1: the course page's KV-cache table from the formulas, per token 512 / 128 / 320 / 504 / 67.5 / 68.6 KiB (Llama 2 7B, 3 8B, 3 70B, 3 405B, DeepSeek-V2, V3), at 128K tokens 64 / 16 / 40 / 63 / 8.44 / 8.58 GiB; DeepSeek-V3 as MHA would be 3.81 MiB per token, MLA is 1/56.9 of that.
- Section 2: the bytes measured on real cache tensors equal the formulas (MHA 4096 B, GQA-2 1024 B, MQA 512 B, MLA 1152 B).
- Section 3: MLA's absorption does not change the result (error about 1e-16 to 3e-07) and decoding caches only d_c + d_R = 80 numbers; the longer the cache the more absorbing pays, 3.6x faster per step at 8192 tokens, although the multiply-adds drop about 30x: a step this small is no longer limited by multiply-adds.
- Section 4: the tiled attention with an online softmax differs from the standard one by 4.6e-14 in float64 and the largest tensor drops from N x N (256 MiB at 8192) to 2 MiB; tiling alone does not reduce the arithmetic, skipping the tiles above the diagonal (causal) does, to 36/64.
- Section 5: sliding-window information travels window - 1 positions per layer (a window of 4 over 4 layers reaches 12); the cache stays at 30,720 B instead of growing with the length.
- Section 6: mean-pooling the heads of an MHA model into GQA lifts the loss from 1.933 to 2.660, 25 more steps bring it to 2.075 and 100 steps to 1.958; mean-pooling and keeping the first head differ by 0.068, less than the 0.328 spread between seeds, so this experiment cannot say which is better, and the output says so.

## Exercises

Run again after a change: the checks that compare with the page's numbers turn into `[FAIL]`; that is expected, read how the numbers themselves move.

1. In section 6 change the 25 in `steps_total, uptrain = (30, 5) if args.quick else (500, 25)` to 100 (so 100 and 400 steps after the conversion): does the loss catch up with the GQA-2 model trained from scratch?
2. Change the window of section 5 from 16 to 64: at how many bytes does the cache stop? Does it agree with the formula (window - 1 tokens)?

## Output

The complete runs below were made on an Apple M2 Max (12 cores, CPU only) with Python 3.12 and PyTorch 2.15.0. On another machine the numbers that involve training can differ in the last digits and timings depend on the machine; the checks use thresholds and tolerances, not bit-for-bit equality.

<details>
<summary>Output of the full run (`labs/architecture/efficient-attention/run.py`)</summary>

```text
# Efficient attention
  course: https://dmxsf.com/wiki/architecture/efficient-attention/
          https://dmxsf.com/en/wiki/architecture/efficient-attention/

## 1. KV cache per token: the formulas against the course page's table
  model         attention             per token (BF16)  128K tokens
  ------------  --------------------  ----------------  -----------
    Llama 2 7B      MHA, 32 KV heads         512.0 KiB    64.00 GiB
    Llama 3 8B       GQA, 8 KV heads         128.0 KiB    16.00 GiB
   Llama 3 70B       GQA, 8 KV heads         320.0 KiB    40.00 GiB
  Llama 3 405B       GQA, 8 KV heads         504.0 KiB    63.00 GiB
   DeepSeek-V2  MLA, d_c=512, d_R=64          67.5 KiB     8.44 GiB
   DeepSeek-V3  MLA, d_c=512, d_R=64          68.6 KiB     8.58 GiB
  [ok] per-token sizes: 512 / 128 / 320 / 504 / 67.5 / 68.6 KiB
  [ok] 128K-token caches: 64 / 16 / 40 / 63 / 8.44 / 8.58 GiB
  [ok] the formula agrees with kv_bytes_per_token on the real Llama configs
  DeepSeek-V3 as MHA: 3.81 MiB per token; MLA caches 68.6 KiB, 1/56.9
  [ok] course page: 3.81 MiB as MHA, 1/56.9 with MLA, and 576 numbers = 2.25 GQA groups

  Tensor parallelism: a KV head cannot be split, so fewer KV heads than GPUs means copies.
  8-way tensor parallel  KV heads per GPU  KV heads stored in total
  ---------------------  ----------------  ------------------------
      MHA (64 KV heads)                 8                        64
       GQA (8 KV heads)                 1                         8
        MQA (1 KV head)                 1                         8
  [ok] with 8 GPUs MQA stores as many KV heads as GQA-8 (the copies cancel the saving)

## 2. The same numbers, measured on real caches
  4 layers, 8 query heads of 16 dims, 100 tokens, float32 (4 bytes)
  design               numbers / token / layer  measured bytes / token
  -------------------  -----------------------  ----------------------
                  MHA                      256                  4096 B
      GQA, 2 KV heads                       64                  1024 B
                  MQA                       32                   512 B
  MLA (d_c=64, d_R=8)                       72                  1152 B
  [ok] measured cache sizes equal the formulas (2 g d_h L x 4 bytes; (d_c + d_R) L x 4 bytes)
  [ok] MLA with d_c = 4 d_h, d_R = d_h/2 caches as much as 2.25 GQA groups, whatever the number of heads

## 3. MLA: caching a latent vector, and never rebuilding the keys
  NumPy toy (d=256, 8 heads, d_c=48): max |explicit - absorbed| = 6.1e-16
  numbers cached per token and layer: MHA 512, MLA 48 (without the RoPE key)
  [ok] the course page's absorption check (error ~1e-16)
  full module with causal masking and decoupled RoPE, float32:
    absorbed vs rebuilt keys, whole sequence : max error 3.0e-07
    absorbed decoding from the latent cache  : max error 3.0e-07
    cache after 48 tokens: 80 numbers per token and layer
  [ok] absorbed decoding reproduces the rebuilt-keys result
  [ok] the cache holds d_c + d_R = 80 numbers per token

  Why absorb? One decode step with the cache already full (random weights, CPU):
  8 heads, d_c=128, median of 15 steps
  cached  rebuild MACs  time     absorbed MACs  time     faster by
  ------  ------------  -------  -------------  -------  ---------
     512        33.9 M  0.43 ms          1.2 M  0.24 ms       1.8x
    2048       135.5 M  0.83 ms          4.5 M  0.29 ms       2.8x
    8192       542.1 M  2.19 ms         17.9 M  0.61 ms       3.6x
  The arithmetic drops by an order of magnitude more than the time does: a step this small is no longer limited by multiply-adds.
  [ok] absorption makes a decode step at least 2x faster at the longest length  (3.6x at 8192 tokens)

## 4. FlashAttention's algorithm: tiles and an online softmax
  N=1000, d=64, tiles of 128: max |standard - tiled| = 4.6e-14 (float64)
  score matrix 7.6 MiB, one tile 128 KiB (float64)
  [ok] the page's numbers: 7.6 MiB for the full matrix, 128 KiB per tile, and an exact result
  [ok] the gradients of q, k and v agree too  (max difference 5.3e-15)

  The largest tensor any operation creates (float32, d=64, tiles of 128):
  N     standard    tiled   
  ----  ----------  --------
   512    1.00 MiB  0.12 MiB
  2048   16.00 MiB  0.50 MiB
  8192  256.00 MiB  2.00 MiB
  [ok] standard attention builds an N x N matrix; the tiled version's biggest tensor is its N x d output

  Matrix-multiply FLOPs, N=1024, d=64, 4 heads (forward):
  version        GFLOPs  relative
  -------------  ------  --------
       standard   1.074    1.0000
          tiled   1.074    1.0000
  tiled, causal   0.604    0.5625
  [ok] tiling alone does not reduce the arithmetic: the same 4 N^2 d per head
  [ok] skipping the tiles above the diagonal (causal) does: 36/64 of the work

## 5. Sliding-window attention: bounded cache, and information still travels
  window 4: the last position that still 'hears' a change to token 0
  layers  last position  layers x (window - 1)
  ------  -------------  ---------------------
       1              3                      3
       2              6                      6
       3              9                      9
       4             12                     12
  [ok] information travels window - 1 positions per layer (a window of 4 over 4 layers reaches 12)
  Mistral 7B: 32 layers x 4096 = 131,072 tokens of theoretical reach; a 32K sequence needs 8x less cache

  Decoding 128 tokens one by one, window 16 (float32, 2 layers):
  tokens so far  full cache  sliding cache
  -------------  ----------  -------------
              8    16,384 B       16,384 B
             16    32,768 B       30,720 B
             32    65,536 B       30,720 B
             64   131,072 B       30,720 B
            128   262,144 B       30,720 B
  [ok] the sliding cache holds window - 1 tokens and gives the same logits as recomputing everything  (max logit difference 7.2e-07)

## 6. From an MHA model to GQA: mean-pool the heads, then train a little
  2 layers, 4 query heads, 500 steps, then 25 more (5%); validation loss in nats per byte over 3 seed(s)
  model                             mean loss  range over seeds
  --------------------------------  ---------  ----------------
                      MHA, trained      1.933     1.927 - 1.937
    MHA, 25 more steps (reference)      1.937     1.930 - 1.946
       GQA-2, trained from scratch      1.935     1.905 - 1.959
         MHA -> GQA-2, mean-pooled      2.660     2.606 - 2.718
                 ... then 25 steps      2.075     2.069 - 2.080
            ... or 100 steps (20%)      1.958     1.948 - 1.963
     MHA -> GQA-2, first head kept      2.592     2.425 - 2.753
    ... then 25 steps (first head)      2.066     2.035 - 2.122
  The two starting points differ by 0.068 nats, less than the seed-to-seed range 0.328:
  this experiment cannot say which is better. (The GQA paper reports mean-pooling ahead on T5-XXL; nothing here contradicts that.)
  [ok] 25 steps repair most of the damage of the conversion (MHA -> GQA-2, mean-pooled)  (loss 2.660 -> 2.075, MHA 1.933)
  [ok] 25 steps repair most of the damage of the conversion (MHA -> GQA-2, first head kept)  (loss 2.592 -> 2.066, MHA 1.933)
  [ok] the more steps after the conversion, the lower the loss  (2.660 > 2.075 > 1.958)
  After 100 steps the converted model is +0.024 nats from the GQA-2 model trained from scratch (range of the latter 1.905 - 1.959); the checkpoint it started from cost 500 steps.

All checks passed. (56.3s)
```

</details>
