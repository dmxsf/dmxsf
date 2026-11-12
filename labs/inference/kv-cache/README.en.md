# KV cache: the memory formula, exactness, the saving, prefill against decode

Course pages: [KV Cache and Inference Performance](https://dmxsf.com/en/wiki/inference/kv-cache/)  
[中文](README.md)

The memory formula against real cache tensors, cached and uncached decoding agreeing exactly, and the saving counted two ways: token positions processed (exact) and time on a CPU (read the ratios), plus the prefill and decode split.

## Run

```bash
uv run python labs/inference/kv-cache/run.py            # full run, a few seconds
uv run python labs/inference/kv-cache/run.py --quick    # a shrunk smoke test, a few seconds
```

You can also write `uv run dmxsf run kv-cache` (and add `--quick`). `--seed N` changes the random seed, default 0. The script prints its results section by section and checks the course page's claims one by one with `[ok]` / `[FAIL]`: the exit status is 0 when all pass and 1 when any fails; checks that only make sense in the full run are marked `[skip]` under `--quick`.

## Code

| File | What it holds |
| --- | --- |
| `labs/inference/kv-cache/run.py` | the lab script, numbered sections |
| `src/dmxsf/inference/kv_cache.py` | KV caches: remember every layer's keys and values so each new token costs one step, not a re-run |

Unit tests: `uv run pytest tests/test_inference_kv_cache.py`

## What to look for

- Section 1: the memory formula and the page's numbers: Llama 3 8B takes 128 KiB per token, 1 GiB per 8,192 tokens and 32 GiB for 32 sequences, more than the weights (16.1 GB); Llama 2 7B without GQA takes 512 KiB per token, 4 times as much; KV heads and precision multiply: MHA fp16 512 KiB, GQA-8 int8 64 KiB, MQA int4 4 KiB.
- Section 2: the formula against real tensors: after a 100-token prefill, 4 / 2 / 1 KV heads hold 409,600 / 204,800 / 102,400 bytes, exactly the formula; a preallocated cache costs its full length from the first token.
- Section 3: cached and uncached decoding agree: the same 64 greedy tokens, logits from the incremental path differ from a full recompute by at most 8.3e-07; the positions processed to write 64 tokens are 6,112 against 128 (48x).
- Section 4: time (read the ratios, not the absolute numbers): a 4.8M-parameter GPT writing 256 tokens; without a cache the time per token (the median of each third of the run) rises with the sequence (10.2 -> 17.0 ms), with one it hardly moves (1.7 -> 1.5 ms) and the whole run is 8.3x faster; prefill costs 0.06 ms per token against 1.72 ms for decoding, 31 times cheaper per token: decoding reads all the weights and the whole cache to produce one token and is bound by memory and overhead, which is why serving systems batch many sequences per step.

## Exercises

Run again after a change: the checks that compare with the page's numbers turn into `[FAIL]`; that is expected, read how the numbers themselves move.

1. Add 0.25 to `for bytes_ in (2, 1, 0.5)` in section 1 and an int2 column to the header `["attention", "fp16", "int8", "int4"]`: how many KiB per token does MQA with a 2-bit cache take?
2. In section 4 change the 256 in `p_big, n_big = (32, 32) if QUICK else (128, 256)` to 384: how much does the per-token time of the last third grow without a cache, and with one?

## Notes

- The milliseconds and GFLOP/s of section 4 are timings of this machine; the checks look only at ratios and directions (a speed-up of at least 3x, prefill cheaper per token), with wide thresholds.

## Output

The complete runs below were made on an Apple M2 Max (12 cores, CPU only) with Python 3.12 and PyTorch 2.15.0. On another machine the numbers that involve training can differ in the last digits and timings depend on the machine; the checks use thresholds and tolerances, not bit-for-bit equality.

<details>
<summary>Output of the full run (`labs/inference/kv-cache/run.py`)</summary>

```text
# KV cache: the memory formula, exactness, the saving, prefill against decode
  course: https://dmxsf.com/wiki/inference/kv-cache/
          https://dmxsf.com/en/wiki/inference/kv-cache/

## 1. The memory formula and the page's numbers (exact)
  fp16 cache; the shape is layers x kv_heads x head_dim:
  model         shape          KiB per token  GiB per 8,192 tokens  GiB for 32 sequences  fp16 weights (GB)
  ------------  -------------  -------------  --------------------  --------------------  -----------------
    Llama 2 7B  32 x 32 x 128            512                  4.00                 128.0               13.5
    Llama 3 8B   32 x 8 x 128            128                  1.00                  32.0               16.1
   Llama 3 70B   80 x 8 x 128            320                  2.50                  80.0              141.1
  Qwen2.5-0.5B    24 x 2 x 64             12                  0.09                   3.0                1.0
  [ok] Llama 3 8B: 131,072 bytes = 128 KiB per token, 1 GiB per 8,192 tokens, 32 GiB for 32 sequences
  [ok] ...which is more than the weights (16.1 GB); Llama 2 7B (32 KV heads) needs 512 KiB per token, 4x as much  (weights 16.06 GB)

  KiB per token for the Llama 3 8B shape, by number of KV heads and cache precision:
  attention             fp16   int8   int4 
  --------------------  -----  -----  -----
     MHA (32 KV heads)  512.0  256.0  128.0
  GQA (8, as released)  128.0   64.0   32.0
               GQA (4)   64.0   32.0   16.0
               MQA (1)   16.0    8.0    4.0
  [ok] fewer KV heads and fewer bits multiply: MHA fp16 512 KiB -> GQA-8 int8 64 KiB -> MQA int4 4 KiB

## 2. The formula against real tensors
  the 0.86M byte-level GPT, fp32, after a 100-token prefill:
  model       bytes in the cache  formula
  ----------  ------------------  -------
  4 KV heads              409600   409600
  2 KV heads              204800   204800
  1 KV heads              102400   102400
  [ok] the cache holds exactly what the formula says (and GQA shrinks it in proportion)
  [ok] a preallocated cache costs its full length from the first token (reserve for the longest request)  (524,288 bytes before any token is stored)

## 3. Cached and uncached decoding agree
  [ok] the same 64 greedy tokens with and without the cache
  [ok] logits from the incremental path equal a full recompute up to float rounding  (max difference 8.3e-07)

  token positions processed to write 64 tokens after a 64-token prompt: without a cache 6,112, with one 128 (48x).
  [ok] counted positions match the page's arithmetic: sum of (P + t) without a cache, P + n with one  (6,112 and 128)

## 4. Time: the saving, and prefill against decode (CPU; read the ratios)
  a 4.8M-parameter GPT (6 layers, d = 256) writing 256 tokens after a 128-token prompt; median ms per new token, by third of the run:
            first third  middle third  last third  whole run (s)
  --------  -----------  ------------  ----------  -------------
  no cache         10.2          14.6        17.0           3.58
  KV cache          1.7           1.6         1.5           0.42
  [ok] the cost per token rises through the run without a cache, and rises more slowly with one  (last third over first: no cache x1.67, cache x0.89)
  [ok] the whole run is at least 3x faster with the cache  (8.3x)
  the whole run is 8.3x faster with the cache (prefill included).

  prefill: 7.1 ms for 128 tokens = 0.06 ms per token, about 171 GFLOP/s;
  decode:  1.72 ms per token, about 5.6 GFLOP/s.
  [ok] prefill handles a token far cheaper than decode does (parallel over the prompt)  (31x cheaper per token)
  decode reads the weights and the whole cache to produce one token; at batch 1 that is
  memory and overhead bound, which is why serving systems batch many sequences per step.

All checks passed. (4.4s)
```

</details>
