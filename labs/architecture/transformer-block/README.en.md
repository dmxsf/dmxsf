# The Transformer block and the modern recipe

Course pages: [Transformer Block Explained: Attention, FFN, Residuals and LayerNorm](https://dmxsf.com/en/wiki/architecture/transformer-block/) · [Modern LLM Architecture: From GPT-2 to Llama, Qwen and DeepSeek](https://dmxsf.com/en/wiki/architecture/modern-llm-architecture/)  
[中文](README.md)

Count a GPT-2 small block (12d²+13d parameters), see where real models keep their parameters, measure signal and gradient by depth for pre-norm and post-norm, then walk from GPT-2 to Llama one change at a time.

## Run

```bash
uv run python labs/architecture/transformer-block/run.py            # full run, about 3 minutes
uv run python labs/architecture/transformer-block/run.py --quick    # a shrunk smoke test, a few seconds
```

You can also write `uv run dmxsf run transformer-block` (and add `--quick`). `--seed N` changes the random seed, default 0. The script prints its results section by section and checks the course page's claims one by one with `[ok]` / `[FAIL]`: the exit status is 0 when all pass and 1 when any fails; checks that only make sense in the full run are marked `[skip]` under `--quick`.

## Code

| File | What it holds |
| --- | --- |
| `labs/architecture/transformer-block/run.py` | the lab script, numbered sections |
| `src/dmxsf/architecture/gpt.py` | A decoder-only Transformer in one configurable class: GPT-2 ... Llama |
| `src/dmxsf/architecture/layers.py` | Normalisation layers and feed-forward blocks |
| `src/dmxsf/architecture/configs.py` | Architecture configs of released checkpoints, with the parameter counts they report |

Unit tests: `uv run pytest tests/test_architecture_gpt.py tests/test_architecture_layers.py`

## What to look for

- Section 1: one GPT-2 small block is 12 d^2 + 13 d = 7,087,872 parameters (feed-forward 66.6%, attention 33.3%); 12 blocks plus the embedding tables make 124,439,808, as on the page.
- Section 2: the parameter breakdown of seven public models, built as shapes only without weights: Llama 3 8B is 70.2% feed-forward, 16.7% attention, 13.1% vocabulary; GQA gives it a KV cache of 128 KiB per token against 512 KiB for Llama 2 7B.
- Section 3: the pre-LN gradient shrinks like 1/sqrt(L) with depth (0.1354 at 48 layers) while the residual stream keeps growing (RMS 7.386); the post-LN gradient does not shrink (0.9117); scaling each residual branch by 1/sqrt(N) keeps the size at 1.136.
- Section 4: swapping GPT-2 for the Llama recipe one change at a time: RoPE is the one clear gain (-0.314 nats), the other changes stay within the spread between seeds (largest 0.063); the full recipe beats the GPT-2 style by 0.231 nats and GQA halves the KV cache per token from 512 B to 256 B with no clear loss.

## Exercises

Run again after a change: the checks that compare with the page's numbers turn into `[FAIL]`; that is expected, read how the numbers themselves move.

1. Most differences in section 4 are smaller than the spread between seeds. Change the 3 in `[args.seed + i for i in range(3)]` (the `seeds` of section 4) to 10 and see which rows emerge from the noise.
2. Set `depths` in section 3 to `(6, 12, 24, 48, 96)`: is the pre-LN "gradient x sqrt(L)" still constant?

## Output

The complete runs below were made on an Apple M2 Max (12 cores, CPU only) with Python 3.12 and PyTorch 2.15.0. On another machine the numbers that involve training can differ in the last digits and timings depend on the machine; the checks use thresholds and tolerances, not bit-for-bit equality.

<details>
<summary>Output of the full run (`labs/architecture/transformer-block/run.py`)</summary>

```text
# The Transformer block and the modern recipe
  course: https://dmxsf.com/wiki/architecture/transformer-block/
          https://dmxsf.com/en/wiki/architecture/transformer-block/
  course: https://dmxsf.com/wiki/architecture/modern-llm-architecture/
          https://dmxsf.com/en/wiki/architecture/modern-llm-architecture/

## 1. One GPT-2 small block: 12 d^2 + 13 d parameters
  part          parameters  share
  ------------  ----------  -----
     attention   2,362,368  33.3%
  feed-forward   4,722,432  66.6%
         norms       3,072   0.0%
  total 7,087,872; the formula 12d^2 + 13d gives 7,087,872
  12 blocks + token table + position table + final norm = 124,439,808
  [ok] the numbers on the course page (GPT-2 small is 124,439,808 parameters)
  [ok] output shape is unchanged and positions 0-5 ignore everything after them

## 2. Where real models keep their parameters (shapes only, no weights)
  model         params  feed-forward  attention  vocab tables  KV cache / token
  ------------  ------  ------------  ---------  ------------  ----------------
   GPT-2 small   0.12B         45.5%      22.8%         31.0%            36 KiB
  Qwen2.5-0.5B   0.49B         63.5%       8.9%         27.6%            12 KiB
    Llama 2 7B   6.74B         64.2%      31.9%          3.9%           512 KiB
    Llama 3 8B   8.03B         70.2%      16.7%         13.1%           128 KiB
    Mistral 7B   7.24B         77.8%      18.5%          3.6%           128 KiB
    Qwen2.5 7B   7.62B         74.9%      10.8%         14.3%            56 KiB
   Llama 3 70B  70.55B         79.9%      17.1%          3.0%           320 KiB
  [ok] all seven totals equal the published parameter counts  (asserted above)
  [ok] course page: Llama 3 8B is 70.2% feed-forward, 16.7% attention, 13.1% vocabulary
  [ok] course page: GPT-2 small spends 31.0% on its token table, Llama 2 7B 3.9%, Llama 3 70B 3.0%
  [ok] course page: KV cache per token is 512 / 128 / 320 KiB for Llama 2 7B / Llama 3 8B / 70B
  Grouped-query attention is the difference: Llama 2 7B caches 32 KV heads, Llama 3 8B 8.

## 3. Pre-norm or post-norm: signal and gradient at initialisation, by depth
  random tokens, d=256, 8 sequences of 64, mean of 8 seed(s)
  RMS  = size of the last layer's output
  grad = norm of the gradient of the last feed-forward's output matrix
  layers  post RMS  post grad  pre RMS  pre grad  pre+s RMS  pre+s grad
  ------  --------  ---------  -------  --------  ---------  ----------
       6     1.000     0.8198    2.347    0.4075      1.137      0.8112
      12     1.000     0.8447    3.415    0.2833      1.136      0.8153
      24     1.000     0.8784    5.029    0.1950      1.137      0.8137
      48     1.000     0.9117    7.386    0.1354      1.136      0.8155
  [ok] post-LN: the gradient does not shrink with depth
  [ok] pre-LN: the gradient shrinks like 1/sqrt(L) (gradient x sqrt(L) stays within 15%)
  [ok] pre-LN: the residual stream keeps growing with depth
  [ok] pre-LN: gradient x RMS is nearly constant (between 0.9 and 1.1)
  [ok] scaling the residual branches by 1/sqrt(N) keeps the size steady (spread under 0.05)  (1.136 to 1.137)
  [ok] all 12 cells agree with the course page's table (RMS within 0.05, gradient within 0.03)  (largest differences: RMS 0.005, gradient 0.005)

## 4. From GPT-2 to Llama, one change at a time
  2 layers, d_model 64, 500 steps (lr 0.003, gradient clipping 1) on Shakespeare bytes, 3 seed(s); validation loss in nats per byte
  variant                                                  parameters  KV / token  mean loss  spread   change vs row above
  -------------------------------------------------------  ----------  ----------  ---------  -------  -------------------
  GPT-2 style: LayerNorm, GELU, learned positions, biases     120,576       512 B      2.071  0.03791                     
                                                + RMSNorm     120,256       512 B      2.080  0.01865          +0.009 same
        + SwiGLU (hidden size cut to keep the MLP's size)     120,168       512 B      2.137  0.01324          +0.057 same
                      + RoPE instead of learned positions     116,072       512 B      1.823  0.04301        -0.314 better
                                              + no biases     114,752       512 B      1.860  0.06328          +0.037 same
                   + grouped-query attention (2 KV heads)     106,560       256 B      1.840  0.02142          -0.020 same
  'spread' is max - min over seeds; a row is called better or worse only when it moves the mean by more than the largest spread, 0.063
  [ok] the full recipe beats GPT-2 style by more than the spread  (0.231 nats)
  [ok] swapping learned positions for RoPE is a clear gain  (0.314 nats)
  [ok] grouped-query attention halves the KV cache without a clear loss  (-0.020 nats)

All checks passed. (302.4s)
```

</details>
