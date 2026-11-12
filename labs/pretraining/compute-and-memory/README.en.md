# Training compute and memory: 6ND, MFU, model state and activations

Course pages: [The Compute and Memory Ledger of Training: FLOPs, MFU and Memory Budgets](https://dmxsf.com/en/wiki/pretraining/compute-and-memory/)  
[中文](README.md)

Every formula on the page checked against a measurement: PyTorch's FLOP counter, the tensors autograd saves for the backward pass, and the optimizer's state; then MFU, GPU hours and a full Llama 2 7B account on 8 x 80 GiB.

## Run

```bash
uv run python labs/pretraining/compute-and-memory/run.py            # full run, a few seconds
uv run python labs/pretraining/compute-and-memory/run.py --quick    # a shrunk smoke test, a few seconds
```

You can also write `uv run dmxsf run compute-and-memory` (and add `--quick`). `--seed N` changes the random seed, default 0. The script prints its results section by section and checks the course page's claims one by one with `[ok]` / `[FAIL]`: the exit status is 0 when all pass and 1 when any fails; checks that only make sense in the full run are marked `[skip]` under `--quick`.

## Code

| File | What it holds |
| --- | --- |
| `labs/pretraining/compute-and-memory/run.py` | the lab script, numbered sections |
| `src/dmxsf/pretraining/accounting.py` | The arithmetic of a training run: FLOPs, MFU, GPU hours, and the bytes of model state and activations |

Unit tests: `uv run pytest tests/test_pretraining_accounting.py`

## What to look for

- Sections 1-2: a linear layer costs 2 * tokens * d_in * d_out forward and two matrix products (dX and dW) backward, exactly twice the forward; a 4-layer small Llama is 2NT plus attention forward, equal to the FLOP (1,750,073,344, the page's number), and the total is 1.083 times 6NT, the whole excess being the attention term.
- Section 3: build only the shapes on the meta device (a 70B model costs no memory) and count parameters: Llama 3 8B has N = 7.50B parameters in matrix products, only 93.5% of the total, so 6 x total over-counts by about 7%.
- Sections 4-5: the attention share of the training FLOPs rises with the sequence length (the 7B shape: 14.0% at 4096, 83.9% at 131,072) and crosses the linear part near 25,200 and 97,800; the measured FLOPs of full recomputation equal Megatron-LM's formula 96 B s l h^2 (1 + s/6h + V/16lh) to the FLOP, and PyTorch's non-reentrant checkpoint skips the last matrix product of each block.
- Section 6: infer MFU from 6ND and a technical report's GPU hours (Llama 2 7B 39.1%, 70B 42.8%); 10,000 A100 hours for a 7B model on 2T tokens cannot be true, 7 times short even at an MFU of 100%. The last table is the MFU of this machine (fp32, CPU): higher with a bigger batch (5.8% -> 24.0%), and it differs by machine.
- Sections 7-9: the model state is 16 bytes per parameter; the activation bytes per token and layer follow a formula, and what PyTorch really saves is 1.35 times that (the formula describes fused kernels); with the three accounts together, Llama 2 7B on 8 x 80 GiB under ZeRO-3 fits 3 sequences per micro-batch (71.00 GiB) but not 4 (90.48 GiB).

## Exercises

Run again after a change: the checks that compare with the page's numbers turn into `[FAIL]`; that is expected, read how the numbers themselves move.

1. In section 4 add 1048576 to `for s in (4096, 8192, 32768, 131072)`: what share of the training FLOPs is attention in a context of one million tokens?
2. Add `("ZeRO-3, micro-batch 2", 2, 3, False)` and `("ZeRO-3, micro-batch 2, recompute", 2, 3, True)` to the configuration list of section 9 and see how much memory micro-batch 2 takes with recomputation and how much that saves.

## Output

The complete runs below were made on an Apple M2 Max (12 cores, CPU only) with Python 3.12 and PyTorch 2.15.0. On another machine the numbers that involve training can differ in the last digits and timings depend on the machine; the checks use thresholds and tolerances, not bit-for-bit equality.

<details>
<summary>Output of the full run (`labs/pretraining/compute-and-memory/run.py`)</summary>

```text
# Training compute and memory: 6ND, MFU, model state and activations
  course: https://dmxsf.com/wiki/pretraining/compute-and-memory/
          https://dmxsf.com/en/wiki/pretraining/compute-and-memory/

## 1. A linear layer: 2 FLOPs per parameter forward, 4 backward
  Y = X W with X of 256 x 512 and W of 512 x 1024 (524,288 weights)
  input                           forward      backward     backward / forward
  ------------------------------  -----------  -----------  ------------------
  activations (needs a gradient)  268,435,456  536,870,912               2.000
    constant data (no dX needed)  268,435,456  268,435,456               1.000
  [ok] forward is 2 * tokens * d_in * d_out  (268,435,456)
  [ok] backward is two matrix products, dX and dW: twice the forward
  [ok] without dX (a constant input) backward is only dW: the 6N rule needs a trainable path below

## 2. Counting a small Llama: the forward pass is 2NT plus attention
  4 layers, hidden 256, 3,414,272 parameters of which N = 3,155,968 take part in matrix products; B x S = 2 x 128 = 256 tokens
  attention  forward        2NT + attention  backward / forward  total / 6NT
  ---------  -------------  ---------------  ------------------  -----------
     manual  1,750,073,344    1,750,073,344               2.000        1.083
       sdpa  1,750,073,344    1,750,073,344               2.038        1.097
  [ok] forward = 2NT + attention, to the FLOP, whichever way attention is computed
  [ok] it is the number on the course page (HF's version adds 8,192 for its rotary matmul)  (1,750,073,344)
  [ok] with the attention matrix spelled out, backward is exactly twice forward
  [ok] a fused (flash-style) kernel redoes the scores in backward: 2.5x on the attention part only  (3,567,255,552 measured, 3,567,255,552 predicted)
  [ok] total / 6NT = 1 + attention / 2NT: the whole excess over 6NT is the attention term  (1.083 = 1 + 0.083)
  6 * (all 3,414,272 parameters) * T would be 0.999x the measured FLOPs: the embedding table costs memory, not FLOPs

## 3. Which parameters are N
  built on the meta device: shapes only, so a 70B model costs nothing
  model         parameters      N (matmul)      N / total  6*total over-counts by
  ------------  --------------  --------------  ---------  ----------------------
   GPT-2 small     124,439,808     123,532,032      99.3%                   +0.7%
  Qwen2.5-0.5B     494,032,768     493,961,216     100.0%                   +0.0%
    Llama 2 7B   6,738,415,616   6,607,077,376      98.1%                   +2.0%
    Llama 3 8B   8,030,261,248   7,504,658,432      93.5%                   +7.0%
    Mistral 7B   7,241,732,096   7,110,393,856      98.2%                   +1.8%
    Qwen2.5 7B   7,615,616,512   7,070,285,824      92.8%                   +7.7%
   Llama 3 70B  70,553,706,496  69,501,714,432      98.5%                   +1.5%
  [ok] Llama 2 7B: N = 6.61B, 98.1% of the total
  [ok] Llama 3 8B: N = 7.50B, 93.5% of the total; 6 * total over-counts by about 7%
  [ok] the quiz's wrong answer is a real number: the block weights alone are 6.98B  (the output matrix (0.53B) is a matrix product and does count)

## 4. The attention term: 12 L H d_h s per token
  Llama 2 7B: L=32, H=32, d_h=128, N=6.61B
  Llama 3 405B: L=126, H=128, d_h=128, N=403.75B
  sequence length s  Llama 2 7B shape  Llama 3 405B shape
  -----------------  ----------------  ------------------
              4,096      14.0 (7.5) %         4.0 (2.1) %
              8,192     24.5 (14.0) %         7.7 (4.0) %
             32,768     56.5 (39.4) %       25.1 (14.4) %
            131,072     83.9 (72.2) %       57.3 (40.1) %
  (share of the training FLOPs per token that is attention; in brackets: lower triangle only)
  [ok] all eight shares are the ones on the page
  attention FLOPs = linear FLOPs at s = 25,204 (7B shape) and 97,789 (405B shape); with a causal mask, twice that
  [ok] the crossover lengths are about 25,200 and 97,800
  [ok] exercise 4: doubling the sequence to 8192 raises the per-token FLOPs by about 14%  (4.61e+10 -> 5.25e+10)

## 5. Recomputation, and the whole-network formula of the Megatron-LM paper
  L=4, h=128, s=128, B=2, V=1000: forward of one layer = 24Bsh^2 + 4Bs^2h = 117,440,512
  blocks are                  measured FLOPs  formula        measured / no recompute
  --------------------------  --------------  -------------  -----------------------
                      stored   1,605,894,144  1,605,894,144                    1.000
      recomputed (reentrant)   2,075,656,192  2,075,656,192                    1.293
  recomputed (non-reentrant)   1,941,438,464  1,941,438,464                    1.209
  [ok] no recomputation: 3 forward passes of the blocks (1 + 2) plus the head, exactly
  [ok] full recomputation is Megatron-LM's F = 96 B s l h^2 (1 + s/6h + V/16lh), exactly  (2,075,656,192)
  [ok] recomputing costs one extra forward pass of the blocks: the blocks' 6 become 8, 4/3  (469,762,048 FLOPs)
  [ok] PyTorch's non-reentrant checkpoint stops early and skips each block's last matmul  (134,217,728 fewer than the paper's formula, which assumes the whole block is redone)

## 6. Time: MFU and GPU hours
  Megatron-Turing NLG 530B, 2240 A100: MFU 0.298.   PaLM 540B, 6144 TPU v4: MFU 0.457
  [ok] the two PaLM-appendix examples give 0.298 and 0.457

  the training-budget widget, reproduced (H100, MFU 40%, nominal N and D):
  Llama 3 405B               FLOPs 3.79e+25  GPU hours 26,617,796  days on 16,384 GPUs 67.7  tokens/s per GPU 163
  DeepSeek-V3 (N = 37B)      FLOPs 3.29e+24  GPU hours 2,307,044  days on 1 GPUs -  tokens/s per GPU 1,782
  Llama 2 7B (A100)          FLOPs 8.4e+22  GPU hours 186,966  days on 1 GPUs -  tokens/s per GPU 2,971
  [ok] Llama 3 405B: 3.79e25 FLOPs, 26.62M GPU hours, 67.7 days on 16,384 GPUs, 163 tokens/s per GPU
  [ok] DeepSeek-V3 needs 1/11.5 of that: its N is the 37B activated, not the 671B stored
  [ok] Llama 2 7B on A100 at 40%: 187 thousand GPU hours (paper: 184,320), 2,971 tokens/s (paper implies 3,014)
  [ok] with full recomputation the same run takes 35.49M GPU hours and its MFU on 6ND shows 30.0%

  what a technical report's GPU hours imply (6ND over hours and the A100's 312 TFLOP/s):
  model                     6ND (FLOPs)  reported GPU hours  TFLOP/s per GPU  of A100 peak
  ------------------------  -----------  ------------------  ---------------  ------------
                Llama 2 7B     8.09e+22             184,320            121.9         39.1%
               Llama 2 13B     1.56e+23             368,640            117.7         37.7%
               Llama 2 34B     4.05e+23           1,038,336            108.3         34.7%
               Llama 2 70B     8.28e+23           1,720,320            133.7         42.8%
  DeepSeek-V3 (37B active)     3.29e+24           2,664,000            342.6   FP8: no MFU
  [ok] the four Llama 2 rows are the page's: 39.1, 37.7, 34.7 and 42.8% of the A100 peak
  [ok] DeepSeek-V3: 342.6 TFLOP/s per H800; its 180K GPU hours per trillion tokens are 3.7 days on 2048 GPUs
  [ok] 10,000 A100 hours for 7B on 2T tokens cannot be true: 7 times short even at MFU 100%  (8.4e+22 FLOPs needed, 1.1e+22 available at the peak)

  MFU on this machine (fp32, CPU), against the best matrix product it can do:
  best 2048^2 matrix product: 1,933 GFLOP/s on 8 threads
  model (layers x hidden)  tokens per step  ms per step  MFU  
  -----------------------  ---------------  -----------  -----
                  4 x 256              128           24   5.8%
                  4 x 256             2048          101  24.0%
                  6 x 512             2048          442  30.1%
  [ok] a larger batch feeds the matrix units better: MFU is higher at 2,048 tokens than at 128  (5.8% -> 24.0%, differs by machine)
  [ok] MFU is a fraction of the peak  (numbers differ by machine)

## 7. Memory: the model state is 16 bytes per parameter
  fp32 Linear(4096, 4096): 16,777,216 parameters; weights 4, grads 4, AdamW state 8 bytes each
  [ok] autocast-style layout: 4 + 4 + 8 = 16 bytes per parameter
  mixed precision: bf16 weights 2, bf16 gradients 2, fp32 master weights 4, fp32 Adam m and v 8 bytes
  [ok] BF16 weights and gradients, FP32 master copy and moments: 2 + 2 + 4 + 8 = 16
  Llama 2 7B, 6,738,415,616 parameters, GiB per GPU:
  scheme                     8 GPUs  64 GPUs
  -------------------------  ------  -------
   ZeRO-0 (nothing sharded)   100.4    100.4
   ZeRO-1 (optimizer state)    34.5     26.3
       ZeRO-2 (+ gradients)    23.5     13.9
  ZeRO-3 / FSDP (+ weights)    12.6      1.6
  [ok] the page's table (100.4 | 34.5 26.3 | 23.5 13.9 | 12.6 1.6)
  [ok] the ZeRO paper's Figure 1 (7.5B, 64 GPUs): 120, 31.4, 16.6, 1.9 GB
  [ok] exercise 3: a 13B model's state is 193.7 GiB, and 24.2 GiB per GPU under ZeRO-3 on 8
  [ok] exercise 2: 13B on 1T tokens, H100 at 40%, 512 GPUs: 54,769 GPU hours, about 4.5 days
  [ok] exercise 1: 6ND for Llama 3 405B is 3.79e25, within 0.3% of the paper's 3.8e25

## 8. Memory: activations grow with the number of tokens
  SwiGLU FFN, h=512, f=1376, bf16, 4096 tokens: 5 tensors saved, 12,032 bytes per token (2h + 8f = 12,032)
  [ok] the FFN saves x, gate, up, silu(gate) and their product: 2h + 8f bytes per token
  fused attention, 8 query heads, 4 KV heads: 3,104 bytes per token; q, k, v and the output are 3,072, the rest is 32
  [ok] attention keeps q, k, v and its output (no S x S matrix) plus a small float32 logsumexp  (logsumexp: 4 bytes per head and token = 32)
  one block of this repo (h=512, 8 heads, 4 KV heads, f=1376, bf16): 24,488 bytes per token in 20 tensors, formula 18,176
  of the measured bytes, 4,136 are float32 (the norms' statistics in float32, the logsumexp); RoPE adds 1,152; repeat_kv copies k and v for every query head: 1,024
  [ok] the formula is a floor for eager PyTorch, which keeps 1.0 - 1.6 times as much  (1.35x: the formula describes fused norm, RoPE and GQA kernels)
  model       per token and layer  per token, whole model  4,096 tokens
  ----------  -------------------  ----------------------  ------------
  Llama 2 7B            150.0 KiB                4.69 MiB     18.75 GiB
  Llama 3 8B            164.0 KiB                5.12 MiB     20.50 GiB
  [ok] the page's table: 150.0 KiB, 4.69 MiB, 18.75 GiB and 164.0 KiB, 5.12 MiB, 20.50 GiB
  [ok] Llama 3's float32 logits for 4,096 tokens: 1.96 GiB
  6 blocks, h=256, 512 tokens (bytes per token): stored 73,136; recomputed 3,072; plus one block's working set 12,189
  [ok] with recomputation the only saved tensors are the blocks' inputs: layers * 2h bytes per token  (3,072 = 6 * 2 * 256 per token)
  saved bytes shrink 4.8x here (6 layers); the formula gives 32 layers of Llama 2 7B:
  [ok] Llama 2 7B, 4,096 tokens: 18.75 GiB become 1.59 GiB, about 1/11.8

## 9. All three accounts together: Llama 2 7B, 8 x 80 GiB, sequence 4096
  configuration                     state   activations  logits  total GiB  80 GiB      
  --------------------------------  ------  -----------  ------  ---------  ------------
             ZeRO-0, micro-batch 1  100.41        18.75    0.73     119.89  does not fit
             ZeRO-3, micro-batch 1   12.55        18.75    0.73      32.03          fits
             ZeRO-3, micro-batch 3   12.55        56.25    2.20      71.00          fits
             ZeRO-3, micro-batch 4   12.55        75.00    2.93      90.48  does not fit
  ZeRO-3, micro-batch 1, recompute   12.55         1.59    0.73      14.87          fits
  [ok] the page's totals: 119.89 (no sharding), 32.03, 71.00 and, with recomputation, 14.87 GiB
  [ok] a 4th sequence per micro-batch (about 90 GiB) no longer fits, a 3rd does
  [ok] with recomputation 29 sequences would fit: 'more than twenty' on the page  (paid for with a third more FLOPs (section 5))

All checks passed. (5.2s)
```

</details>
