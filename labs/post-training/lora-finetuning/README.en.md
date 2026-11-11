# LoRA: the parameter bill, the mechanics, LoRA against full fine-tuning, merging, NF4

Course pages: [LoRA Fine-Tuning in Practice](https://dmxsf.com/en/wiki/post-training/lora-finetuning/)  
[中文](README.md)

LoRA written from scratch and compared with full fine-tuning: the exact parameter bill, learning-rate windows, rank and alpha, which matrices to adapt, how low-rank the learned update is, merging into the weights and into 4-bit (NF4) weights, memory and speed.

## Run

```bash
uv run python labs/post-training/lora-finetuning/run.py            # full run, about 4 minutes
uv run python labs/post-training/lora-finetuning/run.py --quick    # a shrunk smoke test, about 20 seconds
```

You can also write `uv run dmxsf run lora-finetuning` (and add `--quick`). `--seed N` changes the random seed, default 0. The script prints its results section by section and checks the course page's claims one by one with `[ok]` / `[FAIL]`: the exit status is 0 when all pass and 1 when any fails; checks that only make sense in the full run are marked `[skip]` under `--quick`.

This lab uses cached small models: the byte-level base model (842,496 parameters). The first run trains them on TinyShakespeare and caches them in `~/.cache/dmxsf` (set `DMXSF_CACHE` to move it); later runs read them back.

## Code

| File | What it holds |
| --- | --- |
| `labs/post-training/lora-finetuning/run.py` | the lab script, numbered sections |
| `src/dmxsf/post_training/lora.py` | LoRA from scratch: a frozen linear layer plus a trainable low-rank path, merging, and NF4 |

Unit tests: `uv run pytest tests/test_post_training_lora.py`

## What to look for

- Section 1: the parameter bill needs no model. Qwen2.5-0.5B at r=16 is 366,592 per layer, 8,798,208 over 24 layers (the number peft reports); Llama 2 7B at r=8 on q and v is 4,194,304, 0.06%; QLoRA shrinks the base model from 12.55 GiB to 3.60 GiB and adds 0.60 GiB of adapter and optimizer state, where full fine-tuning needs 100.4 GiB.
- Section 2: B starts at zero, so dL/dA = 0 on the first step and only B moves (|dB| 474.7); with A and B both zero nothing is ever learned; with a non-zero B merged into the base weight, the output before and after the merge differs by a relative 5e-07 (float32 rounding).
- Section 3: the learning rate has a window, one for LoRA and one for full fine-tuning. The best LoRA (lr 3e-3) reaches 99.7% and 4.75 bits/byte, the best full fine-tuning 99.0% and 3.73 bits: here LoRA does not 'forget less'; how much is forgotten follows the learning rate and the amount of training, not the name of the method. The spread over seeds is only 0.3 points, so these differences are real.
- Section 4: with alpha fixed at 16, rank 1 and rank 8 both reach 99.7% and rank 64 drops to 94.0%; with alpha = 2r (a fixed multiplier of 2) rank 1 manages only 31.3%. What moves is not capacity but the multiplier alpha / r, that is, the step size.
- Sections 5-6: adapting only q and v (16,384 parameters) gives 65.3%, only the MLP 44.0%, all linear layers (73,728) 99.7%; the update of full fine-tuning is far more concentrated than a random matrix (the top 8 directions hold 75% of q's, a random matrix 21%), but it is not low rank.
- Section 7: the NF4 round trip costs 9.4% of the weight norm (the page: 9.3%-9.7% on Qwen); an update far below the quantization grid is rounded away when merged into already quantized weights (0.017 survives at 0.01 x, against 1.044 when merged in fp32); the four ways of deploying score 99.7% / 99.7% / 99.7% / 99.0% and QLoRA (trained on the NF4 base) 97.7%.

## Exercises

Run again after a change: the checks that compare with the page's numbers turn into `[FAIL]`; that is expected, read how the numbers themselves move.

1. Add `3e-2` to `lora_lrs = (3e-3,) if QUICK else (3e-4, 1e-3, 3e-3, 1e-2)` in section 3: where is the upper edge of LoRA's window, and how do accuracy and forgetting change?
2. Add 4 and 16 to `ranks = (1, 8) if QUICK else (1, 8, 64)` in section 4: what shape does accuracy take against the rank when alpha is fixed?

## Notes

- The task is a synthetic intent classification; a byte-level model of 842,496 parameters is far smaller than a real 0.5B model, so a claim such as 'LoRA forgets less than full fine-tuning' does not hold here and the lab reports it as it is. The milliseconds per step of section 8 are timings of this machine; on a model this small LoRA is not faster.

## Output

The complete runs below were made on an Apple M2 Max (12 cores, CPU only) with Python 3.12 and PyTorch 2.15.0. On another machine the numbers that involve training can differ in the last digits and timings depend on the machine; the checks use thresholds and tolerances, not bit-for-bit equality.

<details>
<summary>Output of the full run (`labs/post-training/lora-finetuning/run.py`)</summary>

```text
# LoRA: the parameter bill, the mechanics, LoRA against full fine-tuning, merging, NF4
  course: https://dmxsf.com/wiki/post-training/lora-finetuning/
          https://dmxsf.com/en/wiki/post-training/lora-finetuning/

## 1. The parameter bill (exact, no model needed)
  Qwen2.5-0.5B (hidden 896, MLP 4864, 24 layers, 2 KV heads of 64), r = 16:
  matrix     d_in -> d_out  LoRA parameters
  ---------  -------------  ---------------
     q_proj     896 -> 896           28,672
     k_proj     896 -> 128           16,384
     v_proj     896 -> 128           16,384
     o_proj     896 -> 896           28,672
  gate_proj    896 -> 4864           92,160
    up_proj    896 -> 4864           92,160
  down_proj    4864 -> 896           92,160
  [ok] per layer 366,592; times 24 layers = 8,798,208 (the number peft reports)  (366,592 x 24 = 8,798,208)
  [ok] Llama 2 7B, r = 8 on q and v: 4,194,304 parameters, 0.06% of the model  (4,194,304 = 0.06% of 6,738,415,616)
  [ok] Llama 2 7B, r = 16 on every linear layer: 39,976,960 parameters, 76.25 MiB in bf16  (39,976,960 parameters, 76.25 MiB)
  quantity                                          value        
  ------------------------------------------------  -------------
  bits per weight: NF4 / NF4 + double quantization  4.500 / 4.127
      base model, bf16 -> NF4 + double quant (GiB)  12.55 -> 3.60
      adapter + gradient + Adam states, r=16 (GiB)           0.60
    full fine-tuning, 16 bytes per parameter (GiB)          100.4
  [ok] QLoRA accounting: 12.55 -> 3.60 GiB base, 0.60 GiB state, against 100.4 GiB for full fine-tuning  (12.55 -> 3.60, 0.60, 100.4)

## 2. Mechanics on the pretrained byte model
  [ok] attn.q_proj: zero start, dL/dA = 0 on step one, A = B = 0 never learns, merge exact  (|lora - base| 0.0, max |dA| 0.0, |dB| 474.72, merged error 4.8e-07)
  [ok] mlp.fc: zero start, dL/dA = 0 on step one, A = B = 0 never learns, merge exact  (|lora - base| 0.0, max |dA| 0.0, |dB| 759.10, merged error 5.8e-07)
  layer        shape       max|lora-base| at init  max|dA| step 1  |dB| step 1  max grad if A=B=0  merge error (rel.)  r(d_in+d_out)
  -----------  ----------  ----------------------  --------------  -----------  -----------------  ------------------  -------------
  attn.q_proj  128 -> 128                     0.0             0.0        474.7                0.0             4.8e-07          2,048
       mlp.fc  128 -> 512                     0.0             0.0        759.1                0.0             5.8e-07          5,120

## 3. Learning rate: LoRA and full fine-tuning each have a window
  method                 learning rate  trained parameters  accuracy  shakespeare bits/byte
  ---------------------  -------------  ------------------  --------  ---------------------
  LoRA r=8 (all linear)         0.0003              73,728     10.0%                   2.62
  LoRA r=8 (all linear)          0.001              73,728     61.7%                   3.05
  LoRA r=8 (all linear)          0.003              73,728     99.7%                   4.75
  LoRA r=8 (all linear)           0.01              73,728     96.3%                   8.41
       full fine-tuning          3e-05             842,496     14.3%                   2.51
       full fine-tuning         0.0001             842,496     78.0%                   2.84
       full fine-tuning         0.0003             842,496     99.0%                   3.73
       full fine-tuning          0.001             842,496     97.0%                   6.28
  (the pretrained model: 2.31 bits/byte on held-out Shakespeare)
  the default LoRA (r=8, alpha=16, lr 3e-3) with other training seeds: 99.7%, 99.7%, 100.0% (range 0.3 points; smaller differences below are not findings)
  [ok] LoRA: the learning rate moves accuracy and forgetting a lot (not a flat curve)  (accuracy 10%-100%, bits/byte 2.62-8.41)
  [ok] full fine-tuning: the learning rate moves accuracy and forgetting a lot (not a flat curve)  (accuracy 14%-99%, bits/byte 2.51-6.28)
  best LoRA: 99.7% / 4.75 bits; best full fine-tuning: 99.0% / 3.73 bits

## 4. Rank and alpha: capacity, or just a bigger learning rate?
  rank  adapter parameters  alpha=16: accuracy  bits/byte  alpha=2r  accuracy  bits/byte
  ----  ------------------  ------------------  ---------  --------  --------  ---------
     1               9,216               99.7%       4.64         2     31.3%       3.31
     8              73,728               99.7%       4.75        16     99.7%       4.75
    64             589,824               94.0%       4.31       128    100.0%       8.45
  alpha / r is the multiplier on the update: with alpha fixed it shrinks as r grows, with
  alpha = 2r it stays 2. Which column moves tells you whether rank matters, or the step size.

## 5. Which matrices: position against parameter count
  adapted matrices  adapter parameters  accuracy  shakespeare bits/byte
  ----------------  ------------------  --------  ---------------------
                qv              16,384     65.3%                   4.14
               mlp              40,960     44.0%                   4.23
               all              73,728     99.7%                   4.75
  [ok] adapter parameters equal r(d_in + d_out) summed over the adapted matrices  (qv 16,384, mlp 40,960, all 73,728)

## 6. How low-rank is what fine-tuning learns?
  share of the squared Frobenius norm of the update in its top-r singular directions,
  averaged over the four blocks:
  update            matrix  r=1    r=2    r=4    r=8    r=16 
  ----------------  ------  -----  -----  -----  -----  -----
  full fine-tuning       q  0.281  0.430  0.582  0.750  0.896
          LoRA r=8       q  0.550  0.774  0.934  1.000  1.000
     random matrix       q  0.031  0.059  0.114  0.211  0.378
  full fine-tuning       v  0.182  0.320  0.511  0.707  0.873
          LoRA r=8       v  0.509  0.742  0.927  1.000  1.000
     random matrix       v  0.030  0.059  0.113  0.210  0.377
  full fine-tuning      fc  0.166  0.262  0.393  0.545  0.726
          LoRA r=8      fc  0.497  0.703  0.903  1.000  1.000
     random matrix      fc  0.017  0.034  0.067  0.128  0.239
  full fine-tuning    proj  0.142  0.233  0.362  0.528  0.726
          LoRA r=8    proj  0.621  0.818  0.955  1.000  1.000
     random matrix    proj  0.017  0.034  0.066  0.126  0.237
  [ok] a full fine-tuning update is far more concentrated than a random matrix (top 8 directions)  (q 0.75 vs 0.21, v 0.71 vs 0.21, fc 0.55 vs 0.13, proj 0.53 vs 0.13)
  [ok] ...but it is not low rank: the top 8 directions hold well under all of it  (q 0.75, v 0.71, fc 0.55, proj 0.53)
  [ok] a rank-8 LoRA update has exactly 8 directions (energy 1.000 at r=8)

## 7. Merging, and merging into 4-bit weights
  NF4 round-trip error of block 0's q_proj: 9.4% of the weight norm
  [ok] NF4 costs roughly a tenth of the weight norm (page: 9.3% to 9.7% on Qwen)  (9.4%)

  share of the update dW that survives, for the LoRA adapter trained above (block 0, three matrices):
  update size     mean |dW|  mean NF4 error  merged into NF4, re-quantized  merged into fp32, then quantized
  --------------  ---------  --------------  -----------------------------  --------------------------------
  0.01 x trained    8.5e-05         4.9e-03                          0.017                             1.044
   0.1 x trained    8.5e-04         4.9e-03                          0.017                             1.010
     1 x trained    8.5e-03         4.9e-03                          0.962                             1.000
    10 x trained    8.5e-02         4.9e-03                          0.995                             0.995
  [ok] an update far below the 4-bit grid spacing is rounded away when merged into an already quantized weight, but survives from fp32  (at 0.01 x the trained size: 0.017 against 1.044)
  [ok] an update well above the spacing mostly survives either way  (at 10 x the trained size: 0.995)

  one adapter, four ways to deploy it (and QLoRA: trained on the quantized base):
  deployment                                      accuracy  shakespeare bits/byte
  ----------------------------------------------  --------  ---------------------
                           fp32 base, no adapter      0.0%                   2.31
               adapter attached to the fp32 base     99.7%                   4.75
                 adapter attached to an NF4 base     99.7%                   4.80
                merged into fp32, then quantized     99.7%                   4.76
                merged into NF4, quantized again     99.0%                   4.66
                            NF4 base, no adapter      0.0%                   2.32
  QLoRA: trained on the NF4 base, attached to it     97.7%                   4.61
  4-bit storage alone moves held-out Shakespeare by +0.016 bits/byte (+0.7%); the page's 0.5B model lost 13% in perplexity.

## 8. Memory and speed
  method                trained parameters  weights+grads+Adam (MiB, fp32)  ms per step (this machine)
  --------------------  ------------------  ------------------------------  --------------------------
      full fine-tuning             842,496                           12.86                          37
  LoRA r=8, all linear              73,728                            4.34                          42
  static memory = one frozen copy of the weights, plus four copies (weights, gradient, two Adam
  moments) of whatever is trained. Activations come on top and shrink the gap, as on the page.
  [ok] LoRA trains a small fraction of the parameters and needs a fraction of the static memory  (8.8% of the parameters)

All checks passed. (224.1s)
```

</details>
