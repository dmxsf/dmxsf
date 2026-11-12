# LoRA 微调

课程页面：[LoRA 微调实战](https://dmxsf.com/wiki/post-training/lora-finetuning/)  
[English](README.en.md)

从零实现 LoRA 并与全量微调对比：精确的参数账本、两者各自的学习率窗口、秩与 alpha、该适配哪些矩阵、学到的更新有多低秩、合并进权重和 4-bit（NF4）权重、显存与速度。

## 运行

```bash
uv run python labs/post-training/lora-finetuning/run.py            # 完整运行，约 4 分钟
uv run python labs/post-training/lora-finetuning/run.py --quick    # 缩小规模的冒烟测试，约 20 秒
```

也可以写成 `uv run dmxsf run lora-finetuning`（同样可以加 `--quick`）。`--seed N` 换随机种子，默认 0。脚本分节打印结果，并用 `[ok]` / `[FAIL]` 逐条检查课程页面里的结论：全部通过时退出码为 0，有一条失败就是 1；只有完整运行才做的检查在 `--quick` 下标为 `[skip]`。

这个实验用到缓存的小模型：字节级基础模型（84 万参数）。第一次运行会先在 TinyShakespeare 上训练并缓存到 `~/.cache/dmxsf`（环境变量 `DMXSF_CACHE` 可以改位置），之后直接读取。

## 代码

| 位置 | 内容 |
| --- | --- |
| `labs/post-training/lora-finetuning/run.py` | 实验脚本，按编号分节 |
| `src/dmxsf/post_training/lora.py` | 从零实现 LoRA：冻结的线性层加可训练的低秩通路、合并与 NF4 |

单元测试：`uv run pytest tests/test_post_training_lora.py`

## 看什么

- 第 1 节：参数账本不需要模型。Qwen2.5-0.5B 的 r=16 每层 366,592、24 层共 8,798,208（peft 报告的数字）；Llama 2 7B 在 q、v 上取 r=8 是 4,194,304，占 0.06%；QLoRA 把基础模型从 12.55 GiB 压到 3.60 GiB，加上 0.60 GiB 的适配器与优化器状态，而全量微调要 100.4 GiB。
- 第 2 节：B 初始为 0，所以第一步 dL/dA = 0，只有 B 先动（|dB| 474.7）；A 和 B 都为 0 则永远学不到东西；给 B 非零值后合并进基础权重，合并前后的输出相对差约 5e-07（float32 舍入）。
- 第 3 节：学习率有窗口，LoRA 和全量微调各一个。最好的 LoRA（lr 3e-3）99.7% 准确率、4.75 比特/字节，最好的全量微调 99.0%、3.73 比特：这里 LoRA 并没有“忘得更少”，忘多少取决于学习率与训练量，不取决于方法的名字。种子之间的波动只有 0.3 个点，所以这里的差别都是真的。
- 第 4 节：alpha 固定为 16 时，rank 1 与 rank 8 都是 99.7%，rank 64 掉到 94.0%；alpha = 2r（放大倍数固定为 2）时 rank 1 只有 31.3%。动的不是容量，而是放大倍数 alpha / r，也就是步长。
- 第 5–6 节：只调 q、v（16,384 个参数）65.3%，只调 MLP 44.0%，全部线性层（73,728 个）99.7%；全量微调的权重更新比随机矩阵集中得多（前 8 个方向占 q 的 75%、随机矩阵是 21%），但并不是低秩的。
- 第 7 节：NF4 的往返误差是权重范数的 9.4%（页面上 Qwen 是 9.3%–9.7%）；合并进已经量化的权重时，远小于量化格距的更新被舍掉（0.01 倍时只剩 0.017，从 fp32 合并则是 1.044）；四种部署方式的准确率为 99.7% / 99.7% / 99.7% / 99.0%，QLoRA（在 NF4 基础模型上训练）97.7%。

## 练习

改了参数再运行：核对页面数字的检查会变成 `[FAIL]`，这是预期的，看数字本身怎么变。

1. 第 3 节 `lora_lrs = (3e-3,) if QUICK else (3e-4, 1e-3, 3e-3, 1e-2)` 里加一个 `3e-2`：LoRA 的窗口上沿在哪里？准确率与遗忘各怎么变？
2. 第 4 节 `ranks = (1, 8) if QUICK else (1, 8, 64)` 里加一个 4 与 16：固定 alpha 时准确率随 rank 的曲线是什么形状？

## 注意

- 任务是合成的意图分类；一个 84 万参数的字节级模型比真实的 0.5B 模型小得多，所以“LoRA 比全量微调忘得少”这类结论在这里不成立，实验如实报告。第 8 节的每步毫秒数是这台机器上的计时，在这么小的模型上 LoRA 并不更快。

## 输出

下面是在 Apple M2 Max（12 核，仅用 CPU）、Python 3.12、PyTorch 2.15.0 上的完整运行结果。换一台机器，涉及训练的数字可能在末位上不同，计时类的数字取决于机器；检查用的是阈值与容差，不要求逐位相同。

<details>
<summary>完整运行的输出（`labs/post-training/lora-finetuning/run.py`）</summary>

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
      full fine-tuning             842,496                           12.86                          55
  LoRA r=8, all linear              73,728                            4.34                          58
  static memory = one frozen copy of the weights, plus four copies (weights, gradient, two Adam
  moments) of whatever is trained. Activations come on top and shrink the gap, as on the page.
  [ok] LoRA trains a small fraction of the parameters and needs a fraction of the static memory  (8.8% of the parameters)

All checks passed. (298.3s)
```

</details>
