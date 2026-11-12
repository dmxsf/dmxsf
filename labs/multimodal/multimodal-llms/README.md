# 多模态大模型

课程页面：[多模态 LLM 原理与应用：图像如何变成 token](https://dmxsf.com/wiki/multimodal/multimodal-llms/)  
[English](README.en.md)

从图像块到视觉 token；这些 token 在 KV Cache 与 FLOPs 上的成本；LLaVA 的投影层与两阶段训练；CLIP 的对比损失，以及在渲染图形上从零训练的小型 CLIP。加 --hf 可在 CIFAR-10 上运行 CLIP ViT-B/32。

## 运行

```bash
uv run python labs/multimodal/multimodal-llms/run.py            # 完整运行，约 110 秒
uv run python labs/multimodal/multimodal-llms/run.py --quick    # 缩小规模的冒烟测试，几秒
```

也可以写成 `uv run dmxsf run multimodal-llms`（同样可以加 `--quick`）。`--seed N` 换随机种子，默认 0。脚本分节打印结果，并用 `[ok]` / `[FAIL]` 逐条检查课程页面里的结论：全部通过时退出码为 0，有一条失败就是 1；只有完整运行才做的检查在 `--quick` 下标为 `[skip]`。

可选参数：

- `--hf`：再在 CIFAR-10 测试集的 10,000 张图上跑真实的 CLIP ViT-B/32（openai/clip-vit-base-patch32，零样本），复现课程页的提示词模板准确率；需要 `uv sync --extra hf`、已下载的模型与数据集。

## 代码

| 位置 | 内容 |
| --- | --- |
| `labs/multimodal/multimodal-llms/run.py` | 实验脚本，按编号分节 |
| `src/dmxsf/multimodal/vision.py` | 图像如何变成 token、token 要花多少：切块、合并、动态分辨率、KV Cache 与 FLOPs |
| `src/dmxsf/multimodal/clip.py` | 迷你 CLIP：对称对比损失、双编码器与玩具图文数据集 |

单元测试：`uv run pytest tests/test_multimodal_vision.py tests/test_multimodal_clip.py`

## 看什么

- 第 1 节：页面的代码（用随机数）：96 个 patch 合成 24 个 4 x 1024 的块，再变成 24 个 2560 维的视觉 token，合并器有 27,269,632 个参数；送进 LLM 的是 29 行（2 个文本 + 24 个视觉 + 3 个文本），占位行换成视觉 token 之后，LLM 分不出哪行来自图像。页面的表也逐项对上：CLIP ViT-B/32 是 49 个，LLaVA-1.5 是 576 个，Qwen2-VL-7B 是 3,996 个 patch 合成 999 个 token，Qwen3-VL-4B 是 3,072 个合成 768 个；图像边长翻倍，patch 数翻 4 倍，视觉编码器里随 patch 数平方增长的注意力对数翻 16 倍。
- 第 2 节：一张图占多少 token、多少 KV Cache：页面表的八行全部重算；每个 token 的 KV 是 512 KiB（LLaVA-1.5，没有 GQA）、56 KiB（Qwen2-VL-7B）、144 KiB（Qwen3-VL-4B）；Qwen2-VL-7B 在 1024 x 768 上的 token 数是 LLaVA-1.5 的 1.73 倍，缓存却不到后者的五分之一（0.19）；一张 1,200 万像素的照片占 32K 上下文窗口的 47% 与 36%；token 上限会把图缩到刚好放得下（至多 256、1,024、4,096 个）。
- 第 3 节：LLaVA 的投影层很小：语言模型 6.74B 个参数，投影器 4,198,400 个（0.06%）或两层 MLP 的 20,979,712 个（0.31%）；用玩具的“视觉编码器 → 投影器 → 语言模型”各做一次反向传播：阶段 1 只有投影器拿到梯度，阶段 2 是投影器和语言模型，视觉编码器始终不动。
- 第 4 节：CLIP 的损失与论文的公式逐项相加得到同一个值（1.740978），对图像和文本对称，温度趋于无穷（scale 0）时恰好是 ln n；八对编造的图文，余弦相似度仿页面上真实 CLIP 的取值：对的配对在 1/tau = 100 下损失 0.4051，错位的配对 7.945，不加温度（1/tau = 1）是 2.002，几乎就是随机水平 ln 8 = 2.079。
- 第 5 节：从零训练的小 CLIP（96 句描述、24 x 24 的渲染图形，随机水平 1.0%，每个设置 300 步、3 个种子）：1/tau = 1 是 67.7%，10 是 94.4%，100 是 79.4%，可学习是 88.5%（种子间 83.0 到 94.3）；负例数：批 2 是 5.2%，8 是 51.8%，32 是 88.5%，96 是 100.0%；可学习的温度从初始的 14.3 走到 22.2。更大的批同时让每一步看到更多的图，这里没有把“更多负例”和“每步更多数据”分开。
- 第 6 节：视觉编码器占多少计算：页面 FLOP 表的八行全部重算；它占的比例从 64 个 token 时的 26% 升到 16,384 个 token 时的 70%，因为编码器的注意力随（4 x token 数）的平方增长；与页面实测的时间占比相差不超过 4 个点，并且从不低于它：这是一次对账，不是预测器。
- 第 7 节：分辨率与字符大小：页面里的准确率（Qwen3-VL-4B，4 比特，MLX，没有重跑）旁边，标出每个格子在模型输入里对应的有效字号和 token 数；有效字号相同的格子准确率相差不超过 3 个点，token 数却相差到 9 倍：决定准确率的是字符的大小，不是 token 数。
- 第 8 节（`--hf`）：真实的 CLIP ViT-B/32 在 CIFAR-10 的 10,000 张测试图上做零样本分类（float32，CPU）：不带模板 0.8735，“a photo of a {}.” 0.8831，“a drawing of a {}.” 0.8967，“a low resolution photo of a {}.” 0.8903，无关的句子 0.8733，7 个模板的平均 0.8917。页面的数字几乎一致，只有“a photo of a {}.” 差一张图（页面是 0.8830，用的是 MPS；在这台机器上把页面的代码原样重跑，`DEV = "mps"`，得到的仍是 0.8830，所以是 CPU 上有一张近似平局的图翻转了）。

## 练习

改了参数再运行：核对页面数字的检查会变成 `[FAIL]`，这是预期的，看数字本身怎么变。

1. 把 `steps = 40 if quick else 300` 改成 1200，再看批 8 的准确率：它这时看过的图片数（8 x 1200）与 300 步时的批 32（32 x 300）一样多，它能追上 88.5% 吗？这样才把“更多负例”和“每步更多数据”分开。
2. 把 `by_batch = {b: run(b, None) for b in (2, 8, 32, 96)}` 改成 `(2, 4, 8, 16, 32, 64, 96)`（批不能超过 96，因为一个批里不放重复的描述）：准确率随负例数的增长像 log(批大小) 吗？

## 注意

- 第 5 节的小 CLIP 是玩具任务（渲染的形状与它们的描述），数字只说明机制；第 7 节的准确率是页面里用真实模型（Qwen3-VL-4B，MLX）测得的，这里没有重跑，也复现不了：只重算了每个格子对应的有效字号与 token 数。

## 输出

下面是在 Apple M2 Max（12 核，仅用 CPU）、Python 3.12、PyTorch 2.15.0 上的完整运行结果。换一台机器，涉及训练的数字可能在末位上不同，计时类的数字取决于机器；检查用的是阈值与容差，不要求逐位相同。

<details>
<summary>离线运行的输出（`labs/multimodal/multimodal-llms/run.py`）</summary>

```text
# Multimodal LLMs: images as tokens
  course: https://dmxsf.com/wiki/multimodal/multimodal-llms/
          https://dmxsf.com/en/wiki/multimodal/multimodal-llms/

## 1. From patches to visual tokens (the page's code, with random numbers)
  patches (96, 1024) -> blocks (24, 4096) -> visual tokens (24, 2560); merger parameters: 27,269,632
  LLM input: (29, 2560) = 2 text + 24 visual + 3 text tokens
  [ok] shapes: 96 patches -> 24 blocks of 4 x 1024 -> 24 visual tokens of 2560; 29 rows reach the LLM
  [ok] the merger has (4096 x 4096 + 4096) + (4096 x 2560 + 2560) = 27,269,632 parameters
  [ok] each visual token is built from one 2 x 2 block of neighbouring patches, in row-major order
  [ok] after the splice the placeholder rows are the visual tokens and the others are table rows: the LLM cannot tell which is which

  Patches in the vision encoder, tokens into the language model:
  model (input)                       patch side  patches  visual tokens
  ----------------------------------  ----------  -------  -------------
             CLIP ViT-B/32 (224x224)          32       49             49
              LLaVA-1.5-7B (336x336)          14      576            576
  Qwen2-VL-7B (1024x768 -> 1036x756)          14     3996            999
  Qwen3-VL-4B (1024x768 -> 1024x768)          16     3072            768
  [ok] the page's table: 49, 576, 3,996 -> 999 and 3,072 -> 768
  [ok] doubling the side of the image quadruples the patches (and the visual tokens), so the vision encoder's attention pairs, which grow with patches squared, grow sixteen-fold  (1024 -> 4096 patches)

## 2. How many tokens does an image take, and what do they cost in KV cache
  model        image      after resize  tokens  KV cache (MiB, FP16)
  -----------  ---------  ------------  ------  --------------------
  Qwen2-VL-7B    224x224       224x224      64                     4
  Qwen2-VL-7B   1024x768      1036x756     999                    55
  Qwen2-VL-7B  1920x1080     1932x1092    2691                   147
  Qwen2-VL-7B  4032x3024     4032x3024   15552                   850
  Qwen3-VL-4B    224x224       256x256      64                     9
  Qwen3-VL-4B   1024x768      1024x768     768                   108
  Qwen3-VL-4B  1920x1080     1920x1088    2040                   287
  Qwen3-VL-4B  4032x3024     4032x3008   11844                  1666
  [ok] all eight rows of the page's table  ([])
  [ok] KV cache per token: 512 KiB (LLaVA-1.5, no GQA), 56 KiB (Qwen2-VL-7B), 144 KiB (Qwen3-VL-4B)
  [ok] LLaVA-1.5 always gives 576 tokens and 288 MiB; Qwen2-VL-7B on 1024x768 has 1.7x the tokens and less than a fifth of the cache  (1.73x the tokens, 0.19 of the cache)
  [ok] a 12-megapixel photo takes 47% and 36% of a 32K context window  ([15552, 11844])
  [ok] a token cap shrinks the image so that it fits: 256, 1024 and 4096 tokens at most  ([234, 972, 4015])
  [ok] for 16 image sizes and both models: both sides are multiples of patch x merge, and the pixel count stays inside [min_pixels, max_pixels]

## 3. LLaVA: the projector is small, and each stage trains different parts
  part                                parameters     share of the 6.74B language model
  ----------------------------------  -------------  ---------------------------------
         Vicuna-7B (LLaMA-7B shapes)  6,738,415,616                               100%
   LLaVA projector, one linear layer      4,198,400                              0.06%
  LLaVA-1.5 projector, two-layer MLP     20,979,712                              0.31%
  [ok] the language model has 6.74 billion parameters; the projector 4,198,400 or 20,979,712, which is 0.06% and 0.31%
  A toy vision encoder -> projector -> language model, one backward pass per stage:
  stage  trainable parameters  vision gets a gradient  projector  language model
  -----  --------------------  ----------------------  ---------  --------------
      1                    54                      no        yes              no
      2                    96                      no        yes             yes
  [ok] stage 1 trains only the projector; stage 2 the projector and the language model; the vision encoder never

## 4. CLIP's loss
  [ok] the loss equals the formula of the paper, summed term by term  (1.740978 vs 1.740978)
  [ok] it is symmetric in images and texts, and at temperature infinity (scale 0) it is exactly ln n
  Eight made-up image/text pairs whose similarities look like the page's real CLIP ones: matched
  pairs 0.21 to 0.33, the others 0.09 to 0.26.
                              loss  
  --------------------------  ------
  correct pairs, 1/tau = 100  0.4051
  shifted pairs, 1/tau = 100   7.945
    correct pairs, 1/tau = 1   2.002
          chance level, ln 8   2.079
  [ok] the cosines are close together, as on the page: a gap of about 0.08 between matched and mismatched pairs  (matched 0.279, mismatched 0.190)
  [ok] without the temperature the softmax is nearly uniform: the loss sits within 0.1 of ln 8
  [ok] multiplied by 100 the same gaps give a loss far below chance, and a shifted pairing is punished hard

## 5. A small CLIP, trained from scratch on rendered shapes
  96 captions ('a small red square on the left', ..., 'a large cyan plus on the right'), 24x24 pictures (3, 24, 24), chance = 1.0%. Zero-shot accuracy: the nearest of the
  96 caption embeddings to a fresh picture, no classifier trained.

  The temperature: batch of 32, 300 steps, 3 seed(s). 'learned' starts at 1/0.07.
                 zero-shot accuracy  range over seeds  final loss  final 1/tau
  -------------  ------------------  ----------------  ----------  -----------
      1/tau = 1               67.7%         59.8-73.3       2.592          1.0
     1/tau = 10               94.4%         92.7-95.5       0.159         10.0
    1/tau = 100               79.4%         75.6-85.5       0.154        100.0
  1/tau learned               88.5%         83.0-94.3       0.100         22.2

  The number of negatives: learned temperature, 300 steps, 3 seed(s).
            zero-shot accuracy  range over seeds  final loss  final 1/tau
  --------  ------------------  ----------------  ----------  -----------
   batch 2                5.2%           4.4-5.6       0.241         13.1
   batch 8               51.8%         48.0-55.8       0.178         17.6
  batch 32               88.5%         83.0-94.3       0.100         22.2
  batch 96              100.0%        99.9-100.0       0.011         22.8
  (A bigger batch also shows more pictures per step, and the same number of steps are taken, so
  'more negatives' and 'more data per step' are not separated here.)
  [ok] every setting beats chance: the temperature runs by a wide margin, a single negative by a few times  (worst temperature run 67.7%, worst batch 5.2%, chance 1.0%)
  [ok] without a temperature (1/tau = 1) the loss cannot get small and the embeddings separate worse than with 1/tau = 100 or a learned one  (67.7% vs 79.4% and 88.5%)
  [ok] the learned temperature moves away from its initial 1/0.07 = 14.3 towards a sharper softmax  (22.2)
  [ok] with only one negative per image (batch 2) the model learns less than with 31  (5.2% vs 88.5%)

## 6. The bill: how much of the compute is the vision encoder
  visual tokens  ViT TFLOPs  LLM TFLOPs  ViT share (FLOPs)  ViT share of time (page)
  -------------  ----------  ----------  -----------------  ------------------------
             64        0.16        0.47                26%                       26%
            256        0.72        1.88                28%                       25%
            576        1.91        4.28                31%                       30%
          1,024        4.12        7.75                35%                       33%
          2,304       13.92       18.31                43%                       39%
          4,096       36.28       34.71                51%                       48%
          9,216      155.86       92.02                63%                       60%
         16,384      461.79      198.22                70%                       70%
  [ok] the page's FLOP table, all eight rows
  [ok] the FLOP share is within 4 points of the measured time share, and never below it: a reconciliation, not a predictor  (gaps [0, 3, 1, 2, 4, 3, 3, 0])
  [ok] the vision encoder's attention term has a coefficient about 5.3 times the language model's  (5.33)
  [ok] the share rises with the number of tokens (26% to 70%) because the encoder's attention grows with the square of 4 x tokens

## 7. Resolution and the size of the characters
  The page's measured accuracy (Qwen3-VL-4B, 4-bit, MLX, ignoring stray punctuation; not rerun) next to what
  each cell means in the model's input: the effective font size and the visual tokens of a 512x512 canvas.
  font   x0.5                 x1                    x2                      x3                    
  -----  -------------------  --------------------  ----------------------  ----------------------
   8 px   4 px, 64 tok: 0.04   8 px, 256 tok: 0.90  16 px, 1,024 tok: 0.99  24 px, 2,304 tok: 0.97
  12 px   6 px, 64 tok: 0.75  12 px, 256 tok: 1.00  24 px, 1,024 tok: 1.00  36 px, 2,304 tok: 1.00
  16 px   8 px, 64 tok: 0.92  16 px, 256 tok: 1.00  32 px, 1,024 tok: 1.00  48 px, 2,304 tok: 1.00
  24 px  12 px, 64 tok: 1.00  24 px, 256 tok: 1.00  48 px, 1,024 tok: 1.00  72 px, 2,304 tok: 1.00
  [ok] the token columns of the page's table are 64, 256, 1,024 and 2,304
  [ok] in the page's table, cells with the same effective font size (font x scale) agree to within 3 points although their token counts differ up to 9-fold: the size of the characters, not the token count, sets the accuracy  (largest spread 0.03)

(8. Rerun the page's CLIP experiment on the real model: add --hf. It needs the extra `hf`, the model and CIFAR-10.)

All checks passed. (109.6s)
```

</details>

<details>
<summary>加上真实模型的输出（`labs/multimodal/multimodal-llms/run.py --hf`）</summary>

```text
# Multimodal LLMs: images as tokens
  course: https://dmxsf.com/wiki/multimodal/multimodal-llms/
          https://dmxsf.com/en/wiki/multimodal/multimodal-llms/

## 1. From patches to visual tokens (the page's code, with random numbers)
  patches (96, 1024) -> blocks (24, 4096) -> visual tokens (24, 2560); merger parameters: 27,269,632
  LLM input: (29, 2560) = 2 text + 24 visual + 3 text tokens
  [ok] shapes: 96 patches -> 24 blocks of 4 x 1024 -> 24 visual tokens of 2560; 29 rows reach the LLM
  [ok] the merger has (4096 x 4096 + 4096) + (4096 x 2560 + 2560) = 27,269,632 parameters
  [ok] each visual token is built from one 2 x 2 block of neighbouring patches, in row-major order
  [ok] after the splice the placeholder rows are the visual tokens and the others are table rows: the LLM cannot tell which is which

  Patches in the vision encoder, tokens into the language model:
  model (input)                       patch side  patches  visual tokens
  ----------------------------------  ----------  -------  -------------
             CLIP ViT-B/32 (224x224)          32       49             49
              LLaVA-1.5-7B (336x336)          14      576            576
  Qwen2-VL-7B (1024x768 -> 1036x756)          14     3996            999
  Qwen3-VL-4B (1024x768 -> 1024x768)          16     3072            768
  [ok] the page's table: 49, 576, 3,996 -> 999 and 3,072 -> 768
  [ok] doubling the side of the image quadruples the patches (and the visual tokens), so the vision encoder's attention pairs, which grow with patches squared, grow sixteen-fold  (1024 -> 4096 patches)

## 2. How many tokens does an image take, and what do they cost in KV cache
  model        image      after resize  tokens  KV cache (MiB, FP16)
  -----------  ---------  ------------  ------  --------------------
  Qwen2-VL-7B    224x224       224x224      64                     4
  Qwen2-VL-7B   1024x768      1036x756     999                    55
  Qwen2-VL-7B  1920x1080     1932x1092    2691                   147
  Qwen2-VL-7B  4032x3024     4032x3024   15552                   850
  Qwen3-VL-4B    224x224       256x256      64                     9
  Qwen3-VL-4B   1024x768      1024x768     768                   108
  Qwen3-VL-4B  1920x1080     1920x1088    2040                   287
  Qwen3-VL-4B  4032x3024     4032x3008   11844                  1666
  [ok] all eight rows of the page's table  ([])
  [ok] KV cache per token: 512 KiB (LLaVA-1.5, no GQA), 56 KiB (Qwen2-VL-7B), 144 KiB (Qwen3-VL-4B)
  [ok] LLaVA-1.5 always gives 576 tokens and 288 MiB; Qwen2-VL-7B on 1024x768 has 1.7x the tokens and less than a fifth of the cache  (1.73x the tokens, 0.19 of the cache)
  [ok] a 12-megapixel photo takes 47% and 36% of a 32K context window  ([15552, 11844])
  [ok] a token cap shrinks the image so that it fits: 256, 1024 and 4096 tokens at most  ([234, 972, 4015])
  [ok] for 16 image sizes and both models: both sides are multiples of patch x merge, and the pixel count stays inside [min_pixels, max_pixels]

## 3. LLaVA: the projector is small, and each stage trains different parts
  part                                parameters     share of the 6.74B language model
  ----------------------------------  -------------  ---------------------------------
         Vicuna-7B (LLaMA-7B shapes)  6,738,415,616                               100%
   LLaVA projector, one linear layer      4,198,400                              0.06%
  LLaVA-1.5 projector, two-layer MLP     20,979,712                              0.31%
  [ok] the language model has 6.74 billion parameters; the projector 4,198,400 or 20,979,712, which is 0.06% and 0.31%
  A toy vision encoder -> projector -> language model, one backward pass per stage:
  stage  trainable parameters  vision gets a gradient  projector  language model
  -----  --------------------  ----------------------  ---------  --------------
      1                    54                      no        yes              no
      2                    96                      no        yes             yes
  [ok] stage 1 trains only the projector; stage 2 the projector and the language model; the vision encoder never

## 4. CLIP's loss
  [ok] the loss equals the formula of the paper, summed term by term  (1.740978 vs 1.740978)
  [ok] it is symmetric in images and texts, and at temperature infinity (scale 0) it is exactly ln n
  Eight made-up image/text pairs whose similarities look like the page's real CLIP ones: matched
  pairs 0.21 to 0.33, the others 0.09 to 0.26.
                              loss  
  --------------------------  ------
  correct pairs, 1/tau = 100  0.4051
  shifted pairs, 1/tau = 100   7.945
    correct pairs, 1/tau = 1   2.002
          chance level, ln 8   2.079
  [ok] the cosines are close together, as on the page: a gap of about 0.08 between matched and mismatched pairs  (matched 0.279, mismatched 0.190)
  [ok] without the temperature the softmax is nearly uniform: the loss sits within 0.1 of ln 8
  [ok] multiplied by 100 the same gaps give a loss far below chance, and a shifted pairing is punished hard

## 5. A small CLIP, trained from scratch on rendered shapes
  96 captions ('a small red square on the left', ..., 'a large cyan plus on the right'), 24x24 pictures (3, 24, 24), chance = 1.0%. Zero-shot accuracy: the nearest of the
  96 caption embeddings to a fresh picture, no classifier trained.

  The temperature: batch of 32, 300 steps, 3 seed(s). 'learned' starts at 1/0.07.
                 zero-shot accuracy  range over seeds  final loss  final 1/tau
  -------------  ------------------  ----------------  ----------  -----------
      1/tau = 1               67.7%         59.8-73.3       2.592          1.0
     1/tau = 10               94.4%         92.7-95.5       0.159         10.0
    1/tau = 100               79.4%         75.6-85.5       0.154        100.0
  1/tau learned               88.5%         83.0-94.3       0.100         22.2

  The number of negatives: learned temperature, 300 steps, 3 seed(s).
            zero-shot accuracy  range over seeds  final loss  final 1/tau
  --------  ------------------  ----------------  ----------  -----------
   batch 2                5.2%           4.4-5.6       0.241         13.1
   batch 8               51.8%         48.0-55.8       0.178         17.6
  batch 32               88.5%         83.0-94.3       0.100         22.2
  batch 96              100.0%        99.9-100.0       0.011         22.8
  (A bigger batch also shows more pictures per step, and the same number of steps are taken, so
  'more negatives' and 'more data per step' are not separated here.)
  [ok] every setting beats chance: the temperature runs by a wide margin, a single negative by a few times  (worst temperature run 67.7%, worst batch 5.2%, chance 1.0%)
  [ok] without a temperature (1/tau = 1) the loss cannot get small and the embeddings separate worse than with 1/tau = 100 or a learned one  (67.7% vs 79.4% and 88.5%)
  [ok] the learned temperature moves away from its initial 1/0.07 = 14.3 towards a sharper softmax  (22.2)
  [ok] with only one negative per image (batch 2) the model learns less than with 31  (5.2% vs 88.5%)

## 6. The bill: how much of the compute is the vision encoder
  visual tokens  ViT TFLOPs  LLM TFLOPs  ViT share (FLOPs)  ViT share of time (page)
  -------------  ----------  ----------  -----------------  ------------------------
             64        0.16        0.47                26%                       26%
            256        0.72        1.88                28%                       25%
            576        1.91        4.28                31%                       30%
          1,024        4.12        7.75                35%                       33%
          2,304       13.92       18.31                43%                       39%
          4,096       36.28       34.71                51%                       48%
          9,216      155.86       92.02                63%                       60%
         16,384      461.79      198.22                70%                       70%
  [ok] the page's FLOP table, all eight rows
  [ok] the FLOP share is within 4 points of the measured time share, and never below it: a reconciliation, not a predictor  (gaps [0, 3, 1, 2, 4, 3, 3, 0])
  [ok] the vision encoder's attention term has a coefficient about 5.3 times the language model's  (5.33)
  [ok] the share rises with the number of tokens (26% to 70%) because the encoder's attention grows with the square of 4 x tokens

## 7. Resolution and the size of the characters
  The page's measured accuracy (Qwen3-VL-4B, 4-bit, MLX, ignoring stray punctuation; not rerun) next to what
  each cell means in the model's input: the effective font size and the visual tokens of a 512x512 canvas.
  font   x0.5                 x1                    x2                      x3                    
  -----  -------------------  --------------------  ----------------------  ----------------------
   8 px   4 px, 64 tok: 0.04   8 px, 256 tok: 0.90  16 px, 1,024 tok: 0.99  24 px, 2,304 tok: 0.97
  12 px   6 px, 64 tok: 0.75  12 px, 256 tok: 1.00  24 px, 1,024 tok: 1.00  36 px, 2,304 tok: 1.00
  16 px   8 px, 64 tok: 0.92  16 px, 256 tok: 1.00  32 px, 1,024 tok: 1.00  48 px, 2,304 tok: 1.00
  24 px  12 px, 64 tok: 1.00  24 px, 256 tok: 1.00  48 px, 1,024 tok: 1.00  72 px, 2,304 tok: 1.00
  [ok] the token columns of the page's table are 64, 256, 1,024 and 2,304
  [ok] in the page's table, cells with the same effective font size (font x scale) agree to within 3 points although their token counts differ up to 9-fold: the size of the characters, not the token count, sets the accuracy  (largest spread 0.03)

## 8. CLIP ViT-B/32 on CIFAR-10, zero-shot (float32, CPU)
  10000 test images; temperature scale 100
  [ok] the learned 1/tau sits at the cap of 100, as the paper clips it
  template                         accuracy  page  
  -------------------------------  --------  ------
                  bare class name    0.8735  0.8735
                 a photo of a {}.    0.8831  0.8830
               a drawing of a {}.    0.8967  0.8967
  a low resolution photo of a {}.    0.8903  0.8903
               unrelated sentence    0.8733  0.8733
              mean of 7 templates    0.8917  0.8917
  [ok] the page's six accuracies, to within two images in 10,000 (the page ran on MPS, this on the CPU: an image that is a near-tie can flip)  (differences {'a photo of a {}.': 0.0001})
  [ok] a template is a hyperparameter: 'a drawing of a {}.' beats 'a photo of a {}.' by more than a point, and an unrelated sentence is no better than the bare class name
  loss of one image per class against the 8 class texts: correct 0.307, shifted 6.587, scale 1 2.027, chance 2.079
  [ok] the page's four loss values  (got {})
  [ok] in every row of the 8 x 8 similarity matrix the maximum is on the diagonal

All checks passed. (182.3s)
```

</details>
