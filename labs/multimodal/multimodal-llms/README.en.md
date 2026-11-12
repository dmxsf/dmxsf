# Multimodal LLMs: images as tokens

Course pages: [Multimodal LLMs: How an Image Becomes Tokens](https://dmxsf.com/en/wiki/multimodal/multimodal-llms/)  
[中文](README.md)

Patches to visual tokens, what the tokens cost in KV cache and FLOPs, LLaVA's projector and two training stages, CLIP's contrastive loss and a small CLIP trained from scratch on rendered shapes. With --hf, CLIP ViT-B/32 on CIFAR-10.

## Run

```bash
uv run python labs/multimodal/multimodal-llms/run.py            # full run, about 110 seconds
uv run python labs/multimodal/multimodal-llms/run.py --quick    # a shrunk smoke test, a few seconds
```

You can also write `uv run dmxsf run multimodal-llms` (and add `--quick`). `--seed N` changes the random seed, default 0. The script prints its results section by section and checks the course page's claims one by one with `[ok]` / `[FAIL]`: the exit status is 0 when all pass and 1 when any fails; checks that only make sense in the full run are marked `[skip]` under `--quick`.

Optional flags:

- `--hf`: also run the real CLIP ViT-B/32 (openai/clip-vit-base-patch32, zero-shot) on the 10,000 CIFAR-10 test images and reproduce the page's accuracies for the prompt templates. Needs `uv sync --extra hf`, the downloaded model and dataset.

## Code

| File | What it holds |
| --- | --- |
| `labs/multimodal/multimodal-llms/run.py` | the lab script, numbered sections |
| `src/dmxsf/multimodal/vision.py` | How an image becomes tokens, and what the tokens cost: patches, merging, dynamic resolution, KV cache, FLOPs |
| `src/dmxsf/multimodal/clip.py` | CLIP in miniature: the symmetric contrastive loss, a dual encoder and a toy image-caption dataset |

Unit tests: `uv run pytest tests/test_multimodal_vision.py tests/test_multimodal_clip.py`

## What to look for

- Section 1: the page's code (with random numbers): 96 patches become 24 blocks of 4 x 1024, then 24 visual tokens of 2560, and the merger has 27,269,632 parameters; the LLM receives 29 rows (2 text + 24 visual + 3 text), and once the placeholders are replaced the LLM cannot tell which row came from an image. The page's table matches too: CLIP ViT-B/32 has 49 tokens, LLaVA-1.5 576, Qwen2-VL-7B turns 3,996 patches into 999 tokens and Qwen3-VL-4B 3,072 into 768; doubling the side of an image quadruples the patches, and the encoder's attention, which grows with the square of the patches, sixteen-fold.
- Section 2: what an image costs in tokens and KV cache: all eight rows of the page's table are recomputed; the KV cache takes 512 KiB per token for LLaVA-1.5 (no GQA), 56 KiB for Qwen2-VL-7B and 144 KiB for Qwen3-VL-4B; on 1024 x 768 Qwen2-VL-7B has 1.73 times LLaVA-1.5's tokens and less than a fifth (0.19) of its cache; a 12-megapixel photo takes 47% and 36% of a 32K context window; a token cap shrinks the image until it fits (at most 256, 1,024 or 4,096 tokens).
- Section 3: LLaVA's projector is small: the language model has 6.74B parameters, the projector 4,198,400 (0.06%) or, as a two-layer MLP, 20,979,712 (0.31%); one backward pass of a toy vision encoder -> projector -> language model per stage: in stage 1 only the projector gets a gradient, in stage 2 the projector and the language model, and the vision encoder never does.
- Section 4: CLIP's loss equals the paper's formula summed term by term (1.740978), is symmetric in images and texts and, at temperature infinity (scale 0), is exactly ln n; eight made-up pairs with cosine similarities like the page's real CLIP ones: matched pairs lose 0.4051 at 1/tau = 100, shifted pairs 7.945, and without the temperature (1/tau = 1) the loss is 2.002, nearly chance level ln 8 = 2.079.
- Section 5: a small CLIP trained from scratch (96 captions, rendered 24 x 24 shapes, chance 1.0%, 300 steps and 3 seeds per setting): 1/tau = 1 gives 67.7%, 10 gives 94.4%, 100 gives 79.4% and a learned temperature 88.5% (83.0 to 94.3 over seeds); the number of negatives: batch 2 gives 5.2%, 8 gives 51.8%, 32 gives 88.5% and 96 gives 100.0%; the learned temperature moves from its initial 14.3 to 22.2. A bigger batch also shows more pictures per step, and 'more negatives' is not separated from 'more data per step' here.
- Section 6: how much of the compute is the vision encoder: all eight rows of the page's FLOP table are recomputed; its share rises from 26% at 64 tokens to 70% at 16,384 because the encoder's attention grows with the square of (4 x tokens); it is within 4 points of the page's measured time share and never below it: a reconciliation, not a predictor.
- Section 7: resolution and the size of the characters: next to the page's measured accuracies (Qwen3-VL-4B, 4-bit, MLX, not rerun) each cell is labelled with the effective font size and the number of tokens it means in the model's input; cells with the same effective font size agree within 3 points although their token counts differ up to 9-fold: the size of the characters, not the token count, sets the accuracy.
- Section 8 (`--hf`): the real CLIP ViT-B/32 on the 10,000 CIFAR-10 test images, zero-shot (float32, CPU): bare class name 0.8735, 'a photo of a {}.' 0.8831, 'a drawing of a {}.' 0.8967, 'a low resolution photo of a {}.' 0.8903, an unrelated sentence 0.8733 and the mean of 7 templates 0.8917. The page's numbers agree, except 'a photo of a {}.' by one image (the page says 0.8830 and ran on MPS; rerunning the page's code unchanged on this machine, with `DEV = "mps"`, gives 0.8830 again, so it is the CPU on which one near-tie image flips).

## Exercises

Run again after a change: the checks that compare with the page's numbers turn into `[FAIL]`; that is expected, read how the numbers themselves move.

1. Change `steps = 40 if quick else 300` to 1200 and read the accuracy of batch 8: it has now seen as many pictures (8 x 1200) as batch 32 had at 300 steps (32 x 300); does it catch up with 88.5%? That separates 'more negatives' from 'more data per step'.
2. Change `by_batch = {b: run(b, None) for b in (2, 8, 32, 96)}` to `(2, 4, 8, 16, 32, 64, 96)` (a batch cannot exceed 96, because a batch holds no repeated caption): does the accuracy grow like log(batch size)?

## Notes

- The small CLIP of section 5 is a toy task (rendered shapes and their captions): its numbers show the mechanism and nothing more. The accuracies of section 7 were measured on the page with a real model (Qwen3-VL-4B, MLX) and are not rerun here: only the effective font size and the token count of each cell are recomputed.

## Output

The complete runs below were made on an Apple M2 Max (12 cores, CPU only) with Python 3.12 and PyTorch 2.15.0. On another machine the numbers that involve training can differ in the last digits and timings depend on the machine; the checks use thresholds and tolerances, not bit-for-bit equality.

<details>
<summary>Output of the offline run (`labs/multimodal/multimodal-llms/run.py`)</summary>

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
<summary>Output with the real models (`labs/multimodal/multimodal-llms/run.py --hf`)</summary>

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
