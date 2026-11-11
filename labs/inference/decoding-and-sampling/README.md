# 解码与采样

课程页面：[解码与采样：温度、Top-k、Top-p 与束搜索](https://dmxsf.com/wiki/inference/decoding-and-sampling/)  
[English](README.en.md)

精确复现课程页的四 token 例子（温度、top-k、top-p、min-p 以及顺序为何重要）；采样是否服从处理后的分布；各个旋钮对真实文本的影响；束搜索、重复惩罚；批次形状是否会改变贪心输出。

## 运行

```bash
uv run python labs/inference/decoding-and-sampling/run.py            # 完整运行，约 10 秒
uv run python labs/inference/decoding-and-sampling/run.py --quick    # 缩小规模的冒烟测试，几秒
```

也可以写成 `uv run dmxsf run decoding-and-sampling`（同样可以加 `--quick`）。`--seed N` 换随机种子，默认 0。脚本分节打印结果，并用 `[ok]` / `[FAIL]` 逐条检查课程页面里的结论：全部通过时退出码为 0，有一条失败就是 1；只有完整运行才做的检查在 `--quick` 下标为 `[skip]`。

这个实验用到缓存的小模型：字节级基础模型（84 万参数）。第一次运行会先在 TinyShakespeare 上训练并缓存到 `~/.cache/dmxsf`（环境变量 `DMXSF_CACHE` 可以改位置），之后直接读取。

## 代码

| 位置 | 内容 |
| --- | --- |
| `labs/inference/decoding-and-sampling/run.py` | 实验脚本，按编号分节 |
| `src/dmxsf/inference/generate.py` | 带 KV Cache 的自回归生成：每个聊天产品都在跑的循环 |

单元测试：`uv run pytest tests/test_inference_generate.py`

## 看什么

- 第 1 节：页面里 logits = [4, 3, 1, 0] 的每一行（温度、top-k、top-p、min-p）都到三位小数一致；顺序有影响：先温度再 top-p 保留 2 个 token，先 top-p 再温度保留 4 个。
- 第 2 节：每种设置抽 200,000 次，频率与处理后的分布相差不超过 4 个标准误（最差 z = 1.8），被截掉的 token 从未被抽到。
- 第 3 节：在字节级 GPT 上看这些设置对文本做了什么：贪心解码最容易循环（不同 4-gram 占 0.64，采样 T ≥ 1 是 0.98）；T = 2 时真实单词只剩 37.4%（T = 1 是 87.1%）；T = 1.5 再加 top-p 0.9 找回真实单词（77.0%，不加是 61.9%）；换种子重跑 T = 1：87.8%，噪声大约一个点。
- 第 4 节：束搜索找到的序列比贪心概率更高（8 个提示全部如此），每字节 0.82 比特，贪心 1.15，真实文本 1.95：最可能的文本并不最自然。
- 第 5 节：256 个符号的词表上，重复惩罚比真实 10 万词表上猛得多（强度 2.0 就让真实单词降到 95.6%，贪心是 99.1%），读作强度的警告而不是配方。
- 第 6 节：同样的 64 个提示，整批跑和逐个跑，logits 最大差 0（逐位相同），贪心选择没有不同：这台 CPU 上批的形状不改变算术。页面的警告（别的硬件上不同批大小会选不同的内核）这里演示不了。

## 练习

改了参数再运行：核对页面数字的检查会变成 `[FAIL]`，这是预期的，看数字本身怎么变。

1. 把 `Z = torch.tensor([[4.0, 3.0, 1.0, 0.0]])` 换成更平的 logits，例如 `[2.0, 1.8, 1.5, 0.0]`：top-p 0.9 还保留几个 token？
2. 第 3 节 `n_prompts, plen, new = (8, 48, 24) if QUICK else (64, 48, 80)` 里改成 `(64, 8, 120)`（模型的上下文是 128 字节，提示加续写不能超过它）：续写更长以后，贪心的循环会更严重吗？T = 1 的真实单词比例会掉吗？

## 输出

下面是在 Apple M2 Max（12 核，仅用 CPU）、Python 3.12、PyTorch 2.15.0 上的完整运行结果。换一台机器，涉及训练的数字可能在末位上不同，计时类的数字取决于机器；检查用的是阈值与容差，不要求逐位相同。

<details>
<summary>完整运行的输出（`labs/inference/decoding-and-sampling/run.py`）</summary>

```text
# Decoding and sampling: temperature, top-k, top-p, min-p, beam search, penalties
  course: https://dmxsf.com/wiki/inference/decoding-and-sampling/
          https://dmxsf.com/en/wiki/inference/decoding-and-sampling/

## 1. The page's example: logits z = [4, 3, 1, 0]
  setting             token 1  token 2  token 3  token 4
  ------------------  -------  -------  -------  -------
             T = 0.5    0.879    0.119    0.002    0.000
               T = 1    0.696    0.256    0.035    0.013
               T = 2    0.509    0.309    0.114    0.069
             top-k 2    0.731    0.269    0.000    0.000
           top-p 0.9    0.731    0.269    0.000    0.000
  T = 0.5, top-p 0.9    0.881    0.119    0.000    0.000
           min-p 0.1    0.731    0.269    0.000    0.000
  [ok] every row matches the page's numbers (three decimals)

  the order matters (top-p 0.99):
  order                         token 1  token 2  token 3  token 4
  ----------------------------  -------  -------  -------  -------
   temperature 0.5, then top-p    0.881    0.119    0.000    0.000
  top-p at T = 1, then T = 0.5    0.879    0.119    0.002    0.000
  [ok] temperature first keeps 2 tokens, nucleus first keeps 4: the order changes the result

## 2. Do the samples follow the processed distribution?
  setting             p1     p2     p3     p4     freq 1  freq 2  freq 3  freq 4  max |z|
  ------------------  -----  -----  -----  -----  ------  ------  ------  ------  -------
             top-k 2  0.731  0.269  0.000  0.000   0.733   0.267   0.000   0.000      1.7
  T = 0.5, top-p 0.9  0.881  0.119  0.000  0.000   0.880   0.120   0.000   0.000      1.8
    T = 2, min-p 0.2  0.547  0.331  0.122  0.000   0.547   0.331   0.123   0.000      0.9
  [ok] 200,000 draws per setting: frequencies within 4 standard errors, dropped tokens never drawn  (worst z = 1.8)

## 3. What the settings do to text (byte-level GPT on TinyShakespeare)
  64 prompts of 48 bytes from the held-out text, 80 new bytes each (about 842 words per setting):
  setting                  words in the training text  distinct 4-grams  bits/byte under the model
  -----------------------  --------------------------  ----------------  -------------------------
           greedy (T = 0)                       99.1%              0.64                       1.15
                  T = 0.5                       98.6%              0.94                       1.34
                    T = 1                       87.1%              0.98                       1.91
                  T = 1.5                       61.9%              0.99                       2.92
                    T = 2                       37.4%              1.00                       4.47
         T = 1, top-p 0.9                       94.5%              0.98                       1.62
        T = 1, min-p 0.05                       94.3%              0.98                       1.63
       T = 1.5, top-p 0.9                       77.0%              0.99                       2.35
  (the real continuation)                       94.1%              0.98                       2.33
  distinct 4-grams: share of byte 4-grams in a continuation that are unique (low = looping).
  bits/byte: how surprised the model (at T = 1) is by the text, which includes its own output.
  [ok] greedy loops more than any sampled setting at T >= 1 (distinct 4-grams)  (0.64 against 0.98)
  [ok] heating up lowers the share of real words (T = 2 against T = 1, by more than 10 points)  (37.4% against 87.1%)
  [ok] cutting the tail at T = 1.5 (top-p 0.9) recovers real words  (77.0% against 61.9%)
  noise: T = 1 again with another seed gives 87.8% real words, 0.98 distinct, 1.96 bits/byte (first run 87.1%, 0.98, 1.91).

## 4. Beam search: the most probable text is not the most natural
  continuation          bits/byte under the model  distinct 4-grams
  --------------------  -------------------------  ----------------
             real text                       1.95                 -
        sampled, T = 1                       1.87              0.98
                greedy                       1.15              0.71
  beam search, 4 beams                       0.82              0.70
  beam search found a strictly more probable sequence than greedy on 8 of 8 prompts (0 ties).
  [ok] greedy and beam output are far more probable than the real text (open-ended text is not the mode)  (real 1.95, greedy 1.15, beam 0.82 bits/byte)
  [ok] no beam is worse than greedy (not a guarantee of the algorithm, only what these prompts show)  (beam better on 8, equal on 0 of 8)

## 5. Repetition penalties on a 256-symbol vocabulary
  greedy decoding with  words in the training text  distinct 4-grams  bits/byte under the model
  --------------------  --------------------------  ----------------  -------------------------
                greedy                       99.1%              0.64                       1.15
        repetition 1.1                       99.7%              0.80                       1.08
        repetition 1.3                       98.3%              0.92                       1.14
        repetition 2.0                       95.6%              0.92                       1.71
         frequency 0.5                       96.6%              0.95                       1.35
          presence 0.5                       99.2%              0.80                       1.10
  A byte-level model has 256 symbols and the prompt already contains every common letter,
  so 'seen' is nearly everything: these penalties are far harsher than on a vocabulary of
  100k tokens. Read the table as a warning about strength, not as a recipe.
  [ok] a very strong repetition penalty (2.0) costs real words against plain greedy, by more than 2 points  (95.6% against 99.1%)

## 6. Does the batch shape change greedy decoding?
  the same 64 prompts, run as one batch and one at a time: max |logit difference| 0.0e+00,
  smallest gap between the best and second-best logit 2.3e-02, greedy choices that differ: 0.
  bit-identical here: on this CPU the arithmetic is the same for every batch shape, so nothing flips.
  The page's warning (a library may pick different kernels, and a different summation order, per
  batch size) is about other hardware; this lab cannot show it.
  [ok] batch shape changes the logits by at most float rounding (relative to the logit scale)  (0.0e+00)

All checks passed. (8.5s)
```

</details>
