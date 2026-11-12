# 位置编码与长上下文

课程页面：[位置编码详解（绝对 → 相对 → RoPE）](https://dmxsf.com/wiki/architecture/positional-encoding/) · [长上下文：位置外推、训练配方与成本](https://dmxsf.com/wiki/architecture/long-context/)  
[English](README.en.md)

正弦编码是一组周期不同的时钟；RoPE 的分数只依赖相对距离；位置插值、NTK-aware、YaRN 与 Llama 3.1 的拉伸配方并排计算；五种位置方案在短文本上训练、在长文本上测试。

## 运行

```bash
uv run python labs/architecture/positional-encoding/run.py            # 完整运行，约 2 分钟
uv run python labs/architecture/positional-encoding/run.py --quick    # 缩小规模的冒烟测试，几秒
```

也可以写成 `uv run dmxsf run positional-encoding`（同样可以加 `--quick`）。`--seed N` 换随机种子，默认 0。脚本分节打印结果，并用 `[ok]` / `[FAIL]` 逐条检查课程页面里的结论：全部通过时退出码为 0，有一条失败就是 1；只有完整运行才做的检查在 `--quick` 下标为 `[skip]`。

## 代码

| 位置 | 内容 |
| --- | --- |
| `labs/architecture/positional-encoding/run.py` | 实验脚本，按编号分节 |
| `src/dmxsf/architecture/position.py` | 位置编码：正弦、RoPE（含上下文扩展的变体）与 ALiBi |

单元测试：`uv run pytest tests/test_architecture_position.py`

## 看什么

- 第 2 节：RoPE 的分数只取决于相对距离：距离 7 的两个位置，不论在 3 与 10、103 与 110 还是 190 与 197，分数都是 −7.5584，与课程页相同。
- 第 3 节：训练长度 4096、目标 32768 时，PI、NTK-aware、YaRN 在“最大没见过的相位”上是 1.00、1.75、1.00（不改动是 8.00），课程页的 YaRN 三段（21 / 25 / 18）和 Llama 3.1 的三段（29 / 6 / 29）都对上了。
- 第 4 节：同一个模型、五种位置方案、训练 64 字节、测试到 256 字节：RoPE 在两倍训练长度之外损失升到 3.4，ALiBi 在各位置都是 1.77 左右；不微调的 NTK-aware 与 YaRN 拉伸 4 倍后，远处位置的损失降到 2.0 与 1.9，线性插值（PI）则连训练过的位置都弄坏了（3.64）。

## 练习

改了参数再运行：核对页面数字的检查会变成 `[FAIL]`，这是预期的，看数字本身怎么变。

1. 第 4 节用 `--seed` 换种子重跑：各方案的排名稳定吗？哪几项差别小于种子之间的波动？
2. 第 3 节把 `dim, base, trained, s = 128, 10_000.0, 4096, 8.0` 里的 `s` 改成 32（目标长度 131072），NTK-aware 与 YaRN 的“最大没见过的相位”还是 1.75 与 1.00 吗？

## 输出

下面是在 Apple M2 Max（12 核，仅用 CPU）、Python 3.12、PyTorch 2.15.0 上的完整运行结果。换一台机器，涉及训练的数字可能在末位上不同，计时类的数字取决于机器；检查用的是阈值与容差，不要求逐位相同。

<details>
<summary>完整运行的输出（`labs/architecture/positional-encoding/run.py`）</summary>

```text
# Position encodings and length extrapolation
  course: https://dmxsf.com/wiki/architecture/positional-encoding/
          https://dmxsf.com/en/wiki/architecture/positional-encoding/
  course: https://dmxsf.com/wiki/architecture/long-context/
          https://dmxsf.com/en/wiki/architecture/long-context/

## 1. Sinusoidal: every pair of dimensions is a clock with its own period
  32 clocks: periods from 6.3 to 47,117 positions
  [ok] moving k positions rotates each pair by a fixed angle (so offsets are easy to learn)

## 2. RoPE: the score depends only on the distance
  m=  3  n= 10  n-m=7  score -7.5584
  m= 13  n= 20  n-m=7  score -7.5584
  m=103  n=110  n-m=7  score -7.5584
  m=190  n=197  n-m=7  score -7.5584
  [ok] the same offset gives the same score at any position (page: -7.5584)
  [ok] RoFormer's adjacent pairing and Llama's rotate_half differ only by a permutation
  d=128, base 10,000: wavelengths run from 6.28 to 54,410 positions
  [ok] the fastest pair turns every 6.28 positions, the slowest every 54,410

## 3. Stretching RoPE: PI, NTK-aware, YaRN and the Llama 3.1 rule
  d=128, base 10,000, trained on 4096, target 32768 (s = 8)
  pairs that turn less than once inside the training length: 18 of 64 (pairs 46 to 63)
  method            pair 0 frequency  slowest pair divided by  largest unseen phase
  ----------------  ----------------  -----------------------  --------------------
  RoPE (no change)             1.000                    1.000                 8.000
                PI            0.1250                    8.000                 1.000
         NTK-aware             1.000                    8.000                 1.753
              YaRN             1.000                    8.000                 1.000
  [ok] the course page's table: 8.00 / 1.00 / 1.75 / 1.00
  YaRN: 21 pairs unchanged, 25 blended, 18 divided by s; attention factor 1.208 (q.k grows 1.46x)
  Llama 3.1 (base 500,000): 29 pairs divided by 8, 6 blended, 29 unchanged
  [ok] segments match the course page: YaRN 21/25/18, Llama 3.1 29/6/29
  NTK-aware base: 82,685   ALiBi slopes (8 heads): 1/2 ... 1/256

## 4. Train short, test long: five position schemes, same model otherwise
  2 layers, d_model 64, 1000 steps on Shakespeare bytes with 64-byte windows, 2 seed(s); loss in nats per byte
  evaluated on 256-byte windows, split by position (position < 64 was seen in training)
  scheme            pos 0-63  pos 64-127  pos 128-255
  ----------------  --------  ----------  -----------
           learned     1.827         n/a          n/a
        sinusoidal     1.990       3.271        3.386
              rope     1.606       1.824        3.428
  rope + linear x4     3.641       3.678        3.708
     rope + ntk x4     1.642       1.637        2.015
    rope + yarn x4     1.759       1.802        1.857
  rope + llama3 x4     1.938       2.059        2.038
             alibi     1.766       1.768        1.775
              none     2.155       2.393        2.505
  (learned positions exist only for the first 64 positions; the four rows below RoPE
   reuse the RoPE models, stretched 4x at test time with no further training)
  [ok] learned positions cannot even run past the training length
  [ok] ALiBi is flat: far positions cost the same as near ones (within 0.1 nats)
  [ok] RoPE collapses beyond about twice the training length (+1 nat)
  [ok] sinusoidal degrades beyond the training length too (+0.5 nats)
  [ok] PI without fine-tuning damages even the positions it was trained on (+0.5 nats)
  [ok] NTK-aware and YaRN stretch RoPE 4x with no training: far positions improve by 0.5+ nats

All checks passed. (156.2s)
```

</details>
