# 注意力机制

课程页面：[Transformer 注意力机制入门](https://dmxsf.com/wiki/architecture/attention-basics/)  
[English](README.en.md)

手算课程页的三位置例子；为什么要除以 sqrt(d)（点积的方差是 d）；掩码、多头与位置；再训练一个必须靠注意力才能完成“复制上次出现之后的内容”的小模型。

## 运行

```bash
uv run python labs/architecture/attention-basics/run.py            # 完整运行，约 35 秒
uv run python labs/architecture/attention-basics/run.py --quick    # 缩小规模的冒烟测试，几秒
```

也可以写成 `uv run dmxsf run attention-basics`（同样可以加 `--quick`）。`--seed N` 换随机种子，默认 0。脚本分节打印结果，并用 `[ok]` / `[FAIL]` 逐条检查课程页面里的结论：全部通过时退出码为 0，有一条失败就是 1；只有完整运行才做的检查在 `--quick` 下标为 `[skip]`。

## 代码

| 位置 | 内容 |
| --- | --- |
| `labs/architecture/attention-basics/run.py` | 实验脚本，按编号分节 |
| `src/dmxsf/architecture/attention.py` | 缩放点积注意力、多头 / 分组查询注意力与因果掩码 |

单元测试：`uv run pytest tests/test_architecture_attention.py`

## 看什么

- 第 1 节：课程页三位置例子的权重 [0.40, 0.20, 0.40] 与输出 [6.4, 4.4]。
- 第 2 节：点积的方差约等于 d（d=1024 时 1,030.6），除以 √d 后回到约 1；不缩放时 softmax 随 d 饱和（最大权重 → 1），缩放后基本不变。
- 第 3 节：因果掩码让每个 token 看不到未来；多头不改变参数量（始终 4d²）；打乱 token 只会打乱输出，注意力本身看不到顺序。
- 第 4 节：一个必须用注意力才能完成“复制上次出现之后的那个 token”的小模型，损失长时间停在随机水平，然后在一小段训练里突然从 3.2 掉到 0.5；第 0 层有“前一个 token”头（0.67），第 1 层有归纳头（0.92），关掉它们损失回到 4.8 以上。

## 练习

改了参数再运行：核对页面数字的检查会变成 `[FAIL]`，这是预期的，看数字本身怎么变。

1. 把第 4 节配置里的 `n_layer=2` 改成 1：模型还能学会复制吗？（提示：归纳头需要两层配合。）
2. 把第 2 节 `for d in (4, 16, 64, 256, 1024)` 加上 4096，方差和最大权重怎么变？

## 输出

下面是在 Apple M2 Max（12 核，仅用 CPU）、Python 3.12、PyTorch 2.15.0 上的完整运行结果。换一台机器，涉及训练的数字可能在末位上不同，计时类的数字取决于机器；检查用的是阈值与容差，不要求逐位相同。

<details>
<summary>完整运行的输出（`labs/architecture/attention-basics/run.py`）</summary>

```text
# Attention, by hand and in a model
  course: https://dmxsf.com/wiki/architecture/attention-basics/
          https://dmxsf.com/en/wiki/architecture/attention-basics/

## 1. The course page's three-position example
  scores / sqrt(2) = [0.71, 0.0, 0.71]
  weights          = [0.4, 0.2, 0.4]
  output           = [6.4, 4.4]
  [ok] weights [0.40, 0.20, 0.40] and output [6.4, 4.4], as on the page

## 2. Why divide by sqrt(d): the dot product has variance d
  d     var(q.k)  var after scaling  max weight, raw  max weight, scaled
  ----  --------  -----------------  ---------------  ------------------
     4     3.945             0.9862           0.5264              0.3422
    16     15.76             0.9851           0.7413              0.3517
    64     64.00              1.000           0.8776              0.3618
   256     260.4              1.017           0.9388              0.3637
  1024   1,030.6              1.006           0.9670              0.3624
  [ok] var(q.k) is about d
  [ok] scaling brings the variance back to about 1
  [ok] without scaling the softmax saturates (largest weight -> 1); with it nothing changes with d

## 3. Masks, heads and positions
  causal attention weights of a 5-token sequence (row = the token that is looking):
    1.00 0.00 0.00 0.00 0.00
    0.27 0.73 0.00 0.00 0.00
    0.01 0.01 0.98 0.00 0.00
    0.08 0.18 0.43 0.30 0.00
    0.01 0.02 0.07 0.03 0.88
  [ok] a token never attends to the future
  [ok] parameters: 4 d^2 whatever the number of heads (d = 64: 16,384)  (1, 4 and 8 heads: [16384])
  [ok] shuffling the tokens only shuffles the outputs: attention alone cannot see order  (max difference 2.4e-07)

## 4. A model that has to use attention: copy what came after the last time
  2 layers x 4 heads, d_model 64, 115,008 parameters; vocabulary 128, ln(vocab) = 4.852
  step  loss on the repeat  loss elsewhere
  ----  ------------------  --------------
   200               4.638           4.875
   400               4.436           4.879
   600               4.209           4.892
   800               4.102           4.888
  1000               3.652           4.883
  1200               3.244           4.894
  1400              0.5145           4.959
  1600              0.4486           4.922
  1800              0.3625           4.918
  2000              0.3294           4.918
  the repeat's loss fell by 2.38 nats between step 1200 and step 1300
  [ok] it starts at chance
  [ok] then falls abruptly, by more than 1.5 nats in a twentieth of the run
  [ok] the repeat is predicted far better than chance  (0.33 nats)
  [ok] everything else stays at chance (it is random)

  attention paid by the tokens inside the repeat, averaged over 256 sequences:
  head            to the previous token  to the token after the earlier copy
  --------------  ---------------------  -----------------------------------
  layer 0 head 0              0.0009116                              0.01513
  layer 0 head 1                 0.6700                            0.0003745
  layer 0 head 2                0.02275                              0.03975
  layer 0 head 3              0.0002441                              0.01278
  layer 1 head 0              4.280e-06                               0.2698
  layer 1 head 1              0.0002490                               0.9216
  layer 1 head 2              1.293e-07                              0.04565
  layer 1 head 3              0.0002228                               0.1041
  [ok] layer 0 has a previous-token head (> 0.3)
  [ok] layer 1 has induction heads (> 0.7)

  loss on the repeat when one head is switched off (all heads on: 0.329):
  head switched off  loss on the repeat
  -----------------  ------------------
     layer 0 head 0              0.5097
     layer 0 head 1               5.195
     layer 0 head 2              0.3171
     layer 0 head 3              0.5210
     layer 1 head 0              0.3607
     layer 1 head 1               4.800
     layer 1 head 2              0.5825
     layer 1 head 3               1.118
  [ok] switching off the right head costs at least 1 nat  (worst 5.19)

All checks passed. (32.4s)
```

</details>
