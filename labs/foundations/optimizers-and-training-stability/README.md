# 优化器与训练稳定性

课程页面：[优化器与训练稳定性：AdamW、学习率调度、混合精度与损失尖峰](https://dmxsf.com/wiki/foundations/optimizers-and-training-stability/)  
[English](README.en.md)

手写 SGD、Adam、AdamW 并在 float64 下与 PyTorch 对拍；Adam 为何在病态二次函数上更快；解耦的权重衰减；学习率调度；混合精度与 FP32 主权重；一个被污染的批次里梯度裁剪和 beta2 各起什么作用。

## 运行

```bash
uv run python labs/foundations/optimizers-and-training-stability/run.py            # 完整运行，几秒
uv run python labs/foundations/optimizers-and-training-stability/run.py --quick    # 缩小规模的冒烟测试，几秒
```

也可以写成 `uv run dmxsf run optimizers-and-training-stability`（同样可以加 `--quick`）。`--seed N` 换随机种子，默认 0。脚本分节打印结果，并用 `[ok]` / `[FAIL]` 逐条检查课程页面里的结论：全部通过时退出码为 0，有一条失败就是 1；只有完整运行才做的检查在 `--quick` 下标为 `[skip]`。

## 代码

| 位置 | 内容 |
| --- | --- |
| `labs/foundations/optimizers-and-training-stability/run.py` | 实验脚本，按编号分节 |
| `src/dmxsf/foundations/optim.py` | 从零实现的 SGD、Adam、AdamW，加上学习率调度与梯度裁剪 |

单元测试：`uv run pytest tests/test_foundations_optim.py`

## 看什么

- 第 1 节：自己写的 SGD、Adam、AdamW 和 `torch.optim` 在 float64 下只差 5.6e-17。
- 第 2 节：在条件数 100 的二次函数上，SGD 要 162 步，动量 103 步，Adam 24 步：Adam 给每个坐标各除一个梯度尺度。
- 第 3 节：AdamW 的两个权重（梯度相差 1000 倍）走同一条轨迹，Adam + L2 却让小梯度的权重缩到接近 0（−0.0104 对 +0.9322）。
- 第 4 节：课程页的学习率数字（余弦与线性平均 0.548，WSD 0.885，约为余弦的 1.6 倍）。
- 第 6 节：纯 BF16 的更新小于半个 BF16 步长，被舍入掉，损失停在 7.281；BF16 计算加 FP32 主权重则达到 0.01978，与纯 FP32 的 0.01943 相当。
- 第 7 节：一个被放大 1000 倍的坏批次，不裁剪时 250 步后损失仍是对照的 1000 倍，梯度裁剪到 1.0 则把峰值压到不裁剪的 1% 以下。

## 练习

改了参数再运行：核对页面数字的检查会变成 `[FAIL]`，这是预期的，看数字本身怎么变。

1. 第 2 节把 SGD 的 `lr=0.019` 改成 0.021（超过稳定上限 2/100）：会怎样？
2. 第 7 节的 `for clip, beta2 in [...]` 里加一项 `(None, 0.99)`：它“忘记”坏批次的速度落在 0.999 与 0.95 之间吗？

## 输出

下面是在 Apple M2 Max（12 核，仅用 CPU）、Python 3.12、PyTorch 2.15.0 上的完整运行结果。换一台机器，涉及训练的数字可能在末位上不同，计时类的数字取决于机器；检查用的是阈值与容差，不要求逐位相同。

<details>
<summary>完整运行的输出（`labs/foundations/optimizers-and-training-stability/run.py`）</summary>

```text
# Optimizers, schedules, mixed precision and loss spikes
  course: https://dmxsf.com/wiki/foundations/optimizers-and-training-stability/
          https://dmxsf.com/en/wiki/foundations/optimizers-and-training-stability/

## 1. Our optimizers against PyTorch's (float64, 100 steps)
  optimizer     max |ours - torch|
  ------------  ------------------
  SGD+momentum             5.6e-17
          Adam             5.6e-17
         AdamW             5.6e-17
  [ok] all three agree with torch.optim to 1e-10

## 2. Why Adam: an ill-conditioned quadratic f = (100 x^2 + y^2) / 2
  optimizer                                        steps until f < 1e-3
  -----------------------------------------------  --------------------
  SGD, lr 0.019 (the largest stable one is 2/100)                   162
                     SGD + momentum 0.9, lr 0.019                   103
                                    Adam, lr 0.05                    24
  The curvature differs 100x between the axes: one learning rate cannot suit both for SGD,
  while Adam divides each coordinate by its own gradient scale.
  [ok] Adam gets there first

## 3. AdamW decouples weight decay; Adam + L2 does not
  AdamW      weights after 300 steps (both started at 1): +0.6921 (gradient x0.01)  +0.6921 (gradient x10)
  Adam + L2  weights after 300 steps (both started at 1): -0.0104 (gradient x0.01)  +0.9322 (gradient x10)
  [ok] AdamW gives the two weights the same trajectory (scale-invariant)
  [ok] Adam + L2 shrinks the small-gradient weight much more than the large-gradient one

## 4. Learning-rate schedules
  schedule  mean    at 25%  at 50%  at 75%  last step
  --------  ------  ------  ------  ------  ---------
    cosine  0.5475  0.9051  0.5872  0.2452     0.1000
    linear  0.5475  0.8106  0.5737  0.3369     0.1000
       wsd  0.8850   1.000   1.000   1.000     0.1000
   invsqrt  0.3722  0.4472  0.3162  0.2582     0.2236
  20,000 steps, 1,000 warmup steps, floor 10% of the peak, WSD decays over the last 20%
  [ok] course page: cosine and linear average 0.548, WSD 0.885 (about 1.6x cosine)
  [ok] course page: inverse square root is 0.447 / 0.316 / 0.258 and ends at 0.224

## 5. The course page's training loop: AdamW + warmup/cosine + clipping
  step   0  lr 3.00e-06  grad_norm 7.770  loss 16.4080
  step  99  lr 3.00e-04  grad_norm 7.666  loss 16.0073
  step 100  lr 3.00e-04  grad_norm 7.664  loss 15.9993
  step 500  lr 1.88e-04  grad_norm 6.959  loss 13.4174
  step 999  lr 3.00e-05  grad_norm 6.674  loss 12.4340
  [ok] same five lines as the course page
  [ok] our loop and torch's end with the same weights  (max difference 6.0e-08)

## 6. Mixed precision: why the master weights stay in FP32
  format  max       smallest normal  machine eps
  ------  --------  ---------------  -----------
    FP32   3.4e+38          1.2e-38      1.2e-07
    FP16  6.55e+04          6.1e-05      0.00098
    BF16  3.39e+38          1.2e-38       0.0078
  [ok] values on the course page
  a gradient of 1e-08: FP16 stores 0; with loss scaling x65536 it survives as 9.997e-09
  [ok] FP16 underflows to 0 without loss scaling
  [ok] loss scaling brings it back (to 3 digits)

  Same regression, same AdamW (lr 1e-3), three ways to hold the weights:
  loss at step 0: 16.41
  weights held in                                                        loss after training
  ---------------------------------------------------------------------  -------------------
                                                                   FP32              0.01943
  BF16 only (the update is smaller than half a BF16 step: rounded away)                7.281
                            BF16 copy for compute + FP32 master weights              0.01978
  [ok] pure BF16 stalls far above FP32  (375x the FP32 loss)
  [ok] BF16 compute with FP32 master weights matches FP32
  model state per parameter: bf16 weight 2 + bf16 grad 2 + fp32 master 4 + m 4 + v 4 = 16 bytes
  a 7B model: 112 GB (104 GiB) before activations
  [ok] 16 bytes per parameter, 112 GB for 7B

## 7. One corrupted batch: what clipping and beta2 do
  one batch at step 400 has labels x1000: its gradient norm is 7,981
  (loss on clean data; 'control' = the same run without the bad batch)
  setting                peak     +50 steps  control  step 699  control
  ---------------------  -------  ---------  -------  --------  -------
  clip off, beta2 0.999  5.3e-01    5.3e-01  4.8e-04   4.2e-01  3.7e-08
  clip 1.0, beta2 0.999  1.8e-03    1.6e-05  1.0e-12   4.3e-13  7.4e-13
   clip off, beta2 0.95  5.6e-02    5.5e-02  2.3e-06   1.9e-05  1.1e-06
   clip 1.0, beta2 0.95  3.6e-03    4.2e-05  7.6e-07   2.2e-06  1.3e-06
  [ok] without clipping the bad batch is not forgotten: 1000x the control loss 250 steps on
  [ok] clipping to norm 1 keeps the peak below 1% of the unclipped peak
  [ok] beta2 = 0.95 forgets the spike sooner than 0.999 (no clipping)

All checks passed. (2.3s)
```

</details>
