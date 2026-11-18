# 预训练流水线

课程页面：[大模型训练流水线总览](https://dmxsf.com/wiki/pretraining/training-pipeline/)  
[English](README.en.md)

花算力之前的两个免费检查；基线训练与监控日志的读法；然后是三个实验：数据配比、从检查点恢复的四种做法（只有一种严格精确）、验证集泄漏进训练数据。

## 运行

```bash
uv run python labs/pretraining/training-pipeline/run.py            # 完整运行，约 3 分钟
uv run python labs/pretraining/training-pipeline/run.py --quick    # 缩小规模的冒烟测试，约 20 秒
```

也可以写成 `uv run dmxsf run training-pipeline`（同样可以加 `--quick`）。`--seed N` 换随机种子，默认 0。脚本分节打印结果，并用 `[ok]` / `[FAIL]` 逐条检查课程页面里的结论：全部通过时退出码为 0，有一条失败就是 1；只有完整运行才做的检查在 `--quick` 下标为 `[skip]`。

## 代码

| 位置 | 内容 |
| --- | --- |
| `labs/pretraining/training-pipeline/run.py` | 实验脚本，按编号分节 |
| `src/dmxsf/pretraining/train.py` | 一个小而明确的训练循环：学习率调度、裁剪、监控与检查点 |
| `src/dmxsf/data.py` | 小语料与批处理，全部离线可用 |

单元测试：`uv run pytest tests/test_pretraining_train.py tests/test_data.py`

## 看什么

- 第 1 节：N(0, 0.02²) 初始化时第一步的损失是 8.01 比特，即 log₂256；N(0, 1) 加共享嵌入则是 47.84 比特（页面的 0.86M 模型是 108.0）。单个批次能被背下来（7.86 → 0.02 比特），说明代码、损失和优化器都没有 bug。
- 第 2 节：监控表里验证损失每次评测都在降（3.89 → 3.24），梯度裁剪是启动阶段的事件：前 100 步有 24% 的步被裁，之后只有 5.0%（这是观察，不是检查：种子 2 上相反，后半程的梯度范数涨到 1.4，几乎每一步都被裁，页面练习 6 里较低的峰值学习率就是这样，训练仍然正常）。
- 第 3 节：数据配比决定模型擅长什么：90% 的散文比 10% 的散文使散文损失低 0.52 比特，代码同理低 0.61 比特；两个领域的平均值几乎不动（3.014 / 3.238 / 2.965），只看一个数字会掩盖这个取舍。
- 第 4 节：权重、优化器状态与步数都恢复时，续训与不中断的运行逐比特相同；少恢复任何一项，权重就不同（差 0.22、0.64、0.44），验证分数却没有标出它们：三者都不比正确续训更差（最差 +0.005 比特，换种子本身就有 0.131）；重启学习率的两组甚至“更好”（−0.132 与 −0.133）：欠训练时分数分不出 bug 与好处。
- 第 5 节：把 4.1 KB 验证集代码混进训练数据，泄漏的那段从 3.582 降到 2.267 比特（低 37%），其余没泄漏的验证集在这个种子上几乎没变（2.908 → 2.964）；页面里更大的泄漏（60 KB 放进 0.86M 参数的模型）连同包的兄弟文件也拉低了，这里的小规模没有看到，也不能因此说没有（见下面的说明）。

## 练习

改了参数再运行：核对页面数字的检查会变成 `[FAIL]`，这是预期的，看数字本身怎么变。

1. 第 3 节的循环 `for share, seed in [(0.9, S0), (0.1, S0), (0.5, SEEDS[-1])]` 里加一项 `(0.99, S0)`：只占 1% 的散文，损失比占 50% 时差多少？
2. 第 5 节把泄漏的量从 4096 字节改成 40 KB，没泄漏的那部分验证集代码会不会开始被拉低？

## 注意

- 有两条关于效应大小的说法是观察，换种子或换机器就会变，不成立时输出 `[differs]`：第 2 节的“梯度裁剪是启动阶段的事件”（种子 2 上相反），以及第 5 节的“没泄漏的那部分验证集代码不受影响”。后者在五次运行里（Apple M2 Max 上的种子 0、1、2，GitHub Linux 运行器上 Python 3.10 与 3.12 各一次）相对对照组移动了 +0.06、+0.34、+0.28、+0.30 与 −0.41 比特/字节，方向不固定；而没有泄漏的基线与对照这两次训练之间，在其中四次里本来就差 0.04 到 0.32 比特。所以 4 KB 的泄漏在这个规模上既看不出溢出，也不能证明没有溢出：一次训练分不开溢出与训练噪声。
- 这个实验把当前 Python 解释器自带的标准库源码当作“代码语料”，所以换一个 Python 版本，涉及代码语料的数字会略有不同。

## 输出

下面是在 Apple M2 Max（12 核，仅用 CPU）、Python 3.12、PyTorch 2.15.0 上的完整运行结果。换一台机器，涉及训练的数字可能在末位上不同，计时类的数字取决于机器；检查用的是阈值与容差，不要求逐位相同。

<details>
<summary>完整运行的输出（`labs/pretraining/training-pipeline/run.py`）</summary>

```text
# The pretraining pipeline on a laptop: monitoring, data mix, resuming, leakage
  course: https://dmxsf.com/wiki/pretraining/training-pipeline/
          https://dmxsf.com/en/wiki/pretraining/training-pipeline/

## 1. Two free checks before you train
  the page's model: 858,880 parameters; one batch of 16 x 128 bytes, 60 steps of AdamW
  initialisation                   loss at step 0 (bits)  first step below 8 bits  clipped steps
  -------------------------------  ---------------------  -----------------------  -------------
        N(0, 0.02^2), GPT-2 style                   8.01                        1       16 of 50
  N(0, 1), nn.Embedding's default                  47.84                       30       50 of 50
  [ok] with a sensible initialisation the first loss is log2(256) = 8 bits  (8.01)
  [ok] with N(0, 1) and tied embeddings the model starts far worse than uniform guessing  (47.8 bits; the page's 0.86M model starts at 108.0)

  one batch, 300 steps, constant learning rate: loss 7.86 -> 0.02 -> 0.02 -> 0.02 bits (steps 1, 100, 200, 300)
  [ok] the model can memorise a single batch: the code, the loss and the optimizer all work  (7.86 -> 0.02 bits)

## 2. A baseline run: 1200 steps, 50% prose + 50% code, and how to read its log
  124,672-parameter byte-level GPT, 2,048 bytes per step (2.46 MB in all); prose 1.06 MB (TinyShakespeare, last 5% held out), code 10.81 MB (Python standard library, held-out files)
  step  lr       train loss  val prose  val code  grad norm (median)  clipped  
  ----  -------  ----------  ---------  --------  ------------------  ---------
   200  5.7e-03        3.93       3.91      3.86                0.50  25 of 200
   400  4.8e-03        3.65       3.73      3.64                0.42   0 of 200
   600  3.4e-03        3.53       3.63      3.48                0.43   3 of 200
   800  2.0e-03        3.40       3.50      3.38                0.44   1 of 200
  1000  9.8e-04        3.31       3.37      3.26                0.57   6 of 200
  1200  6.0e-04        3.20       3.28      3.19                0.77  44 of 200
  largest gradient norm before clipping: 5.9 at step 21; clipped in 24% of the first 100 steps against 5.0% of the rest
  [ok] the loss starts at log2(256) = 8 bits  (7.93)
  [ok] validation loss falls at every evaluation (prose + code)  (3.89 > 3.69 > 3.55 > 3.44 > 3.32 > 3.24)
  [ok] no overfitting yet: the validation mean is close to the training loss  (validation mean 3.24, training 3.20 bits)
  [ok] clipping is a start-up event: the largest gradient norm comes in the first 100 steps and clipped steps are more frequent early than later  (peak at step 21; 24% of the first 100 steps against 5.0% of the rest)

## 3. Experiment 1: the data mix decides what the model is good at
  prose : code  seed  prose bpb  code bpb  mean 
  ------------  ----  ---------  --------  -----
     90% : 10%     0      2.800     3.228  3.014
     50% : 50%     0      3.285     3.190  3.238
     10% : 90%     0      3.316     2.614  2.965
     50% : 50%     1      3.152     3.061  3.106
  [ok] a 90% share beats a 10% share for that domain  (gain 0.52 bits on prose, 0.61 bits on code)
  [ok] ...by more than the seed range of the 50:50 runs  (seed range 0.13 bits on prose, 0.13 bits on code)
  the mean over both domains hardly moves (3.014 / 3.238 / 2.965 for 90:10 / 50:50 / 10:90, seed range 0.131): one number would have hidden the trade

## 4. Experiment 2: resuming from step 600
  resume                                 prose bpb  code bpb  mean   max |w - baseline|  loss, first 20 steps
  -------------------------------------  ---------  --------  -----  ------------------  --------------------
                 uninterrupted baseline      3.285     3.190  3.238                   0                 3.524
             weights + optimizer + step      3.285     3.190  3.238            0.00e+00                 3.524
                           weights only      3.292     3.193  3.243            2.21e-01                 3.564
  forgot the step (data and lr restart)      3.193     3.017  3.105            6.38e-01                 3.562
                data right, lr restarts      3.178     3.030  3.104            4.42e-01                 3.514
  [ok] with weights, optimizer state and step restored, the resumed run equals the uninterrupted one bit for bit  (largest weight difference 0.0)
  [ok] each of the other three ends with different weights  (0.22, 0.64, 0.44)
  [ok] ...and the validation score does not flag them: none is worse than the correct resume by more than a change of seed  (worst +0.005 bits for the three, 0.131 between seeds 0 and 1)
  the runs that restart the learning rate even score *better* than the correct resume (forgot the step (data and lr restart): -0.132; data right, lr restarts: -0.133): at this step the model is still under-trained, so a higher learning rate helps.
  The score cannot tell a bug from a benefit.

## 5. Experiment 3: validation text in the training data
  leaked: the first 4.1 KB of the held-out code; the remaining 567 KB never enter training. 20% of the training windows come from the extra data.
  model                              leaked chunk (bpb)  rest of held-out code (bpb)
  ---------------------------------  ------------------  ---------------------------
            baseline: no extra data               3.828                        3.145
     control: 4 KB of training code               3.582                        2.908
  leaked: the 4 KB of held-out code               2.267                        2.964
  [ok] the leaked text scores far better than the same model without the leak  (3.582 -> 2.267 bits: 37% lower, and the model did not become 30% better at code)
  [ok] the rest of the held-out code stays where the control had it (no spillover at this size)  (2.908 -> 2.964; the page's larger leak (60 KB into 0.86M parameters) lifted sibling files in the same package)

All checks passed. (209.0s)
```

</details>
