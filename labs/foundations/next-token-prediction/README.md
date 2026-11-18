# 下一个词预测与信息论

课程页面：[从语言建模到下一个 token 预测](https://dmxsf.com/wiki/foundations/next-token-prediction/) · [信息论回顾：熵、交叉熵与 KL 散度](https://dmxsf.com/wiki/foundations/information-theory/)  
[English](README.en.md)

在课程页的玩具语料上数二元组，核对熵、交叉熵与 KL 散度的关系；在莎士比亚文本上看字符 n 元组在数据耗尽前的收益；再用梯度下降训练一个神经二元组，让它学出同一张计数表。

## 运行

```bash
uv run python labs/foundations/next-token-prediction/run.py            # 完整运行，几秒
uv run python labs/foundations/next-token-prediction/run.py --quick    # 缩小规模的冒烟测试，几秒
```

也可以写成 `uv run dmxsf run next-token-prediction`（同样可以加 `--quick`）。`--seed N` 换随机种子，默认 0。脚本分节打印结果，并用 `[ok]` / `[FAIL]` 逐条检查课程页面里的结论：全部通过时退出码为 0，有一条失败就是 1；只有完整运行才做的检查在 `--quick` 下标为 `[skip]`。

## 代码

| 位置 | 内容 |
| --- | --- |
| `labs/foundations/next-token-prediction/run.py` | 实验脚本，按编号分节 |
| `src/dmxsf/foundations/information.py` | 熵、交叉熵与 KL 散度 |
| `src/dmxsf/foundations/language_model.py` | 最简单的几种语言模型：n-gram 与神经 bigram |

单元测试：`uv run pytest tests/test_foundations_information.py tests/test_foundations_language_model.py`

## 看什么

- 第 1 节：课程页玩具语料上的 bigram 概率（P(sat|cat)=0.379）与困惑度（自然顺序 2.37，打乱词序 31.23）逐个对上。
- 第 2 节：H(p, q) = H(p) + KL(p‖q)，KL 不对称；两个分词方式不同的模型困惑度是 7.39 和 2.72，每字节比特数却同为 0.721：比较模型要用每字节比特数。
- 第 3 节：字符 n-gram 在莎士比亚上的验证集 bits/char 从 1 阶的 4.849 降到 5 阶的 2.590，6 阶又升到 2.935，上下文越长、计数越稀，过拟合就越重。
- 第 4 节：神经网络 bigram 从 ln 256 = 5.5452 出发，梯度下降收敛到计数表的最优值 H(next|prev) = 2.4522。

## 练习

改了参数再运行：核对页面数字的检查会变成 `[FAIL]`，这是预期的，看数字本身怎么变。

1. 把第 3 节 `for n in range(1, 5 if args.quick else 7):` 里的 7 改成 10，看训练集和验证集的 bits/char 各自怎么走。
2. 第 2 节里把 `q = [0.02, 0.02, 0.02, 0.02, 0.92]` 改成别的分布，观察 KL(p‖q) 和 KL(q‖p) 哪个对错误更敏感。

## 输出

下面是在 Apple M2 Max（12 核，仅用 CPU）、Python 3.12、PyTorch 2.15.0 上的完整运行结果。换一台机器，涉及训练的数字可能在末位上不同，计时类的数字取决于机器；检查用的是阈值与容差，不要求逐位相同。

<details>
<summary>完整运行的输出（`labs/foundations/next-token-prediction/run.py`）</summary>

```text
# Next-token prediction and information theory
  course: https://dmxsf.com/wiki/foundations/next-token-prediction/
          https://dmxsf.com/en/wiki/foundations/next-token-prediction/
  course: https://dmxsf.com/wiki/foundations/information-theory/
          https://dmxsf.com/en/wiki/foundations/information-theory/

## 1. A bigram model on the course page's toy corpus
  V = 9
  P(sat | cat) = 0.379
  P(mat | the) = 0.159
  P(cat | the) = 0.304
  perplexity, natural order  = 2.37
  perplexity, shuffled words = 31.23
  [ok] numbers match the course page

## 2. Entropy, cross-entropy and KL divergence (nats)
  quantity                 value
  -----------------------  -----
                     H(p)  1.303
                  H(p, q)  3.721
               KL(p || q)  2.417
               KL(q || p)  2.500
  perplexity exp(H(p, q))  41.29
  [ok] H(p, q) = H(p) + KL(p || q)
  [ok] KL is not symmetric
  [ok] with a one-hot target, cross-entropy is -ln(probability of the true token)  (0.511 nats)
  Same perplexity is not same quality when tokenizers differ; bits per byte is:
    model A: perplexity 7.39, bits per byte 0.721
    model B: perplexity 2.72, bits per byte 0.721

## 3. Character n-grams on Shakespeare: context helps until the counts run out
  n  contexts seen  train bits/char  val bits/char
  -  -------------  ---------------  -------------
  1              1            4.751          4.849
  2             65            3.583          3.585
  3           1391            2.821          3.002
  4          11412            2.213          2.613
  5          49692            1.923          2.590
  6         137272            1.804          2.935
  [ok] each of the first orders beats the one before
  [ok] long contexts overfit: order 6 is worse than the best order  (best 2.59, order 6 2.94)
  order 2 samples: 'Sheny,\nOKISTh witero wes, sy:\nCEd erowiveroneey y bl g medars fonoouchead To yfr owof ulmemphels\n\nug\n'
  order 5 samples: "Seconduit,\nthough bears doleful shall alread looking\nperfect\nMadam,' quondard.\n\nProduce forge the Dukes "

## 4. A neural bigram, trained by gradient descent, learns the count table
  loss at step 0     : 5.5452   (ln 256 = 5.5452)
  loss at the end    : 2.4523
  H(next | prev)     : 2.4522   (the best any bigram can do)
  [ok] starts at ln V: the uniform guess
  [ok] ends at the maximum-likelihood optimum
  after 'q' the model expects  'u': 1.00
  after 't' the model expects  'h': 0.34, ' ': 0.25, 'o': 0.09

All checks passed. (3.1s)
```

</details>
