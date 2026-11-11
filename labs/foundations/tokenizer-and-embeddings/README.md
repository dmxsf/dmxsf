# 分词器与词嵌入

课程页面：[词嵌入与 Tokenizer 原理](https://dmxsf.com/wiki/foundations/tokenizer-and-embeddings/)  
[English](README.en.md)

手写 BPE（课程页的玩具语料），实现永不出现 <unk> 的字节级 BPE，看训练文本如何决定一种语言的 token 成本；嵌入表等价于一次矩阵乘法；以及嵌入表在整个模型里占多大比例。

## 运行

```bash
uv run python labs/foundations/tokenizer-and-embeddings/run.py            # 完整运行，几秒
uv run python labs/foundations/tokenizer-and-embeddings/run.py --quick    # 缩小规模的冒烟测试，几秒
```

也可以写成 `uv run dmxsf run tokenizer-and-embeddings`（同样可以加 `--quick`）。`--seed N` 换随机种子，默认 0。脚本分节打印结果，并用 `[ok]` / `[FAIL]` 逐条检查课程页面里的结论：全部通过时退出码为 0，有一条失败就是 1；只有完整运行才做的检查在 `--quick` 下标为 `[skip]`。

## 代码

| 位置 | 内容 |
| --- | --- |
| `labs/foundations/tokenizer-and-embeddings/run.py` | 实验脚本，按编号分节 |
| `labs/foundations/tokenizer-and-embeddings/parallel_text.py` | 第 3 节用的 24 对中英平行句子 |
| `src/dmxsf/foundations/tokenizer.py` | 从零实现的字节级 BPE |

单元测试：`uv run pytest tests/test_foundations_tokenizer.py`

## 看什么

- 第 1 节：课程页玩具语料上 BPE 的合并顺序（`e+s → es` 9 次、`es+t → est` 9 次……）与切分结果一一对上。
- 第 2 节：字节级 BPE 对任何字符串都可逆（含 emoji 和中日韩文字）；词表从 256 增大到 1024，每个 token 覆盖的字节从 1.00 升到 2.37，注意力代价降到 0.18，嵌入参数却从 32,768 涨到 131,072。
- 第 3 节：只用英文训练的分词器，每个汉字要 2.75 个 token，与词表大小无关；加入中文数据后中文变便宜、英文变贵，词表是共享的预算。
- 第 5 节：嵌入表占模型的比例：GPT-2 small 31.0%、Qwen2.5-0.5B 27.6%、Llama 3 8B 13.1%，与课程页一致。

## 练习

改了参数再运行：核对页面数字的检查会变成 `[FAIL]`，这是预期的，看数字本身怎么变。

1. 把 `max_vocab` 和第 2 节的词表大小列表都加到 2048，bytes/token 还在涨吗？嵌入参数呢？
2. 第 3 节的 24 对句子在同目录的 `parallel_text.py` 里；换成你自己的语言（日语、阿拉伯语……）看分词器的“偏科”。

## 输出

下面是在 Apple M2 Max（12 核，仅用 CPU）、Python 3.12、PyTorch 2.15.0 上的完整运行结果。换一台机器，涉及训练的数字可能在末位上不同，计时类的数字取决于机器；检查用的是阈值与容差，不要求逐位相同。

<details>
<summary>完整运行的输出（`labs/foundations/tokenizer-and-embeddings/run.py`）</summary>

```text
# Tokenizers and embeddings
  course: https://dmxsf.com/wiki/foundations/tokenizer-and-embeddings/
          https://dmxsf.com/en/wiki/foundations/tokenizer-and-embeddings/

## 1. BPE by hand: the course page's toy corpus
   1  e + s -> es  (9 times)
   2  es + t -> est  (9 times)
   3  ▁ + l -> ▁l  (7 times)
   4  ▁l + o -> ▁lo  (7 times)
   5  ▁lo + w -> ▁low  (7 times)
   6  ▁ + n -> ▁n  (6 times)
   7  ▁n + e -> ▁ne  (6 times)
   8  ▁ne + w -> ▁new  (6 times)
   9  ▁new + est -> ▁newest  (6 times)
  10  ▁ + w -> ▁w  (3 times)
  ['▁low', 'est'] ['▁new', 'e', 'r'] ['▁w', 'i', 'd', 'e', 'r']
  [ok] same merge table and segmentations as the course page

## 2. Byte-level BPE: pre-tokenization and a tokenizer that never says <unk>
  chunks: ['Hello', ',', ' world', '!', ' It', "'s", ' 12345678', '.']
  trained 768 merges on 300,000 characters in 5.8s
  first merges: [' t', 'he', 'ou', ' a', ' s', ' w', 'in', ' m']
  longest tokens: [' Buckingham', ' Servingman', 'CORIOLANUS', 'BUCKINGHAM', ' ELIZABETH']
  [ok] encode -> decode is lossless for any string, including emoji and CJK
  'Hello' -> [b'H', b'ell', b'o']     ' Hello' -> [b' H', b'ell', b'o']
  '你好' -> 6 tokens for 6 bytes

  A bigger vocabulary means shorter sequences (held-out Shakespeare, 20,000 characters):
  vocab  tokens  bytes/token  attention cost (x bytes)  embedding params (d=128)
  -----  ------  -----------  ------------------------  ------------------------
    256   20000        1.000                     1.000                     32768
    320   13458        1.486                    0.4528                     40960
    384   11706        1.709                    0.3426                     49152
    512   10381        1.927                    0.2694                     65536
    768    9065        2.206                    0.2054                     98304
   1024    8434        2.371                    0.1778                    131072
  [ok] each larger vocabulary compresses better
  [ok] a byte-level vocabulary of 256 is one token per byte

## 3. What the tokenizer saw decides what a language costs
  held-out text: 186 Chinese characters (522 bytes) and 107 English words (604 bytes)
  E = trained on English only, E+Z = English and Chinese, Z = Chinese only; same vocabulary size within a row
  vocab  zh tok/char E  E+Z    Z      en bytes/tok E  E+Z    Z    
  -----  -------------  -----  -----  --------------  -----  -----
    320          2.753  2.253  2.022           1.646  1.462  1.049
    384          2.753  2.022  1.801           1.893  1.692  1.049
    512          2.753  1.839  1.608           2.127  1.936  1.049
  [ok] an English-only tokenizer pays about 2.75 tokens per Chinese character at any vocab size
  [ok] putting Chinese in the training data makes Chinese cheaper ...
  [ok] ... and English more expensive: the vocabulary is a shared budget
  (24 sentence pairs written for this lab; the direction matters, not the exact ratio.
   On the course page GPT-2 needs 50 tokens for a 24-character sentence, Qwen2.5 only 16.)

## 4. From token id to vector: an embedding table is a matrix multiply
  [ok] lookup == one-hot @ weight matrix
  one backward pass touched 7 of 1000 rows: only tokens that appeared are trained
  [ok] only the rows of the tokens that appeared get gradient

## 5. How much of a model is its embedding table?
  model               total params   table V x d  copies  share
  ------------------  -------------  -----------  ------  -----
         GPT-2 small    124,439,808   38,597,376       1  31.0%
        Qwen2.5-0.5B    494,032,768  136,134,656       1  27.6%
          Llama 3 8B  8,030,261,248  525,336,576       2  13.1%
  our byte-level GPT        858,880       32,768       1   3.8%
  [ok] matches the course page: 31.0%, 27.6%, 13.1%

All checks passed. (6.8s)
```

</details>
