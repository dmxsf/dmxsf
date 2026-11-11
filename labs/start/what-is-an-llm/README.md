# 大模型是什么：一个小到能读完的完整循环

课程页面：[大模型是什么：从“预测下一个词”到通用助手](https://dmxsf.com/wiki/start/what-is-an-llm/) · [大模型学习路线图：应用、工程与研究三条路线](https://dmxsf.com/wiki/start/learning-roadmap/)  
[English](README.en.md)

在一个 842,496 参数、能整个打开的 GPT 上跑通完整循环：预测一个字节、接上去、再预测；从分词、嵌入到 softmax、训练目标和三个关键数字（参数、token、上下文）。末尾是全部课程页面与实验的对照表。

## 运行

```bash
uv run python labs/start/what-is-an-llm/run.py            # 完整运行，几秒
uv run python labs/start/what-is-an-llm/run.py --quick    # 缩小规模的冒烟测试，几秒
```

也可以写成 `uv run dmxsf run what-is-an-llm`（同样可以加 `--quick`）。`--seed N` 换随机种子，默认 0。脚本分节打印结果，并用 `[ok]` / `[FAIL]` 逐条检查课程页面里的结论：全部通过时退出码为 0，有一条失败就是 1；只有完整运行才做的检查在 `--quick` 下标为 `[skip]`。

这个实验用到缓存的小模型：字节级基础模型（84 万参数）。第一次运行会先在 TinyShakespeare 上训练并缓存到 `~/.cache/dmxsf`（环境变量 `DMXSF_CACHE` 可以改位置），之后直接读取。

## 代码

| 位置 | 内容 |
| --- | --- |
| `labs/start/what-is-an-llm/run.py` | 实验脚本，按编号分节 |
| `src/dmxsf/models.py` | 小型预训练字节级模型：训练一次、缓存下来，供预训练之后的各个实验共用 |
| `src/dmxsf/inference/generate.py` | 带 KV Cache 的自回归生成：每个聊天产品都在跑的循环 |
| `src/dmxsf/foundations/tokenizer.py` | 从零实现的字节级 BPE |
| `src/dmxsf/foundations/information.py` | 熵、交叉熵与 KL 散度 |

单元测试：`uv run pytest tests/test_models.py tests/test_inference_generate.py tests/test_foundations_tokenizer.py tests/test_foundations_information.py`

## 看什么

- 第 1 节：自己写的“读全文、预测一个字节、接上去、重复”循环，和库里带 KV Cache 的 `generate` 写出同样的 60 个字节；不用缓存时写 128 个字节要重读 10,048 个字节，用缓存只要 142 个，差 70.8 倍。
- 第 2 节：同一句话在三种分词器下是三组不同的 id；每个 BPE 在自己训练的领域里每个 token 覆盖的字节更多（莎士比亚 1.84，Python 代码 2.05）；“你好，世界”在两个 BPE 下都是 15 个字节、15 个 token，没见过的文字退回字节。
- 第 4 节：同一组分数在温度 0.3、1、1.5 下的熵是 0.55、3.90、4.71 比特；采样给出不同的话，贪心每次都是同一句。
- 第 5 节：什么都不知道时一个字节要 8 比特，读几分钟莎士比亚后在没见过的文本上降到 2.308；词中间的字节只要 1.50 比特，一个词的首字母却要 4.13 比特：难的是“下一个词是什么”。
- 第 7 节：30 个实验覆盖课程的 35 页，三条学习路线（应用、工程、研究）按周列出该跑哪几个。

## 练习

改了参数再运行：核对页面数字的检查会变成 `[FAIL]`，这是预期的，看数字本身怎么变。

1. 把 `run.py` 里的 `PROMPT` 换成 `"JULIET:\nO "` 或别的开头，看第 4 节的分布和采样结果怎么变。
2. 第 2 节的文本是写在脚本里的；换成一段你自己的中文或代码，比较两个 BPE 各用多少个 token。
3. 用 `--seed 1`、`--seed 2` 各跑一遍：哪些数字变了，哪些没变？（提示：模型是缓存的，变的只有采样。）

## 注意

- 这个实验把当前 Python 解释器自带的标准库源码当作“代码语料”，所以换一个 Python 版本，涉及代码语料的数字会略有不同。

## 输出

下面是在 Apple M2 Max（12 核，仅用 CPU）、Python 3.12、PyTorch 2.15.0 上的完整运行结果。换一台机器，涉及训练的数字可能在末位上不同，计时类的数字取决于机器；检查用的是阈值与容差，不要求逐位相同。

<details>
<summary>完整运行的输出（`labs/start/what-is-an-llm/run.py`）</summary>

```text
# What is an LLM? The whole loop, small enough to read
  course: https://dmxsf.com/wiki/start/what-is-an-llm/
          https://dmxsf.com/en/wiki/start/what-is-an-llm/
  course: https://dmxsf.com/wiki/start/learning-roadmap/
          https://dmxsf.com/en/wiki/start/learning-roadmap/
  model: 4 layers, width 128, 4 heads, 842,496 parameters, context 128 bytes; read 9,830,400 bytes of Shakespeare

## 1. The loop: read everything so far, predict one token, append, repeat
  'ROMEO:\nWhat is the state? what say the state?\n\nPOMPEY:\nWhat is the state?\n\n'
  [ok] the hand-written loop and the library's KV-cached `generate` produce the same 60 bytes
  [ok] round i feeds the model the prompt plus everything generated: the bytes read add up to 15 x 60 + 60 x 59 / 2  (2,670 bytes)

  Bytes the model must read to write n bytes after a 15-byte prompt:
  n    re-read every round  with a KV cache  ratio
  ---  -------------------  ---------------  -----
   16                  360               30  12.0x
   32                  976               46  21.2x
   64                2,976               78  38.2x
  128               10,048              142  70.8x
  (A KV cache keeps the keys and values of what was already read; see labs/inference/kv-cache.)

## 2. Stop 1: text becomes tokens
  Tokens needed by two tokenizers (vocabulary 600, trained on 100,000 bytes each):
  text                   bytes  BPE Shakespeare  BPE code
  ---------------------  -----  ---------------  --------
    'unbelievably good'     17               10        12
  'To be, or not to be'     19                7         8
          'def main():'     11                8         4
         'Hello, world'     12                6         9
                '你好，世界'     15               15        15

  'unbelievably good', piece by piece:
    bytes only           u | n | b | e | l | i | e | v | a | b | l | y |   | g | o | o | d
                         ids [117, 110, 98, 101, 108, 105, 101, 118, 97, 98, 108, 121, 32, 103, 111, 111, 100]
    BPE on Shakespeare   un | b | e | l | ie | v | a | b | ly |  good
                         ids [384, 98, 101, 108, 500, 118, 97, 98, 359, 479]
    BPE on Python code   un | b | e | li | e | v | ab | l | y |  g | o | od
                         ids [367, 98, 101, 347, 101, 118, 324, 108, 121, 555, 111, 350]
  [ok] ids belong to a tokenizer: the same text becomes different ids under each of the three

  Bytes per token on held-out text (more is better: fewer tokens for the same text):
  tokenizer           Shakespeare  Python code
  ------------------  -----------  -----------
  BPE on Shakespeare         1.84         1.41
  BPE on Python code         1.55         2.05
  [ok] a tokenizer is best on the text it was trained on: each BPE beats the other one on its own domain
  [ok] text the tokenizer never saw falls back to bytes: 5 Chinese characters are 15 UTF-8 bytes and 15 tokens under both BPEs, while English gets shorter  (Chinese [15, 15] tokens; English [6, 9] tokens for 12 bytes)

## 3. Stop 2: ids become vectors, and the Transformer reads them
  part                                   parameters
  -------------------------------------  ----------
              embedding table (256 x d)      32,768
               position table (128 x d)      16,384
                   4 Transformer blocks     793,088
                             final norm         256
  output matrix (shared with the table)           0
  [ok] the parts add up to the model's size; the output matrix is the embedding table used backwards (tied), so it is counted once  (842,496 = 842,496)
  [ok] an id is a row of the embedding table
  [ok] the model only looks backwards: changing bytes 40-63 changes no prediction at positions 0-39, and does change those after

## 4. Stop 3: the output is a probability distribution
  After 'ROMEO:\nWhat is ' the six most likely next bytes:
  byte  probability
  ----  -----------
   't'        0.284
   'n'        0.093
   'm'        0.088
   'a'        0.072
   'h'        0.068
   's'        0.061
  [ok] softmax of the 256 scores is a distribution: exp(z_v) / sum of exp(z_u), and it sums to 1
  [ok] greedy decoding takes the most likely byte  ('t' with probability 0.284)

  The same scores at different temperatures (decoding does not change the model):
  temperature  entropy (bits)  top probability
  -----------  --------------  ---------------
          0.3            0.55            0.929
          0.7            2.96            0.464
            1            3.90            0.284
          1.5            4.71            0.158
  [ok] a higher temperature flattens the distribution: entropy up, top probability down
  Sampling at temperature 1 with three seeds, then greedy twice:
    seed 0: "some a God's sweet? Had off yo"
    seed 1: 'such not me soundly the counte'
    seed 2: 'thy praise to stands crue?\n\nEX'
    greedy : 'the state? what say the state?'
  [ok] the same question gets different answers when sampled, and the same answer when greedy

## 5. Why predicting the next byte is the whole training objective
  [ok] the probability of a text is the product of the next-byte probabilities: one forward pass over the whole text gives the same log-probability as predicting byte by byte  (log P = -183.529 nats (-264.8 bits for 127 bytes))
  [ok] the training loss is the average of -log p over positions  (1.445 nats = 2.085 bits per byte)
  held-out Shakespeare: 2.308 bits per byte; knowing nothing, 256 bytes equally likely: 8.000
  [ok] reading Shakespeare for a few minutes cuts the cost of a byte from 8 bits to under 2.5  (2.308)

  Where the bits go: the average cost of a byte, by the kind of byte (held-out text):
  byte                    count  bits
  ----------------------  -----  ----
           inside a word   7332  1.50
  first letter of a word   1885  4.13
    first byte of a line    530  3.43
             punctuation    712  2.97
  [ok] the model has learned spelling: a byte inside a word is far cheaper than the first letter of the next word, where what is said next has to be guessed  (1.50 against 4.13 bits)

## 6. Parameters, tokens, context: the three numbers
  Memory for the weights alone (parameters x bytes per parameter):
  model       FP32     BF16     INT8     4-bit  
  ----------  -------  -------  -------  -------
  this model  3.37 MB  1.68 MB  0.84 MB  0.42 MB
  a 7B model  28.0 GB  14.0 GB   7.0 GB   3.5 GB
  [ok] weights take parameters x bytes per parameter: measured on this model's own tensors, FP32 is 4 bytes each and BF16 is 2  (3,369,984 and 1,684,992 bytes)
  [ok] the same formula gives the page's numbers for a 7B model: 14 GB in BF16 and 3.5 GB at 4 bits (the KV cache and activations come on top)

  Training tokens: 2400 steps x 32 windows x 128 bytes = 9,830,400 bytes read, 9.3 passes over
  the 1,059,624-byte training text, 11.7 bytes per parameter.
  [ok] the context window is a hard limit: this model's positions stop at 128, and a 129-byte input is refused

## 7. The stages of the page, and where each lives in this repository
  stage                  sections                                labs  course pages
  ---------------------  --------------------------------------  ----  ------------
           Pre-training  foundations, architecture, pretraining    12            15
          Post-training                           post-training     4             4
  Inference and serving                               inference     5             6
        Using the model    applications, multimodal, evaluation     8             8
  [ok] the 30 labs cover all 35 pages of the course, each page at least once

  AI application developer (8 weeks): 10 labs, in the order of the weeks
    week  1: what-is-an-llm, prompt-engineering
    week  2: decoding-and-sampling, llm-evaluation, eval-statistics
    week  4: rag
    week  5: agents-and-tool-calling
    week  6: continuous-batching, kv-cache
    week  7: quantization

  LLM engineer (12 weeks): 17 labs, in the order of the weeks
    week  1: next-token-prediction, tokenizer-and-embeddings
    week  2: attention-basics
    week  3: transformer-block, positional-encoding
    week  5: training-pipeline, scaling-laws
    week  6: distributed-training
    week  7: sft-data, lora-finetuning
    week  8: preference-optimization
    week  9: kv-cache, efficient-attention
    week 10: quantization, continuous-batching
    week 11: llm-evaluation, eval-statistics

  LLM researcher (16 weeks): 12 labs, in the order of the weeks
    week  2: attention-basics
    week  3: scaling-laws
    week  4: transformer-block
    week  5: mixture-of-experts
    week  7: efficient-attention, continuous-batching
    week  8: positional-encoding
    week  9: sft-data
    week 10: preference-optimization
    week 11: reasoning-rl
    week 12: llm-evaluation
    week 13: eval-statistics
  [ok] every lab on the map exists in this checkout  ([])

  List them all: dmxsf list. Run one: dmxsf run kv-cache --quick. A route: dmxsf roadmap engineer.

All checks passed. (3.4s)
```

</details>
