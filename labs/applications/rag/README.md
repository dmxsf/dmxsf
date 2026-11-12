# RAG：逐个零件

课程页面：[RAG 系统搭建实战](https://dmxsf.com/wiki/applications/rag/)  
[English](README.en.md)

切块、相似度与索引大小、按公式实现的 BM25、排名融合（RRF）、检索指标与上下文拼装，离线复现课程页的数字；加 --bge 还会运行稠密检索器与重排序模型。

## 运行

```bash
uv run python labs/applications/rag/run.py            # 完整运行，几秒
uv run python labs/applications/rag/run.py --quick    # 缩小规模的冒烟测试，几秒
```

也可以写成 `uv run dmxsf run rag`（同样可以加 `--quick`）。`--seed N` 换随机种子，默认 0。脚本分节打印结果，并用 `[ok]` / `[FAIL]` 逐条检查课程页面里的结论：全部通过时退出码为 0，有一条失败就是 1；只有完整运行才做的检查在 `--quick` 下标为 `[skip]`。

可选参数：

- `--lang en|zh`：用哪个语言版本课程页的语料与查询（默认 `en`）。
- `--bge`：再运行真实的稠密检索器（BAAI/bge-small-en-v1.5 或 -zh-v1.5）和重排序模型（BAAI/bge-reranker-base），复现课程页的全部检索指标；需要 `uv sync --extra hf` 和已下载的模型。

## 代码

| 位置 | 内容 |
| --- | --- |
| `labs/applications/rag/run.py` | 实验脚本，按编号分节 |
| `src/dmxsf/applications/rag.py` | 检索增强生成，逐件拆开：切块、BM25、RRF、指标与上下文拼装 |
| `src/dmxsf/applications/rag_data.py` | RAG 页的小语料、查询与示例文档（每种语言一套） |

单元测试：`uv run pytest tests/test_applications_rag.py`

## 看什么

- 第 1 节：页面里那份虚构的报销制度（354 个字、15 句，最长 45 字）：固定窗口 120 字、不重叠时切断 2 句；重叠不是越大越好，切断的句数随重叠 5、10、15、20 是 1、0、1、0；只有重叠不小于最长的句子（窗口也更长）才保证不切断句子，代价是存 1.51 倍；按句或按段切分从不切断句子。
- 第 2 节：单位向量上内积就是余弦、距离平方是 2 − 2cos，三者给出同一个排序；不归一化时内积的排序与余弦不同（前 20 名只有 9 个相同）；索引大小表与页面一致。
- 第 3 节：从公式写出的 BM25：“报销需要上传哪些材料？”的前三名与得分（7.42 / 5.46 / 2.46）与页面一致；err-1041 与问题无关，却因为只匹配了“需要”进了前三：中文的二元组跨词也会得分；词频饱和的上限是 k1 + 1 = 2.2。
- 第 4 节：RRF 只用名次，不用分数：页面的例子里稠密排序 [acc-lock, acc-reset, err-1042]、BM25 只有 [err-1044]，融合后 [acc-lock, err-1044, acc-reset]；离线的词级 BM25 对字符三元组 BM25 对 RRF：0.92 / 0.50 / 0.83，融合反而更差：两个词面检索器会一起出错，融合要加的是一个不同类型的检索器。
- 第 5–6 节：Hit@k、Recall@k、MRR 由名次表算出，与页面一致（一个查询是 8.3 个点，没有置信区间）；上下文预算 100 个字时，提示里装进了排第一的 exp-advance 和无关的 err-1045：预算不是相关性检验，按相对分数截断也救不回排错的第一名。
- 第 7 节（`--bge`）：BAAI/bge-small-zh-v1.5 与 BAAI/bge-reranker-base 的结果，每个查询每个系统的名次都与页面的表一致：中文的 12 个查询上，稠密检索 Hit@1 是 0.75，词面 BM25 是 0.92，RRF 0.83，重排序 1.00。在这份中文样例上词面检索比稠密检索强，融合没有超过 BM25，只有重排序做到了全对。

## 练习

改了参数再运行：核对页面数字的检查会变成 `[FAIL]`，这是预期的，看数字本身怎么变。

1. 第 6 节里 `"prompt_budget": 100`（中文）改成 200：提示里还会多装进哪些块？它们与问题相关吗？
2. 第 4 节 `for k in (1, 10, 60, 1000)` 里加一个 5：RRF 里第一名比第五名值钱多少倍？

## 注意

- 第 7 节要加 `--bge`（需要 `hf` 额外依赖，以及两个 BAAI 模型）；中英文各用自己的样例文档与 12 个查询完整跑了一遍，数字互不借用。文档是自己编的演示数据，不是真实公司的制度。

## 输出

下面是在 Apple M2 Max（12 核，仅用 CPU）、Python 3.12、PyTorch 2.15.0 上的完整运行结果。换一台机器，涉及训练的数字可能在末位上不同，计时类的数字取决于机器；检查用的是阈值与容差，不要求逐位相同。

<details>
<summary>离线运行的输出（`labs/applications/rag/run.py --lang zh`）</summary>

```text
# RAG, piece by piece
  course: https://dmxsf.com/wiki/applications/rag/
          https://dmxsf.com/en/wiki/applications/rag/

## 1. Chunking: where the sentences go
  The page's made-up expense policy (354 characters): 15 sentences, the longest 45 characters. A sentence is 'cut' when no chunk holds it whole.
  strategy   size  overlap  chunks  avg chars  cut  stored/doc
  ---------  ----  -------  ------  ---------  ---  ----------
      fixed   120        0       3      118.0    2       1.00x
      fixed    60        0       6      58.83    4       1.00x
      fixed   120        5       4      92.25    1       1.04x
      fixed   120       10       4      95.00    0       1.07x
      fixed   120       15       4      99.75    1       1.12x
      fixed   120       20       4      103.2    0       1.16x
      fixed   120      100      13      119.4    0       4.34x
      fixed   120       45       5      106.6    0       1.51x
   sentence   120        0       4      87.75    0       1.00x
  paragraph   120        0       5      69.20    0       1.00x
  paragraph   400        0       5      69.20    0       1.00x
  [ok] the document has 15 sentences, the longest 45 characters, as the widget counts
  [ok] every (chunks, cut sentences) pair matches what the widget shows  ([])
  [ok] the redundancy figures match  (('fixed', 120, 20): 1.16x, ('fixed', 120, 100): 4.34x)
  [ok] more overlap does not mean fewer cut sentences (not monotonic)  (overlaps 5, 10, 15, 20 cut [1, 0, 1, 0])
  [ok] overlap >= longest sentence (with a longer window) leaves no sentence cut: no exception in the grid  (91 (size, overlap) pairs, window 46+, overlap 45+)
  [ok] a window shorter than the longest sentence cuts something whatever the overlap  (windows 40 and 32, overlaps 0..)
  [ok] splitting by sentence or paragraph never cuts a sentence (sizes 40..400)
  What the guarantee costs here: window 120, overlap 45 stores 1.51x the document.

## 2. Similarity, normalisation and the size of an index
  [ok] on unit vectors the inner product is the cosine
  [ok] and the squared distance is 2 - 2 * cosine
  [ok] so inner product, cosine and Euclidean distance give one and the same ranking
  [ok] without normalising, the inner product ranks differently from the cosine (long vectors win)  (9 of the top 20 agree)

  index memory, vectors only (GB = 10^9 bytes):
  vectors     dim   float32 GB  int8 GB
  ----------  ----  ----------  -------
     100,000   384        0.15     0.04
   1,000,000   768        3.07     0.77
  10,000,000  1024       40.96    10.24
  [ok] float32 and int8 sizes match the page (0.15/0.04, 3.07/0.77, 40.96/10.24)

## 3. BM25 from the formula
  query: "报销需要上传哪些材料？"   (23 chunks, average length 27.5 tokens)
  chunk         BM25   what it matched (term, contribution)
  ------------  -----  ------------------------------------
  exp-material  7.418             上传 2.75, 报销 1.91, 材料 2.75
    exp-submit  5.456                      报销 2.66, 销需 2.79
      err-1041  2.464                               需要 2.46
  [ok] the top three and their scores are the page's  (exp-material 7.42, exp-submit 5.46, err-1041 2.46)
  [ok] err-1041 has nothing to do with the question yet is in the top three: it matched only ['需要']
  BM25 is a word-matching signal, not understanding: a filler word of the question (or, in Chinese, a bigram that
  crosses a word boundary) is enough to score.
  [ok] the IDF stays positive even for a term in every chunk: ln(1 + 0.5 / (N + 0.5)) > 0  (0.134 for N=3)
  term-frequency factor f(k1+1)/(f+k1) for f = 1, 2, 5, 10, 100 (length-normalised chunk): 1.00, 1.38, 1.77, 1.96, 2.17  -> the limit is k1+1 = 2.2
  [ok] term frequency saturates: ten occurrences count less than twice one, a hundred less than k1+1 = 2.2 times
  [ok] length normalisation (b=0.75) scores the long chunk lower than the short one for the same term; with b=0 they tie  (0.25 against 0.14)

## 4. Rank fusion: RRF uses ranks, not scores
  query: "错误码 E1044 怎么处理"
    dense (top 3 as printed on the page): ['acc-lock', 'acc-reset', 'err-1042']
    BM25 (only chunks with a score above 0): ['err-1044']
    fused (top 3): ['acc-lock', 'err-1044', 'acc-reset']
  [ok] RRF turns the page's two lists into the page's fused top three
  [ok] rank 1 in a list is worth exactly 1/61 with k=60, and a chunk missing from a list adds nothing

  how much more a first place is worth than a fifth or a fiftieth place, by k:
  k     rank 1 / rank 5  rank 1 / rank 50
  ----  ---------------  ----------------
     1            3.000            25.500
    10            1.364             5.455
    60            1.066             1.803
  1000            1.004             1.049
  [ok] a small k lets the first place dominate; k=60 keeps lower ranks in play

  word BM25, character-trigram BM25 and their RRF on this page's 12 queries (offline):
  system  hit@1  hit@3  recall@5  mrr@5
  ------  -----  -----  --------  -----
    word   0.92   0.92      0.92   0.92
   char3   0.50   0.50      0.50   0.50
     rrf   0.83   0.92      0.92   0.88
  Fusion pays when the two lists fail on different queries and the added list is good enough; two
  lexical scorers mostly fail together, and one query is 8.3 points of Hit@1, so read this as arithmetic,
  not as a ranking of scorers. The page's gain comes from fusing a dense retriever (section 7).
  [ok] adding raw scores of different scales (the second divided by 1000, like a cosine) always keeps the first scorer's top chunk
  [ok] RRF gives identical fused lists whatever the scale of a score (here: x10^6)

## 5. Retrieval metrics: Hit@k, Recall@k, MRR
  [ok] BM25 offline: the right chunk's rank for Q1..Q12 is the page's BM25 column  ([1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
  the page's rank tables (dense, rrf, rerank come from the page; bm25 was just recomputed), scored:
  system  hit@1  hit@3  recall@5  mrr@5
  ------  -----  -----  --------  -----
   dense   0.75   0.83      0.92   0.79
    bm25   0.92   0.92      0.92   0.92
     rrf   0.83   0.92      1.00   0.90
  rerank   1.00   1.00      1.00   1.00
  [ok] the metric code turns the page's rank tables into the page's metric tables  ([])
  [ok] with one right chunk per query, Hit@5 equals Recall@5, and MRR is at most Hit@1 + (1 - Hit@1) / 2
  One query is worth 8.3 points of Hit@1 (12 queries), and there are no confidence intervals here: a mechanism demo, not a benchmark.
  [ok] one query is 8.3 points

## 6. Context assembly: a budget is not a relevance test
  BM25 alone, budget 100 characters, query: "借款最多能借多少，什么时候要还？"
    * exp-advance    2.75  (in the prompt)
    * err-1045       2.55  (in the prompt)
      remote-policy  2.04  
  [ok] the prompt holds the first chunks that fit the budget, numbered, and stops at the first one that does not  (2 chunks, 83 characters)
  The right chunk (exp-advance) is at rank 1 here: BM25 alone gets this question right (the English one, full of stopwords, is the hard one).
  A relative score cut cannot fix a wrong first place, it only keeps the neighbours of whatever came first:
    keep scores >= 0.5 x best -> ['exp-advance', 'err-1045', 'remote-policy']
    keep scores >= 0.7 x best -> ['exp-advance', 'err-1045', 'remote-policy']
  [ok] a higher relative threshold keeps fewer chunks, but the chunk ranked first always stays

## 7. The neural parts: dense retrieval and the reranker (--bge)
  [skip] dense retrieval, RRF with BM25, reranking: the page's four-system comparison  (run with --bge (needs the 'hf' extra and the two BAAI models))

All checks passed. (0.0s)
```

</details>

<details>
<summary>加上真实模型的输出（`labs/applications/rag/run.py --lang zh --bge`）</summary>

```text
# RAG, piece by piece
  course: https://dmxsf.com/wiki/applications/rag/
          https://dmxsf.com/en/wiki/applications/rag/

## 1. Chunking: where the sentences go
  The page's made-up expense policy (354 characters): 15 sentences, the longest 45 characters. A sentence is 'cut' when no chunk holds it whole.
  strategy   size  overlap  chunks  avg chars  cut  stored/doc
  ---------  ----  -------  ------  ---------  ---  ----------
      fixed   120        0       3      118.0    2       1.00x
      fixed    60        0       6      58.83    4       1.00x
      fixed   120        5       4      92.25    1       1.04x
      fixed   120       10       4      95.00    0       1.07x
      fixed   120       15       4      99.75    1       1.12x
      fixed   120       20       4      103.2    0       1.16x
      fixed   120      100      13      119.4    0       4.34x
      fixed   120       45       5      106.6    0       1.51x
   sentence   120        0       4      87.75    0       1.00x
  paragraph   120        0       5      69.20    0       1.00x
  paragraph   400        0       5      69.20    0       1.00x
  [ok] the document has 15 sentences, the longest 45 characters, as the widget counts
  [ok] every (chunks, cut sentences) pair matches what the widget shows  ([])
  [ok] the redundancy figures match  (('fixed', 120, 20): 1.16x, ('fixed', 120, 100): 4.34x)
  [ok] more overlap does not mean fewer cut sentences (not monotonic)  (overlaps 5, 10, 15, 20 cut [1, 0, 1, 0])
  [ok] overlap >= longest sentence (with a longer window) leaves no sentence cut: no exception in the grid  (91 (size, overlap) pairs, window 46+, overlap 45+)
  [ok] a window shorter than the longest sentence cuts something whatever the overlap  (windows 40 and 32, overlaps 0..)
  [ok] splitting by sentence or paragraph never cuts a sentence (sizes 40..400)
  What the guarantee costs here: window 120, overlap 45 stores 1.51x the document.

## 2. Similarity, normalisation and the size of an index
  [ok] on unit vectors the inner product is the cosine
  [ok] and the squared distance is 2 - 2 * cosine
  [ok] so inner product, cosine and Euclidean distance give one and the same ranking
  [ok] without normalising, the inner product ranks differently from the cosine (long vectors win)  (9 of the top 20 agree)

  index memory, vectors only (GB = 10^9 bytes):
  vectors     dim   float32 GB  int8 GB
  ----------  ----  ----------  -------
     100,000   384        0.15     0.04
   1,000,000   768        3.07     0.77
  10,000,000  1024       40.96    10.24
  [ok] float32 and int8 sizes match the page (0.15/0.04, 3.07/0.77, 40.96/10.24)

## 3. BM25 from the formula
  query: "报销需要上传哪些材料？"   (23 chunks, average length 27.5 tokens)
  chunk         BM25   what it matched (term, contribution)
  ------------  -----  ------------------------------------
  exp-material  7.418             上传 2.75, 报销 1.91, 材料 2.75
    exp-submit  5.456                      报销 2.66, 销需 2.79
      err-1041  2.464                               需要 2.46
  [ok] the top three and their scores are the page's  (exp-material 7.42, exp-submit 5.46, err-1041 2.46)
  [ok] err-1041 has nothing to do with the question yet is in the top three: it matched only ['需要']
  BM25 is a word-matching signal, not understanding: a filler word of the question (or, in Chinese, a bigram that
  crosses a word boundary) is enough to score.
  [ok] the IDF stays positive even for a term in every chunk: ln(1 + 0.5 / (N + 0.5)) > 0  (0.134 for N=3)
  term-frequency factor f(k1+1)/(f+k1) for f = 1, 2, 5, 10, 100 (length-normalised chunk): 1.00, 1.38, 1.77, 1.96, 2.17  -> the limit is k1+1 = 2.2
  [ok] term frequency saturates: ten occurrences count less than twice one, a hundred less than k1+1 = 2.2 times
  [ok] length normalisation (b=0.75) scores the long chunk lower than the short one for the same term; with b=0 they tie  (0.25 against 0.14)

## 4. Rank fusion: RRF uses ranks, not scores
  query: "错误码 E1044 怎么处理"
    dense (top 3 as printed on the page): ['acc-lock', 'acc-reset', 'err-1042']
    BM25 (only chunks with a score above 0): ['err-1044']
    fused (top 3): ['acc-lock', 'err-1044', 'acc-reset']
  [ok] RRF turns the page's two lists into the page's fused top three
  [ok] rank 1 in a list is worth exactly 1/61 with k=60, and a chunk missing from a list adds nothing

  how much more a first place is worth than a fifth or a fiftieth place, by k:
  k     rank 1 / rank 5  rank 1 / rank 50
  ----  ---------------  ----------------
     1            3.000            25.500
    10            1.364             5.455
    60            1.066             1.803
  1000            1.004             1.049
  [ok] a small k lets the first place dominate; k=60 keeps lower ranks in play

  word BM25, character-trigram BM25 and their RRF on this page's 12 queries (offline):
  system  hit@1  hit@3  recall@5  mrr@5
  ------  -----  -----  --------  -----
    word   0.92   0.92      0.92   0.92
   char3   0.50   0.50      0.50   0.50
     rrf   0.83   0.92      0.92   0.88
  Fusion pays when the two lists fail on different queries and the added list is good enough; two
  lexical scorers mostly fail together, and one query is 8.3 points of Hit@1, so read this as arithmetic,
  not as a ranking of scorers. The page's gain comes from fusing a dense retriever (section 7).
  [ok] adding raw scores of different scales (the second divided by 1000, like a cosine) always keeps the first scorer's top chunk
  [ok] RRF gives identical fused lists whatever the scale of a score (here: x10^6)

## 5. Retrieval metrics: Hit@k, Recall@k, MRR
  [ok] BM25 offline: the right chunk's rank for Q1..Q12 is the page's BM25 column  ([1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
  the page's rank tables (dense, rrf, rerank come from the page; bm25 was just recomputed), scored:
  system  hit@1  hit@3  recall@5  mrr@5
  ------  -----  -----  --------  -----
   dense   0.75   0.83      0.92   0.79
    bm25   0.92   0.92      0.92   0.92
     rrf   0.83   0.92      1.00   0.90
  rerank   1.00   1.00      1.00   1.00
  [ok] the metric code turns the page's rank tables into the page's metric tables  ([])
  [ok] with one right chunk per query, Hit@5 equals Recall@5, and MRR is at most Hit@1 + (1 - Hit@1) / 2
  One query is worth 8.3 points of Hit@1 (12 queries), and there are no confidence intervals here: a mechanism demo, not a benchmark.
  [ok] one query is 8.3 points

## 6. Context assembly: a budget is not a relevance test
  BM25 alone, budget 100 characters, query: "借款最多能借多少，什么时候要还？"
    * exp-advance    2.75  (in the prompt)
    * err-1045       2.55  (in the prompt)
      remote-policy  2.04  
  [ok] the prompt holds the first chunks that fit the budget, numbered, and stops at the first one that does not  (2 chunks, 83 characters)
  The right chunk (exp-advance) is at rank 1 here: BM25 alone gets this question right (the English one, full of stopwords, is the hard one).
  A relative score cut cannot fix a wrong first place, it only keeps the neighbours of whatever came first:
    keep scores >= 0.5 x best -> ['exp-advance', 'err-1045', 'remote-policy']
    keep scores >= 0.7 x best -> ['exp-advance', 'err-1045', 'remote-policy']
  [ok] a higher relative threshold keeps fewer chunks, but the chunk ranked first always stays

## 7. The neural parts: dense retrieval and the reranker (--bge)
  BAAI/bge-small-zh-v1.5: 23 chunks -> (23, 512); query "登录口令忘了怎么办？"
    0.555  acc-lock
    0.509  err-1041
    0.506  acc-reset
  [ok] the embedding matrix has unit rows
  [ok] the identity |q - d|^2 = 2 - 2 q.d holds for the real embeddings
  [ok] the top three by cosine are the page's  ([('acc-lock', 0.555), ('err-1041', 0.509), ('acc-reset', 0.506)])

  rank of the right chunk (- = not in the top 5):
  query  dense  bm25  rrf  rerank
  -----  -----  ----  ---  ------
     Q1      1     1    1       1
     Q2      3     -    4       1
     Q3      1     1    1       1
     Q4      1     1    1       1
     Q5      1     1    1       1
     Q6      -     1    2       1
     Q7      1     1    1       1
     Q8      5     1    1       1
     Q9      1     1    1       1
    Q10      1     1    1       1
    Q11      1     1    1       1
    Q12      1     1    1       1

  system  hit@1  hit@3  recall@5  mrr@5
  ------  -----  -----  --------  -----
   dense   0.75   0.83      0.92   0.79
    bm25   0.92   0.92      0.92   0.92
     rrf   0.83   0.92      1.00   0.90
  rerank   1.00   1.00      1.00   1.00
  [ok] every rank of every system matches the page's table  (dense, bm25, rrf and rerank, 12 queries)
  [ok] and so do the metrics

  query: "登录口令忘了怎么办？"
    fused    ['acc-lock', 'err-1041', 'vpn-install', 'acc-reset', 'exp-material']
    reranked ['acc-reset', 'acc-lock', 'err-1041', 'vpn-install', 'exp-material']
  [ok] fusion and reranking order the page's example as printed

  the prompt of the page's context-assembly example (each line cut at 60 characters):
    只依据下列资料回答，资料里没有答案就说“资料中没有”，并在句末用 [编号] 标注来源。
    
    资料：
    [1] 借款需提前 3 个工作日申请，单次不超过 10000 元，出差结束后 7 个工作日内必须冲销。
    [2] 单笔采购金额超过 5000 元须提供至少三家供应商的比价单，并经部门负责人审批。
    
    问题：借款最多能借多少，什么时候要还？
  [ok] the chunks that fit the budget are the page's: the right one first, then one that has nothing to do with it  (['exp-advance', 'buy-threshold'])

All checks passed. (5.1s)
```

</details>
