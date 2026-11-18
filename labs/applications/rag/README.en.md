# RAG, piece by piece

Course pages: [RAG in Practice: Chunking, Hybrid Retrieval, Reranking and Evaluation](https://dmxsf.com/en/wiki/applications/rag/)  
[中文](README.md)

Chunking, similarity and index size, BM25 from the formula, rank fusion (RRF), retrieval metrics and context assembly, reproducing the page's numbers offline; with --bge the dense retriever and the reranker run too.

## Run

```bash
uv run python labs/applications/rag/run.py            # full run, a few seconds
uv run python labs/applications/rag/run.py --quick    # a shrunk smoke test, a few seconds
```

You can also write `uv run dmxsf run rag` (and add `--quick`). `--seed N` changes the random seed, default 0. The script prints its results section by section and checks the course page's claims one by one with `[ok]` / `[FAIL]`: the exit status is 0 when all pass and 1 when any fails; checks that only make sense in the full run are marked `[skip]` under `--quick`.

Optional flags:

- `--lang en|zh`: which language's corpus and queries of the course page to use (default `en`).
- `--bge`: also run the real dense retriever (BAAI/bge-small-en-v1.5 or -zh-v1.5) and the reranker (BAAI/bge-reranker-base) and reproduce all the page's retrieval metrics. Needs `uv sync --extra hf` and the downloaded models.

## Code

| File | What it holds |
| --- | --- |
| `labs/applications/rag/run.py` | the lab script, numbered sections |
| `src/dmxsf/applications/rag.py` | Retrieval-augmented generation, piece by piece: chunking, BM25, RRF, metrics, context assembly |
| `src/dmxsf/applications/rag_data.py` | The small corpora, queries and sample documents of the RAG page (one set per language) |

Unit tests: `uv run pytest tests/test_applications_rag.py`

## What to look for

- Section 1: the page's made-up expense policy (1131 characters, 19 sentences, the longest 133): fixed windows of 120 characters with no overlap cut 9 of the 19 sentences; more overlap is not monotonically better, the cut count at overlaps 5, 10, 15, 20 is 7, 8, 9, 8; only an overlap at least as long as the longest sentence (with a longer window) leaves no sentence cut, and it stores 2.64 times the document; splitting by sentence or paragraph never cuts one.
- Section 2: on unit vectors the inner product is the cosine and the squared distance is 2 - 2 cos, so the three give one ranking; without normalising the inner product ranks differently (9 of the top 20 agree); the index-size table matches the page.
- Section 3: BM25 written from the formula: the top three of 'Which documents must I upload with an expense claim?' and their scores (5.32 / 3.46 / 3.34) are the page's; comp-time has nothing to do with the question yet is in the top three, matched only by 'must' and 'which'; term-frequency saturation tops out at k1 + 1 = 2.2.
- Section 4: RRF uses ranks, not scores: in the page's example the dense list [err-1044, err-1043, err-1042] and the BM25 list [err-1042] fuse to [err-1042, err-1044, err-1043]; offline, word BM25, character-trigram BM25 and their RRF score Hit@1 0.67 / 0.67 / 0.58: no gain, two lexical retrievers fail together, and fusion pays when the added list is of a different kind.
- Sections 5-6: Hit@k, Recall@k and MRR computed from the rank tables agree with the page (one query is 8.3 points, no confidence intervals); with a context budget of 300 characters the right chunk, at rank 3 for this question, does not make it into the prompt, which holds comp-time and vpn-cert instead: a budget is not a relevance test and a relative score cut cannot fix a wrong first place.
- Section 7 (`--bge`): BAAI/bge-small-en-v1.5 and BAAI/bge-reranker-base: every rank of every system for every query matches the page's table: on the 12 queries dense retrieval has Hit@1 0.83, BM25 0.67, RRF 0.92, reranking 0.92 (MRR 0.96).

## Exercises

Run again after a change: the checks that compare with the page's numbers turn into `[FAIL]`; that is expected, read how the numbers themselves move.

1. In section 6 change `"prompt_budget": 300` (English) to 600: which chunks enter the prompt now, and are they relevant to the question?
2. Add 5 to `for k in (1, 10, 60, 1000)` in section 4: how many times is a first place worth a fifth place in RRF?

## Notes

- Section 7 needs `--bge` (the `hf` extra and the two BAAI models); the English and the Chinese runs each used their own sample document and their own 12 queries, and no number is borrowed from the other. The document is made-up demo data, not a real company's policy.

## Output

The complete runs below were made on an Apple M2 Max (12 cores, CPU only) with Python 3.12 and PyTorch 2.15.0. On another machine the numbers that involve training can differ in the last digits and timings depend on the machine; the checks use thresholds and tolerances, not bit-for-bit equality.

<details>
<summary>Output of the offline run (`labs/applications/rag/run.py --lang en`)</summary>

```text
# RAG, piece by piece
  course: https://dmxsf.com/wiki/applications/rag/
          https://dmxsf.com/en/wiki/applications/rag/

## 1. Chunking: where the sentences go
  The page's made-up expense policy (1131 characters): 19 sentences, the longest 133 characters. A sentence is 'cut' when no chunk holds it whole.
  strategy   size  overlap  chunks  avg chars  cut  stored/doc
  ---------  ----  -------  ------  ---------  ---  ----------
      fixed   120        0      10      112.9    9       1.00x
      fixed    60        0      19      59.16   11       1.00x
      fixed   120        5      10      117.4    7       1.04x
      fixed   120       10      11      111.5    8       1.09x
      fixed   120       15      11      116.1    9       1.13x
      fixed   120       20      12      112.2    8       1.20x
      fixed   120      100      52      119.5    4       5.50x
      fixed   200      133      15      199.2    0       2.64x
   sentence   120        0      11      101.8    0       1.00x
  paragraph   120        0      13      85.77    0       1.00x
  paragraph   400        0       5      224.6    0       1.00x
  [ok] the document has 19 sentences, the longest 133 characters, as the widget counts
  [ok] every (chunks, cut sentences) pair matches what the widget shows  ([])
  [ok] the redundancy figures match  (('fixed', 120, 20): 1.20x, ('fixed', 120, 100): 5.50x, ('fixed', 200, 133): 2.64x)
  [ok] more overlap does not mean fewer cut sentences (not monotonic)  (overlaps 5, 10, 15, 20 cut [7, 8, 9, 8])
  [ok] overlap >= longest sentence (with a longer window) leaves no sentence cut: no exception in the grid  (741 (size, overlap) pairs, window 134+, overlap 133+)
  [ok] a window shorter than the longest sentence cuts something whatever the overlap  (windows 128 and 76, overlaps 0..)
  [ok] splitting by sentence or paragraph never cuts a sentence (sizes 40..400)
  What the guarantee costs here: window 200, overlap 133 stores 2.64x the document.

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
  query: "Which documents must I upload with an expense claim?"   (23 chunks, average length 21.4 tokens)
  chunk         BM25   what it matched (term, contribution)
  ------------  -----  ------------------------------------
  exp-material  5.320                   an 1.49, claim 3.83
     comp-time  3.464                 must 0.87, which 2.60
    exp-submit  3.344               expense 2.37, must 0.97
  [ok] the top three and their scores are the page's  (exp-material 5.32, comp-time 3.46, exp-submit 3.34)
  [ok] comp-time has nothing to do with the question yet is in the top three: it matched only ['must', 'which']
  BM25 is a word-matching signal, not understanding: a filler word of the question (or, in Chinese, a bigram that
  crosses a word boundary) is enough to score.
  [ok] the IDF stays positive even for a term in every chunk: ln(1 + 0.5 / (N + 0.5)) > 0  (0.134 for N=3)
  term-frequency factor f(k1+1)/(f+k1) for f = 1, 2, 5, 10, 100 (length-normalised chunk): 1.00, 1.38, 1.77, 1.96, 2.17  -> the limit is k1+1 = 2.2
  [ok] term frequency saturates: ten occurrences count less than twice one, a hundred less than k1+1 = 2.2 times
  [ok] length normalisation (b=0.75) scores the long chunk lower than the short one for the same term; with b=0 they tie  (0.25 against 0.14)

## 4. Rank fusion: RRF uses ranks, not scores
  query: "E1042"
    dense (top 3 as printed on the page): ['err-1044', 'err-1043', 'err-1042']
    BM25 (only chunks with a score above 0): ['err-1042']
    fused (top 3): ['err-1042', 'err-1044', 'err-1043']
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
    word   0.67   0.92      0.92   0.78
   char3   0.67   0.83      1.00   0.78
     rrf   0.58   0.92      0.92   0.74
  Fusion pays when the two lists fail on different queries and the added list is good enough; two
  lexical scorers mostly fail together, and one query is 8.3 points of Hit@1, so read this as arithmetic,
  not as a ranking of scorers. The page's gain comes from fusing a dense retriever (section 7).
  [ok] adding raw scores of different scales (the second divided by 1000, like a cosine) always keeps the first scorer's top chunk
  [ok] RRF gives identical fused lists whatever the scale of a score (here: x10^6)

## 5. Retrieval metrics: Hit@k, Recall@k, MRR
  [ok] BM25 offline: the right chunk's rank for Q1..Q12 is the page's BM25 column  ([1, 2, 2, 0, 1, 1, 1, 1, 1, 1, 3, 1])
  the page's rank tables (dense, rrf, rerank come from the page; bm25 was just recomputed), scored:
  system  hit@1  hit@3  recall@5  mrr@5
  ------  -----  -----  --------  -----
   dense   0.83   1.00      1.00   0.90
    bm25   0.67   0.92      0.92   0.78
     rrf   0.92   1.00      1.00   0.94
  rerank   0.92   1.00      1.00   0.96
  [ok] the metric code turns the page's rank tables into the page's metric tables  ([])
  [ok] with one right chunk per query, Hit@5 equals Recall@5, and MRR is at most Hit@1 + (1 - Hit@1) / 2
  One query is worth 8.3 points of Hit@1 (12 queries), and there are no confidence intervals here: a mechanism demo, not a benchmark.
  [ok] one query is 8.3 points

## 6. Context assembly: a budget is not a relevance test
  BM25 alone, budget 300 characters, query: "How much can I borrow in advance and when must I pay it back?"
    * comp-time      6.41  (in the prompt)
    * vpn-cert       4.64  (in the prompt)
      exp-advance    4.01  
      exp-material   3.30  
      err-1041       2.99  
  [ok] the prompt holds the first chunks that fit the budget, numbered, and stops at the first one that does not  (2 chunks, 280 characters)
  The right chunk (exp-advance) is at rank 3 here; a stopword-heavy question makes BM25 alone a poor ranker.
  A relative score cut cannot fix a wrong first place, it only keeps the neighbours of whatever came first:
    keep scores >= 0.5 x best -> ['comp-time', 'vpn-cert', 'exp-advance', 'exp-material']
    keep scores >= 0.7 x best -> ['comp-time', 'vpn-cert']
  [ok] a higher relative threshold keeps fewer chunks, but the chunk ranked first always stays

## 7. The neural parts: dense retrieval and the reranker (--bge)
  [skip] dense retrieval, RRF with BM25, reranking: the page's four-system comparison  (run with --bge (needs the 'hf' extra and the two BAAI models))

All checks passed. (0.3s)
```

</details>

<details>
<summary>Output with the real models (`labs/applications/rag/run.py --lang en --bge`)</summary>

```text
# RAG, piece by piece
  course: https://dmxsf.com/wiki/applications/rag/
          https://dmxsf.com/en/wiki/applications/rag/

## 1. Chunking: where the sentences go
  The page's made-up expense policy (1131 characters): 19 sentences, the longest 133 characters. A sentence is 'cut' when no chunk holds it whole.
  strategy   size  overlap  chunks  avg chars  cut  stored/doc
  ---------  ----  -------  ------  ---------  ---  ----------
      fixed   120        0      10      112.9    9       1.00x
      fixed    60        0      19      59.16   11       1.00x
      fixed   120        5      10      117.4    7       1.04x
      fixed   120       10      11      111.5    8       1.09x
      fixed   120       15      11      116.1    9       1.13x
      fixed   120       20      12      112.2    8       1.20x
      fixed   120      100      52      119.5    4       5.50x
      fixed   200      133      15      199.2    0       2.64x
   sentence   120        0      11      101.8    0       1.00x
  paragraph   120        0      13      85.77    0       1.00x
  paragraph   400        0       5      224.6    0       1.00x
  [ok] the document has 19 sentences, the longest 133 characters, as the widget counts
  [ok] every (chunks, cut sentences) pair matches what the widget shows  ([])
  [ok] the redundancy figures match  (('fixed', 120, 20): 1.20x, ('fixed', 120, 100): 5.50x, ('fixed', 200, 133): 2.64x)
  [ok] more overlap does not mean fewer cut sentences (not monotonic)  (overlaps 5, 10, 15, 20 cut [7, 8, 9, 8])
  [ok] overlap >= longest sentence (with a longer window) leaves no sentence cut: no exception in the grid  (741 (size, overlap) pairs, window 134+, overlap 133+)
  [ok] a window shorter than the longest sentence cuts something whatever the overlap  (windows 128 and 76, overlaps 0..)
  [ok] splitting by sentence or paragraph never cuts a sentence (sizes 40..400)
  What the guarantee costs here: window 200, overlap 133 stores 2.64x the document.

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
  query: "Which documents must I upload with an expense claim?"   (23 chunks, average length 21.4 tokens)
  chunk         BM25   what it matched (term, contribution)
  ------------  -----  ------------------------------------
  exp-material  5.320                   an 1.49, claim 3.83
     comp-time  3.464                 must 0.87, which 2.60
    exp-submit  3.344               expense 2.37, must 0.97
  [ok] the top three and their scores are the page's  (exp-material 5.32, comp-time 3.46, exp-submit 3.34)
  [ok] comp-time has nothing to do with the question yet is in the top three: it matched only ['must', 'which']
  BM25 is a word-matching signal, not understanding: a filler word of the question (or, in Chinese, a bigram that
  crosses a word boundary) is enough to score.
  [ok] the IDF stays positive even for a term in every chunk: ln(1 + 0.5 / (N + 0.5)) > 0  (0.134 for N=3)
  term-frequency factor f(k1+1)/(f+k1) for f = 1, 2, 5, 10, 100 (length-normalised chunk): 1.00, 1.38, 1.77, 1.96, 2.17  -> the limit is k1+1 = 2.2
  [ok] term frequency saturates: ten occurrences count less than twice one, a hundred less than k1+1 = 2.2 times
  [ok] length normalisation (b=0.75) scores the long chunk lower than the short one for the same term; with b=0 they tie  (0.25 against 0.14)

## 4. Rank fusion: RRF uses ranks, not scores
  query: "E1042"
    dense (top 3 as printed on the page): ['err-1044', 'err-1043', 'err-1042']
    BM25 (only chunks with a score above 0): ['err-1042']
    fused (top 3): ['err-1042', 'err-1044', 'err-1043']
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
    word   0.67   0.92      0.92   0.78
   char3   0.67   0.83      1.00   0.78
     rrf   0.58   0.92      0.92   0.74
  Fusion pays when the two lists fail on different queries and the added list is good enough; two
  lexical scorers mostly fail together, and one query is 8.3 points of Hit@1, so read this as arithmetic,
  not as a ranking of scorers. The page's gain comes from fusing a dense retriever (section 7).
  [ok] adding raw scores of different scales (the second divided by 1000, like a cosine) always keeps the first scorer's top chunk
  [ok] RRF gives identical fused lists whatever the scale of a score (here: x10^6)

## 5. Retrieval metrics: Hit@k, Recall@k, MRR
  [ok] BM25 offline: the right chunk's rank for Q1..Q12 is the page's BM25 column  ([1, 2, 2, 0, 1, 1, 1, 1, 1, 1, 3, 1])
  the page's rank tables (dense, rrf, rerank come from the page; bm25 was just recomputed), scored:
  system  hit@1  hit@3  recall@5  mrr@5
  ------  -----  -----  --------  -----
   dense   0.83   1.00      1.00   0.90
    bm25   0.67   0.92      0.92   0.78
     rrf   0.92   1.00      1.00   0.94
  rerank   0.92   1.00      1.00   0.96
  [ok] the metric code turns the page's rank tables into the page's metric tables  ([])
  [ok] with one right chunk per query, Hit@5 equals Recall@5, and MRR is at most Hit@1 + (1 - Hit@1) / 2
  One query is worth 8.3 points of Hit@1 (12 queries), and there are no confidence intervals here: a mechanism demo, not a benchmark.
  [ok] one query is 8.3 points

## 6. Context assembly: a budget is not a relevance test
  BM25 alone, budget 300 characters, query: "How much can I borrow in advance and when must I pay it back?"
    * comp-time      6.41  (in the prompt)
    * vpn-cert       4.64  (in the prompt)
      exp-advance    4.01  
      exp-material   3.30  
      err-1041       2.99  
  [ok] the prompt holds the first chunks that fit the budget, numbered, and stops at the first one that does not  (2 chunks, 280 characters)
  The right chunk (exp-advance) is at rank 3 here; a stopword-heavy question makes BM25 alone a poor ranker.
  A relative score cut cannot fix a wrong first place, it only keeps the neighbours of whatever came first:
    keep scores >= 0.5 x best -> ['comp-time', 'vpn-cert', 'exp-advance', 'exp-material']
    keep scores >= 0.7 x best -> ['comp-time', 'vpn-cert']
  [ok] a higher relative threshold keeps fewer chunks, but the chunk ranked first always stays

## 7. The neural parts: dense retrieval and the reranker (--bge)
  BAAI/bge-small-en-v1.5: 23 chunks -> (23, 384); query "I forgot my login credentials, what now?"
    0.727  acc-reset
    0.657  err-1041
    0.656  acc-lock
  [ok] the embedding matrix has unit rows
  [ok] the identity |q - d|^2 = 2 - 2 q.d holds for the real embeddings
  [ok] the top three by cosine are the page's  ([('acc-reset', 0.727), ('err-1041', 0.657), ('acc-lock', 0.656)])

  rank of the right chunk (- = not in the top 5):
  query  dense  bm25  rrf  rerank
  -----  -----  ----  ---  ------
     Q1      1     1    1       1
     Q2      1     2    1       1
     Q3      1     2    1       1
     Q4      1     -    3       2
     Q5      3     1    1       1
     Q6      2     1    1       1
     Q7      1     1    1       1
     Q8      1     1    1       1
     Q9      1     1    1       1
    Q10      1     1    1       1
    Q11      1     3    1       1
    Q12      1     1    1       1

  system  hit@1  hit@3  recall@5  mrr@5
  ------  -----  -----  --------  -----
   dense   0.83   1.00      1.00   0.90
    bm25   0.67   0.92      0.92   0.78
     rrf   0.92   1.00      1.00   0.94
  rerank   0.92   1.00      1.00   0.96
  [ok] every rank of every system matches the page's table  (dense, bm25, rrf and rerank, 12 queries)
  [ok] and so do the metrics

  query: "What do I need to bring for a day off when I feel ill?"
    fused    ['leave-annual', 'comp-time', 'leave-sick', 'backup']
    reranked ['exp-material', 'leave-sick', 'comp-time', 'leave-annual']
  [ok] fusion and reranking order the page's example as printed

  the prompt of the page's context-assembly example (each line cut at 60 characters):
    Answer only from the sources below. If they do not contain t...
    
    Sources:
    [1] An advance must be requested 3 working days ahead, up to...
    [2] Annual leave depends on total years of work: 5 days for ...
    
    Question: How much can I borrow in advance and when must I p...
  [ok] the chunks that fit the budget are the page's: the right one first, then one that has nothing to do with it  (['exp-advance', 'leave-annual'])

All checks passed. (6.4s)
```

</details>
