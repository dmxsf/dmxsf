"""Lab: RAG, piece by piece. Chunking, similarity and index size, BM25, rank fusion, retrieval metrics.

Course page: https://dmxsf.com/wiki/applications/rag/

Everything that does not need a neural network runs offline and reproduces the numbers on the page:
the chunking widget's counts, the index-size table, BM25's scores on the page's corpus, RRF on the page's
rankings, and the retrieval metrics. With `--bge` (extra `hf`: `uv run --extra hf ...`) the dense
retriever (BAAI/bge-small) and the reranker (BAAI/bge-reranker-base) are run as well, and the page's whole
four-system comparison is reproduced rank for rank. `--lang zh` switches to the Chinese page's corpus.

    uv run python labs/applications/rag/run.py [--lang en|zh] [--bge] [--seed N]
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from dmxsf import lab
from dmxsf.applications import rag, rag_data

# What the page prints, per language. Chunking: (chunks, cut sentences) for (strategy, size, overlap), and
# the redundancy the text quotes. Ranks: where the right chunk landed for Q1..Q12 (0 = not in the top 5).
PAGE = {
    "en": {
        "sentences": 19,
        "longest": 133,
        "guarantee": (200, 133),
        "chunking": {
            ("fixed", 120, 0): (10, 9),
            ("fixed", 60, 0): (19, 11),
            ("fixed", 120, 5): (10, 7),
            ("fixed", 120, 10): (11, 8),
            ("fixed", 120, 15): (11, 9),
            ("fixed", 120, 20): (12, 8),
            ("fixed", 120, 100): (52, 4),
            ("fixed", 200, 133): (15, 0),
            ("sentence", 120, 0): (11, 0),
            ("paragraph", 120, 0): (13, 0),
            ("paragraph", 400, 0): (5, 0),
        },
        "redundancy": {
            ("fixed", 120, 20): 1.20,
            ("fixed", 120, 100): 5.50,
            ("fixed", 200, 133): 2.64,
        },
        "ranks": {
            "dense": [1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 1],
            "bm25": [1, 2, 2, 0, 1, 1, 1, 1, 1, 1, 3, 1],
            "rrf": [1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1],
            "rerank": [1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        },
        "metrics": {
            "dense": (0.83, 1.00, 1.00, 0.90),
            "bm25": (0.67, 0.92, 0.92, 0.78),
            "rrf": (0.92, 1.00, 1.00, 0.94),
            "rerank": (0.92, 1.00, 1.00, 0.96),
        },
        "bm25_query": "Which documents must I upload with an expense claim?",
        "bm25_top": [("exp-material", 5.32), ("comp-time", 3.46), ("exp-submit", 3.34)],
        "bm25_noise": (
            "comp-time",
            {"must", "which"},
        ),  # unrelated to the question, scores on filler
        "embed_query": "I forgot my login credentials, what now?",
        "embed_top": [("acc-reset", 0.727), ("err-1041", 0.657), ("acc-lock", 0.656)],
        "fusion_query": "E1042",
        # the first three of the dense list as printed (the other two places do not change the top three)
        "dense_head": ["err-1044", "err-1043", "err-1042"],
        "bm25_list": ["err-1042"],
        "fused_head": ["err-1042", "err-1044", "err-1043"],
        "rerank_query": "What do I need to bring for a day off when I feel ill?",
        "rerank_pool": ["leave-annual", "comp-time", "leave-sick", "backup"],
        "reranked": ["exp-material", "leave-sick", "comp-time", "leave-annual"],
        "prompt_query": "How much can I borrow in advance and when must I pay it back?",
        "prompt_budget": 300,
        "prompt_picked": ["exp-advance", "leave-annual"],
    },
    "zh": {
        "sentences": 15,
        "longest": 45,
        "guarantee": (120, 45),
        "chunking": {
            ("fixed", 120, 0): (3, 2),
            ("fixed", 60, 0): (6, 4),
            ("fixed", 120, 5): (4, 1),
            ("fixed", 120, 10): (4, 0),
            ("fixed", 120, 15): (4, 1),
            ("fixed", 120, 20): (4, 0),
            ("fixed", 120, 100): (13, 0),
            ("fixed", 120, 45): (5, 0),
            ("sentence", 120, 0): (4, 0),
            ("paragraph", 120, 0): (5, 0),
            ("paragraph", 400, 0): (5, 0),
        },
        "redundancy": {("fixed", 120, 20): 1.16, ("fixed", 120, 100): 4.34},
        "ranks": {
            "dense": [1, 3, 1, 1, 1, 0, 1, 5, 1, 1, 1, 1],
            "bm25": [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "rrf": [1, 4, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1],
            "rerank": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        },
        "metrics": {
            "dense": (0.75, 0.83, 0.92, 0.79),
            "bm25": (0.92, 0.92, 0.92, 0.92),
            "rrf": (0.83, 0.92, 1.00, 0.90),
            "rerank": (1.00, 1.00, 1.00, 1.00),
        },
        "bm25_query": "报销需要上传哪些材料？",
        "bm25_top": [("exp-material", 7.42), ("exp-submit", 5.46), ("err-1041", 2.46)],
        "bm25_noise": (
            "err-1041",
            {"需要"},
        ),  # unrelated to the question, scores on one shared bigram
        "embed_query": "登录口令忘了怎么办？",
        "embed_top": [("acc-lock", 0.555), ("err-1041", 0.509), ("acc-reset", 0.506)],
        "fusion_query": "错误码 E1044 怎么处理",
        "dense_head": ["acc-lock", "acc-reset", "err-1042"],
        "bm25_list": ["err-1044"],
        "fused_head": ["acc-lock", "err-1044", "acc-reset"],
        "rerank_query": "登录口令忘了怎么办？",
        "rerank_pool": ["acc-lock", "err-1041", "vpn-install", "acc-reset", "exp-material"],
        "reranked": ["acc-reset", "acc-lock", "err-1041", "vpn-install", "exp-material"],
        "prompt_query": "借款最多能借多少，什么时候要还？",
        "prompt_budget": 100,
        "prompt_picked": [
            "exp-advance",
            "buy-threshold",
        ],  # the page: the advance rule, then purchasing
    },
}
METRIC_NAMES = ("hit@1", "hit@3", "recall@5", "mrr@5")


def add_args(parser) -> None:
    parser.add_argument(
        "--lang", choices=["en", "zh"], default="en", help="which page's corpus (default en)"
    )
    parser.add_argument(
        "--bge", action="store_true", help="also run the real embedding model and reranker"
    )


def main() -> None:
    args, report = lab.start("RAG, piece by piece", wiki=["applications/rag"], add_args=add_args)
    lang = args.lang
    page = PAGE[lang]
    chunks, queries = rag_data.CHUNKS[lang], rag_data.QUERIES[lang]
    ids, texts = list(chunks), list(chunks.values())
    gold = [ids.index(g) for _, g in queries]
    bm25 = rag.BM25(texts, lang)
    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------------------------------------
    report.section("1. Chunking: where the sentences go")
    doc = rag_data.SAMPLE_DOCS[lang]
    every = rag.sentences(doc)
    longest = max(e - s for s, e in every)
    report.say(
        f"  The page's made-up expense policy ({len(doc)} characters): {len(every)} sentences, the longest "
        f"{longest} characters. A sentence is 'cut' when no chunk holds it whole."
    )
    rows = []
    got = {}
    for (strategy, size, overlap), _ in page["chunking"].items():
        rep = rag.evaluate_chunks(doc, rag.chunk(doc, strategy, size, overlap))
        got[(strategy, size, overlap)] = rep
        rows.append(
            [
                strategy,
                size,
                overlap,
                rep.count,
                rep.average,
                len(rep.cut),
                f"{rep.redundancy:.2f}x",
            ]
        )
    report.table(["strategy", "size", "overlap", "chunks", "avg chars", "cut", "stored/doc"], rows)
    report.check(
        f"the document has {page['sentences']} sentences, the longest {page['longest']} characters, as the widget counts",
        (len(every), longest) == (page["sentences"], page["longest"]),
    )
    mismatch = [
        k for k, (n, cut) in page["chunking"].items() if (got[k].count, len(got[k].cut)) != (n, cut)
    ]
    report.check(
        "every (chunks, cut sentences) pair matches what the widget shows",
        not mismatch,
        str(mismatch),
    )
    off = [k for k, v in page["redundancy"].items() if f"{got[k].redundancy:.2f}" != f"{v:.2f}"]
    report.check(
        "the redundancy figures match",
        not off,
        ", ".join(f"{k}: {v:.2f}x" for k, v in page["redundancy"].items()),
    )

    cuts = [len(got[("fixed", 120, o)].cut) for o in (5, 10, 15, 20)]
    report.check(
        "more overlap does not mean fewer cut sentences (not monotonic)",
        any(b > a for a, b in itertools.pairwise(cuts))
        and any(b < a for a, b in itertools.pairwise(cuts)),
        f"overlaps 5, 10, 15, 20 cut {cuts}",
    )
    size_g, over_g = page["guarantee"]
    # the guarantee: a window longer than the longest sentence, and an overlap at least that long
    violations = 0
    cells = 0
    for size in range(longest + 1, 3 * longest, 7):
        for overlap in range(longest, size, 7):
            cells += 1
            violations += bool(rag.evaluate_chunks(doc, rag.chunk(doc, "fixed", size, overlap)).cut)
    report.check(
        "overlap >= longest sentence (with a longer window) leaves no sentence cut: no exception in the grid",
        violations == 0,
        f"{cells} (size, overlap) pairs, window {longest + 1}+, overlap {longest}+",
    )
    short_ok = [
        (size, overlap)
        for size in (longest - 5, longest // 2 + 10)
        for overlap in range(0, size, 5)
        if not rag.evaluate_chunks(doc, rag.chunk(doc, "fixed", size, overlap)).cut
    ]
    report.check(
        "a window shorter than the longest sentence cuts something whatever the overlap",
        not short_ok,
        f"windows {longest - 5} and {longest // 2 + 10}, overlaps 0..",
    )
    never = all(
        not rag.evaluate_chunks(doc, rag.chunk(doc, s, size, 0)).cut
        for s in ("sentence", "paragraph")
        for size in range(40, 401, 10)
    )
    report.check("splitting by sentence or paragraph never cuts a sentence (sizes 40..400)", never)
    report.say(
        f"  What the guarantee costs here: window {size_g}, overlap {over_g} stores "
        f"{got[('fixed', size_g, over_g)].redundancy:.2f}x the document."
    )

    # ------------------------------------------------------------------------------------------------
    report.section("2. Similarity, normalisation and the size of an index")
    n_vec, dim = 2000, 64
    x = rng.normal(size=(n_vec, dim)) * rng.uniform(
        0.2, 5.0, size=(n_vec, 1)
    )  # vectors of very different norms
    q = rng.normal(size=dim)
    unit = x / np.linalg.norm(x, axis=1, keepdims=True)
    qn = q / np.linalg.norm(q)
    cos_raw = (x @ q) / (np.linalg.norm(x, axis=1) * np.linalg.norm(q))
    dot_unit = unit @ qn
    dist_unit = ((unit - qn) ** 2).sum(1)
    report.check(
        "on unit vectors the inner product is the cosine",
        np.allclose(cos_raw, dot_unit, atol=1e-12),
    )
    report.check(
        "and the squared distance is 2 - 2 * cosine",
        np.allclose(dist_unit, 2 - 2 * dot_unit, atol=1e-12),
    )
    same = np.array_equal(np.argsort(-dot_unit), np.argsort(dist_unit)) and np.array_equal(
        np.argsort(-dot_unit), np.argsort(-cos_raw)
    )
    report.check(
        "so inner product, cosine and Euclidean distance give one and the same ranking", same
    )
    top_cos = set(np.argsort(-cos_raw)[:20])
    top_dot = set(np.argsort(-(x @ q))[:20])
    report.check(
        "without normalising, the inner product ranks differently from the cosine (long vectors win)",
        len(top_cos & top_dot) < 20,
        f"{len(top_cos & top_dot)} of the top 20 agree",
    )
    rows = []
    for n, d in [(10**5, 384), (10**6, 768), (10**7, 1024)]:
        rows.append([f"{n:,}", d, f"{rag.index_gb(n, d, 4):.2f}", f"{rag.index_gb(n, d, 1):.2f}"])
    report.say("\n  index memory, vectors only (GB = 10^9 bytes):")
    report.table(["vectors", "dim", "float32 GB", "int8 GB"], rows)
    report.check(
        "float32 and int8 sizes match the page (0.15/0.04, 3.07/0.77, 40.96/10.24)",
        [(r[2], r[3]) for r in rows] == [("0.15", "0.04"), ("3.07", "0.77"), ("40.96", "10.24")],
    )

    # ------------------------------------------------------------------------------------------------
    report.section("3. BM25 from the formula")
    scores = bm25.scores(page["bm25_query"])
    best = [int(j) for j in np.argsort(-scores)[:3]]
    report.say(
        f'  query: "{page["bm25_query"]}"   ({bm25.n} chunks, average length {bm25.avgdl:.1f} tokens)'
    )
    rows = []
    for j in best:
        contrib = ", ".join(
            f"{t} {bm25.term_score(t, j):.2f}"
            for t in bm25.matched(page["bm25_query"])
            if bm25.term_score(t, j) > 0
        )
        rows.append([ids[j], float(scores[j]), contrib])
    report.table(["chunk", "BM25", "what it matched (term, contribution)"], rows)
    report.check(
        "the top three and their scores are the page's",
        [(ids[j], round(float(scores[j]), 2)) for j in best] == page["bm25_top"],
        ", ".join(f"{i} {s:.2f}" for i, s in page["bm25_top"]),
    )
    noise_id, noise_terms = page["bm25_noise"]
    noise_j = ids.index(noise_id)
    matched = {t for t in bm25.matched(page["bm25_query"]) if bm25.term_score(t, noise_j) > 0}
    report.check(
        f"{noise_id} has nothing to do with the question yet is in the top three: it matched only {sorted(matched)}",
        noise_j in best and matched == noise_terms,
    )
    report.say(
        "  BM25 is a word-matching signal, not understanding: a filler word of the question (or, in Chinese, a bigram that\n"
        "  crosses a word boundary) is enough to score."
    )
    tiny = rag.BM25(["a b"] * 3, "en")  # "a" is in every chunk
    report.check(
        "the IDF stays positive even for a term in every chunk: ln(1 + 0.5 / (N + 0.5)) > 0",
        abs(tiny.idf("a") - math.log(1 + 0.5 / 3.5)) < 1e-12 and tiny.idf("a") > 0,
        f"{tiny.idf('a'):.3f} for N=3",
    )
    k1 = bm25.k1
    sat = [f * (k1 + 1) / (f + k1) for f in (1, 2, 5, 10, 100)]
    report.say(
        "  term-frequency factor f(k1+1)/(f+k1) for f = 1, 2, 5, 10, 100 (length-normalised chunk): "
        + ", ".join(f"{x:.2f}" for x in sat)
        + f"  -> the limit is k1+1 = {k1 + 1:.1f}"
    )
    report.check(
        "term frequency saturates: ten occurrences count less than twice one, a hundred less than k1+1 = 2.2 times",
        sat[3] < 2 * sat[0] and sat[-1] < k1 + 1,
    )
    shorter = rag.BM25(
        ["error code e1042 means the pool is full", "error code e1042 " + "filler " * 40], "en"
    )
    s_short, s_long = shorter.scores("e1042")
    flat = rag.BM25(
        ["error code e1042 means the pool is full", "error code e1042 " + "filler " * 40],
        "en",
        b=0.0,
    )
    report.check(
        "length normalisation (b=0.75) scores the long chunk lower than the short one for the same term; with b=0 they tie",
        s_short > s_long and abs(flat.scores("e1042")[0] - flat.scores("e1042")[1]) < 1e-12,
        f"{s_short:.2f} against {s_long:.2f}",
    )

    # ------------------------------------------------------------------------------------------------
    report.section("4. Rank fusion: RRF uses ranks, not scores")
    dense_head = [ids.index(c) for c in page["dense_head"]]
    unknown_tail = [
        -1,
        -2,
    ]  # the page's dense list has five places; the last two cannot move the top three
    dense_list = dense_head + unknown_tail
    bm25_list = [ids.index(c) for c in page["bm25_list"]]
    fused = rag.rrf([dense_list, bm25_list])
    report.say(f'  query: "{page["fusion_query"]}"')
    report.say(f"    dense (top 3 as printed on the page): {page['dense_head']}")
    report.say(f"    BM25 (only chunks with a score above 0): {page['bm25_list']}")
    report.say(f"    fused (top 3): {[ids[j] for j in fused[:3]]}")
    report.check(
        "RRF turns the page's two lists into the page's fused top three",
        [ids[j] for j in fused[:3]] == page["fused_head"],
    )
    report.check(
        "rank 1 in a list is worth exactly 1/61 with k=60, and a chunk missing from a list adds nothing",
        abs(sum(1 / (60 + p) for p in (1,)) - 1 / 61) < 1e-15
        and rag.rrf([[7], [8]]) == [7, 8]
        and rag.rrf([[7, 9], [9]])[0] == 9,
    )
    rows = []
    for k in (1, 10, 60, 1000):
        rows.append(
            [k, f"{(1 / (k + 1)) / (1 / (k + 5)):.3f}", f"{(1 / (k + 1)) / (1 / (k + 50)):.3f}"]
        )
    report.say("\n  how much more a first place is worth than a fifth or a fiftieth place, by k:")
    report.table(["k", "rank 1 / rank 5", "rank 1 / rank 50"], rows)
    report.check(
        "a small k lets the first place dominate; k=60 keeps lower ranks in play",
        float(rows[0][1]) > 2 and float(rows[2][1]) < 1.1,
    )

    # two lexical scorers on very different scales: add their raw scores, or fuse their ranks
    gram = rag.BM25(texts, "char3")  # a second lexical signal: overlapping character trigrams
    follows_word, fused_lists, fused_scaled, ranks_two = (
        0,
        [],
        [],
        {"word": [], "char3": [], "rrf": []},
    )
    for (qt, _), g in zip(queries, gold, strict=True):
        a, b = bm25.scores(qt), gram.scores(qt)
        follows_word += int(np.argmax(a + b / 1000.0) == np.argmax(a))
        list_a, list_b = rag.top(a, positive=True), rag.top(b, positive=True)
        fused_lists.append(rag.rrf([list_a, list_b]))
        fused_scaled.append(rag.rrf([list_a, rag.top(b * 1e6, positive=True)]))
        ranks_two["word"].append(rag.rank_of(list_a, g))
        ranks_two["char3"].append(rag.rank_of(list_b, g))
        ranks_two["rrf"].append(rag.rank_of(fused_lists[-1], g))
    report.say(
        "\n  word BM25, character-trigram BM25 and their RRF on this page's 12 queries (offline):"
    )
    rows = [
        [n, *(f"{rag.retrieval_metrics(r)[m]:.2f}" for m in METRIC_NAMES)]
        for n, r in ranks_two.items()
    ]
    report.table(["system", *METRIC_NAMES], rows)
    report.say(
        "  Fusion pays when the two lists fail on different queries and the added list is good enough; two\n"
        "  lexical scorers mostly fail together, and one query is 8.3 points of Hit@1, so read this as arithmetic,\n"
        "  not as a ranking of scorers. The page's gain comes from fusing a dense retriever (section 7)."
    )
    report.check(
        "adding raw scores of different scales (the second divided by 1000, like a cosine) always keeps the first scorer's top chunk",
        follows_word == len(queries),
    )
    report.check(
        "RRF gives identical fused lists whatever the scale of a score (here: x10^6)",
        fused_scaled == fused_lists,
    )

    # ------------------------------------------------------------------------------------------------
    report.section("5. Retrieval metrics: Hit@k, Recall@k, MRR")
    bm25_ranks = [
        rag.rank_of(rag.top(bm25.scores(qt), positive=True), g)
        for (qt, _), g in zip(queries, gold, strict=True)
    ]
    report.check(
        "BM25 offline: the right chunk's rank for Q1..Q12 is the page's BM25 column",
        bm25_ranks == page["ranks"]["bm25"],
        str(bm25_ranks),
    )
    rows = []
    wrong = []
    for name, ranks in page["ranks"].items():
        met = rag.retrieval_metrics(ranks)
        rows.append([name, *(f"{met[m]:.2f}" for m in METRIC_NAMES)])
        if tuple(round(met[m], 2) for m in METRIC_NAMES) != page["metrics"][name]:
            wrong.append(name)
    report.say(
        "  the page's rank tables (dense, rrf, rerank come from the page; bm25 was just recomputed), scored:"
    )
    report.table(["system", *METRIC_NAMES], rows)
    report.check(
        "the metric code turns the page's rank tables into the page's metric tables",
        not wrong,
        str(wrong),
    )
    met = {n: rag.retrieval_metrics(r) for n, r in page["ranks"].items()}
    report.check(
        "with one right chunk per query, Hit@5 equals Recall@5, and MRR is at most Hit@1 + (1 - Hit@1) / 2",
        all(m["mrr@5"] <= m["hit@1"] + (1 - m["hit@1"]) / 2 + 1e-12 for m in met.values()),
    )
    report.say(
        f"  One query is worth {100 / len(queries):.1f} points of Hit@1 (12 queries), and there are no confidence "
        "intervals here: a mechanism demo, not a benchmark."
    )
    report.check("one query is 8.3 points", abs(100 / len(queries) - 8.33) < 0.01)

    # ------------------------------------------------------------------------------------------------
    report.section("6. Context assembly: a budget is not a relevance test")
    query = page["prompt_query"]
    s = bm25.scores(query)
    order = rag.top(s, 5, positive=True)
    prompt, picked = rag.build_prompt(query, texts, order, page["prompt_budget"], lang)
    report.say(f'  BM25 alone, budget {page["prompt_budget"]} characters, query: "{query}"')
    for j in order:
        report.say(
            f"    {'*' if j in picked else ' '} {ids[j]:14s} {s[j]:.2f}  {'(in the prompt)' if j in picked else ''}"
        )
    gold_j = ids.index("exp-advance")
    report.check(
        "the prompt holds the first chunks that fit the budget, numbered, and stops at the first one that does not",
        sum(len(texts[j]) for j in picked) <= page["prompt_budget"]
        and prompt.count("\n[") == len(picked)
        and picked == order[: len(picked)],
        f"{len(picked)} chunks, {sum(len(texts[j]) for j in picked)} characters",
    )
    gold_rank = rag.rank_of(order, gold_j)
    verdict = (
        "; a stopword-heavy question makes BM25 alone a poor ranker."
        if gold_rank > 1
        else ": BM25 alone gets this question right (the English one, full of stopwords, is the hard one)."
    )
    report.say(
        f"  The right chunk (exp-advance) is at rank {gold_rank} here{verdict}\n  A relative score cut cannot "
        "fix a wrong first place, it only keeps the neighbours of whatever came first:"
    )
    for frac in (0.5, 0.7):
        kept = rag.above_threshold(s, order, frac)
        report.say(f"    keep scores >= {frac:.1f} x best -> {[ids[j] for j in kept]}")
    report.check(
        "a higher relative threshold keeps fewer chunks, but the chunk ranked first always stays",
        len(rag.above_threshold(s, order, 0.7)) <= len(rag.above_threshold(s, order, 0.5))
        and rag.above_threshold(s, order, 0.99)[:1] == order[:1],
    )

    # ------------------------------------------------------------------------------------------------
    report.section("7. The neural parts: dense retrieval and the reranker (--bge)")
    if not args.bge:
        report.skip(
            "dense retrieval, RRF with BM25, reranking: the page's four-system comparison",
            "run with --bge (needs the 'hf' extra and the two BAAI models)",
        )
        report.finish()
        return
    dense = rag.BgeDense(lang)
    reranker = rag.BgeReranker()
    d_matrix = dense.embed(texts)
    q_vec = dense.embed([page["embed_query"]])[0]
    sims = d_matrix @ q_vec
    top3 = [(ids[j], round(float(sims[j]), 3)) for j in np.argsort(-sims)[:3]]
    report.say(
        f'  {dense.name}: {d_matrix.shape[0]} chunks -> {d_matrix.shape}; query "{page["embed_query"]}"'
    )
    for name, value in top3:
        report.say(f"    {value:.3f}  {name}")
    report.check(
        "the embedding matrix has unit rows",
        np.allclose(np.linalg.norm(d_matrix, axis=1), 1.0, atol=1e-5),
    )
    report.check(
        "the identity |q - d|^2 = 2 - 2 q.d holds for the real embeddings",
        np.allclose(((d_matrix - q_vec) ** 2).sum(1), 2 - 2 * sims, atol=1e-5),
    )
    report.check(
        "the top three by cosine are the page's", top3 == page["embed_top"], str(page["embed_top"])
    )

    systems = rag.systems(texts, lang, dense, reranker)
    ranks = {
        n: [rag.rank_of(f(qt), g) for (qt, _), g in zip(queries, gold, strict=True)]
        for n, f in systems.items()
    }
    report.say("\n  rank of the right chunk (- = not in the top 5):")
    report.table(
        ["query", *ranks],
        [[f"Q{i + 1}", *(r[i] or "-" for r in ranks.values())] for i in range(len(queries))],
    )
    rows = [
        [n, *(f"{rag.retrieval_metrics(r)[m]:.2f}" for m in METRIC_NAMES)] for n, r in ranks.items()
    ]
    report.say("")
    report.table(["system", *METRIC_NAMES], rows)
    report.check(
        "every rank of every system matches the page's table",
        ranks == page["ranks"],
        "dense, bm25, rrf and rerank, 12 queries",
    )
    report.check(
        "and so do the metrics",
        all(
            tuple(round(rag.retrieval_metrics(r)[m], 2) for m in METRIC_NAMES) == page["metrics"][n]
            for n, r in ranks.items()
        ),
    )

    pool = systems["rrf"](page["rerank_query"])
    reranked = reranker.rerank(page["rerank_query"], texts, pool)
    report.say(f'\n  query: "{page["rerank_query"]}"')
    report.say(f"    fused    {[ids[j] for j in pool[: len(page['rerank_pool'])]]}")
    report.say(f"    reranked {[ids[j] for j in reranked[: len(page['reranked'])]]}")
    report.check(
        "fusion and reranking order the page's example as printed",
        [ids[j] for j in pool[: len(page["rerank_pool"])]] == page["rerank_pool"]
        and [ids[j] for j in reranked[: len(page["reranked"])]] == page["reranked"],
    )

    prompt, picked = rag.build_prompt(
        page["prompt_query"],
        texts,
        systems["rerank"](page["prompt_query"]),
        page["prompt_budget"],
        lang,
    )
    report.say(
        "\n  the prompt of the page's context-assembly example (each line cut at 60 characters):"
    )
    for line in prompt.splitlines():
        report.say("    " + (line[:60] + ("..." if len(line) > 60 else "")))
    report.check(
        "the chunks that fit the budget are the page's: the right one first, then one that has nothing to do with it",
        [ids[j] for j in picked] == page["prompt_picked"],
        str(page["prompt_picked"]),
    )
    report.finish()


if __name__ == "__main__":
    main()
