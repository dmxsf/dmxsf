import math

import numpy as np
import pytest

from dmxsf.applications import rag, rag_data


def test_the_data_has_the_pages_shape():
    for lang in ("en", "zh"):
        chunks, queries = rag_data.CHUNKS[lang], rag_data.QUERIES[lang]
        assert len(chunks) == 23 and len(queries) == 12
        assert all(gold in chunks for _, gold in queries)
    assert list(rag_data.CHUNKS["en"]) == list(rag_data.CHUNKS["zh"])  # same ids, separate texts


def test_sentences_split_on_stops_and_line_breaks():
    text = "One. Two!\nThree"
    spans = rag.sentences(text)
    assert [text[s:e] for s, e in spans] == ["One.", "Two!", "Three"]
    cjk = "第一句。第二句？“引用。”第三句"
    assert [cjk[s:e] for s, e in rag.sentences(cjk)] == [
        "第一句。",
        "第二句？",
        "“引用。”",
        "第三句",
    ]
    assert rag.sentences("") == []
    assert rag.sentences("3.14 is pi") == [(0, 10)]  # a dot not followed by a space is not a stop


def test_fixed_windows_and_their_overlap():
    text = "abcdefghij" * 3
    assert rag.fixed_chunks(text, 10, 0) == [(0, 10), (10, 20), (20, 30)]
    assert rag.fixed_chunks(text, 10, 5) == [(0, 10), (5, 15), (10, 20), (15, 25), (20, 30)]
    assert rag.fixed_chunks("", 10, 0) == []
    assert rag.fixed_chunks("abc", 10, 0) == [(0, 3)]


def test_sentence_chunks_keep_sentences_whole_and_repeat_the_overlap():
    units = [(0, 10), (11, 20), (21, 30), (31, 40)]
    assert rag.sentence_chunks(units, 25, 0) == [(0, 20), (21, 40)]
    assert rag.sentence_chunks(units, 25, 9) == [
        (0, 20),
        (11, 30),
        (21, 40),
    ]  # the last sentence is repeated
    assert rag.sentence_chunks([(0, 50)], 25, 0) == [(0, 50)]  # a long sentence stays whole


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_chunking_never_cuts_a_sentence_by_sentence_or_paragraph(lang):
    doc = rag_data.SAMPLE_DOCS[lang]
    for strategy in ("sentence", "paragraph"):
        for size in range(40, 401, 20):
            assert not rag.evaluate_chunks(doc, rag.chunk(doc, strategy, size)).cut


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_the_overlap_guarantee(lang):
    doc = rag_data.SAMPLE_DOCS[lang]
    longest = max(e - s for s, e in rag.sentences(doc))
    for size in range(longest + 1, 2 * longest, 11):
        for overlap in range(longest, size, 11):
            assert not rag.evaluate_chunks(doc, rag.chunk(doc, "fixed", size, overlap)).cut
    for overlap in range(0, longest - 5, 7):  # a window shorter than the longest sentence
        assert rag.evaluate_chunks(doc, rag.chunk(doc, "fixed", longest - 5, overlap)).cut


def test_the_widgets_numbers_on_its_sample_documents():
    zh = rag_data.SAMPLE_DOCS["zh"]
    rep = rag.evaluate_chunks(zh, rag.chunk(zh, "fixed", 120, 0))
    assert (rep.count, len(rep.cut), rep.total) == (3, 2, 15)
    rep = rag.evaluate_chunks(zh, rag.chunk(zh, "fixed", 120, 100))
    assert (rep.count, round(rep.redundancy, 2)) == (13, 4.34)
    en = rag_data.SAMPLE_DOCS["en"]
    rep = rag.evaluate_chunks(en, rag.chunk(en, "fixed", 120, 0))
    assert (rep.count, len(rep.cut), rep.total) == (10, 9, 19)
    rep = rag.evaluate_chunks(en, rag.chunk(en, "paragraph", 400))
    assert (rep.count, round(rep.average)) == (5, 225)
    with pytest.raises(ValueError):
        rag.chunk(en, "semantic", 100)


def test_tokenizers():
    assert rag.tokenize("Error E1042: pool full!") == ["error", "e1042", "pool", "full"]
    assert rag.tokenize("密码重置 E1042", "zh") == ["e1042", "密码", "码重", "重置"]
    assert rag.tokenize("password", "char3") == ["pas", "ass", "ssw", "swo", "wor", "ord"]
    assert rag.tokenize("a b", "char3") == []  # no window of three characters without a space


def test_bm25_idf_saturation_and_length_normalisation():
    tiny = rag.BM25(["a b"] * 3, "en")
    assert tiny.idf("a") == pytest.approx(math.log(1 + 0.5 / 3.5)) and tiny.idf("a") > 0
    docs = ["pool pool pool pool pool pool", "pool", "other words here", "more other words"]
    bm = rag.BM25(docs, "en", b=0.0)
    scores = bm.scores("pool")
    assert scores[0] < scores[1] * (bm.k1 + 1)  # six occurrences are worth less than 2.2 times one
    assert scores[0] > scores[1] and scores[2] == 0
    short = rag.BM25(["x y z", "x " + "filler " * 30], "en")
    assert short.scores("x")[0] > short.scores("x")[1]  # same term count, longer chunk, lower score
    assert short.matched("X and z") == ["x", "z"]


def test_bm25_matches_a_direct_computation():
    docs = ["red apple pie", "green apple", "blue sky", "red sky at night"]
    bm = rag.BM25(docs, "en", k1=1.5, b=0.6)
    n, avgdl = 4, (3 + 2 + 2 + 4) / 4
    expect = []
    for d in docs:
        words = d.split()
        total = 0.0
        for t in ("red", "sky"):
            df = sum(t in x.split() for x in docs)
            f = words.count(t)
            if f:
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                total += idf * f * 2.5 / (f + 1.5 * (1 - 0.6 + 0.6 * len(words) / avgdl))
        expect.append(total)
    assert bm.scores("red sky") == pytest.approx(expect)
    assert bm.scores("red red red") == pytest.approx(
        bm.scores("red")
    )  # a repeated query term counts once


def test_top_and_rank_of():
    s = np.array([0.1, 0.9, 0.0, 0.5])
    assert rag.top(s, 3) == [1, 3, 0]
    assert rag.top(s, 3, positive=True) == [1, 3, 0]
    assert rag.top(np.array([0.0, 0.0, 2.0]), 3, positive=True) == [2]
    assert rag.rank_of([4, 2, 9], 2) == 2
    assert rag.rank_of([4, 2, 9], 7) == 0
    assert rag.rank_of([4, 2, 9], 9, k=2) == 0  # outside the first k


def test_rrf():
    assert rag.rrf([[1, 2, 3], [3]])[0] == 3  # 1/61 + 1/63 beats 1/61 alone
    assert rag.rrf([[7], [8]]) == [7, 8]  # a tie keeps the order of first appearance
    assert rag.rrf([[1, 2], [2, 1]]) == [1, 2]
    scores = {"a": 1 / 61 + 1 / 62, "b": 1 / 62 + 1 / 61}
    assert scores["a"] == scores["b"]  # both orders add the same two terms
    fused = rag.rrf([["x", "y"], ["y"]], k=1)
    assert fused == ["y", "x"]
    assert rag.rrf([]) == []


def test_retrieval_metrics():
    m = rag.retrieval_metrics([1, 2, 3, 4, 0])
    assert m["hit@1"] == pytest.approx(0.2) and m["hit@3"] == pytest.approx(0.6)
    assert m["recall@5"] == pytest.approx(0.8)
    assert m["mrr@5"] == pytest.approx((1 + 1 / 2 + 1 / 3 + 1 / 4) / 5)
    # the page's English dense row
    dense = rag.retrieval_metrics([1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 1])
    assert [round(dense[k], 2) for k in ("hit@1", "hit@3", "recall@5", "mrr@5")] == [
        0.83,
        1.0,
        1.0,
        0.9,
    ]


def test_the_page_prompt_and_threshold():
    texts = ["a" * 40, "b" * 40, "c" * 40]
    prompt, picked = rag.build_prompt("why?", texts, [2, 0, 1], budget=90)
    assert picked == [
        2,
        0,
    ]  # the third would reach 120 > 90, and the loop stops at the first that does not fit
    assert (
        prompt.endswith("Question: why?")
        and "[1] " + "c" * 40 in prompt
        and "[2] " + "a" * 40 in prompt
    )
    zh, got = rag.build_prompt("什么？", ["甲" * 10], [0], budget=100, lang="zh")
    assert got == [0] and zh.endswith("问题：什么？") and "资料：\n[1] " in zh
    scores = np.array([4.0, 3.0, 1.0])
    assert rag.above_threshold(scores, [0, 1, 2], 0.5) == [0, 1]
    assert rag.above_threshold(scores, [], 0.5) == []


def test_index_size():
    assert round(rag.index_gb(10**6, 768, 4), 2) == 3.07
    assert round(rag.index_gb(10**7, 1024, 1), 2) == 10.24
