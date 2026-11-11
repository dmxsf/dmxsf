"""Rules, MinHash + LSH, and n-gram contamination against known ground truth."""

import numpy as np
import pytest

from dmxsf.post_training import data_quality as dq


def test_rules_flag_the_obvious_defects():
    rec = lambda i="", o="fine answer here": {"input": i, "output": o}  # noqa: E731
    assert dq.RULES["empty output"](rec(o="  "))
    assert dq.RULES["output <= 2 words"](rec(o="Yes."))
    assert not dq.RULES["output <= 2 words"](rec(o="Yes it is."))
    assert dq.RULES["placeholder text"](rec(i="<noinput>")) and dq.RULES["placeholder text"](
        rec(o="N/A")
    )
    assert dq.RULES["input is only a URL"](rec(i=" https://example.com/x "))
    assert not dq.RULES["input is only a URL"](rec(i="see https://example.com/x for context"))
    assert dq.RULES["refusal-like phrase"](rec(o="As an AI language model, I cannot do that."))
    assert not dq.RULES["refusal-like phrase"](rec(o="Paris is the capital of France."))


def test_shingles_and_jaccard():
    a, b = dq.shingles("one two three four five"), dq.shingles("one two three four six")
    assert len(a) == 3 and dq.jaccard(a, b) == pytest.approx(2 / 4)  # {123,234} shared, 4 in union
    assert dq.jaccard(a, a) == 1 and dq.jaccard(a, dq.shingles("x y z w v")) == 0
    assert len(dq.shingles("short")) == 1  # fewer than k words: one shingle


def test_minhash_agreement_estimates_jaccard():
    pool = (
        np.random.default_rng(0).integers(0, 2**32, 900).tolist()
    )  # shingle hashes look like this
    a, b = set(pool[:600]), set(pool[300:])  # 300 shared, 900 in the union -> Jaccard 1/3
    sigs = dq.signatures([a, b], 4000)
    assert (sigs[0] == sigs[1]).mean() == pytest.approx(1 / 3, abs=0.04)


def test_s_curve():
    assert dq.candidate_probability(0, 16, 4) == 0 and dq.candidate_probability(1, 16, 4) == 1
    assert dq.candidate_probability(0.8, 16, 4) > 0.999
    assert dq.candidate_probability(0.5, 16, 4) == pytest.approx(1 - (1 - 0.5**4) ** 16)
    curve = [dq.candidate_probability(j / 10, 16, 4) for j in range(11)]
    assert curve == sorted(curve)


def test_near_duplicates_finds_planted_pairs_only():
    rng = np.random.default_rng(0)
    vocab = [f"w{i}" for i in range(500)]
    texts = [" ".join(rng.choice(vocab, 40)) for _ in range(60)]
    texts.append(texts[3].replace("w", "w", 1) + " extra")  # a one-word edit of text 3
    texts.append(texts[10])  # an exact copy of text 10
    pairs = dq.near_duplicates(texts)
    assert {(i, j) for i, j, _ in pairs} == {(3, 60), (10, 61)}
    assert dq.removable(len(texts), pairs) == 2


def test_removable_counts_extras_per_cluster():
    pairs = [(0, 1, 1.0), (1, 2, 0.9), (3, 4, 1.0)]
    assert dq.removable(6, pairs) == 3  # cluster {0,1,2} keeps one, cluster {3,4} keeps one


def test_contamination_detects_verbatim_overlap_but_not_paraphrases():
    question = (
        "which gas do plants absorb from the air during photosynthesis to build sugars from water"
    )
    paraphrase = "during photosynthesis plants take in what gas from the atmosphere to make sugar out of water"
    train = ["a long essay " + question + " and more text afterwards", "unrelated text " * 20]
    tests = [
        question,
        paraphrase,
        "something else entirely about trains and timetables in europe today",
    ]
    assert dq.contaminated(train, tests) == {0}
