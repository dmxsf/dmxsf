import numpy as np
import torch

from dmxsf.architecture.configs import small_gpt
from dmxsf.architecture.gpt import GPT
from dmxsf.evaluation import contamination as C

QUESTION = (
    "Janet has 3 apples, buys 5 more, and then gives 2 to her friend. "
    "How many apples does Janet have left?"
)
REPHRASED = (
    "After buying five more apples to add to her original three, Janet hands two over to a "
    "friend. What is the number of apples she still holds?"
)
UNRELATED = (
    "A train leaves the station at 9 am and travels at 80 km per hour. How far has it gone by noon?"
)


def test_the_pages_demo():
    corpus = (
        "Practice problems for grade school students. Problem 7: "
        + QUESTION
        + " Answer: 6. Problem 8: ..."
    )
    grams = C.ngrams(corpus, 13)
    hits = [len(C.ngrams(t, 13) & grams) for t in (QUESTION, REPHRASED, UNRELATED)]
    assert hits == [9, 0, 0]


def test_ngrams_ignore_case_and_punctuation():
    assert C.ngrams("Hello, WORLD! Again?", 2) == {("hello", "world"), ("world", "again")}
    assert C.ngrams("one two", 3) == set()
    assert C.words("K-1783 isn't it") == ["k", "1783", "isn", "t", "it"]


def test_ngram_size_follows_the_gpt3_recipe():
    long = ["w " * 40] * 10
    short = ["w " * 3] * 10
    assert C.ngram_size(long) == 13  # capped at 13
    assert C.ngram_size(short) == 8  # floored at 8
    assert C.ngram_size(["w " * 10] * 20) == 10


def test_items_are_unique_and_reproducible():
    a, b = C.make_items(60, seed=3), C.make_items(60, seed=3)
    assert a == b
    assert len({i.code for i in a}) == 60
    assert all(len(i.answer) == 3 and i.answer.isdigit() for i in a)
    assert C.make_items(5, seed=4) != C.make_items(5, seed=5)


def test_paraphrases_are_invisible_to_a_long_ngram_but_the_template_is_not():
    items = C.make_items(30, seed=0)
    paraphrases = " ".join(" ".join(i.paraphrases) for i in items)
    verbatim = " ".join(i.question for i in items[:10])
    assert C.ngram_size([i.question for i in items]) == 13
    for n in (13, 8):
        corpus = C.ngrams(paraphrases, n)
        assert all(C.overlap(i.question, corpus, n) == 0 for i in items)  # no wording in common
    # the verbatim items are caught by 13-grams, and only they
    corpus = C.ngrams(verbatim, 13)
    assert all(C.overlap(i.question, corpus, 13) > 0 for i in items[:10])
    assert all(C.overlap(i.question, corpus, 13) == 0 for i in items[10:])
    # at N = 8 the boilerplate of the template matches every item, leaked or not
    corpus = C.ngrams(verbatim, 8)
    assert all(C.overlap(i.question, corpus, 8) > 0 for i in items[10:])


def test_lines_end_with_the_answer():
    item = C.make_items(1)[0]
    assert C.lines(item, "question") == [f"{item.question} A: {item.answer}\n"]
    forms = C.lines(item, "paraphrase")
    assert len(forms) == len(C.PARAPHRASES)
    assert all(line.endswith(f" A: {item.answer}\n") and item.code in line for line in forms)
    assert C.prompt(item).endswith(" A: ")


def test_leak_training_runs_and_scoring_has_the_right_shape():
    torch.manual_seed(0)
    model = GPT(small_gpt(n_layer=1, d_model=32, n_head=2, block_size=128))
    before = [p.detach().clone() for p in model.parameters()]
    items = C.make_items(6, seed=1)
    lines = [ln for i in items for ln in C.lines(i, "question")]
    rows = C.train_with_leak(model, lines, steps=3, seed=0)
    assert [r["step"] for r in rows] == [1, 2, 3]
    assert any(not torch.equal(b, p) for b, p in zip(before, model.parameters(), strict=True))
    scored = C.answer_nats(model, items)
    assert scored.shape == (6, 2)
    assert (scored[:, 0] > 0).all() and set(np.unique(scored[:, 1])) <= {0.0, 1.0}
