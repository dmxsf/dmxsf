"""Byte-level BPE. The first test reproduces the merge table printed on the course page
https://dmxsf.com/wiki/foundations/tokenizer-and-embeddings/."""

import random

import pytest

from dmxsf.data import load_shakespeare
from dmxsf.foundations.tokenizer import (
    BPETokenizer,
    learn_merges,
    merge_pair,
    pretokenize,
)


def _chars_to_ids(corpus: dict[str, int]):
    """Give every character an id, so the generic merge loop can run on the page's example."""
    chars = sorted({c for w in corpus for c in w})
    ident = {c: i for i, c in enumerate(chars)}
    words = {tuple(ident[c] for c in w): n for w, n in corpus.items()}
    return ident, words


def test_merge_table_matches_course_page() -> None:
    corpus = {"▁low": 5, "▁lower": 2, "▁newest": 6, "▁widest": 3}
    ident, words = _chars_to_ids(corpus)
    names = {i: c for c, i in ident.items()}
    merges = learn_merges(words, 10, first_id=len(ident))
    shown = []
    for i, (a, b) in enumerate(merges):
        names[len(ident) + i] = names[a] + names[b]
        shown.append((names[a], names[b]))
    assert shown == [
        ("e", "s"),
        ("es", "t"),
        ("▁", "l"),
        ("▁l", "o"),
        ("▁lo", "w"),
        ("▁", "n"),
        ("▁n", "e"),
        ("▁ne", "w"),
        ("▁new", "est"),
        ("▁", "w"),
    ]

    def encode(word: str) -> list[str]:
        symbols = [ident[c] for c in word]
        for i, pair in enumerate(merges):
            symbols = merge_pair(symbols, pair, len(ident) + i)
        return [names[s] for s in symbols]

    assert encode("▁lowest") == ["▁low", "est"]
    assert encode("▁newer") == ["▁new", "e", "r"]
    assert encode("▁wider") == ["▁w", "i", "d", "e", "r"]


def _random_text(rng: random.Random) -> str:
    pools = [
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "0123456789",
        " \t\n\r  ",
        "!?.,;:'\"()[]{}-_/\\@#$%^&*+=~`<>|",
        "中文汉字大模型算法Ωπ😀🚀é",
    ]
    return "".join(rng.choice(rng.choice(pools)) for _ in range(rng.randint(0, 60)))


def test_pretokenize_partitions_any_string() -> None:
    rng = random.Random(0)
    for _ in range(300):
        text = _random_text(rng)
        assert "".join(pretokenize(text)) == text


def test_round_trip_is_lossless_for_any_string() -> None:
    tok = BPETokenizer.train(load_shakespeare()[:30_000].decode(), vocab_size=300)
    rng = random.Random(1)
    for _ in range(300):
        text = _random_text(rng)
        assert tok.decode(tok.encode(text)) == text


def test_untrained_tokenizer_is_just_utf8_bytes() -> None:
    tok = BPETokenizer()
    assert tok.encode("héllo") == list("héllo".encode())
    assert tok.vocab_size == 256


def test_training_compresses_text_and_is_deterministic() -> None:
    text = load_shakespeare()[:60_000].decode()
    a = BPETokenizer.train(text, vocab_size=356)
    b = BPETokenizer.train(text, vocab_size=356)
    assert a.merges == b.merges
    assert a.vocab_size == 356
    ids = a.encode(text)
    assert len(text.encode()) / len(ids) > 1.5  # bytes per token
    assert a.decode(ids) == text


def test_encode_applies_merges_in_learned_order() -> None:
    """The fast encoder (earliest-learned pair first) equals applying every merge in turn."""
    text = load_shakespeare()[:40_000].decode()
    tok = BPETokenizer.train(text, vocab_size=320)

    def sequential(chunk: str) -> list[int]:
        ids = list(chunk.encode())
        for i, pair in enumerate(tok.merges):
            ids = merge_pair(ids, pair, 256 + i)
        return ids

    for chunk in pretokenize(text[5_000:9_000]):
        assert list(tok._encode_chunk(chunk)) == sequential(chunk)


def test_save_and_load(tmp_path) -> None:
    tok = BPETokenizer.train("low lower lowest newest widest " * 20, vocab_size=270)
    tok.save(tmp_path / "bpe.json")
    again = BPETokenizer.load(tmp_path / "bpe.json")
    assert again.merges == tok.merges
    assert again.encode("lowest") == tok.encode("lowest")


def test_chinese_costs_more_tokens_than_english_in_an_english_tokenizer() -> None:
    tok = BPETokenizer.train(load_shakespeare()[:80_000].decode(), vocab_size=400)
    english, chinese = "the model reads text", "大模型读取文本"
    assert len(tok.encode(chinese)) == len(chinese.encode())  # no Chinese merges: 3 bytes each
    assert len(english.encode()) / len(tok.encode(english)) > 1.3  # English is compressed


def test_vocab_size_must_cover_the_bytes() -> None:
    with pytest.raises(ValueError):
        BPETokenizer.train("abc", vocab_size=100)
