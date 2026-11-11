"""Byte-level BPE, written from scratch (course page: tokenizer and embeddings).

Ids 0-255 are the raw bytes of UTF-8 text, so *any* string can be encoded and nothing is ever
"unknown". Id ``256 + i`` is the ``i``-th merge learned in training. The learning loop,
:func:`learn_merges`, is generic: it only sees sequences of integer symbols.
"""

from __future__ import annotations

import itertools
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

# Merges never cross a chunk boundary: text is first cut into words, numbers, punctuation and
# whitespace, like the regex of GPT-2 (which needs the third-party ``regex`` package for
# \p{L}; this version uses only the standard library). Every character falls in exactly one
# chunk, so ``"".join(chunks) == text`` for any string.
PRETOKENIZE = re.compile(r"'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+")


def pretokenize(text: str) -> list[str]:
    """Cut ``text`` into chunks inside which BPE merges may happen."""
    return PRETOKENIZE.findall(text)


def merge_pair(symbols: Sequence[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every (non-overlapping, left to right) occurrence of ``pair`` by ``new_id``."""
    out: list[int] = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


def learn_merges(
    words: Mapping[tuple[int, ...], int],
    num_merges: int,
    first_id: int,
    counts: list[int] | None = None,
) -> list[tuple[int, int]]:
    """The BPE training loop.

    ``words`` maps each distinct word (a tuple of symbol ids) to how often it occurs. Every
    round counts adjacent symbol pairs over all words, weighted by frequency, merges the most
    frequent pair into a new symbol (id ``first_id + round``) and repeats. Ties go to the pair
    seen first, so training is deterministic. If ``counts`` is a list, the frequency of each
    chosen pair is appended to it.
    """
    words = {tuple(w): n for w, n in words.items()}
    merges: list[tuple[int, int]] = []
    for step in range(num_merges):
        pairs: Counter[tuple[int, int]] = Counter()
        for symbols, n in words.items():
            for a, b in itertools.pairwise(symbols):
                pairs[(a, b)] += n
        if not pairs:
            break
        pair, count = max(pairs.items(), key=lambda item: item[1])  # max keeps the first of a tie
        merges.append(pair)
        if counts is not None:
            counts.append(count)
        merged: dict[tuple[int, ...], int] = {}
        for symbols, n in words.items():
            key = tuple(merge_pair(symbols, pair, first_id + step))
            merged[key] = merged.get(key, 0) + n
        words = merged
    return merges


class BPETokenizer:
    """Byte-level BPE tokenizer: ``train`` on text, then ``encode`` and ``decode``."""

    def __init__(self, merges: Sequence[tuple[int, int]] = ()) -> None:
        self.merges = [tuple(m) for m in merges]
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self._cache: dict[str, tuple[int, ...]] = {}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    @classmethod
    def train(cls, text: str, vocab_size: int) -> BPETokenizer:
        """Learn ``vocab_size - 256`` merges from ``text``."""
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 (the raw bytes)")
        words = Counter(tuple(chunk.encode("utf-8")) for chunk in pretokenize(text))
        return cls(learn_merges(words, vocab_size - 256, first_id=256))

    def _encode_chunk(self, chunk: str) -> tuple[int, ...]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # Of the neighbouring pairs, apply the one that was learned earliest.
            pair = min(itertools.pairwise(ids), key=lambda p: self.ranks.get(p, math.inf))
            if pair not in self.ranks:
                break
            ids = merge_pair(ids, pair, 256 + self.ranks[pair])
        self._cache[chunk] = tuple(ids)
        return self._cache[chunk]

    def encode(self, text: str) -> list[int]:
        """Text to token ids."""
        return [i for chunk in pretokenize(text) for i in self._encode_chunk(chunk)]

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        return b"".join(self.vocab[i] for i in ids)

    def decode(self, ids: Iterable[int]) -> str:
        """Token ids to text (bytes that are not valid UTF-8 become U+FFFD)."""
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    def tokens(self, text: str) -> list[bytes]:
        """The tokens of ``text`` as byte strings, handy for printing."""
        return [self.vocab[i] for i in self.encode(text)]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"merges": self.merges}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["merges"])
