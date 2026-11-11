"""Checking an instruction dataset: rules, near-duplicates (MinHash + LSH) and n-gram contamination.

The three tools of the sft-data page's "health check" section, written so that each can be tested
against a ground truth: a rule is a function of one record, near-duplicate detection is MinHash
signatures cut into bands (pairs that share a band are candidates, then exact Jaccard decides),
and contamination is the overlap of word n-grams between training text and test questions.

Course page: https://dmxsf.com/wiki/post-training/sft-data/
"""

from __future__ import annotations

import re
import zlib
from collections import defaultdict

import numpy as np


def norm(s: str) -> str:
    """Lower-case and collapse whitespace: the key for exact-duplicate detection."""
    return re.sub(r"\s+", " ", s.strip().lower())


REFUSAL = re.compile(
    r"as an ai( language)? model|as a language model|i'm sorry|i am sorry|"
    r"i cannot|i can't|i'm not able|i am not able|i'm unable",
    re.I,
)
URL_ONLY = re.compile(r"^\s*(https?://\S+|www\.\S+)\s*$")
PLACEHOLDER = {
    "<noinput>",
    "<no input>",
    "no input",
    "noinput",
    "n/a",
    "<no output>",
    "<picture attached>",
    "none",
}
RULES = {
    "empty output": lambda r: not r["output"].strip(),
    "output <= 2 words": lambda r: len(r["output"].split()) <= 2,
    "placeholder text": lambda r: (
        norm(r["output"]) in PLACEHOLDER or norm(r["input"]) in PLACEHOLDER
    ),
    "input is only a URL": lambda r: bool(URL_ONLY.match(r["input"])),
    "refusal-like phrase": lambda r: bool(REFUSAL.search(r["output"])),
}


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def shingles(text: str, k: int = 3) -> set[int]:
    """Hashes of the word k-grams of ``text`` (a text shorter than k words is one shingle)."""
    w = words(text)
    return {zlib.crc32(" ".join(w[i : i + k]).encode()) for i in range(max(1, len(w) - k + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


PRIME = 4294967291  # the largest prime below 2**32


def signatures(sets: list[set[int]], n_hashes: int, seed: int = 0) -> np.ndarray:
    """MinHash signatures, one row per set. P(two signature entries agree) = Jaccard similarity."""
    rng = np.random.RandomState(seed)
    a = rng.randint(1, 1 << 30, n_hashes).astype(np.uint64)
    b = rng.randint(0, 1 << 30, n_hashes).astype(np.uint64)
    out = np.empty((len(sets), n_hashes), dtype=np.uint64)
    for i, s in enumerate(sets):
        x = np.fromiter(s, dtype=np.uint64)[:, None]
        out[i] = ((x * a + b) % np.uint64(PRIME)).min(axis=0)
    return out


def candidate_pairs(sigs: np.ndarray, bands: int, rows: int, max_bucket: int = 50) -> set:
    """Pairs of rows whose signatures agree on all ``rows`` entries of at least one band."""
    buckets = defaultdict(list)
    for i, sig in enumerate(sigs):
        for k in range(bands):
            buckets[k, tuple(sig[k * rows : (k + 1) * rows])].append(i)
    cand = set()
    for ids in buckets.values():
        if 1 < len(ids) <= max_bucket:  # skip giant buckets (boilerplate)
            cand |= {(i, j) for p, i in enumerate(ids) for j in ids[p + 1 :]}
    return cand


def candidate_probability(jac: float, bands: int, rows: int) -> float:
    """The LSH S-curve: chance that two texts with Jaccard ``jac`` share at least one band."""
    return 1 - (1 - jac**rows) ** bands


def near_duplicates(
    texts: list[str], threshold: float = 0.8, bands: int = 16, rows: int = 4
) -> list[tuple[int, int, float]]:
    """MinHash + LSH banding: ``(i, j, jaccard)`` for pairs with Jaccard >= threshold."""
    sets = [shingles(t) for t in texts]
    cand = candidate_pairs(signatures(sets, bands * rows), bands, rows)
    out = []
    for i, j in cand:
        jac = jaccard(sets[i], sets[j])
        if jac >= threshold:
            out.append((i, j, jac))
    return sorted(out)


def removable(n: int, pairs: list[tuple[int, int, float]]) -> int:
    """Rows to drop if one row is kept per cluster of near-duplicates (union-find)."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, _ in pairs:
        parent[find(i)] = find(j)
    return sum(find(i) != i for i in range(n))


def ngrams(text: str, n: int = 13) -> set[str]:
    w = words(text)
    return {" ".join(w[i : i + n]) for i in range(len(w) - n + 1)}


def contaminated(train_texts: list[str], test_texts: list[str], n: int = 13) -> set[int]:
    """Indices of test texts that share at least one word n-gram with any training text."""
    index = defaultdict(set)
    for q, text in enumerate(test_texts):
        for g in ngrams(text, n):
            index[g].add(q)
    hit: set[int] = set()
    for text in train_texts:
        for g in ngrams(text, n):
            hit |= index.get(g, set())
    return hit
