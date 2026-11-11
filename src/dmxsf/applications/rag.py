"""Retrieval-augmented generation, piece by piece: chunking, BM25, RRF, metrics, context assembly.

Course page: https://dmxsf.com/wiki/applications/rag/

Everything except the two neural models is plain Python and NumPy and runs offline: the chunkers are a
port of the page's chunking widget, BM25 and RRF are the page's code, and the retrieval metrics are the
page's. The dense retriever (``BAAI/bge-small-*``) and the cross-encoder reranker
(``BAAI/bge-reranker-base``) are wrapped in classes that import ``transformers`` only when built, so the
rest of the module needs nothing beyond NumPy (install the ``hf`` extra to use them).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass

import numpy as np

Span = tuple[int, int]  # [start, end) in characters

# --- chunking: a port of the page's chunking widget -----------------------------------------------

_CLOSERS = "。！？!?\"'”’」』)）"


def _trimmed(text: str, start: int, end: int) -> Span | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def sentences(text: str) -> list[Span]:
    """Sentence spans: a sentence ends at 。！？!? (and at ``. `` in English) or at a line break."""
    spans: list[Span] = []
    start, i, n = 0, 0, len(text)

    def push(end: int) -> None:
        span = _trimmed(text, start, end)
        if span:
            spans.append(span)

    while i < n:
        c = text[i]
        if c == "\n":
            push(i)
            start = i + 1
        elif c in "。！？!?" or (c == "." and (i + 1 == n or text[i + 1].isspace())):
            j = i + 1
            while j < n and text[j] in _CLOSERS:
                j += 1
            push(j)
            start = j
            i = j - 1
        i += 1
    push(n)
    return spans


def fixed_chunks(text: str, size: int, overlap: int) -> list[Span]:
    """Windows of ``size`` characters that advance by ``size - overlap``, whatever is in them."""
    step = max(1, size - overlap)
    chunks: list[Span] = []
    start = 0
    while start < len(text):
        span = _trimmed(text, start, min(len(text), start + size))
        if span:
            chunks.append(span)
        if start + size >= len(text):
            break
        start += step
    return chunks


def sentence_chunks(units: list[Span], size: int, overlap: int) -> list[Span]:
    """Pack whole sentences up to ``size`` characters. The next chunk repeats the last sentences that
    fit in ``overlap``. A sentence longer than ``size`` stays whole."""
    chunks: list[Span] = []
    current: list[Span] = []

    def flush() -> None:
        if current:
            chunks.append((current[0][0], current[-1][1]))

    for unit in units:
        if current and unit[1] - current[0][0] > size:
            flush()
            carry: list[Span] = []
            kept = 0
            for prev in reversed(current):
                length = prev[1] - prev[0]
                if kept + length > overlap:
                    break
                carry.insert(0, prev)
                kept += length
            while carry and unit[1] - carry[0][0] > size:
                carry.pop(0)
            current = carry
        current.append(unit)
    flush()
    return chunks


def paragraph_chunks(text: str, size: int) -> list[Span]:
    """One chunk per blank-line-separated block; blocks longer than ``size`` are packed by sentence."""
    blocks: list[Span] = []
    last = 0
    for m in re.finditer(r"\n\s*\n", text):
        block = _trimmed(text, last, m.start())
        if block:
            blocks.append(block)
        last = m.end()
    tail = _trimmed(text, last, len(text))
    if tail:
        blocks.append(tail)
    every = sentences(text)
    out: list[Span] = []
    for b in blocks:
        if b[1] - b[0] <= size:
            out.append(b)
        else:
            inside = [s for s in every if s[0] >= b[0] and s[1] <= b[1]]
            out += sentence_chunks(inside, size, 0)
    return out


def chunk(text: str, strategy: str, size: int, overlap: int = 0) -> list[Span]:
    if strategy == "fixed":
        return fixed_chunks(text, size, overlap)
    if strategy == "sentence":
        return sentence_chunks(sentences(text), size, overlap)
    if strategy == "paragraph":
        return paragraph_chunks(text, size)
    raise ValueError(f"unknown strategy {strategy!r}")


@dataclass
class ChunkReport:
    count: int
    average: float  # characters per chunk
    redundancy: float  # characters stored in all chunks / characters in the document (blanks aside)
    total: int  # sentences in the document
    cut: list[Span]  # sentences that no single chunk contains in full


def evaluate_chunks(text: str, chunks: list[Span]) -> ChunkReport:
    every = sentences(text)
    cut = [s for s in every if not any(c[0] <= s[0] and s[1] <= c[1] for c in chunks)]
    stored = sum(e - s for s, e in chunks)

    def letters(s: str) -> int:
        return len(re.sub(r"\s", "", s))

    kept = sum(letters(text[s:e]) for s, e in chunks)
    return ChunkReport(
        count=len(chunks),
        average=stored / len(chunks) if chunks else 0.0,
        redundancy=kept / max(1, letters(text)),
        total=len(every),
        cut=cut,
    )


# --- sparse retrieval: BM25 ----------------------------------------------------------------------


def tokenize(text: str, lang: str = "en") -> list[str]:
    """Lowercase words and numbers (no stemming, no stop words). ``zh`` adds the adjacent two-character
    pieces (bigrams) of every Han run, since there are no spaces to split on. ``char3`` is a different
    lexical signal: every run of three characters, so that "credential" and "credentials" overlap."""
    if lang == "char3":
        flat = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", text.lower())
        return [flat[i : i + 3] for i in range(len(flat) - 2) if " " not in flat[i : i + 3]]
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if lang == "zh":
        for run in re.findall(r"[\u4e00-\u9fff]+", text):
            tokens += [run[i : i + 2] for i in range(len(run) - 1)]
    return tokens


class BM25:
    """Okapi BM25 with Lucene's non-negative IDF, ``ln(1 + (N - n + 0.5) / (n + 0.5))``."""

    def __init__(
        self, texts: Sequence[str], lang: str = "en", k1: float = 1.2, b: float = 0.75
    ) -> None:
        self.lang, self.k1, self.b = lang, k1, b
        self.docs = [Counter(tokenize(t, lang)) for t in texts]
        self.lens = [sum(d.values()) for d in self.docs]
        self.n = len(self.docs)
        self.avgdl = sum(self.lens) / self.n
        self.df = Counter(t for d in self.docs for t in d)  # chunks containing the term

    def idf(self, term: str) -> float:
        return math.log(1 + (self.n - self.df[term] + 0.5) / (self.df[term] + 0.5))

    def term_score(self, term: str, j: int) -> float:
        """What one query term adds to chunk ``j``."""
        d = self.docs[j]
        if term not in d:
            return 0.0
        norm = d[term] + self.k1 * (1 - self.b + self.b * self.lens[j] / self.avgdl)
        return self.idf(term) * d[term] * (self.k1 + 1) / norm

    def matched(self, query: str) -> list[str]:
        return sorted(set(tokenize(query, self.lang)) & set(self.df))

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n)
        for t in self.matched(query):
            for j in range(self.n):
                out[j] += self.term_score(t, j)
        return out


# --- fusion and ranking metrics ------------------------------------------------------------------


def top(scores: np.ndarray, k: int = 5, positive: bool = False) -> list[int]:
    """Indices of the ``k`` best scores; with ``positive`` only those above zero."""
    order = [int(j) for j in np.argsort(-scores)]
    return [j for j in order if scores[j] > 0 or not positive][:k]


def rrf(rankings: Sequence[Sequence[Hashable]], k: int = 60) -> list[Hashable]:
    """Reciprocal rank fusion: each list adds ``1 / (k + rank)``; a chunk missing from a list adds
    nothing. Ties keep the order in which the chunks first appeared."""
    fused: Counter = Counter()
    for ranking in rankings:
        for pos, j in enumerate(ranking, start=1):
            fused[j] += 1 / (k + pos)
    return [j for j, _ in fused.most_common()]


def rank_of(ranking: Sequence[int], gold: int, k: int = 5) -> int:
    """Rank (from 1) of the gold chunk within the first ``k``, or 0 if it is not there."""
    ranking = list(ranking)[:k]
    return ranking.index(gold) + 1 if gold in ranking else 0


def retrieval_metrics(ranks: Sequence[int]) -> dict[str, float]:
    """Hit@1, Hit@3, Recall@5 and MRR@5 from the ranks of the single right chunk (0 = missed)."""
    n = len(ranks)
    return {
        "hit@1": sum(x == 1 for x in ranks) / n,
        "hit@3": sum(1 <= x <= 3 for x in ranks) / n,
        "recall@5": sum(x > 0 for x in ranks) / n,
        "mrr@5": sum(1 / x for x in ranks if x) / n,
    }


# --- the index and the prompt --------------------------------------------------------------------


def index_gb(n: int, dim: int, bytes_per_number: float) -> float:
    """Memory of ``n`` vectors of ``dim`` numbers, in 10^9 bytes (the vectors only)."""
    return n * dim * bytes_per_number / 1e9


INSTRUCTION = {
    "en": (
        "Answer only from the sources below. If they do not contain the answer, say so, "
        "and cite sources as [number]."
    ),
    "zh": "只依据下列资料回答，资料里没有答案就说“资料中没有”，并在句末用 [编号] 标注来源。",
}


def build_prompt(
    query: str, texts: Sequence[str], ranking: Sequence[int], budget: int, lang: str = "en"
) -> tuple[str, list[int]]:
    """Take chunks in rank order while they fit in ``budget`` characters, number them, and wrap them in
    an instruction. Returns the prompt and the indices that made it in."""
    picked: list[int] = []
    used = 0
    for j in ranking:
        if used + len(texts[j]) > budget:
            break
        picked.append(j)
        used += len(texts[j])
    context = "\n".join(f"[{n}] {texts[j]}" for n, j in enumerate(picked, start=1))
    if lang == "zh":
        return f"{INSTRUCTION[lang]}\n\n资料：\n{context}\n\n问题：{query}", picked
    return f"{INSTRUCTION[lang]}\n\nSources:\n{context}\n\nQuestion: {query}", picked


def above_threshold(scores: np.ndarray, ranking: Sequence[int], fraction: float) -> list[int]:
    """Keep the chunks of ``ranking`` whose score is at least ``fraction`` of the best one's. A relative
    cut like this has to be calibrated on your own data; the number says nothing in general."""
    best = scores[ranking[0]] if ranking else 0.0
    return [j for j in ranking if scores[j] >= fraction * best]


# --- the two neural models (optional, imported lazily) ---------------------------------------------

DENSE_MODELS = {"en": "BAAI/bge-small-en-v1.5", "zh": "BAAI/bge-small-zh-v1.5"}
RERANK_MODEL = "BAAI/bge-reranker-base"


def _need_transformers():
    try:
        import torch
        import transformers
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "this needs the 'hf' extra: uv sync --extra hf (or pip install transformers)"
        ) from exc
    return torch, transformers


class BgeDense:
    """``BAAI/bge-small-{en,zh}-v1.5``: the ``[CLS]`` hidden state of the last layer, L2-normalised,
    exactly as the model card says. No query instruction is used."""

    def __init__(self, lang: str = "en") -> None:
        self.torch, tf = _need_transformers()
        self.name = DENSE_MODELS[lang]
        self.tok = tf.AutoTokenizer.from_pretrained(self.name)
        self.model = tf.AutoModel.from_pretrained(self.name).eval()

    def embed(self, batch: Sequence[str]) -> np.ndarray:
        x = self.tok(list(batch), padding=True, truncation=True, return_tensors="pt")
        with self.torch.no_grad():
            cls = self.model(**x).last_hidden_state[:, 0]
        return self.torch.nn.functional.normalize(cls, dim=-1).numpy()


class BgeReranker:
    """``BAAI/bge-reranker-base``: a cross-encoder that reads the (query, chunk) pair jointly."""

    def __init__(self) -> None:
        self.torch, tf = _need_transformers()
        self.tok = tf.AutoTokenizer.from_pretrained(RERANK_MODEL)
        self.model = tf.AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL).eval()

    def rerank(self, query: str, texts: Sequence[str], candidates: Sequence[int]) -> list[int]:
        x = self.tok(
            [query] * len(candidates),
            [texts[j] for j in candidates],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with self.torch.no_grad():
            scores = self.model(**x).logits.view(-1)
        order = scores.argsort(descending=True)
        return [candidates[i] for i in order]


def systems(
    texts: Sequence[str], lang: str, dense: BgeDense | None, reranker: BgeReranker | None
) -> dict[str, Callable[[str], list[int]]]:
    """The four systems of the page's comparison. Dense and rerank need the models."""
    bm25 = BM25(texts, lang)
    out: dict[str, Callable[[str], list[int]]] = {}
    if dense is not None:
        d_matrix = dense.embed(texts)

        def dense_rank(q: str) -> list[int]:
            return top(d_matrix @ dense.embed([q])[0])

        out["dense"] = dense_rank
    out["bm25"] = lambda q: top(bm25.scores(q), positive=True)
    if dense is not None:
        out["rrf"] = lambda q: rrf([out["dense"](q), out["bm25"](q)])
        if reranker is not None:
            out["rerank"] = lambda q: reranker.rerank(q, texts, out["rrf"](q))
    return out
