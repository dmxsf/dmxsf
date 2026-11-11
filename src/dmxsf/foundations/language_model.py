"""Language models in their simplest forms (course page: next-token prediction).

* :class:`NGramLM`: estimate ``P(next | previous n-1 tokens)`` by counting, with add-k smoothing.
* :class:`NeuralBigram`: the same bigram table, but learned by gradient descent on the
  cross-entropy loss. It converges to the count-based answer, which shows what "training a
  language model" optimises.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Hashable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class NGramLM:
    """Count-based n-gram model with add-k smoothing.

    ``P(x | c) = (count(c, x) + k) / (count(c) + k V)``, where ``c`` is the previous ``n-1``
    tokens and ``V`` the number of distinct tokens seen in training. Without smoothing
    (``k = 0``) any unseen combination gets probability 0 and the perplexity is infinite.
    """

    def __init__(self, tokens: Sequence[Hashable], n: int = 2, k: float = 0.1) -> None:
        if n < 1:
            raise ValueError("n must be at least 1")
        self.n, self.k = n, k
        self.vocab = sorted(set(tokens), key=repr)
        self.counts: dict[tuple, Counter] = defaultdict(Counter)
        for i in range(n - 1, len(tokens)):
            context = tuple(tokens[i - n + 1 : i])
            self.counts[context][tokens[i]] += 1

    def prob(self, nxt: Hashable, *context: Hashable) -> float:
        """P(nxt | context); ``context`` holds the previous ``n-1`` tokens (oldest first)."""
        if len(context) != self.n - 1:
            raise ValueError(f"need {self.n - 1} context token(s), got {len(context)}")
        row = self.counts.get(context)
        total = sum(row.values()) if row else 0
        count = row[nxt] if row else 0
        denominator = total + self.k * len(self.vocab)
        return (count + self.k) / denominator if denominator > 0 else 0.0

    def nll(self, tokens: Sequence[Hashable]) -> float:
        """Mean negative log-likelihood (nats per predicted token) of ``tokens``."""
        n = self.n
        total, steps = 0.0, 0
        for i in range(n - 1, len(tokens)):
            p = self.prob(tokens[i], *tokens[i - n + 1 : i])
            if p <= 0:
                return math.inf  # one impossible token makes the whole text impossible
            total -= math.log(p)
            steps += 1
        return total / steps

    def perplexity(self, tokens: Sequence[Hashable]) -> float:
        return math.exp(self.nll(tokens))

    def sample(self, context: Sequence[Hashable], length: int, rng: np.random.Generator) -> list:
        """Sample ``length`` tokens after ``context`` (which needs at least ``n-1`` tokens)."""
        out = list(context)
        for _ in range(length):
            key = tuple(out[len(out) - self.n + 1 :]) if self.n > 1 else ()
            row = self.counts.get(key)
            if not row:  # a context never seen in training: back off to the first token
                out.append(self.vocab[int(rng.integers(len(self.vocab)))])
                continue
            tokens = list(row)
            weights = np.array([row[t] for t in tokens], dtype=float)
            out.append(tokens[int(rng.choice(len(tokens), p=weights / weights.sum()))])
        return out[len(context) :]


def bigram_counts(ids: np.ndarray, vocab_size: int) -> np.ndarray:
    """``C[a, b]``: how often token ``b`` follows token ``a``."""
    flat = ids[:-1].astype(np.int64) * vocab_size + ids[1:].astype(np.int64)
    return np.bincount(flat, minlength=vocab_size * vocab_size).reshape(vocab_size, vocab_size)


def conditional_entropy(counts: np.ndarray) -> float:
    """H(next | prev) in nats: the lowest loss any bigram model can reach on these counts.

    It is the cross-entropy of the maximum-likelihood table ``C[a, b] / C[a, :].sum()``.
    """
    n = counts.sum()
    rows = counts.sum(axis=1, keepdims=True)
    mask = counts > 0
    ratio = np.where(mask, counts / np.maximum(rows, 1), 1.0)
    return float(-(counts[mask] / n * np.log(ratio[mask])).sum())


class NeuralBigram(nn.Module):
    """A bigram model as a neural network: row ``a`` of ``table`` holds the logits after ``a``."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)
        nn.init.zeros_(self.table.weight)  # start from the uniform distribution

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.table(idx)


def fit_neural_bigram(
    counts: np.ndarray, steps: int = 300, lr: float = 0.5
) -> tuple[NeuralBigram, list[float]]:
    """Train a :class:`NeuralBigram` by full-batch gradient descent.

    The loss over *all* pairs is the count-weighted cross-entropy, so the table of counts is
    all the data we need. Returns the model and its loss after every step (in nats).
    """
    c = torch.as_tensor(counts, dtype=torch.float32)
    weights = c / c.sum()
    model = NeuralBigram(c.shape[0])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for _ in range(steps):
        logp = F.log_softmax(model.table.weight, dim=-1)
        loss = -(weights * logp).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(loss.item())
    return model, history
