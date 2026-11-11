"""Benchmark contamination: n-gram detection and a miniature leak you can measure.

Course page: https://dmxsf.com/wiki/evaluation/llm-evaluation/

The detector is the GPT-3 recipe (Brown et al., 2020, appendix C): flag a benchmark question when it
shares an N-gram of words with any training document. The experiment trains a small byte-level
model on a stream that contains some benchmark items verbatim, some only as paraphrases, and some
not at all, then asks every item. Each answer is a random three-digit code, so the model can know
it only by having seen it.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from dmxsf.architecture.gpt import GPT
from dmxsf.pretraining.train import finetune_on_lines

# --- n-gram overlap -------------------------------------------------------------------------------


def words(text: str) -> list[str]:
    """Lower-case words with punctuation dropped (the detector ignores both)."""
    return re.findall(r"[a-z0-9]+", text.lower())


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    w = words(text)
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


def ngram_size(questions: list[str], low: int = 8, high: int = 13, percentile: float = 5) -> int:
    """GPT-3's N: the 5th percentile of the question lengths in words, between 8 and 13."""
    lengths = [len(words(q)) for q in questions]
    return int(min(high, max(low, np.percentile(lengths, percentile))))


def overlap(question: str, corpus_ngrams: set[tuple[str, ...]], n: int) -> int:
    """How many N-grams of the question also occur in the training text."""
    return len(ngrams(question, n) & corpus_ngrams)


# --- a benchmark whose answers can only be memorised --------------------------------------------


QUESTION = "What is the access code of vault {code} in the north wing of the old archive?"
PARAPHRASES = (
    "Vault {code} sits in the old archive's north wing; which access code opens it?",
    "Which code opens {code}, the vault in the archive's northern section?",
    "To unlock vault {code} at the archive you need which access code?",
    "Archive note: north-wing vault {code} is locked. What is its code?",
)


@dataclass(frozen=True)
class Item:
    code: str  # a unique identifier such as "K-1783"
    answer: str  # three random digits

    @property
    def question(self) -> str:
        """The wording the benchmark uses."""
        return QUESTION.format(code=self.code)

    @property
    def paraphrases(self) -> list[str]:
        """The same item worded in other ways, with the same identifier."""
        return [p.format(code=self.code) for p in PARAPHRASES]


def make_items(n: int, seed: int = 0) -> list[Item]:
    rng = random.Random(seed)
    seen: set[str] = set()
    items = []
    while len(items) < n:
        code = f"{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}-{rng.randrange(1000, 10000)}"
        if code not in seen:
            seen.add(code)
            items.append(Item(code=code, answer=f"{rng.randrange(1000):03d}"))
    return items


def lines(item: Item, form: str) -> list[str]:
    """Training lines for ``form``: "question" (the benchmark wording) or "paraphrase" (all the
    other wordings). Every line ends with the answer."""
    texts = [item.question] if form == "question" else item.paraphrases
    return [f"{t} A: {item.answer}\n" for t in texts]


def prompt(item: Item) -> str:
    return f"{item.question} A: "


# --- training with a leak ------------------------------------------------------------------------


def train_with_leak(
    model: GPT, lines: list[str], steps: int, seed: int = 0
) -> list[dict[str, float]]:
    """Continue training ``model`` on batches that mix random Shakespeare windows with windows of
    the leaked lines (8 and 24 rows per batch, see ``finetune_on_lines``)."""
    return finetune_on_lines(model, lines, steps, seed=seed, line_rows=24, clean_rows=8, lr=3e-3)


# --- asking the benchmark ------------------------------------------------------------------------


@torch.no_grad()
def answer_nats(model: GPT, items: list[Item], ask: Callable[[Item], str] = prompt) -> np.ndarray:
    """Per item: the loss (nats per digit) of its three answer digits given the prompt ``ask(item)``,
    and whether greedy decoding gets all three digits right. Returns an array of shape (n, 2)."""
    model.eval()
    out = []
    for item in items:
        ids = torch.tensor([list((ask(item) + item.answer).encode())])
        logits = model(ids[:, :-1])[0, -3:]
        target = ids[0, -3:]
        nats = F.cross_entropy(logits, target).item()
        # all three argmaxes right under teacher forcing is the same as greedy decoding being right
        out.append((nats, float(bool((logits.argmax(-1) == target).all()))))
    return np.array(out)
