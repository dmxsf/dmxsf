"""A synthetic customer-message task: the stand-in for banking77 in the post-training labs.

The post-training pages fine-tune a 0.5B model on banking77 (77 banking intents). A laptop-sized
byte-level model cannot read English that well, so the labs use a *generated* task with the same
shape: a short message in, an intent label out. Twelve intents are the product of three topics
(card, transfer, account) and four problems (missing, declined, fee, limit); every intent can be
phrased in 64 ways (8 topic words x 8 problem phrases), plus an opener and a closer. A model that
has seen one phrasing of an intent has not seen the others, which is what makes the number of
training examples matter, as it does on real data.

The data is made up. Nothing here is a statement about real customer messages.

Course pages: https://dmxsf.com/wiki/post-training/sft-data/
              https://dmxsf.com/wiki/post-training/lora-finetuning/
"""

from __future__ import annotations

import itertools

import numpy as np
import torch

from dmxsf.inference.generate import generate

TOPICS = {
    "card": (
        "card",
        "debit card",
        "credit card",
        "visa",
        "mastercard",
        "plastic",
        "new card",
        "bank card",
    ),
    "transfer": (
        "transfer",
        "wire",
        "bank transfer",
        "payment",
        "remittance",
        "money order",
        "deposit",
        "sepa payment",
    ),
    "account": (
        "account",
        "balance",
        "login",
        "profile",
        "app login",
        "password",
        "savings",
        "statement",
    ),
}
PROBLEMS = {
    "missing": (
        "has not arrived",
        "never showed up",
        "is missing",
        "did not come",
        "is lost",
        "has not turned up",
        "is nowhere",
        "went missing",
    ),
    "declined": (
        "was declined",
        "got rejected",
        "does not work",
        "keeps failing",
        "was refused",
        "got blocked",
        "is not accepted",
        "bounced",
    ),
    "fee": (
        "has a strange fee",
        "cost too much",
        "charged me extra",
        "has a hidden charge",
        "is overpriced",
        "has a bad rate",
        "billed twice",
        "has extra costs",
    ),
    "limit": (
        "hit the limit",
        "is capped",
        "has a daily limit",
        "limits me",
        "maxed out",
        "is over the cap",
        "has a low ceiling",
        "is restricted",
    ),
}
OPENERS = ("", "hi, ", "hello ", "pls ", "hey, ")
CLOSERS = ("", " today", " again", " asap", ", why?")
LABELS = tuple(f"{t}_{p}" for t in TOPICS for p in PROBLEMS)
WIDTH = 56  # messages are padded with spaces to this width so that every prompt has the same length
PROMPT = "Classify the message.\nMessage: {text:<56}\nIntent:"
STOP = ord("\n")

Example = tuple[str, str]  # (message, label)


def all_texts(label: str) -> list[str]:
    """Every message the generator can produce for ``label`` (1,600 of them)."""
    topic, problem = label.split("_")
    texts = [
        f"{o}my {t} {p}{c}"
        for t, p, o, c in itertools.product(TOPICS[topic], PROBLEMS[problem], OPENERS, CLOSERS)
    ]
    assert max(len(t) for t in texts) <= WIDTH
    return texts


def make_dataset(
    per_class: int, seed: int, exclude: set[str] | frozenset[str] = frozenset()
) -> list[Example]:
    """``per_class`` distinct messages for each intent, drawn at random, none from ``exclude``."""
    rng = np.random.default_rng(seed)
    out = []
    for label in LABELS:
        pool = [t for t in all_texts(label) if t not in exclude]
        picks = rng.choice(len(pool), size=per_class, replace=False)
        out += [(pool[i], label) for i in picks]
    order = rng.permutation(len(out))
    return [out[i] for i in order]


def render(text: str, template: str = PROMPT) -> bytes:
    """The prompt the model sees. Training and inference must use exactly this."""
    return template.format(text=text).encode()


def answer_bytes(label: str) -> bytes:
    """The supervised answer: the label and a newline, which is the 'stop' signal."""
    return f" {label}\n".encode()


@torch.no_grad()
def predict(model, texts: list[str], max_new: int = 24, template: str = PROMPT) -> list[str]:
    """Greedy answers for ``texts`` (stopping at the newline), as strings."""
    prompts = torch.tensor([list(render(t, template)) for t in texts])
    out = generate(model, prompts, max_new, stop=STOP)
    answers = []
    for row in out.tolist():
        cut = row.index(STOP) if STOP in row else len(row)
        answers.append(bytes(row[:cut]).decode("utf-8", errors="replace").strip())
    return answers


def score(model, examples: list[Example], template: str = PROMPT) -> dict[str, float]:
    """Exact-match accuracy, and the share of answers that are one of the twelve labels."""
    answers = predict(model, [t for t, _ in examples], template=template)
    correct = sum(a == y for a, (_, y) in zip(answers, examples, strict=True))
    valid = sum(a in LABELS for a in answers)
    return {"accuracy": correct / len(examples), "valid": valid / len(examples)}
