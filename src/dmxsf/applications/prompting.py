"""Prompting, measured: how a prompt is rendered, how a reply is scored, and how far a small eval set
can be trusted.

Course page: https://dmxsf.com/wiki/applications/prompt-engineering/

The page tunes a ticket-triage prompt against a tiny eval set. This module holds what does not depend on a
particular model: the chat template (ChatML, as Qwen2.5 renders it), the strict and lenient scorers, the
word-problem answer parser, common-prefix length for prefix caching, and the exact statistics of picking the
best of several prompts on a small dev set.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from math import comb

LABELS = ("faq", "order_lookup", "escalation")
DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def render_chatml(messages: Sequence[dict], add_generation_prompt: bool = True) -> str:
    """Qwen2.5's chat template without tools: ``<|im_start|>role`` newline, content, ``<|im_end|>``
    newline for every message, a default system message when the first one is not a system message,
    and an open assistant turn at the end."""
    msgs = list(messages)
    if not msgs or msgs[0]["role"] != "system":
        msgs = [{"role": "system", "content": DEFAULT_SYSTEM}, *msgs]
    text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
    return text + ("<|im_start|>assistant\n" if add_generation_prompt else "")


def common_prefix(a: Sequence, b: Sequence) -> int:
    """How many leading items two sequences share: the part a prefix cache can reuse."""
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


# --- scoring the triage output --------------------------------------------------------------------


def parse(text: str) -> dict | None:
    """Strict: the whole output is one JSON object with the right fields, types and values."""
    try:
        obj = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    if (
        isinstance(obj, dict)
        and obj.get("task_type") in LABELS
        and isinstance(obj.get("requires_human"), bool)
    ):
        return obj
    return None


def label_of(text: str) -> str | None:
    """Lenient: the label word that appears last in the output, whatever the format around it."""
    hits = [(text.rfind(w), w) for w in LABELS if w in text]
    return max(hits)[1] if hits else None


def build(system: str, shots: Sequence[tuple[str, str]] = ()) -> Callable[[str], list[dict]]:
    """Ticket -> messages. Few-shot examples are written as user / assistant turns."""

    def make(ticket: str) -> list[dict]:
        msgs = [{"role": "system", "content": system}]
        for text, gold in shots:
            answer = {"task_type": gold, "requires_human": gold == "escalation"}
            msgs.append({"role": "user", "content": text})
            msgs.append({"role": "assistant", "content": json.dumps(answer)})
        msgs.append({"role": "user", "content": ticket})
        return msgs

    return make


def score_reply(text: str, gold: str) -> dict:
    """The three judgments of the page: ``valid`` (usable by code), ``label`` (right class, any format)
    and ``both`` (end to end)."""
    obj = parse(text)
    return {
        "valid": obj is not None,
        "label": label_of(text) == gold,
        "both": obj is not None and obj["task_type"] == gold,
    }


def evaluate(
    chat: Callable[[list[dict], int], tuple[str, int]],
    make: Callable[[str], list[dict]],
    cases: Sequence[tuple[str, str]],
    max_new_tokens: int = 60,
) -> list[dict]:
    """Run ``chat(messages, max_new_tokens) -> (text, n_tokens)`` on every ticket and score it."""
    rows = []
    for ticket, gold in cases:
        text, n = chat(make(ticket), max_new_tokens)
        rows.append(
            {"ticket": ticket, "gold": gold, "text": text, "n": n, **score_reply(text, gold)}
        )
    return rows


def summarize(rows: Sequence[dict]) -> dict:
    n = len(rows)
    out = {k: sum(r[k] for r in rows) for k in ("valid", "label", "both")}
    return {**out, "n": n, "tokens": sum(r["n"] for r in rows) / n}


def format_slips(rows: Sequence[dict]) -> tuple[int, int]:
    """(replies wrapped in a code fence, replies that write the boolean as a string)."""
    fence = sum("```" in r["text"] for r in rows)
    string_bool = sum(bool(re.search(r'"requires_human":\s*"', r["text"])) for r in rows)
    return fence, string_bool


# --- the word problems ------------------------------------------------------------------------------


def numbers(text: str) -> list[str]:
    return re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))


def answer_number(text: str, marker: str = "Answer:") -> float | None:
    """The first number after the last ``marker``; with no marker, the last number in the text."""
    if marker in text:
        nums = numbers(text.rsplit(marker, 1)[1])
        return float(nums[0]) if nums else None
    nums = numbers(text)
    return float(nums[-1]) if nums else None


# --- how far a small eval set can be trusted -----------------------------------------------------


def binomial_pmf(n: int, p: float) -> list[float]:
    return [comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(n + 1)]


def expected_best_of(m: int, n: int, p: float) -> float:
    """Expected accuracy of the best of ``m`` prompts on an ``n``-item dev set when every prompt is
    really ``p`` accurate and the items are independent. Exact: E[max] = sum over k of P(max >= k)."""
    pmf = binomial_pmf(n, p)
    cdf = [sum(pmf[: k + 1]) for k in range(n + 1)]
    expected_max = sum(1 - cdf[k - 1] ** m for k in range(1, n + 1))
    return expected_max / n


def p_gap_at_least(n: int, p: float, gap: int) -> float:
    """P(|X - Y| >= gap) for two independent counts of ``n`` items at the same accuracy ``p``: how often
    one prompt scores ``gap`` items differently on two splits by luck alone."""
    pmf = binomial_pmf(n, p)
    return sum(pmf[x] * pmf[y] for x in range(n + 1) for y in range(n + 1) if abs(x - y) >= gap)
