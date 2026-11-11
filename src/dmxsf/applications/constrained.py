"""Constrained decoding, structured output and the function-calling loop.

Course page: https://dmxsf.com/wiki/applications/structured-output-and-function-calling/

A grammar becomes a character-level automaton (DFA). From it we precompute, for every state, which
tokens can be written and where they lead (the index of Outlines, Willard and Louf, 2023). At every
step the logits of all other tokens are set to -inf. Nothing here needs a library beyond the
standard library, except ``ByteConstraint``, which applies the same mask to the byte-level GPT.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from collections.abc import Callable, Sequence
from fractions import Fraction

import torch
import torch.nn.functional as F

# --- the grammar: {"name": "<lowercase letters>+", "age": <digits>+} -------------------------------

SEGMENTS = [
    ("lit", '{"name": "'),
    ("cls", "abcdefghijklmnopqrstuvwxyz"),
    ("lit", '", "age": '),
    ("cls", "0123456789"),
    ("lit", "}"),
]

Edges = list[dict[str, int]]


def build_dfa(segments: Sequence[tuple]) -> tuple[Edges, int]:
    """Returns ``(edges, accept)``: ``edges[state][char]`` is the next state, no entry means illegal.

    A segment is ``("lit", text)`` for fixed text, ``("cls", chars)`` for a character class that
    must occur at least once and then loops on itself, or ``("rep", chars, lo, hi)`` for between
    ``lo`` and ``hi`` characters of the class (so the output has to end). The next segment may
    start from any state in which the previous one could have stopped (the "frontier")."""
    edges: Edges = [{}]
    frontier = [0]

    def new_state() -> int:
        edges.append({})
        return len(edges) - 1

    for seg in segments:
        kind, spec = seg[0], seg[1]
        if kind == "lit":
            cur = None
            for ch in spec:
                nxt = new_state()
                for s in frontier if cur is None else [cur]:
                    edges[s][ch] = nxt
                cur = nxt
            frontier = [cur]
        elif kind == "cls":
            loop = new_state()
            for s in frontier:
                for ch in spec:
                    edges[s][ch] = loop
            for ch in spec:
                edges[loop][ch] = loop
            frontier = [loop]
        else:
            lo, hi = seg[2], seg[3]
            chain = [new_state() for _ in range(hi)]  # chain[k] = k + 1 characters written
            for k, state in enumerate(chain):
                for ch in spec:
                    for s in frontier if k == 0 else [chain[k - 1]]:
                        edges[s][ch] = state
            frontier = chain[lo - 1 :] + (frontier if lo == 0 else [])
    if len(frontier) != 1:
        raise ValueError("the grammar must end in a single accepting state")
    return edges, frontier[0]


# fmt: off
VOCAB = ['{', '{"', '"', 'name', 'age', '":', '": ', '": "', '",', '", "', ' ', '}',
         'al', 'ice', 'alice', 'bob', 'Bob', 'thirty', '30', '3', '7', '42',
         'Sure', 'Here', '\n', '```', 'json', ',', ':', '<eos>']
# fmt: on


def build_index(edges: Edges, accept: int, vocab: Sequence[str]) -> list[dict[int, int]]:
    """``index[state] = {token_id: state after the whole token}``, for tokens that can be written
    in full from that state. The end token is allowed only in the accepting state."""
    index = []
    for state in range(len(edges)):
        row = {}
        for i, tok in enumerate(vocab):
            if tok == "<eos>":
                if state == accept:
                    row[i] = state
                continue
            s: int | None = state
            for ch in tok:  # a token is several characters: the automaton must eat all of them
                s = edges[s].get(ch)
                if s is None:
                    break
            if s is not None:
                row[i] = s
        index.append(row)
    return index


def sample(logits: Sequence[float], allowed, rng: random.Random) -> int:
    """Softmax over the allowed tokens only (the others count as -inf), then draw one."""
    masked = [z if i in allowed else -math.inf for i, z in enumerate(logits)]
    top = max(masked)
    weights = [math.exp(z - top) for z in masked]
    return rng.choices(range(len(logits)), weights=weights)[0]


class ToyConstraint:
    """The page's toy: a vocabulary of 30 tokens, a language model that knows nothing (random
    logits), and the grammar above. Whether the output is valid depends only on the mask."""

    def __init__(self) -> None:
        self.edges, self.accept = build_dfa(SEGMENTS)
        self.vocab = VOCAB
        self.index = build_index(self.edges, self.accept, self.vocab)

    def generate(self, rng: random.Random, constrained: bool, max_steps: int) -> str:
        state, out = 0, []
        for _ in range(max_steps):
            logits = [rng.gauss(0, 2) for _ in self.vocab]
            allowed = self.index[state].keys() if constrained else range(len(self.vocab))
            token = sample(logits, allowed, rng)
            if constrained:
                state = self.index[state][token]
            if self.vocab[token] == "<eos>":
                break
            out.append(self.vocab[token])
        return "".join(out)

    def judge(self, text: str) -> str:
        """ "complete", "prefix" (legal so far but unfinished) or "invalid"."""
        s: int | None = 0
        for ch in text:
            s = self.edges[s].get(ch)
            if s is None:
                return "invalid"
        if s == self.accept:
            json.loads(text)  # a real JSON parser reads it too
            return "complete"
        return "prefix"

    def outputs(self, constrained: bool, budget: int, runs: int = 2000, seed: int = 0) -> list[str]:
        rng = random.Random(seed)
        return [self.generate(rng, constrained, budget) for _ in range(runs)]

    def tally(self, constrained: bool, budget: int, runs: int = 2000, seed: int = 0) -> Counter:
        return Counter(self.judge(t) for t in self.outputs(constrained, budget, runs, seed))


# --- the stepwise mask changes the distribution ---------------------------------------------------


def toy_distortion() -> dict[str, dict[str, Fraction]]:
    """The page's minimal example, in exact fractions. The model writes A or B with probability
    1/2 each. After A the only legal token is x, to which it gives 1/10; after B the only legal
    token is y, to which it gives 9/10."""
    half, p_x, p_y = Fraction(1, 2), Fraction(1, 10), Fraction(9, 10)
    joint = {"Ax": half * p_x, "By": half * p_y}
    total = sum(joint.values())
    return {
        "renormalised over legal sequences": {k: v / total for k, v in joint.items()},
        "step-by-step mask": {"Ax": half, "By": half},  # each later step has one legal token: 1
    }


# --- the same mask on a real (byte-level) language model -----------------------------------------


class ByteConstraint:
    """Mask the byte-level GPT with the character DFA above: the vocabulary is the 256 bytes, so
    each byte is a token and the automaton's edges are the allowed set."""

    def __init__(self, model, segments=SEGMENTS) -> None:
        self.model = model
        self.edges, self.accept = build_dfa(segments)

    @torch.no_grad()
    def generate(
        self,
        prompt: bytes,
        max_new: int,
        constrained: bool = True,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> tuple[bytes, float]:
        """Write up to ``max_new`` bytes after ``prompt``. Constrained, it stops in the accepting
        state; unconstrained, it writes freely (greedy at temperature 0). Returns the text and
        ``sum(log Z)``: how much probability the mask took away from the model, summed over the steps
        (always 0 for the unconstrained run)."""
        self.model.eval()
        ids = list(prompt)
        out: list[int] = []
        state, pressure = 0, 0.0
        for _ in range(max_new):
            logits = self.model(torch.tensor([ids]))[0, -1].float()
            if constrained:
                allowed = [ord(ch) for ch in self.edges[state]]
                if not allowed:  # accepting state: nothing more may be written
                    break
                logp = F.log_softmax(logits, -1)
                pressure += torch.logsumexp(logp[allowed], 0).item()
                mask = torch.full_like(logits, -math.inf)
                mask[allowed] = 0.0
                logits = logits + mask
            if temperature == 0:
                token = int(logits.argmax())
            else:
                probs = F.softmax(logits / temperature, -1)
                token = int(torch.multinomial(probs, 1, generator=generator))
            if constrained:
                state = self.edges[state][chr(token)]
            ids.append(token)
            out.append(token)
        return bytes(out), pressure

    def is_valid(self, text: bytes) -> bool:
        s: int | None = 0
        for b in text:
            s = self.edges[s].get(chr(b))
            if s is None:
                return False
        return s == self.accept


@torch.no_grad()
def finite_language_distributions(
    model, prefix: bytes, words: Sequence[str], end: str = '"'
) -> tuple[dict[str, tuple[float, float]], float]:
    """For a finite set of legal outputs (each word followed by ``end``), the probability of every
    word two ways, from the same model:

    * sequence level: the model's probability of the whole word, renormalised over the words;
    * step by step: at each step keep only the bytes that continue some word and renormalise.

    Returns ``({word: (sequence_level, step_by_step)}, valid_mass)``, where ``valid_mass`` is the
    probability the model itself gives to writing any legal word. The two distributions differ
    whenever the legal branches leave different amounts of probability on the table later on.
    """
    model.eval()
    texts = [w + end for w in words]
    next_logp: dict[str, torch.Tensor] = {}  # log-probabilities of the next byte after each prefix
    for text in texts:
        ids = torch.tensor([list(prefix) + list(text.encode())])
        logp = F.log_softmax(model(ids[:, :-1])[0].float(), -1)
        for k in range(len(text)):
            next_logp.setdefault(text[:k], logp[len(prefix) - 1 + k])
    joint, stepwise = {}, {}
    for w, text in zip(words, texts, strict=True):
        joint[w] = sum(next_logp[text[:k]][ord(text[k])].item() for k in range(len(text)))
        p = 1.0
        for k in range(len(text)):
            legal = sorted({t[k] for t in texts if t.startswith(text[:k]) and len(t) > k})
            row = next_logp[text[:k]]
            z = sum(math.exp(row[ord(c)].item()) for c in legal)
            p *= math.exp(row[ord(text[k])].item()) / z
        stepwise[w] = p
    total = sum(math.exp(v) for v in joint.values())
    return {w: (math.exp(joint[w]) / total, stepwise[w]) for w in words}, total


# --- JSON Schema, strict mode, and the validate-and-retry loop ------------------------------------

UNSUPPORTED = {"allOf", "not", "dependentRequired", "dependentSchemas", "if", "then", "else"}


def strict_problems(schema: dict, path: str = "$", depth: int = 1) -> list[str]:
    """What a ``strict: true`` function schema would be refused for (OpenAI's documented subset,
    November 2026): every object needs ``additionalProperties: false`` and every property must be
    listed in ``required``; the root must be an object; some keywords are unsupported; at most 10
    levels of nesting."""
    problems = []
    if path == "$" and schema.get("type") != "object":
        problems.append("$: the root must be an object")
    if depth > 10:
        problems.append(f"{path}: nested more than 10 levels")
    for key in sorted(UNSUPPORTED & schema.keys()):
        problems.append(f"{path}: keyword '{key}' is not supported")
    if schema.get("type") == "object":
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is not False:
            problems.append(f"{path}: additionalProperties must be false")
        for name in props:
            if name not in schema.get("required", []):
                problems.append(f"{path}.{name}: every property must be listed in required")
        for name, sub in props.items():
            problems += strict_problems(sub, f"{path}.{name}", depth + 1)
    if schema.get("type") == "array" and "items" in schema:
        problems += strict_problems(schema["items"], f"{path}[]", depth + 1)
    for i, sub in enumerate(schema.get("anyOf", [])):
        problems += strict_problems(sub, f"{path}.anyOf[{i}]", depth + 1)
    return problems


_TYPES = {
    "string": str,
    "boolean": bool,
    "integer": int,
    "number": (int, float),
    "null": type(None),
    "array": list,
    "object": dict,
}


def validate(schema: dict, value, path: str = "") -> list[str]:
    """A small JSON Schema validator for the subset used here: type (or a list of types), enum,
    properties, required, additionalProperties and array items. Returns readable errors."""
    errors = []
    kinds = schema.get("type")
    if kinds is not None:
        kinds = [kinds] if isinstance(kinds, str) else kinds
        ok = any(
            isinstance(value, _TYPES[k])
            and not (k in ("integer", "number") and isinstance(value, bool))
            for k in kinds
        )
        if not ok:
            return [f"{path or '<whole>'}: expected {'/'.join(kinds)}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        options = ", ".join(repr(o) for o in schema["enum"])
        errors.append(f"{path or '<whole>'}: input should be one of {options}")
    if isinstance(value, dict):
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path + '.' if path else ''}{name}: field required")
        props = schema.get("properties", {})
        for name, sub in value.items():
            if name in props:
                errors += validate(props[name], sub, f"{path + '.' if path else ''}{name}")
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path + '.' if path else ''}{name}: extra field not permitted")
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            errors += validate(schema["items"], item, f"{path}[{i}]")
    return errors


def ask_with_retries(
    model: Callable[[list[dict]], str],
    question: str,
    schema: dict,
    max_retries: int = 3,
    log: list[str] | None = None,
):
    """Call ``model``, validate its reply against ``schema`` and send the errors back, at most
    ``max_retries`` times. Returns the parsed object; raises RuntimeError when it never validates."""
    messages = [{"role": "user", "content": question}]
    for attempt in range(max_retries + 1):
        text = model(messages)
        try:
            value = json.loads(text)
            errors = validate(schema, value)
        except json.JSONDecodeError as e:
            errors = [f"<whole>: invalid JSON: {e.msg} at column {e.colno}"]
        if not errors:
            return value
        if log is not None:
            log.append(f"attempt {attempt + 1} rejected: {errors}")
        messages += [
            {"role": "assistant", "content": text},
            {
                "role": "user",
                "content": "validation failed: " + "; ".join(errors) + ". Reply with JSON only.",
            },
        ]
    raise RuntimeError("still invalid after the retries: hand over to a human or degrade")


# --- the function-calling loop ------------------------------------------------------------------


def run_tool(registry: dict[str, Callable], call: dict) -> str:
    """Arguments are untrusted input: parse, dispatch, catch failures, and return errors to the
    model as the result of the call instead of crashing."""
    try:
        args = json.loads(call["arguments"])
        result = registry[call["name"]](**args)
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        result = {"error": f"{type(e).__name__}: {e}"}
    return json.dumps(result, ensure_ascii=False)


def run_loop(
    model: Callable[[list], list], registry: dict[str, Callable], question: str, max_turns: int = 5
):
    """The message loop of function calling (OpenAI Responses shapes): the model asks for calls,
    your code runs them, the outputs go back as new items, until it answers in text."""
    items: list[dict] = [{"role": "user", "content": question}]
    for _ in range(max_turns):
        output = model(items)
        items += output
        calls = [x for x in output if x["type"] == "function_call"]
        if not calls:
            return output[-1]["content"][0]["text"], items
        for call in calls:
            items.append(
                {"type": "function_call_output", "call_id": call["call_id"],
                 "output": run_tool(registry, call)}
            )  # fmt: skip
    raise RuntimeError("more turns than allowed")
