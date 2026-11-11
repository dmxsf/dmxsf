import json
import math
import random
from fractions import Fraction

import pytest
import torch

from dmxsf.applications.constrained import (
    SEGMENTS,
    VOCAB,
    ByteConstraint,
    ToyConstraint,
    ask_with_retries,
    build_dfa,
    finite_language_distributions,
    run_loop,
    run_tool,
    sample,
    strict_problems,
    toy_distortion,
    validate,
)


def accepts(edges, accept, text):
    s = 0
    for ch in text:
        s = edges[s].get(ch)
        if s is None:
            return False
    return s == accept


def test_the_dfa_accepts_exactly_the_grammar():
    edges, accept = build_dfa(SEGMENTS)
    assert len(edges) == 24
    assert accepts(edges, accept, '{"name": "alice", "age": 30}')
    assert not accepts(edges, accept, '{"name": "Alice", "age": 30}')  # capital letter
    assert not accepts(edges, accept, '{"name": "", "age": 30}')  # at least one letter
    assert not accepts(edges, accept, '{"name": "bob", "age": }')  # at least one digit
    assert not accepts(edges, accept, '{"name": "bob", "age": 3')  # unfinished


def test_bounded_repeats_force_the_output_to_end():
    segments = [("lit", "<"), ("rep", "ab", 2, 3), ("lit", ">"), ("rep", "01", 0, 1), ("lit", ".")]
    edges, accept = build_dfa(segments)
    for good in ["<ab>.", "<aba>.", "<aba>1.", "<ab>0."]:
        assert accepts(edges, accept, good), good
    for bad in ["<a>.", "<abab>.", "<ab>11.", "<ab>", "ab>."]:
        assert not accepts(edges, accept, bad), bad
    # after three letters the only legal character is the closing bracket
    s = 0
    for ch in "<aba":
        s = edges[s][ch]
    assert list(edges[s]) == [">"]


def test_the_index_lists_whole_tokens_and_eos_only_when_done():
    toy = ToyConstraint()
    assert sum(len(r) for r in toy.index) == 49
    names = lambda state: [VOCAB[i] for i in toy.index[state]]  # noqa: E731
    assert names(0) == ["{", '{"']
    assert names(23) == ["<eos>"]
    assert all("<eos>" not in names(s) for s in range(23))
    # a token is several characters: from the state after {"name the token '": "' writes a quote,
    # a colon, a space and a quote in one go and lands four states further on
    after_name = {VOCAB[i]: nxt for i, nxt in toy.index[6].items()}
    assert after_name == {'"': 7, '":': 8, '": ': 9, '": "': 10}
    first = {VOCAB[i]: nxt for i, nxt in toy.index[0].items()}
    assert first['{"'] == first["{"] + 1


def test_sample_never_picks_a_masked_token():
    rng = random.Random(0)
    logits = [0.0] * 10
    picks = {sample(logits, {2, 5}, rng) for _ in range(500)}
    assert picks == {2, 5}
    logits[7] = 50.0  # huge logit, but masked
    assert 7 not in {sample(logits, {1, 2}, rng) for _ in range(200)}


def test_the_toy_model_is_always_valid_with_the_mask_and_never_without():
    toy = ToyConstraint()
    assert toy.tally(True, 80, runs=300)["complete"] == 300
    free = toy.tally(False, 40, runs=300)
    assert free["complete"] == 0 and free["invalid"] > 250
    assert toy.judge('{"name": "bob", "age": 42}') == "complete"
    assert toy.judge('{"name": "bob"') == "prefix"
    assert toy.judge("Sure") == "invalid"


def test_the_toy_distortion_is_exact():
    d = toy_distortion()
    assert d["renormalised over legal sequences"] == {"Ax": Fraction(1, 10), "By": Fraction(9, 10)}
    assert d["step-by-step mask"] == {"Ax": Fraction(1, 2), "By": Fraction(1, 2)}


class TableModel(torch.nn.Module):
    """A stand-in language model: the next-byte distribution is a lookup on the bytes so far."""

    def __init__(self, table, default_prefix_len):
        super().__init__()
        self.table, self.skip = table, default_prefix_len

    def forward(self, ids):
        out = torch.full((1, ids.shape[1], 256), -30.0)
        for t in range(ids.shape[1]):
            key = bytes(ids[0, self.skip : t + 1].tolist())
            for byte, p in self.table[key].items():
                out[0, t, byte] = math.log(p)
        return out


def test_the_stepwise_mask_changes_the_distribution_of_a_real_model_interface():
    A, B, x, y, z, sp = (ord(c) for c in "ABxyz ")
    table = {
        b"": {A: 0.5, B: 0.5},
        b"A": {x: 0.1, z: 0.9},
        b"B": {y: 0.9, z: 0.1},
        b"Ax": {sp: 1.0},
        b"By": {sp: 1.0},
    }
    model = TableModel(table, default_prefix_len=1)
    result, valid_mass = finite_language_distributions(model, b">", ["Ax", "By"], end=" ")
    assert result["Ax"][0] == pytest.approx(0.1, abs=1e-6)
    assert result["By"][0] == pytest.approx(0.9, abs=1e-6)
    assert result["Ax"][1] == pytest.approx(0.5, abs=1e-6)
    assert result["By"][1] == pytest.approx(0.5, abs=1e-6)
    assert valid_mass == pytest.approx(0.5 * 0.1 + 0.5 * 0.9, abs=1e-6)


def test_byte_constraint_always_produces_a_valid_output_with_a_random_model():
    from dmxsf.architecture.configs import small_gpt
    from dmxsf.architecture.gpt import GPT

    torch.manual_seed(0)
    model = GPT(small_gpt(n_layer=1, d_model=32, n_head=2, block_size=128))
    grammar = [("lit", "{"), ("rep", "abc", 1, 3), ("lit", ":"), ("rep", "09", 1, 2), ("lit", "}")]
    c = ByteConstraint(model, grammar)
    gen = torch.Generator().manual_seed(0)
    for _ in range(5):
        out, pressure = c.generate(b"hello\n", 20, constrained=True, generator=gen)
        assert c.is_valid(out)
        assert pressure <= 0  # log of a probability mass
    free, zero = c.generate(b"hello\n", 8, constrained=False, temperature=0)
    assert zero == 0.0 and len(free) == 8


STRICT = {
    "type": "object",
    "properties": {"a": {"type": "string"}, "b": {"type": ["integer", "null"]}},
    "required": ["a", "b"],
    "additionalProperties": False,
}


def test_strict_problems():
    assert strict_problems(STRICT) == []
    loose = {"type": "object", "properties": {"a": {"type": "string"}}, "required": []}
    assert len(strict_problems(loose)) == 2
    assert strict_problems({"type": "array"}) == ["$: the root must be an object"]
    assert any("allOf" in p for p in strict_problems({**STRICT, "allOf": []}))


def test_validate_reports_each_problem():
    assert validate(STRICT, {"a": "x", "b": None}) == []
    assert validate(STRICT, {"a": "x", "b": 3}) == []
    assert validate(STRICT, {"a": 1, "b": True, "c": 0}) == [
        "a: expected string, got int",
        "b: expected integer/null, got bool",
        "c: extra field not permitted",
    ]
    assert validate(STRICT, {"b": 1}) == ["a: field required"]
    enum = {"type": "string", "enum": ["low", "high"]}
    assert validate(enum, "mid") == ["<whole>: input should be one of 'low', 'high'"]
    assert validate({"type": "array", "items": {"type": "integer"}}, [1, "x"]) == [
        "[1]: expected integer, got str"
    ]


def test_retries_send_the_errors_back_and_stop():
    seen = []

    def model(messages):
        seen.append(messages[-1]["content"])
        return "oops" if len(seen) == 1 else json.dumps({"a": "ok", "b": None})

    log = []
    assert ask_with_retries(model, "q", STRICT, log=log) == {"a": "ok", "b": None}
    assert len(log) == 1 and "invalid JSON" in log[0]
    assert seen[1].startswith("validation failed:")
    with pytest.raises(RuntimeError):
        ask_with_retries(lambda _m: "{}", "q", STRICT, max_retries=2)


def test_tool_loop():
    def add(x: int, y: int) -> dict:
        return {"sum": x + y}

    registry = {"add": add}
    assert json.loads(run_tool(registry, {"name": "add", "arguments": '{"x": 1, "y": 2}'})) == {
        "sum": 3
    }
    for bad in ('{"x": 1}', "not json", '{"x": 1, "y": 2, "z": 3}'):
        assert "error" in json.loads(run_tool(registry, {"name": "add", "arguments": bad}))
    assert "error" in json.loads(run_tool(registry, {"name": "nope", "arguments": "{}"}))

    def model(items):
        if not any(i.get("type") == "function_call_output" for i in items):
            return [
                {
                    "type": "function_call",
                    "call_id": "1",
                    "name": "add",
                    "arguments": '{"x": 2, "y": 5}',
                }
            ]
        total = json.loads(items[-1]["output"])["sum"]
        return [{"type": "message", "content": [{"type": "output_text", "text": f"{total}"}]}]

    answer, items = run_loop(model, registry, "2+5?")
    assert answer == "7" and [i.get("type", i.get("role")) for i in items] == [
        "user", "function_call", "function_call_output", "message"
    ]  # fmt: skip
    forever = lambda _items: [  # noqa: E731
        {"type": "function_call", "call_id": "1", "name": "add", "arguments": '{"x": 1, "y": 1}'}
    ]
    with pytest.raises(RuntimeError):
        run_loop(forever, registry, "loop", max_turns=3)
