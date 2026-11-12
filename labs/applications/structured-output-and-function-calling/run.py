"""Lab: structured output and function calling. Constrained decoding from the automaton up.

Course page: https://dmxsf.com/wiki/applications/structured-output-and-function-calling/

The page's standard-library demo is reproduced token for token (24 states, 49 usable (state, token)
pairs, 1,928 invalid outputs without the mask and none with it). Then the same mask is put on real
models: the exact cost of stepwise masking on a finite language, and two regimes on a byte-level
GPT, one that has never seen the format and one that has, to see how hard the mask has to push.
The last section is the validate-and-retry loop and the function-calling loop, with a scripted
model so every message is reproducible.

    uv run python labs/applications/structured-output-and-function-calling/run.py [--quick] [--seed N]
"""

from __future__ import annotations

import copy
import itertools
import json
import random

import torch

from dmxsf import lab, models
from dmxsf.applications.constrained import (
    ByteConstraint,
    ToyConstraint,
    ask_with_retries,
    finite_language_distributions,
    run_loop,
    strict_problems,
    toy_distortion,
    validate,
)
from dmxsf.pretraining.train import finetune_on_lines

NAMES = (
    "romeo juliet hamlet ophelia macbeth othello iago portia lear cordelia puck oberon titania "
    "prospero ariel brutus cassius antony caesar viola orsino falstaff"
).split()
FEW_SHOT = b'{"name": "romeo", "age": 24}\n{"name": "tybalt", "age": 31}\n'
GRAMMAR = [
    ("lit", '{"name": "'),
    ("rep", "abcdefghijklmnopqrstuvwxyz", 3, 8),
    ("lit", '", "age": '),
    ("rep", "0123456789", 1, 2),
    ("lit", "}"),
]


def get_order_status(order_id: str) -> dict:
    return {"order_id": order_id, "status": "charged, awaiting confirmation"}


def fake_model(items: list[dict]) -> list[dict]:
    """A scripted model: first a call with a wrong argument name (strict mode would have prevented
    it), then the repaired call once it has seen the error, then the answer."""
    outputs = [x for x in items if x.get("type") == "function_call_output"]
    call = {"type": "function_call", "name": "get_order_status"}
    if not outputs:
        return [{**call, "call_id": "call_1", "arguments": '{"order": "A123"}'}]
    if "error" in json.loads(outputs[-1]["output"]):
        return [{**call, "call_id": "call_2", "arguments": '{"order_id": "A123"}'}]
    status = json.loads(outputs[-1]["output"])["status"]
    text = f"Order A123 is currently: {status}."
    return [
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}
    ]


def main() -> None:
    args, report = lab.start(
        "Structured output: constrained decoding, its cost, and the function-calling loop",
        wiki=["applications/structured-output-and-function-calling"],
    )
    QUICK = args.quick

    def full_check(name: str, ok: bool, detail: str = "") -> None:
        if QUICK:
            report.skip(name, "needs the full run")
        else:
            report.check(name, ok, detail)

    # -----------------------------------------------------------------------------------------
    report.section("1. The automaton, the index and the mask (the page's demo, token for token)")
    toy = ToyConstraint()
    size = sum(len(r) for r in toy.index)
    report.say(
        f"  DFA states {len(toy.edges)}, vocabulary {len(toy.vocab)}, "
        f"usable (state, token) pairs {size} of {len(toy.edges) * len(toy.vocab)}"
    )
    for state in (0, 1, 10, 21, 22, 23):
        report.say(f"  state {state:2d} allows {[toy.vocab[i] for i in toy.index[state]]}")
    report.check(
        "24 states x 30 tokens = 720 pairs, of which only 49 can be written",
        (len(toy.edges), len(toy.vocab), size) == (24, 30, 49),
    )
    rng = random.Random(7)
    samples = [toy.generate(rng, True, 40) for _ in range(3)]
    for text in samples:
        report.say(f"  constrained sample: {text!r}")
    report.check(
        "the three constrained samples of the page (seed 7)",
        samples
        == [
            '{"name": "agealage", "age": 3074273730423374233}',
            '{"name": "alicealicealicebobicename", "age": 42}',
            '{"name": "alicenameicebob", "age": 73307}',
        ],
    )
    runs = 200 if QUICK else 2000
    rows, tallies = [], {}
    for constrained, budget in [(False, 40), (True, 40), (True, 80)]:
        tallies[constrained, budget] = toy.tally(constrained, budget, runs=runs)
        t = tallies[constrained, budget]
        rows.append(
            [
                "yes" if constrained else "no",
                budget,
                t["complete"],
                t["prefix"],
                t["invalid"],
            ]
        )
    report.say(f"\n  A model that knows nothing (random logits), {runs:,} generations each:")
    report.table(["mask", "max steps", "complete", "legal prefix", "invalid"], rows)
    if QUICK:
        report.check(
            "with the mask nothing is ever invalid; without it nothing is ever complete",
            tallies[True, 40]["invalid"] == 0
            and tallies[True, 80]["invalid"] == 0
            and tallies[False, 40]["complete"] == 0,
        )
    else:
        report.check(
            "the page's tallies: no mask 1,928 invalid and 72 prefixes; mask 1,987 complete and 13 prefixes at "
            "40 steps; all 2,000 complete at 80",
            dict(tallies[False, 40]) == {"invalid": 1928, "prefix": 72}
            and dict(tallies[True, 40]) == {"complete": 1987, "prefix": 13}
            and dict(tallies[True, 80]) == {"complete": 2000},
        )
        empty = toy.outputs(False, 40).count("")
        report.say(
            f"  of the 72 unconstrained prefixes, {empty} were empty (the model stopped at once)"
        )
        report.check(
            "the page says 68 of those 72 prefixes were an immediate end-of-text",
            empty == 68,
            f"{empty}",
        )
    report.say(
        "\n  The unfinished outputs at 40 steps are the mask's guarantee ending: every step is legal, but\n"
        "  nothing forces the model to finish. Raise the budget to 80 steps and they complete."
    )

    # -----------------------------------------------------------------------------------------
    report.section("2. The stepwise mask is not the model's distribution over legal outputs")
    dist = toy_distortion()
    report.table(
        ["how legal outputs are drawn", "P(Ax)", "P(By)"],
        [[k, str(v["Ax"]), str(v["By"])] for k, v in dist.items()],
    )
    report.check(
        "the page's example: 10% / 90% over the legal sequences, 50% / 50% with the stepwise mask",
        [str(v["Ax"]) for v in dist.values()] == ["1/10", "1/2"]
        and [str(v["By"]) for v in dist.values()] == ["9/10", "1/2"],
    )

    base = models.pretrained("base", quick=QUICK, seed=args.seed)
    words = ["love", "lord", "lady", "death", "dead", "king", "kind"]
    prefix = b"ROMEO:\nO, speak again, bright angel! My "
    table, valid_mass = finite_language_distributions(base, prefix, words, end=" ")
    rows = [[w, f"{100 * a:.1f}%", f"{100 * b:.1f}%"] for w, (a, b) in table.items()]
    tv = 0.5 * sum(abs(a - b) for a, b in table.values())
    report.say(f"\n  The byte-level GPT after {prefix.decode()!r}, legal outputs = 7 words:")
    report.table(["word", "renormalised over legal words", "stepwise mask"], rows)
    report.say(
        f"  the model itself gives these 7 words {100 * valid_mass:.3g}% of its probability together;\n"
        f"  total variation distance between the two columns: {tv:.3f}"
    )
    report.check(
        "both columns are probability distributions over the 7 words",
        all(abs(sum(col) - 1) < 1e-6 for col in zip(*table.values(), strict=True)),
    )
    report.check(
        "the two ways of drawing a legal word disagree by more than 0.05 in total variation",
        tv > 0.05,
        f"{tv:.3f}",
    )

    # -----------------------------------------------------------------------------------------
    report.section("3. The mask on a real model: how much of the format does the model know?")
    stages = [0, 40] if QUICK else [0, 20, 40, 80, 300]  # steps of training on JSON lines
    lines_rng = random.Random(args.seed)
    lines = [
        f'{{"name": "{lines_rng.choice(NAMES)}", "age": {lines_rng.randrange(7, 100)}}}\n'
        for _ in range(400)
    ]
    n_samples = 12 if QUICK else 100
    rows, got = [], {}
    for steps in stages:
        model = copy.deepcopy(base)
        if steps:
            finetune_on_lines(model, lines, steps, seed=args.seed)
        c = ByteConstraint(model, GRAMMAR)
        gen = torch.Generator().manual_seed(args.seed)
        valid_free = 0
        for _ in range(n_samples):
            out, _ = c.generate(FEW_SHOT, 40, constrained=False, temperature=1.0, generator=gen)
            valid_free += c.is_valid(out.split(b"\n")[0])
        gen = torch.Generator().manual_seed(args.seed)
        valid_masked, pressure, shown = 0, [], []
        for _ in range(n_samples):
            out, p = c.generate(FEW_SHOT, 40, constrained=True, temperature=1.0, generator=gen)
            valid_masked += c.is_valid(out)
            pressure.append(-p)
            shown.append(out.decode())
        got[steps] = (
            valid_free / n_samples,
            valid_masked / n_samples,
            sum(pressure) / n_samples,
            shown,
        )
        rows.append(
            [
                steps,
                f"{100 * got[steps][0]:.0f}%",
                f"{100 * got[steps][1]:.0f}%",
                f"{got[steps][2]:.2f}",
            ]
        )
    report.say(f"  400 lines such as {lines[0].strip()} are the training text of the tuned models.")
    report.say(
        '  grammar {"name": "[a-z]{3,8}", "age": [0-9]{1,2}}, '
        f"{n_samples} samples per row at temperature 1, two-line few-shot prompt:"
    )
    report.table(["steps on JSON", "valid, no mask", "valid, mask", "mask pressure (nats)"], rows)
    for steps in (stages[0], stages[-1]):
        report.say(f"  masked samples after {steps} steps: {got[steps][3][:3]}")
    report.say(
        "\n  mask pressure = the log-probability the model itself gave to the bytes the mask allowed,\n"
        "  summed over the steps and negated: how much probability the mask had to move."
    )
    report.check(
        "with the mask every output is valid, whatever the model has seen",
        all(v[1] == 1.0 for v in got.values()),
    )
    full_check(
        "the model that never saw the format almost never writes it unaided (5% or fewer valid)",
        got[0][0] <= 0.05,
        f"{got[0][0]:.2f}",
    )
    full_check(
        "the more of the format the model knows, the less the mask has to push",
        all(got[a][2] > got[b][2] for a, b in itertools.pairwise(stages)),
        " > ".join(f"{got[s][2]:.2f}" for s in stages),
    )
    full_check(
        "unaided validity does not fall with training, and the longest-trained model is valid 95%+",
        all(got[a][0] <= got[b][0] + 0.1 for a, b in itertools.pairwise(stages))
        and got[stages[-1]][0] >= 0.95,
        ", ".join(f"{100 * got[s][0]:.0f}%" for s in stages),
    )
    report.say(
        "\n  A valid output is not a good output: the mask guarantees the shape, not that the name is a\n"
        "  name or the age is plausible. Compare the masked samples of the first and last row."
    )

    # -----------------------------------------------------------------------------------------
    report.section("4. Schemas, strict mode, validation and retries")
    triage = {
        "type": "object",
        "properties": {
            "reason": {"type": "string"},
            "priority": {"type": "string", "enum": ["low", "medium", "high"]},
            "requires_human": {"type": "boolean"},
        },
        "required": ["reason", "priority", "requires_human"],
        "additionalProperties": False,
    }
    loose = {  # what a type-to-schema tool produces without extra="forbid" and with a default
        "type": "object",
        "properties": {"reason": {"type": "string"}, "note": {"type": "string"}},
        "required": ["reason"],
    }
    fixed = {
        "type": "object",
        "properties": {"reason": {"type": "string"}, "note": {"type": ["string", "null"]}},
        "required": ["reason", "note"],
        "additionalProperties": False,
    }
    nested_bad = {
        "type": "object",
        "properties": {
            "items": {"type": "array", "items": {"type": "object", "properties": {"a": {}}}}
        },
        "required": ["items"],
        "additionalProperties": False,
        "allOf": [],
    }
    rows = []
    for name, schema in [
        ("triage", triage),
        ("loose", loose),
        ("fixed", fixed),
        ("nested_bad", nested_bad),
    ]:
        problems = strict_problems(schema)
        rows.append([name, len(problems), "; ".join(problems) or "-"])
    report.table(["schema", "strict-mode problems", "what"], rows)
    report.check(
        "triage and fixed pass; the loose schema (no additionalProperties, optional field) does not",
        strict_problems(triage) == []
        and strict_problems(fixed) == []
        and len(strict_problems(loose)) == 2,
    )
    report.check(
        "an unsupported keyword and a nested object without additionalProperties are reported",
        len(strict_problems(nested_bad)) == 3,
        f"{len(strict_problems(nested_bad))} problems",
    )

    replies = iter(
        [
            '好的，结果如下：{"priority": "high", "requires_human": true}',
            '{"priority": "urgent", "requires_human": true}',
            '{"reason": "payment taken but order unpaid: needs a ledger check", "priority": "high", '
            '"requires_human": true}',
        ]
    )
    log: list[str] = []
    result = ask_with_retries(
        lambda _messages: next(replies),
        "Order charged but shown unpaid. What now?",
        triage,
        log=log,
    )
    for entry in log:
        report.say(f"  {entry}")
    report.say(f"  accepted: {result}")
    report.check(
        "the scripted model needs three attempts; the errors name the field and the problem",
        len(log) == 2
        and "invalid JSON" in log[0]
        and "reason: field required" in log[1]
        and "priority: input should be one of" in log[1]
        and result["priority"] == "high",
    )
    report.check(
        "validate catches extra fields and wrong types",
        validate(triage, {"reason": 1, "priority": "low", "requires_human": True, "x": 0})
        == ["reason: expected string, got int", "x: extra field not permitted"],
    )
    try:
        ask_with_retries(lambda _m: "never json", "q", triage, max_retries=2)
        gave_up = False
    except RuntimeError:
        gave_up = True
    report.check("after the allowed retries it gives up instead of looping forever", gave_up)

    # The function-calling loop with a scripted model (OpenAI Responses message shapes).
    answer, items = run_loop(
        fake_model, {"get_order_status": get_order_status}, "What is the status of order A123?"
    )
    for x in items:
        kind = x.get("type", x.get("role"))
        detail = x.get("arguments") or x.get("output") or x.get("content")
        if kind == "message":
            detail = detail[0]["text"]
        report.say(f"  {kind:20s} {detail}")
    kinds = [x.get("type", x.get("role")) for x in items]
    report.check(
        "the loop: a bad call comes back as an error result, the model repairs it, then answers",
        kinds
        == [
            "user",
            "function_call",
            "function_call_output",
            "function_call",
            "function_call_output",
            "message",
        ]
        and "TypeError" in items[2]["output"]
        and answer.endswith("charged, awaiting confirmation.")
        and json.loads(items[4]["output"])["order_id"] == "A123",
    )
    try:
        run_loop(
            lambda _items: [
                {
                    "type": "function_call",
                    "call_id": "c",
                    "name": "get_order_status",
                    "arguments": '{"order_id": "A1"}',
                }
            ],
            {"get_order_status": get_order_status},
            "loop forever",
            max_turns=3,
        )
        stopped = False
    except RuntimeError:
        stopped = True
    report.check("a model that keeps calling tools is cut off at the turn limit", stopped)
    report.finish()


if __name__ == "__main__":
    main()
