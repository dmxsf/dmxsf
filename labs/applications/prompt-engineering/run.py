"""Lab: Prompting, measured. The chat template, the scorers, prefix caching and a 12-ticket eval set.

Course page: https://dmxsf.com/wiki/applications/prompt-engineering/

Everything that does not need a model runs offline: the chat template rendered by hand and compared with
what the real Qwen2.5 tokenizer printed on the page, the strict and lenient scorers on invented replies,
what a prefix cache can reuse under two prompt layouts, the twelve word problems re-solved in code, and
the exact statistics of picking the best of several prompts on a small dev set. With `--hf` the page's own
experiments are rerun on Qwen/Qwen2.5-1.5B-Instruct (greedy decoding, float32, CPU; several GB of weights
and about a quarter of an hour) and every number is compared with the page. `--lang zh` switches to the
Chinese page's tickets and prompts.

    uv run python labs/applications/prompt-engineering/run.py [--lang en|zh] [--hf] [--seed N]
"""

from __future__ import annotations

import json
import re
import time

import numpy as np

from dmxsf import lab
from dmxsf.applications import prompt_data, prompting
from dmxsf.data import encode_text
from dmxsf.evaluation import stats
from dmxsf.inference.serving import block_hashes

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# What the page prints, per language: token counts of the two renderings, the prefix experiment
# (request length, shared prefix), the schema-free probe, the V1-V3 table (valid, label, both, tokens),
# format slips (fenced, boolean-as-string), the judgments that were wrong, the word problems and the CoT
# triage table.
PAGE = {
    "en": {
        "tokens": (33, 40),
        "prefix": {"rules first": (163, 147), "ticket first": (179, 24)},
        "probe": (["status", "description"], True, False),
        "table": {
            ("V1", "dev"): (0, 11, 0, 23),
            ("V1", "test"): (0, 10, 0, 22),
            ("V2", "dev"): (7, 12, 7, 19),
            ("V2", "test"): (9, 10, 8, 19),
            ("V3", "dev"): (12, 12, 12, 15),
            ("V3", "test"): (12, 12, 12, 15),
        },
        "slips": {"V1": (24, 0), "V2": (3, 5), "V3": (0, 0)},
        "wrong": {
            "V2": [
                ("order_lookup", "faq", "Has the return request for order E5005 been approved?"),
                (
                    "escalation",
                    "faq",
                    "I never placed an order but got a payment text message. Was my account stolen?",
                ),
            ],
            "V3": [],
        },
        "math": {"direct": (0, 10), "step": (11, 164)},
        "cot": {"dev": (0, 11, 0, 51), "test": (0, 12, 0, 52)},
        "marker": "Answer:",
        "replies": {
            "invented fields": '{"status": "unpaid", "description": "card charged, order unpaid"}',
        },
        "math_replies": [
            ("12 + 30 = 42.\nAnswer: 1,280 dollars", 1280.0),
            ("The tank ends at 640 liters.", 640.0),
            ("2 * 60 = 120\nAnswer: about 3.5 km (2 + 1.5)", 3.5),
            ("Net change is -7 per minute.", -7.0),
            ("I cannot tell.", None),
        ],
    },
    "zh": {
        "tokens": (34, 41),
        "prefix": {"rules first": (162, 148), "ticket first": (178, 24)},
        "probe": (["问题", "解决方案"], False, False),
        "table": {
            ("V1", "dev"): (0, 10, 0, 23),
            ("V1", "test"): (0, 11, 0, 22),
            ("V2", "dev"): (6, 11, 6, 20),
            ("V2", "test"): (10, 12, 10, 19),
            ("V3", "dev"): (12, 11, 11, 15),
            ("V3", "test"): (12, 12, 12, 15),
        },
        "slips": {"V1": (19, 19), "V2": (6, 4), "V3": (0, 0)},
        "wrong": {
            "V2": [("escalation", "faq", "订单已扣款但还是显示未支付，怎么办？")],
            "V3": [("order_lookup", "faq", "我昨天下的单，什么时候能发货？")],
        },
        "math": {"direct": (1, 19), "step": (12, 118)},
        "cot": {"dev": (0, 10, 0, 40), "test": (0, 11, 0, 38)},
        "marker": "答案：",
        "replies": {
            "invented fields": '{"问题": "订单已扣款但显示未支付", "解决方案": "请稍后刷新"}',
        },
        "math_replies": [
            ("12 + 30 = 42。\n答案：1,280 元", 1280.0),
            ("水池最后有 640 升。", 640.0),
            ("2 * 60 = 120\n答案：约 3.5 千米（2 + 1.5）", 3.5),
            ("每分钟净变化是 -7 升。", -7.0),
            ("我算不出来。", None),
        ],
    },
}

# The twelve word problems solved again, in the order of the page (the same problems in both languages).
SOLUTIONS = [
    23 - 8 + 3 * 12,  # apples: 23, give 8 away, then 3 boxes of 12
    4 * 18 - 15 + 6,  # books: 4 shelves of 18, 15 out, 6 back
    120 - 120 * 3 / 4 + 30,  # buns: 3/4 of the 120 sold, 30 more made
    60 * 2.5 + 80 * 1.5,  # km: 60 km/h for 2.5 h, then 80 km/h for 1.5 h
    200 * 80 / 100 - 30,  # price: 20% off 200, then a coupon of 30
    (45 - 7) / 2,  # girls: boys + girls = 45 and boys = girls + 7
    14 * 6 + 59,  # pages: read 14 * 6, 59 left
    5 * 24 - 37 - 2 * 24,  # cartons: 5 cases, 37 drunk, 2 cases given away
    500 - (15 - 8) * 20,  # litres: 15 out and 8 in per minute for 20 minutes
    3 * 24 - 20 - 31,  # the third number: the sum is 3 * 24
    150 + (25 - 10) * 12,  # savings: 25 in and 10 out per week for 12 weeks
    (18 + 12) * 2 / 3,  # trees: perimeter 60 m, one every 3 m
]


def add_args(parser) -> None:
    parser.add_argument(
        "--lang", choices=["en", "zh"], default="en", help="which page's tickets (default en)"
    )
    parser.add_argument(
        "--hf",
        action="store_true",
        help=f"also rerun the page's experiments on {MODEL} (needs the hf extra; slow)",
    )


def main() -> None:
    args, report = lab.start(
        "Prompting, measured", wiki=["applications/prompt-engineering"], add_args=add_args
    )
    lang = args.lang
    page, d = PAGE[lang], prompt_data.DATA[lang]
    dev, shots = d["dev"], d["shots"]
    system = d["v1"] + d["rules"] + d["strict"]
    makers = {
        "V1": prompting.build(d["v1"]),
        "V2": prompting.build(system),
        "V3": prompting.build(system, shots),
    }
    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------------------------------------
    report.section("1. What the model reads: the chat template")
    msgs = [
        {"role": "system", "content": d["demo_system"]},
        {"role": "user", "content": d["demo_ticket"]},
    ]
    with_system = prompting.render_chatml(msgs)
    without = prompting.render_chatml(msgs[1:])
    report.say("  With no system message, the rendering is (written by prompting.render_chatml):")
    for line in without.rstrip("\n").split("\n"):
        report.say("    " + line)
    report.check(
        "the hand-written renderer gives the text the real Qwen2.5 tokenizer printed on the page, "
        "with a system message",
        with_system == d["render_with"],
    )
    report.check("and without one", without == d["render_without"])
    report.check(
        "with no system message, a default one is added; with one, it is not",
        prompting.DEFAULT_SYSTEM in without and prompting.DEFAULT_SYSTEM not in with_system,
    )
    report.check(
        "the text ends with an open assistant turn: the model continues from there",
        with_system.endswith("<|im_start|>assistant\n"),
    )

    first = dev[0][0]
    rows = []
    for name, make in makers.items():
        prompt_msgs = make(first)
        text = prompting.render_chatml(prompt_msgs)
        rows.append([name, len(prompt_msgs), text.count("<|im_start|>"), len(text.encode("utf-8"))])
    report.say(
        "\n  The same ticket under the three prompts of the page. Examples are just more turns:"
    )
    report.table(["prompt", "messages", "<|im_start|>", "bytes"], rows)
    report.check(
        "V3's three examples add six turns (user, assistant), and each assistant turn is plain JSON",
        rows[2][1] - rows[1][1] == 6
        and all(
            isinstance(json.loads(m["content"]), dict)
            for m in makers["V3"](first)
            if m["role"] == "assistant"
        ),
    )

    # ------------------------------------------------------------------------------------------------
    report.section("2. Stable content first: what a prefix cache can reuse")
    report.say(
        "  A server can reuse the KV cache of the leading part of a prompt when it is identical, token for\n"
        "  token. Here a 'token' is a UTF-8 byte and a block is 16 of them (the page counts Qwen tokens;\n"
        "  section 6 does too), so the proportions, not the absolute numbers, are what carries over."
    )
    tickets = (dev[0][0], dev[8][0])
    layouts = {
        "rules first": lambda t: [
            {"role": "system", "content": system},
            {"role": "user", "content": t},
        ],
        "ticket first": lambda t: [{"role": "user", "content": t + "\n\n" + system}],
    }
    head = f"<|im_start|>system\n{prompting.DEFAULT_SYSTEM}<|im_end|>\n<|im_start|>user\n"
    rows, shared_of = [], {}
    for name, make in layouts.items():
        ids = [encode_text(prompting.render_chatml(make(t))) for t in tickets]
        shared = prompting.common_prefix(*ids)
        blocks = prompting.common_prefix(block_hashes(ids[0]), block_hashes(ids[1]))
        shared_of[name] = shared
        rows.append([name, len(ids[0]), shared, f"{shared / len(ids[0]):.0%}", blocks])
    report.table(["layout", "request", "shared", "of request", "cacheable blocks"], rows)
    report.check(
        "rules first: the two requests share most of their text",
        shared_of["rules first"] > 0.8 * rows[0][1],
        f"{shared_of['rules first']} of {rows[0][1]} bytes",
    )
    report.check(
        "ticket first: they share only the template's own opening (default system message and the "
        "user marker), none of the rules",
        shared_of["ticket first"] == len(head.encode("utf-8")),
        f"{shared_of['ticket first']} bytes = the {len(head.encode('utf-8'))}-byte head",
    )

    stamps = ("09:41:07", "09:41:08")
    placed = {
        "clock at the top": lambda s: f"Current time: {s}\n" + system,
        "clock at the end": lambda s: system + f"\nCurrent time: {s}",
    }
    rows = []
    for name, make in placed.items():
        ids = [
            encode_text(
                prompting.render_chatml([{"role": "system", "content": make(s)}, *msgs[1:]])
            )
            for s in stamps
        ]
        shared = prompting.common_prefix(*ids)
        blocks = prompting.common_prefix(block_hashes(ids[0]), block_hashes(ids[1]))
        rows.append([name, len(ids[0]), shared, f"{shared / len(ids[0]):.0%}", blocks])
    report.say(
        "\n  One changing line in the system prompt (the clock), same ticket, one second apart:"
    )
    report.table(["placement", "request", "shared", "of request", "cacheable blocks"], rows)
    report.check(
        "a changing line near the top spoils the cache for everything behind it; at the end it costs little",
        rows[0][4] < rows[1][4] / 4,
        f"{rows[0][4]} vs {rows[1][4]} cacheable blocks",
    )

    fixed, varying, requests_per_day, hit = 1200, 80, 100_000, 0.9
    full = requests_per_day * (fixed + varying)
    cached = requests_per_day * (fixed * (1 - hit) + varying)
    report.say(
        f"\n  The page's exercise: a {fixed}-token fixed prefix, {varying} varying tokens, "
        f"{requests_per_day:,} requests a day."
    )
    report.table(
        ["hit rate", "tokens to compute", "saved"],
        [
            [
                f"{h:.0%}",
                f"{requests_per_day * (fixed * (1 - h) + varying):,.0f}",
                f"{1 - (fixed * (1 - h) + varying) / (fixed + varying):.1%}",
            ]
            for h in (0.0, 0.5, 0.9, 0.99)
        ],
    )
    report.check(
        "at a 90% hit rate: 128,000,000 tokens without the cache, 20,000,000 with it, about 84% fewer",
        (round(full), round(cached), round(1 - cached / full, 2))
        == (128_000_000, 20_000_000, 0.84),
    )

    # ------------------------------------------------------------------------------------------------
    report.section("3. Scoring a reply: format and judgment are separate failures")
    replies = [
        ("bare JSON", '{"task_type": "faq", "requires_human": false}', "faq"),
        ("fenced JSON", '```json\n{"task_type": "faq", "requires_human": false}\n```', "faq"),
        ("boolean as string", '{"task_type": "faq", "requires_human": "false"}', "faq"),
        (
            "analysis, then JSON",
            'Analysis: a general policy question.\n{"task_type": "faq", "requires_human": false}',
            "faq",
        ),
        ("wrong label", '{"task_type": "faq", "requires_human": false}', "escalation"),
        (
            "right, then a remark",
            '{"task_type": "faq", "requires_human": false}\n(this is not an escalation)',
            "faq",
        ),
        ("extra field", '{"task_type": "faq", "requires_human": false, "confidence": 0.9}', "faq"),
        ("invented fields", page["replies"]["invented fields"], "faq"),
    ]
    rows, got = [], {}
    for name, text, gold in replies:
        s = prompting.score_reply(text, gold)
        got[name] = (s["valid"], s["label"], s["both"])
        rows.append([name, gold, *("yes" if v else "no" for v in got[name])])
    report.say(
        "  Invented replies, one for each way the page's outputs went wrong (hand-made, not model output):"
    )
    report.table(["reply", "gold", "valid", "label", "both"], rows)
    want = {
        "bare JSON": (True, True, True),
        "fenced JSON": (False, True, False),
        "boolean as string": (False, True, False),
        "analysis, then JSON": (False, True, False),
        "wrong label": (True, False, False),
        "right, then a remark": (False, False, False),
        "extra field": (True, True, True),
        "invented fields": (False, False, False),
    }
    report.check(
        "a fence, a string for the boolean and text around the JSON each fail 'valid' but keep 'label': "
        "a format failure, not a wrong judgment",
        all(got[k] == want[k] for k in ("fenced JSON", "boolean as string", "analysis, then JSON")),
    )
    report.check(
        "a clean JSON with the wrong class is the opposite: valid, but not label",
        got["wrong label"] == want["wrong label"],
    )
    report.check(
        "the scorers are code, and err too: the lenient reader takes the last class word, so a trailing "
        "'this is not an escalation' turns a right answer into a wrong one",
        got["right, then a remark"] == want["right, then a remark"]
        and prompting.label_of(replies[5][1]) == "escalation",
    )
    report.check(
        "parse checks only the two fields, so an extra field passes (a choice to make on purpose)",
        got["extra field"] == want["extra field"],
    )
    report.check(
        "invented field names pass nothing", got["invented fields"] == want["invented fields"]
    )

    # ------------------------------------------------------------------------------------------------
    report.section("4. The twelve word problems, solved again in code")
    math = d["math"]
    cut = 22 if lang == "zh" else 48  # characters; Chinese ones are twice as wide
    rows = [
        [i, question[:cut] + ("..." if len(question) > cut else ""), f"{gold:g}", f"{mine:g}"]
        for i, ((question, gold), mine) in enumerate(zip(math, SOLUTIONS, strict=True))
    ]
    report.table(["#", "problem", "page", "recomputed"], rows)
    report.check(
        "all twelve answers of the page are right",
        [g for _, g in math] == SOLUTIONS,
        f"{sum(g == s for (_, g), s in zip(math, SOLUTIONS, strict=True))}/12",
    )
    report.check(
        "the tank problem turns on the sign of the net change: -7 per minute gives 360, +7 gives 640",
        500 - 7 * 20 == 360 and 500 + 7 * 20 == 640 and SOLUTIONS[8] == 360,
    )
    marker = page["marker"]
    rows, ok = [], True
    for text, want_value in page["math_replies"]:
        value = prompting.answer_number(text, marker)
        ok &= value == want_value
        rows.append([text.replace("\n", " | "), "none" if value is None else f"{value:g}"])
    report.say(
        f"\n  How `answer_number` reads a reply (the number after the last '{marker}', else the last number):"
    )
    report.table(["reply", "read as"], rows)
    report.check("it reads those replies as intended", ok)
    pitfall = prompting.answer_number(f"{marker} 2,5", marker)
    report.check(
        "but it strips commas as thousands separators, so a decimal comma ('Answer: 2,5') is read as 25: "
        "a scorer's blind spot",
        pitfall == 25.0,
        f"read as {pitfall:g}",
    )

    # ------------------------------------------------------------------------------------------------
    report.section("5. How far a 12-ticket eval set can be trusted")
    n_dev = len(dev)
    rows = [
        [n, f"{100 / n:.1f}", f"{100 * stats.half_width(0.8, n):.1f}"]
        for n in (12, 24, 100, 500, 2000)
    ]
    report.say(
        "  One more right answer, and the 95% interval of an accuracy of 80% (normal approximation):"
    )
    report.table(["tickets", "one item (points)", "interval +- (points)"], rows)
    report.check(
        "on 12 tickets one ticket is 8.3 points and the interval is about +-23 points",
        f"{100 / 12:.1f}" == "8.3" and round(100 * stats.half_width(0.8, 12)) == 23,
    )

    dev_v2, test_v2 = page["table"][("V2", "dev")][2], page["table"][("V2", "test")][2]
    gap = abs(dev_v2 - test_v2)
    pooled = (dev_v2 + test_v2) / (2 * n_dev)
    report.say(
        f"\n  The page's V2 prompt got {dev_v2}/{n_dev} right end to end on dev and {test_v2}/{n_dev} on test: "
        f"{gap} apart. If its true accuracy\n  were {pooled:.0%} on both splits, the chance that two splits of "
        f"{n_dev} differ by at least g items by luck alone is:"
    )
    rows = [[g, f"{prompting.p_gap_at_least(n_dev, pooled, g):.0%}"] for g in range(1, 6)]
    report.table(["g", "chance"], rows)
    chance = prompting.p_gap_at_least(n_dev, pooled, max(gap, 1))
    report.check(
        f"the page's own dev/test gap ({gap}) is within what luck produces (more than 5% likely)",
        chance > 0.05,
        f"{chance:.0%}",
    )

    trials = 4_000 if args.quick else 40_000
    rows, exact_values = [], {}
    for p in (0.6, 0.75, 0.9):
        for m in (1, 3, 10, 30):
            best = prompting.expected_best_of(m, n_dev, p)
            exact_values[p, m] = best
            rows.append([f"{p:.0%}", m, f"{best:.1%}", f"{best - p:+.1%}"])
    report.say(
        f"\n  Pick the best of m prompts on a {n_dev}-ticket dev set when every prompt is really p accurate "
        "(exact, independent items):"
    )
    report.table(["true accuracy", "m prompts", "best prompt's dev score", "overstated by"], rows)
    report.check(
        "with one prompt the dev score is unbiased; with more, the best one looks better than it is",
        all(abs(exact_values[p, 1] - p) < 1e-12 for p in (0.6, 0.75, 0.9))
        and all(
            exact_values[p, a] < exact_values[p, b]
            for p in (0.6, 0.75, 0.9)
            for a, b in ((1, 3), (3, 10), (10, 30))
        ),
    )
    m, p = 10, 0.75
    best_dev = rng.binomial(n_dev, p, size=(trials, m)).max(axis=1) / n_dev
    report.check(
        f"a simulation agrees with the exact figure ({trials:,} rounds, m={m}, p={p:.0%})",
        abs(best_dev.mean() - exact_values[p, m]) < 0.01,
        f"{best_dev.mean():.3f} vs {exact_values[p, m]:.3f}",
    )

    true_p = np.linspace(0.60, 0.85, m)  # ten prompts that really do differ
    dev_scores = rng.binomial(n_dev, true_p, size=(trials, m)) + rng.random((trials, m)) * 1e-3
    winner = dev_scores.argmax(axis=1)  # the tiny noise breaks ties at random
    fresh = rng.binomial(n_dev, true_p[winner]) / n_dev  # the winner on a new 12-ticket test set
    winner_dev = np.floor(dev_scores.max(axis=1)) / n_dev
    report.say(
        f"\n  Ten prompts whose true accuracies run from {true_p[0]:.0%} to {true_p[-1]:.0%}; "
        f"choose by the {n_dev}-ticket dev score ({trials:,} rounds):"
    )
    report.table(
        ["quantity", "value"],
        [
            ["winner's dev score", f"{winner_dev.mean():.1%}"],
            ["winner's true accuracy", f"{true_p[winner].mean():.1%}"],
            ["winner's score on a fresh test set", f"{fresh.mean():.1%}"],
            ["the best prompt's true accuracy", f"{true_p[-1]:.1%}"],
            ["the truly best prompt wins", f"{(winner == m - 1).mean():.0%}"],
        ],
    )
    report.check(
        "the winner's dev score overstates it; a fresh test set sees its true accuracy; and the truly "
        "best prompt wins well under half the time",
        winner_dev.mean() - fresh.mean() > 0.05
        and abs(fresh.mean() - true_p[winner].mean()) < 0.01
        and (winner == m - 1).mean() < 0.5,
    )

    p_a, p_b = 11 / 12, 10 / 12
    rows = [[f"{rho:.1f}", stats.questions_needed(p_a, p_b, rho)] for rho in (0.0, 0.5, 0.7)]
    report.say(
        "\n  Tickets needed to tell 11/12 from 10/12 (the page's exercise), 80% power, 5% level:"
    )
    report.table(["correlation of the two prompts", "tickets"], rows)
    report.check(
        "a one-ticket improvement on 12 is far below what 12 tickets can resolve",
        rows[0][1] > 100 and rows[2][1] < rows[0][1],
        f"{rows[2][1]} to {rows[0][1]} tickets",
    )

    # ------------------------------------------------------------------------------------------------
    if args.hf:
        run_model(args, report, lang, page, d, makers, system)
    else:
        report.say(
            "\n(6. Rerun the page's experiments on a real model: add --hf. It downloads "
            f"{MODEL} and takes a while.)"
        )
    report.finish()


def run_model(args, report, lang, page, d, makers, system) -> None:
    from dmxsf.applications.hf import HFChat

    report.section(f"6. The page's experiments, rerun on {MODEL} (float32, CPU, greedy)")
    t0 = time.perf_counter()
    llm = HFChat(MODEL)
    report.say(f"  loaded in {time.perf_counter() - t0:.0f}s")
    dev, test, math = d["dev"], d["test"], d["math"]
    if args.quick:
        dev, test = dev[:3], []
    marker = page["marker"]

    # the chat template and its token counts
    msgs = [
        {"role": "system", "content": d["demo_system"]},
        {"role": "user", "content": d["demo_ticket"]},
    ]
    texts = [llm.render(msgs), llm.render(msgs[1:])]
    counts = tuple(llm.count(t) for t in texts)
    report.check(
        "the tokenizer's template gives the text of section 1",
        texts == [d["render_with"], d["render_without"]],
    )
    report.check(
        "token counts with and without a system message",
        counts == page["tokens"],
        f"{counts}, page {page['tokens']}",
    )

    # prefix sharing in tokens
    tickets = (d["dev"][0][0], d["dev"][8][0])
    layouts = {
        "rules first": lambda t: [
            {"role": "system", "content": system},
            {"role": "user", "content": t},
        ],
        "ticket first": lambda t: [{"role": "user", "content": t + "\n\n" + system}],
    }
    got = {}
    for name, make in layouts.items():
        ids = [llm.tok(llm.render(make(t))).input_ids for t in tickets]
        got[name] = (len(ids[0]), prompting.common_prefix(*ids))
    report.say(
        "  prefix, in Qwen tokens: "
        + "; ".join(f"{k}: request {a}, shared {b}" for k, (a, b) in got.items())
    )
    report.check(
        "the prefix experiment of the page", got == page["prefix"], f"page {page['prefix']}"
    )

    # no schema
    reply, _ = llm.generate(
        [
            {"role": "system", "content": d["demo_system"]},
            {"role": "user", "content": d["dev"][8][0]},
        ]
    )
    probe = (
        re.findall(r'"([^"]+)"\s*:', reply),
        "```" in reply,
        prompting.parse(reply) is not None,
    )
    report.say(
        f"  asked for 'JSON only' with no schema: fields {probe[0]}, fence {probe[1]}, parse accepts {probe[2]}"
    )
    report.check(
        "the schema-free probe of the page", probe == page["probe"], f"page {page['probe']}"
    )

    # V1-V3
    def chat(messages, max_new_tokens):
        return llm.generate(messages, max_new_tokens)

    runs, rows = {}, []
    for name, make in makers.items():
        for split, cases in (("dev", dev), ("test", test)):
            if not cases:
                continue
            runs[name, split] = prompting.evaluate(chat, make, cases)
            s = prompting.summarize(runs[name, split])
            rows.append(
                [
                    name,
                    split,
                    *(f"{s[k]}/{s['n']}" for k in ("valid", "label", "both")),
                    f"{s['tokens']:.0f}",
                ]
            )
    report.table(["name", "split", "valid", "label", "both", "tokens"], rows)
    if args.quick:
        report.skip("the page's V1-V3 table", "--quick runs three dev tickets only")
        return
    mismatch = []
    for key, want in page["table"].items():
        s = prompting.summarize(runs[key])
        mine = (s["valid"], s["label"], s["both"], int(f"{s['tokens']:.0f}"))
        if mine != want:
            mismatch.append((key, mine, want))
    report.check("V1-V3 reproduce the page's table", not mismatch, str(mismatch))
    slips = {
        name: prompting.format_slips(runs[name, "dev"] + runs[name, "test"]) for name in makers
    }
    report.say("  (fenced, boolean as string): " + str(slips))
    report.check("format slips per prompt", slips == page["slips"], f"page {page['slips']}")
    wrong = {
        name: [
            (r["gold"], prompting.label_of(r["text"]), r["ticket"])
            for r in runs[name, "dev"] + runs[name, "test"]
            if not r["label"]
        ]
        for name in ("V2", "V3")
    }
    report.check("the judgments that stay wrong in V2 and V3", wrong == page["wrong"], str(wrong))
    bad = [r for r in runs["V2", "dev"] if not r["valid"]]
    report.check(
        "V2's first format error is a fenced JSON",
        bool(bad)
        and bad[0]["text"]
        == '```json\n{\n  "task_type": "faq",\n  "requires_human": false\n}\n```',
    )

    # word problems
    answers = {}
    for mode, sys_key, budget in (("direct", "sys_direct", 40), ("step", "sys_cot", 300)):
        results = []
        for question, gold in math:
            text, n = llm.generate(
                [{"role": "system", "content": d[sys_key]}, {"role": "user", "content": question}],
                budget,
            )
            results.append((prompting.answer_number(text, marker) == gold, n, text))
        answers[mode] = results
        right = sum(r[0] for r in results)
        tokens = sum(r[1] for r in results) / len(results)
        report.say(f"  {mode}: {right}/{len(math)} right, {tokens:.0f} tokens on average")
        report.check(
            f"word problems, {mode}",
            (right, int(f"{tokens:.0f}")) == page["math"][mode],
            f"page {page['math'][mode]}",
        )
    missed = [
        (i, prompting.answer_number(r[2], marker))
        for i, r in enumerate(answers["step"])
        if not r[0]
    ]
    report.say(f"  step by step, missed: {missed}")

    # chain of thought on the triage task
    make_cot = prompting.build(d["v1"] + d["rules"] + d["cot"])
    for split, cases in (("dev", dev), ("test", test)):
        rows = prompting.evaluate(chat, make_cot, cases, max_new_tokens=160)
        s = prompting.summarize(rows)
        mine = (s["valid"], s["label"], s["both"], int(f"{s['tokens']:.0f}"))
        report.say(
            f"  CoT {split}: valid {s['valid']}/{s['n']}, label {s['label']}/{s['n']}, tokens {s['tokens']:.0f}"
        )
        report.check(
            f"CoT triage on {split}", mine == page["cot"][split], f"page {page['cot'][split]}"
        )


if __name__ == "__main__":
    main()
