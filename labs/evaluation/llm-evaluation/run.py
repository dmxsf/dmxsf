"""Lab: benchmark contamination. What an n-gram detector catches, what it misses, and why it matters.

Course page: https://dmxsf.com/wiki/evaluation/llm-evaluation/

Part one is the page's detector, run on its own example. Part two builds a benchmark whose answers
are random three-digit codes, so a model can know an answer only if it has seen it. A small
byte-level model is then trained on a stream in which a third of the benchmark appears verbatim,
a third appears only as paraphrases, and a third not at all. We ask every item and see which of
them the detector flags, how much each kind of leak inflates the score, and what the GPT-3 style
"re-evaluate on the clean subset" check would conclude.

    uv run python labs/evaluation/llm-evaluation/run.py [--quick] [--seed N]

The full run trains two models of 1,000 steps each (about 1.5 minutes per model on a laptop CPU).
"""

from __future__ import annotations

import copy

import numpy as np

from dmxsf import lab, models
from dmxsf.evaluation import contamination as C
from dmxsf.evaluation.stats import half_width

GROUPS = ("never leaked", "leaked verbatim", "leaked as paraphrases")
BITS = np.log(2)


def main() -> None:
    args, report = lab.start(
        "Benchmark contamination: an n-gram detector, and a leak you can measure",
        wiki=["evaluation/llm-evaluation"],
    )
    QUICK = args.quick

    def full_check(name: str, ok: bool, detail: str = "") -> None:
        if QUICK:
            report.skip(name, "needs the full run")
        else:
            report.check(name, ok, detail)

    def full_observe(name: str, ok: bool, detail: str = "") -> None:
        if QUICK:
            report.skip(name, "needs the full run")
        else:
            report.observe(name, ok, detail)

    # -----------------------------------------------------------------------------------------
    report.section("1. The page's detector on its own example")
    question = (
        "Janet has 3 apples, buys 5 more, and then gives 2 to her friend. "
        "How many apples does Janet have left?"
    )
    rephrased = (
        "After buying five more apples to add to her original three, Janet hands two over to a "
        "friend. What is the number of apples she still holds?"
    )
    unrelated = (
        "A train leaves the station at 9 am and travels at 80 km per hour. "
        "How far has it gone by noon?"
    )
    corpus = (
        "Practice problems for grade school students. Problem 7: "
        + question
        + " Answer: 6. Problem 8: ..."
    )
    grams = C.ngrams(corpus, 13)
    hits = {}
    for name, text in [("verbatim", question), ("rephrased", rephrased), ("unrelated", unrelated)]:
        hits[name] = C.overlap(text, grams, 13)
        report.say(
            f"  {name:<9}  13-gram hits {hits[name]:2d}  ->  {'dirty' if hits[name] else 'clean'}"
        )
    report.check(
        "9 hits for the verbatim question; the rephrased one, same meaning, has none",
        list(hits.values()) == [9, 0, 0],
    )

    # -----------------------------------------------------------------------------------------
    report.section("2. How big is a benchmark's error bar?")
    report.table(
        ["benchmark", "questions", "worst-case 95% half-width (accuracy 50%)"],
        [
            [name, size, f"+-{100 * half_width(0.5, size):.1f} points"]
            for name, size in [("HumanEval", 164), ("GSM8K", 1319), ("MMLU", 14079)]
        ],
    )
    report.say(
        "  (the formula and the paired comparison are in the eval-statistics lab; keep these\n"
        "   widths in mind when reading the table below, which has 48 questions per group)"
    )

    # -----------------------------------------------------------------------------------------
    report.section("3. A benchmark whose answers can only be memorised")
    per_group = 12 if QUICK else 48
    steps = 60 if QUICK else 1000
    seeds = [args.seed] if QUICK else [args.seed, args.seed + 1]
    sample = C.make_items(1, seed=args.seed)[0]
    report.say(f"  benchmark wording : {sample.question}")
    report.say(f"  one paraphrase    : {sample.paraphrases[0]}")
    report.say(f"  training line     : {C.lines(sample, 'paraphrase')[0].rstrip()}")
    report.say("  the answer is three random digits, so a guess is right 1 time in 1,000.\n")

    all_items = C.make_items(3 * per_group, seed=args.seed)
    n_gram = C.ngram_size([item.question for item in all_items])
    report.check(
        "the GPT-3 recipe (5th percentile of the question length, between 8 and 13) gives N = 13",
        n_gram == 13,
        f"{len(C.words(sample.question))} words per question",
    )

    def flagged_rate(items: list[C.Item], leak_text: str, n: int) -> float:
        corpus = C.ngrams(leak_text, n)
        return float(np.mean([C.overlap(item.question, corpus, n) > 0 for item in items]))

    def groups_of(items: list[C.Item]) -> list[list[C.Item]]:
        return [items[:per_group], items[per_group : 2 * per_group], items[2 * per_group :]]

    rows, rates = [], {}
    for seed in seeds:
        held, verbatim, paraphrased = groups_of(C.make_items(3 * per_group, seed=seed))
        text = "".join(
            [ln for i in verbatim for ln in C.lines(i, "question")]
            + [ln for i in paraphrased for ln in C.lines(i, "paraphrase")]
        )
        for n in (13, 8):
            rates[seed, n] = [flagged_rate(g, text, n) for g in (held, verbatim, paraphrased)]
    for n in (13, 8):
        r = rates[seeds[0], n]
        rows.append([f"N = {n}"] + [f"{100 * x:.0f}%" for x in r])
    report.say("  Share of benchmark questions the detector flags as dirty:")
    report.table(["detector", *GROUPS], rows)
    report.check(
        "N = 13 flags exactly the verbatim leak: the paraphrases pass as clean",
        all(rates[s, 13] == [0.0, 1.0, 0.0] for s in seeds),
    )
    report.check(
        "N = 8 flags everything: the benchmark's own template ('in the north wing of the old "
        "archive') is in the leaked lines",
        all(rates[s, 8] == [1.0, 1.0, 1.0] for s in seeds),
    )

    # -----------------------------------------------------------------------------------------
    report.section("4. Train with the leak, then ask the benchmark")
    base = models.pretrained("base", quick=QUICK, seed=args.seed)
    results = []  # per seed: {group: (n, 2) array}, format results
    formats = {
        "as in training": C.prompt,
        "'Answer:' instead of 'A:'": lambda i: f"{i.question} Answer: ",
        "'Question:' in front": lambda i: f"Question: {i.question} A: ",
        "no space after 'A:'": lambda i: f"{i.question} A:",
    }
    for seed in seeds:
        held, verbatim, paraphrased = groups_of(C.make_items(3 * per_group, seed=seed))
        lines = [ln for i in verbatim for ln in C.lines(i, "question")] + [
            ln for i in paraphrased for ln in C.lines(i, "paraphrase")
        ]
        model = copy.deepcopy(base)
        report.say(f"  seed {seed}: training {steps} steps on {len(lines)} leaked lines ...")
        C.train_with_leak(model, lines, steps, seed=seed)
        scored = [C.answer_nats(model, g) for g in (held, verbatim, paraphrased)]
        by_format = {
            name: float(C.answer_nats(model, verbatim, ask)[:, 1].mean())
            for name, ask in formats.items()
        }
        results.append(
            {"scored": scored, "formats": by_format, "groups": (held, verbatim, paraphrased)}
        )

    report.say(
        "\n  Exact match of all three digits, and the loss per digit in bits (guessing = 3.32):"
    )
    rows = []
    for seed, res in zip(seeds, results, strict=True):
        for name, s in zip(GROUPS, res["scored"], strict=True):
            rows.append(
                [seed, name, f"{100 * s[:, 1].mean():.1f}%", f"{s[:, 0].mean() / BITS:.2f}"]
            )
    report.table(["seed", "group", "exact match", "bits per digit"], rows)
    acc = [[float(s[:, 1].mean()) for s in res["scored"]] for res in results]
    # How much a 1,000-step run memorises, and how far a paraphrase carries, moves with the seed and
    # the machine (verbatim 62% to 100%, paraphrases 0% to 40% where this was tried): observations.
    full_observe(
        "the verbatim leak is almost fully retained (80% or more exact answers) in every run",
        all(a[1] >= 0.8 for a in acc),
        ", ".join(f"{100 * a[1]:.0f}%" for a in acc),
    )
    full_check(
        "questions never leaked are answered at chance level (at most 2% exact)",
        all(a[0] <= 0.02 for a in acc),
        ", ".join(f"{100 * a[0]:.1f}%" for a in acc),
    )
    full_observe(
        "paraphrased leaks transfer to the benchmark wording: at least 5 points above never-leaked",
        all(a[2] >= a[0] + 0.05 for a in acc),
        ", ".join(f"{100 * a[2]:.1f}% against {100 * a[0]:.1f}%" for a in acc),
    )
    full_check(
        "the paraphrase leak is weaker than the verbatim one",
        all(a[2] < a[1] for a in acc),
    )
    if not QUICK and not all(a[2] >= a[0] + 0.05 for a in acc):
        report.say(
            "  A paraphrase leak did not carry over in every run: whether a model that saw only\n"
            "  paraphrases answers the benchmark's own wording depends on the run."
        )

    # -----------------------------------------------------------------------------------------
    report.section("5. The clean-subset check, applied to this benchmark")
    rows, clean_accs, full_accs = [], [], []
    for seed, res in zip(seeds, results, strict=True):
        held, verbatim, paraphrased = res["groups"]
        everything = held + verbatim + paraphrased
        scored = np.concatenate(res["scored"])
        text = "".join(
            [ln for i in verbatim for ln in C.lines(i, "question")]
            + [ln for i in paraphrased for ln in C.lines(i, "paraphrase")]
        )
        corpus = C.ngrams(text, n_gram)
        dirty = np.array([C.overlap(i.question, corpus, n_gram) > 0 for i in everything])
        full, flagged, clean = (
            float(scored[:, 1].mean()),
            float(scored[dirty, 1].mean()),
            float(scored[~dirty, 1].mean()),
        )
        clean_accs.append(clean)
        full_accs.append(full)
        rows.append(
            [seed, f"{100 * full:.1f}%", f"{100 * flagged:.1f}%", f"{100 * clean:.1f}%", "0.1%"]
        )
    report.table(
        ["seed", "full benchmark", "flagged dirty", "'clean' subset", "pure guessing"], rows
    )
    if not QUICK:
        report.say(
            "  The clean subset scores below the full benchmark, so the check does notice the\n"
            f"  contamination. But it is still {min(clean_accs) / 0.001:.0f} to {max(clean_accs) / 0.001:.0f} times above "
            "guessing: the paraphrased leak is in there,\n"
            "  labelled clean. Here we know the guessing rate; on a real benchmark you do not."
        )
    full_check(
        "the 'clean' subset scores below the full benchmark: the check notices the contamination",
        all(c < f for c, f in zip(clean_accs, full_accs, strict=True)),
        ", ".join(
            f"{100 * c:.1f}% against {100 * f:.1f}%"
            for c, f in zip(clean_accs, full_accs, strict=True)
        ),
    )
    full_observe(
        "...yet it is at least 10 times above guessing (the paraphrased leak is in it)",
        all(c >= 0.01 for c in clean_accs),
        ", ".join(f"{100 * c:.1f}%" for c in clean_accs),
    )

    # -----------------------------------------------------------------------------------------
    report.section("6. The pattern of the scores: where a leak is fragile")
    names = list(formats)
    rows = [[name] + [f"{100 * res['formats'][name]:.1f}%" for res in results] for name in names]
    report.say("  Exact match on the verbatim-leaked questions, asked in different formats:")
    report.table(["prompt", *[f"seed {s}" for s in seeds]], rows)
    kept = {
        name: [res["formats"][name] / res["formats"]["as in training"] for res in results]
        for name in ("'Answer:' instead of 'A:'", "no space after 'A:'")
    }
    # seen: dropping the space leaves 0% of the leaked score on every model; 'Answer:' leaves 12% to 58%
    # (the model keys on the string 'A:', how much a nearby marker still helps varies)
    full_check(
        "dropping the space after 'A:' cuts the leaked score to less than half",
        all(k < 0.5 for k in kept["no space after 'A:'"]),
        ", ".join(f"{100 * k:.0f}% kept" for k in kept["no space after 'A:'"]),
    )
    full_observe(
        "...and so does writing 'Answer:' instead of 'A:'",
        all(k < 0.5 for k in kept["'Answer:' instead of 'A:'"]),
        ", ".join(f"{100 * k:.0f}% kept" for k in kept["'Answer:' instead of 'A:'"]),
    )
    full_check(
        "adding text in front of the question changes almost nothing (within 20%)",
        all(
            res["formats"]["'Question:' in front"] > 0.8 * res["formats"]["as in training"]
            for res in results
        ),
    )
    if not QUICK:
        report.say(
            "\n  A score that collapses when 'A:' becomes 'Answer:', or when a space goes missing, is the\n"
            "  memory of a string rather than the ability to answer. But the same model does not mind a\n"
            "  new prefix: it keys on what comes right before the answer. Format sensitivity is an\n"
            "  indirect sign, and a probe has to touch the right place to show it."
        )
    report.finish()


if __name__ == "__main__":
    main()
