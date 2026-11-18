"""Lab: Benchmark contamination: detect wording overlap and audit slices.

Wiki pages
  Chinese: https://dmxsf.com/wiki/evaluation/benchmark-contamination/
  English: https://dmxsf.com/en/wiki/evaluation/benchmark-contamination/
"""

from dmxsf import lab
from dmxsf.evaluation.contamination import lines, make_items, ngram_size, ngrams, overlap


def main() -> None:
    args, report = lab.start(
        "Benchmark contamination: detect wording overlap and audit slices",
        wiki=["evaluation/benchmark-contamination"],
    )
    report.section("1. Detect verbatim leaks and inspect paraphrases")
    items = make_items(24, seed=args.seed)
    verbatim, paraphrased, clean = items[:8], items[8:16], items[16:]
    width = ngram_size([item.question for item in items])
    corpus = set()
    for slice_items, form in [(verbatim, "question"), (paraphrased, "paraphrase")]:
        for item in slice_items:
            for text in lines(item, form):
                corpus.update(ngrams(text, width))
    hits = [
        sum(overlap(item.question, corpus, width) > 0 for item in group)
        for group in (verbatim, paraphrased, clean)
    ]
    report.check(
        "the long exact window catches this verbatim slice and misses the paraphrased slice",
        hits == [8, 0, 0],
        f"{width}-gram; hits by slice {hits}",
    )
    report.section("2. Report results by data slice")
    before, after = (4, 4, 4), (8, 6, 4)  # illustrative task scores, not measured model behavior
    report.table(
        ["slice", "count", "baseline correct", "candidate correct"],
        [
            [name, 8, a, b]
            for name, a, b in zip(("verbatim", "paraphrase", "clean"), before, after, strict=True)
        ],
    )
    report.check(
        "the demonstration aggregate gain is 25 points and the clean gain is zero",
        (sum(after) - sum(before)) / len(items) == 0.25 and after[-1] == before[-1],
    )
    report.finish()


if __name__ == "__main__":
    main()
