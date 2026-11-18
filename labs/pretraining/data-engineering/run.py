"""Lab: Data engineering: build and audit a dedup filter.

Wiki pages
  Chinese: https://dmxsf.com/wiki/pretraining/data-engineering/
  English: https://dmxsf.com/en/wiki/pretraining/data-engineering/
"""

from dmxsf import lab
from dmxsf.pretraining.dedup import banding_probability, jaccard, shingles


def main() -> None:
    _args, report = lab.start(
        "Data engineering: build and audit a dedup filter", wiki=["pretraining/data-engineering"]
    )
    report.section("1. Exact matching and near duplicates")
    words = "amber bronze coral denim elm flax gold hazel ivory jade khaki lilac"
    dup = words.replace("flax", "fern")
    left, right = shingles(words, 3), shingles(dup, 3)
    overlap = jaccard(left, right)
    report.check(
        "exact-match misses a one-word edit, shingle matching sees seven shared triples",
        words != dup and len(left & right) == 7 and len(left | right) == 13,
        f"Jaccard {overlap:.4f}",
    )
    report.check(
        "disjoint short documents have zero overlap",
        jaccard(left, shingles("violet white yellow", 3)) == 0,
    )
    report.section("2. Read the banding curve")
    similarities = [0.5, 0.75, 0.9, 1.0]
    chances = [banding_probability(s, 14, 8) for s in similarities]
    report.table(
        ["Jaccard", "candidate probability"],
        [[s, f"{p:.2%}"] for s, p in zip(similarities, chances, strict=True)],
    )
    report.check(
        "14 bands of 8 rows favor close pairs but have no hard cutoff",
        0 < chances[0] < chances[1] < chances[2] < chances[3] == 1,
    )
    report.finish()


if __name__ == "__main__":
    main()
