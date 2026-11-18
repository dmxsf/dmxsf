"""Lab: Privacy audit: exposure for synthetic canaries.

Wiki pages
  Chinese: https://dmxsf.com/wiki/safety/memorization-and-privacy/
  English: https://dmxsf.com/en/wiki/safety/memorization-and-privacy/
"""

import math

from dmxsf import lab
from dmxsf.evaluation.privacy import canary_rank, exposure


def main() -> None:
    _args, report = lab.start(
        "Privacy audit: exposure for synthetic canaries", wiki=["safety/memorization-and-privacy"]
    )
    report.section("1. Distinguish score rank from exact output")
    candidate_count = 10_000
    scores = [0.0] * candidate_count
    scores[0], scores[1] = 2.0, 1.0
    rank = canary_rank(scores, 1)
    bits = exposure(candidate_count, rank)
    report.check(
        "the synthetic canary ranks second and has about 12.29 bits of exposure",
        rank == 2 and round(bits, 2) == 12.29 and scores.index(max(scores)) != 1,
        f"rank {rank}, {bits:.2f} bits",
    )
    scores[2] = scores[1]
    report.check("a tie receives the last rank in the tied group", canary_rank(scores, 1) == 3)
    report.section("2. Use a uniform rank as a reference")
    reference = (
        math.fsum(exposure(candidate_count, r) for r in range(1, candidate_count + 1))
        / candidate_count
    )
    report.check(
        "the uniform-rank reference is about 1.44 bits",
        round(reference, 2) == 1.44,
        f"{reference:.4f} bits",
    )
    report.finish()


if __name__ == "__main__":
    main()
