"""Lab: Test-time compute: independent draws, reranking and majority voting.

Wiki pages
  Chinese: https://dmxsf.com/wiki/frontier/test-time-compute/
  English: https://dmxsf.com/en/wiki/frontier/test-time-compute/
"""

import math

import numpy as np

from dmxsf import lab
from dmxsf.evaluation.test_time import at_least_one, binary_majority


def main() -> None:
    args, report = lab.start(
        "Test-time compute: independent draws, reranking and majority voting",
        wiki=["frontier/test-time-compute"],
    )
    report.section("1. More samples only help when the draws add information")
    p, count = 0.3, 5
    oracle, majority = at_least_one(p, count), binary_majority(p, count)
    correlated = p
    report.check(
        "independent oracle coverage is 0.83193, binary majority is 0.16308",
        math.isclose(oracle, 0.83193) and math.isclose(majority, 0.16308),
        f"oracle {oracle:.5f}, majority {majority:.5f}",
    )
    rng = np.random.default_rng(args.seed)
    first = rng.random((30_000, 1)) < p
    shared = np.broadcast_to(first, (len(first), count))
    report.check(
        "repeating the same draw leaves coverage at the single-draw rate",
        np.array_equal(shared.any(axis=1), first[:, 0])
        and abs(shared.any(axis=1).mean() - correlated) < 0.02,
    )
    report.section("2. A noisy reranker loses attainable coverage")
    draws = rng.random((30_000, count)) < p
    selector = draws + rng.normal(0, 0.8, draws.shape)
    chosen = draws[np.arange(len(draws)), selector.argmax(axis=1)]
    report.check(
        "the selector helps over one sample and falls short of the oracle",
        p < chosen.mean() < draws.any(axis=1).mean(),
        f"reranked {chosen.mean():.3f}; oracle {draws.any(axis=1).mean():.3f}",
    )
    report.finish()


if __name__ == "__main__":
    main()
