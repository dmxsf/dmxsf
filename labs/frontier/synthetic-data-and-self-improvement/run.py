"""Lab: Synthetic data: rare-event loss and verified selection.

Wiki pages
  Chinese: https://dmxsf.com/wiki/frontier/synthetic-data-and-self-improvement/
  English: https://dmxsf.com/en/wiki/frontier/synthetic-data-and-self-improvement/
"""

import itertools

import numpy as np

from dmxsf import lab
from dmxsf.pretraining.synthetic import missing_mass, replacement_chain, verify_additions


def main() -> None:
    args, report = lab.start(
        "Synthetic data: rare-event loss and verified selection",
        wiki=["frontier/synthetic-data-and-self-improvement"],
    )
    report.section("1. Once a category disappears, pure replacement cannot rediscover it")
    p = np.array([0.8, 0.15, 0.05])
    first = missing_mass(p, 20)
    rng = np.random.default_rng(args.seed)
    chains = [replacement_chain(p, 20, 6, rng) for _ in range(1500)]
    observed = np.mean([p @ (history[1] == 0) for history in chains])
    report.check(
        "first-generation missing mass matches the exact occupancy formula",
        abs(observed - first) < 0.005,
        f"{observed:.4f} vs {first:.4f}",
    )
    report.check(
        "support never grows along a replacement chain",
        all(
            all(
                not np.any((left == 0) & (right > 0)) for left, right in itertools.pairwise(history)
            )
            for history in chains
        ),
    )
    report.section("2. Inject a correctness signal")
    examples = [(2, 5, 7), (4, 3, 4), (1, 8, 9), (7, 4, 9)]
    accepted = verify_additions(examples)
    report.check(
        "independent arithmetic verification retains only correctly labelled examples",
        accepted == [examples[0], examples[2]],
        f"{len(accepted)}/{len(examples)} accepted",
    )
    report.finish()


if __name__ == "__main__":
    main()
