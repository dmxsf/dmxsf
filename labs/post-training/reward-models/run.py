"""Lab: Reward models: pairwise scores and label noise.

Wiki pages
  Chinese: https://dmxsf.com/wiki/post-training/reward-models/
  English: https://dmxsf.com/en/wiki/post-training/reward-models/
"""

import numpy as np

from dmxsf import lab
from dmxsf.post_training.reward_models import fit_bradley_terry, label_noise_rates


def main() -> None:
    args, report = lab.start(
        "Reward models: pairwise scores and label noise", wiki=["post-training/reward-models"]
    )
    report.section("1. Recover relative scores from pairwise choices")
    truth = np.array([-1.0, 0.0, 1.0])
    difference = truth[:, None] - truth[None, :]
    wins = 1000 / (1 + np.exp(-difference))
    np.fill_diagonal(wins, 0)
    fitted = fit_bradley_terry(wins)
    gap = np.max(np.abs(fitted - truth))
    report.check(
        "relative scores are recovered up to a single global shift",
        gap < 1e-8 and abs(fitted.sum()) < 1e-12,
        f"max gap {gap:.1e}",
    )
    report.section("2. Independent symmetric label flips")
    ceiling, agreement = label_noise_rates(0.1)
    rng = np.random.default_rng(args.seed)
    flips = rng.random((2, 40_000)) < 0.1
    observed = np.mean(flips[0] == flips[1])
    report.check(
        "10% flips yield 90% single-label ceiling and 82% two-label agreement",
        abs(ceiling - 0.90) < 1e-12
        and abs(agreement - 0.82) < 1e-12
        and abs(observed - agreement) < 0.01,
        f"observed agreement {observed:.3f}",
    )
    report.finish()


if __name__ == "__main__":
    main()
