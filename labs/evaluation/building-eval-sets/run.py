"""Lab: Evaluation sets: noisy labels, agreement and item information.

Wiki pages
  Chinese: https://dmxsf.com/wiki/evaluation/building-eval-sets/
  English: https://dmxsf.com/en/wiki/evaluation/building-eval-sets/
"""

import numpy as np

from dmxsf import lab
from dmxsf.evaluation.eval_sets import item_information, noisy_accuracy
from dmxsf.evaluation.judge import cohens_kappa


def main() -> None:
    args, report = lab.start(
        "Evaluation sets: noisy labels, agreement and item information",
        wiki=["evaluation/building-eval-sets"],
    )
    report.section("1. What an independently wrong label does to scores")
    rng = np.random.default_rng(args.seed)
    n = 80_000
    model_wrong = rng.random(n) >= 0.8
    label_wrong = rng.random(n) < 0.1
    model = np.where(model_wrong, rng.integers(1, 4, size=n), 0)
    label = np.where(label_wrong, rng.integers(1, 4, size=n), 0)
    predicted = noisy_accuracy(0.8, 0.1, 4)
    measured = np.mean(model == label)
    report.check(
        "the simulation reproduces predicted agreement near 72.7%",
        abs(predicted - measured) < 0.01 and round(predicted, 3) == 0.727,
        f"{measured:.3f} vs {predicted:.3f}",
    )
    shrink = 1 - 4 * 0.0649 / 3
    report.check(
        "the article's 6.49% label error requires about 1.198x as many items",
        round(shrink**-2, 3) == 1.198,
        f"{shrink**-2:.3f}",
    )
    report.section("2. Agreement and information")
    observed, chance, kappa = cohens_kappa(np.array([[45, 5], [5, 45]]))
    report.check(
        "90% raw agreement with balanced marginals gives kappa 0.8",
        np.allclose([observed, chance, kappa], [0.9, 0.5, 0.8]),
    )
    report.check(
        "the 2PL information peaks at difficulty with value discrimination squared over four",
        item_information(0.8, 0.8, 2) == 1 and item_information(2.0, 0.8, 2) < 1,
    )
    report.finish()


if __name__ == "__main__":
    main()
