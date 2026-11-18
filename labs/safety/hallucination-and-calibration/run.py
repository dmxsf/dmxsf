"""Lab: Calibration: withheld answers, confidence and accuracy.

Wiki pages
  Chinese: https://dmxsf.com/wiki/safety/hallucination-and-calibration/
  English: https://dmxsf.com/en/wiki/safety/hallucination-and-calibration/
"""

import numpy as np

from dmxsf import lab
from dmxsf.evaluation.calibration import binary_nll, binary_probability, selective_rates


def main() -> None:
    args, report = lab.start(
        "Calibration: withheld answers, confidence and accuracy",
        wiki=["safety/hallucination-and-calibration"],
    )
    report.section("1. Use calibration data to choose temperature, then evaluate separately")
    rng = np.random.default_rng(args.seed)
    truth = rng.normal(size=30_000)
    labels = (rng.random(len(truth)) < binary_probability(truth, 1)).astype(int)
    logits = 3 * truth
    calibration, test = slice(0, 15_000), slice(15_000, None)
    temperature = min(
        (1.0, 2.0, 3.0, 4.0), key=lambda t: binary_nll(logits[calibration], labels[calibration], t)
    )
    before = binary_nll(logits[test], labels[test], 1)
    after = binary_nll(logits[test], labels[test], temperature)
    report.check(
        "temperature reduces held-out negative log likelihood and preserves class decisions",
        after < before and np.array_equal(logits[test] > 0, logits[test] / temperature > 0),
        f"T={temperature:.0f}, {before:.3f} -> {after:.3f}",
    )
    report.section("2. Abstain below a confidence threshold")
    prob = binary_probability(logits[test], temperature)
    confidence = np.maximum(prob, 1 - prob)
    correct = (prob >= 0.5) == labels[test]
    full = selective_rates(confidence, correct, 0.5)
    strict = selective_rates(confidence, correct, 0.8)
    report.check(
        "in this synthetic world abstention reduces coverage and improves answered accuracy",
        0 < strict[0] < full[0] == 1 and strict[1] > full[1],
        f"coverage {full[0]:.2%} -> {strict[0]:.2%}, accuracy {full[1]:.2%} -> {strict[1]:.2%}",
    )
    report.check(
        "no retained answers means no conditional accuracy",
        selective_rates(confidence, correct, 1.1) == (0.0, None),
    )
    report.finish()


if __name__ == "__main__":
    main()
