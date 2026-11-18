"""Lab: Interpretability: superposed features and causal patching.

Wiki pages
  Chinese: https://dmxsf.com/wiki/frontier/interpretability/
  English: https://dmxsf.com/en/wiki/frontier/interpretability/
"""

import numpy as np

from dmxsf import lab
from dmxsf.architecture.interpretability import feature_scores, patch_feature


def main() -> None:
    _args, report = lab.start(
        "Interpretability: superposed features and causal patching",
        wiki=["frontier/interpretability"],
    )
    report.section("1. More features than representation dimensions")
    features = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]])
    clean, corrupted = np.array([2.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0])
    single = feature_scores(features, np.array([1.0, 0.0, 0.0]))
    report.check(
        "activating only the first feature creates a 0.6 projection onto the second",
        np.allclose(single, [1.0, 0.6, 0.0]),
        str(single),
    )
    report.section("2. Intervene and compare with an irrelevant control")
    original = feature_scores(features, clean)
    baseline = feature_scores(features, corrupted)
    targeted = feature_scores(features, patch_feature(clean, corrupted, 0))
    control = feature_scores(features, patch_feature(clean, corrupted, 2))
    report.check(
        "patching the removed activation restores the output vector",
        np.array_equal(targeted, original) and not np.array_equal(baseline, original),
        f"{baseline} -> {targeted}",
    )
    report.check(
        "patching an activation unchanged by corruption leaves the baseline intact",
        np.array_equal(control, baseline),
    )
    report.say(
        "The activations are named by construction; a real model would require a separate feature-discovery step."
    )
    report.finish()


if __name__ == "__main__":
    main()
