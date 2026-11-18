"""Lab: Neural networks: gradients and XOR.

Wiki pages
  Chinese: https://dmxsf.com/wiki/foundations/neural-network-basics/
  English: https://dmxsf.com/en/wiki/foundations/neural-network-basics/
"""

import numpy as np

from dmxsf import lab
from dmxsf.foundations.neural_network import loss_and_gradients, xor_fit


def main() -> None:
    args, report = lab.start(
        "Neural networks: gradients and XOR", wiki=["foundations/neural-network-basics"]
    )
    report.section("1. Backpropagation by finite differences")
    rng = np.random.default_rng(args.seed)
    x, w1, w2 = rng.normal(size=(5, 3)), rng.normal(size=(3, 4)), rng.normal(size=(4, 2))
    labels = np.array([0, 1, 1, 0, 1])
    loss, (g1, g2) = loss_and_gradients(x, labels, w1, w2)
    gaps = []
    for weights, gradient in [(w1, g1), (w2, g2)]:
        for index in np.ndindex(weights.shape):
            original = weights[index]
            weights[index] = original + 1e-5
            plus = loss_and_gradients(x, labels, w1, w2)[0]
            weights[index] = original - 1e-5
            minus = loss_and_gradients(x, labels, w1, w2)[0]
            weights[index] = original
            gaps.append(abs((plus - minus) / 2e-5 - gradient[index]))
    report.check(
        "every weight agrees with finite differences",
        max(gaps) < 1e-8,
        f"loss {loss:.4f}, max gap {max(gaps):.1e}",
    )
    report.section("2. What the nonlinearity changes")
    linear = xor_fit(args.seed, 0)
    hidden = [xor_fit(args.seed + offset, 4) for offset in range(3)]
    report.check(
        "a linear score cannot classify all four XOR points", linear[1] < 4, f"{linear[1]}/4"
    )
    report.check(
        "tanh can learn all four points",
        any(hits == 4 and loss < 0.1 for loss, hits in hidden),
        str([(round(loss, 4), hits) for loss, hits in hidden]),
    )
    report.finish()


if __name__ == "__main__":
    main()
