"""Lab: Diffusion and flow matching: verify a Gaussian path.

Wiki pages
  Chinese: https://dmxsf.com/wiki/multimodal/diffusion-and-flow-matching/
  English: https://dmxsf.com/en/wiki/multimodal/diffusion-and-flow-matching/
"""

import numpy as np

from dmxsf import lab
from dmxsf.multimodal.diffusion_flow import forward_noise, gaussian_flow


def main() -> None:
    args, report = lab.start(
        "Diffusion and flow matching: verify a Gaussian path",
        wiki=["multimodal/diffusion-and-flow-matching"],
    )
    report.section("1. The variance of a closed-form noising step")
    rng = np.random.default_rng(args.seed)
    clean = rng.normal(0, 0.8, 80_000)
    noise = rng.normal(size=len(clean))
    observed = forward_noise(clean, noise, 0.25).var()
    expected = 0.25 * 0.8**2 + 0.75
    report.check(
        "the measured variance matches the analytical 0.91",
        abs(observed - expected) < 0.02,
        f"{observed:.4f} vs {expected:.4f}",
    )
    report.check(
        "the noise schedule endpoints return the original signal and pure noise",
        np.array_equal(forward_noise(clean, noise, 1), clean)
        and np.array_equal(forward_noise(clean, noise, 0), noise),
    )
    report.section("2. More Euler evaluations reduce discretization error")
    basis = np.array([-1.0, 1.0])  # exact unit standard deviation without sampling noise
    steps = (8, 16, 32, 64)
    measures = [gaussian_flow(basis, 0.8, n).std() for n in steps]
    errors = [abs(v - 0.8) for v in measures]
    report.table(
        ["Euler steps", "output std", "absolute error"],
        [[n, f"{s:.5f}", f"{e:.5f}"] for n, s, e in zip(steps, measures, errors, strict=True)],
    )
    report.check(
        "for this flow, Euler estimates converge toward target std 0.8",
        errors[0] > errors[1] > errors[2] > errors[3] and errors[-1] < 0.02,
    )
    report.finish()


if __name__ == "__main__":
    main()
